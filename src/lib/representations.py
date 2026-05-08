from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.data_typing import Sample


def load_model(model_name: str, device: str, quantization: str | None = None):
    """Load a causal LM and its tokenizer onto *device*.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"google/gemma-2-2b"``).
    device : str
        Target device string (``"cpu"``, ``"cuda"``, etc.).
    quantization : str | None
        ``"4bit"``, ``"8bit"``, or ``None`` (default bfloat16).
        4/8-bit require ``bitsandbytes``.

    Returns
    -------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    """
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def extract_activations_multilayer(
    samples: list[Sample],
    model,
    tokenizer,
    layer_indices: list[int],
    device: str,
    batch_size: int = 8,
) -> dict[tuple[str, str, int], torch.Tensor]:
    """Extract mean content-token activations at multiple transformer layers in one pass.

    Runs prompts through *model* in batches with ``output_hidden_states=True`` so
    every requested layer is captured per forward pass. For each ``(trait,
    intensity, layer)`` group, mean-pools over content tokens (excluding BOS,
    EOS, and pad tokens) and averages across prompts in the group.

    Parameters
    ----------
    samples : list[Sample]
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    layer_indices : list[int]
        Indices into ``model.model.layers``.
    device : str
    batch_size : int

    Returns
    -------
    dict[tuple[str, str, str, int], torch.Tensor]
        Maps ``(trait, intensity, scenario_id, layer)`` to a mean activation vector
        ``(hidden_dim,)``. Samples sharing the same scenario_id+intensity are averaged.
    """
    layers = sorted(set(layer_indices))
    special_ids = {
        tid for tid in (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        if tid is not None
    }

    bucket: dict[tuple[str, str, str, int], list[torch.Tensor]] = {}

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        texts = [s.prompt for s in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple length num_layers+1

        token_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"].bool()
        valid = attention_mask.clone()
        for sid in special_ids:
            valid &= token_ids != sid
        empty_rows = valid.sum(dim=1) == 0
        if empty_rows.any():
            valid[empty_rows] = attention_mask[empty_rows]

        weights = valid.to(hidden_states[0].dtype).unsqueeze(-1)  # (B, L, 1)
        counts = weights.sum(dim=1).clamp(min=1)  # (B, 1)

        # Pool every requested layer on-device, stack, then a single cross-device copy.
        per_layer = [
            (hidden_states[layer + 1] * weights).sum(dim=1) / counts
            for layer in layers
        ]
        stacked = torch.stack(per_layer, dim=0).detach().to("cpu", dtype=torch.float32)
        # stacked: (num_layers, B, D)

        for li, layer in enumerate(layers):
            for i, sample in enumerate(batch):
                key = (sample.trait, sample.intensity, sample.scenario_id, layer)
                bucket.setdefault(key, []).append(stacked[li, i])

    return {k: torch.stack(v).mean(dim=0) for k, v in bucket.items()}


def extract_activations(
    samples: list[Sample],
    model,
    tokenizer,
    layer_index: int,
    device: str,
) -> dict[tuple[str, str], torch.Tensor]:
    """Mean content-token activations at a single layer.

    Thin wrapper over :func:`extract_activations_multilayer` for one layer.
    Returns a dict keyed by ``(trait, intensity)`` (no layer index).
    """
    multi = extract_activations_multilayer(
        samples, model, tokenizer, [layer_index], device
    )
    return {(t, i, s): vec for (t, i, s, _), vec in multi.items()}


def extract_activations_last_token_chat_multilayer(
    samples: list[Sample],
    model,
    tokenizer,
    layer_indices: list[int],
    device: str,
    batch_size: int = 8,
    user_instruction: str = "Rate the politeness of the following message:",
) -> dict[tuple[str, str, str, int], torch.Tensor]:
    """Wrap each paraphrase as the assistant turn of a chat and read the
    last-token hidden state at each requested layer.

    If the tokenizer exposes a ``chat_template``, it is used; otherwise a
    minimal "User: ...\\nAssistant: ..." formatting is applied. The last-token
    activation is the residual stream right after the model has integrated the
    full assistant turn — the standard probing site in representation
    engineering work, and typically more informative than mean-pooled content
    tokens for trait-style probes.
    """
    layers = sorted(set(layer_indices))
    has_template = bool(getattr(tokenizer, "chat_template", None))

    def _render(text: str) -> str:
        if has_template:
            msgs = [
                {"role": "user", "content": user_instruction},
                {"role": "assistant", "content": text},
            ]
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        return f"User: {user_instruction}\nAssistant: {text}"

    bucket: dict[tuple[str, str, str, int], list[torch.Tensor]] = {}
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        texts = [_render(s.prompt) for s in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple length num_layers+1

        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1  # (B,)
        b_idx = torch.arange(attn.size(0), device=device)

        per_layer = [hidden_states[layer + 1][b_idx, last_idx, :] for layer in layers]
        stacked = torch.stack(per_layer, dim=0).detach().to("cpu", dtype=torch.float32)

        for li, layer in enumerate(layers):
            for i, sample in enumerate(batch):
                key = (sample.trait, sample.intensity, sample.scenario_id, layer)
                bucket.setdefault(key, []).append(stacked[li, i])

    return {k: torch.stack(v).mean(dim=0) for k, v in bucket.items()}


def extract_activations_last_token_chat(
    samples: list[Sample],
    model,
    tokenizer,
    layer_index: int,
    device: str,
    batch_size: int = 8,
    user_instruction: str = "Rate the politeness of the following message:",
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Single-layer wrapper for :func:`extract_activations_last_token_chat_multilayer`."""
    multi = extract_activations_last_token_chat_multilayer(
        samples, model, tokenizer, [layer_index], device,
        batch_size=batch_size, user_instruction=user_instruction,
    )
    return {(t, i, s): vec for (t, i, s, _), vec in multi.items()}


def save_activations(
    activations: dict[tuple[str, str, str], torch.Tensor],
    out_dir: Path,
) -> None:
    """Save each activation tensor to ``out_dir/<trait>__<intensity>__<scenario_id>.pt``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (trait, intensity, scenario_id), tensor in activations.items():
        torch.save(tensor, out_dir / f"{trait}__{intensity}__{scenario_id}.pt")


def load_activations(act_dir: Path) -> dict[tuple[str, str, str], torch.Tensor]:
    """Load activation tensors saved by :func:`save_activations`.

    Parameters
    ----------
    act_dir : Path
        Directory containing ``<trait>__<intensity>__<scenario_id>.pt`` files.

    Returns
    -------
    dict[tuple[str, str, str], torch.Tensor]
        Maps ``(trait, intensity, scenario_id)`` to its activation vector.
    """
    result = {}
    for path in sorted(act_dir.glob("*.pt")):
        trait, intensity, scenario_id = path.stem.split("__")
        result[(trait, intensity, scenario_id)] = torch.load(path, weights_only=True)
    return result

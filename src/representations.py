from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import Sample


def load_model(model_name: str, device: str):
    """Load a causal LM and its tokenizer onto *device*.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"google/gemma-2-2b"``).
    device : str
        Target device string (``"cpu"``, ``"cuda"``, etc.).

    Returns
    -------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval().to(device)
    return model, tokenizer


def extract_activations(
    samples: list[Sample],
    model,
    tokenizer,
    layer_index: int,
    device: str,
) -> dict[tuple[str, str], torch.Tensor]:
    """Extract mean last-token activations at a given transformer layer.

    For each ``(trait, intensity)`` group, runs all corresponding prompts through
    *model*, captures the hidden state at *layer_index*, takes the last-token vector,
    and returns the mean across prompts in that group.

    Parameters
    ----------
    samples : list[Sample]
        Prompts with trait/intensity labels.
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    layer_index : int
        Index into ``model.model.layers``.
    device : str

    Returns
    -------
    dict[tuple[str, str], torch.Tensor]
        Maps ``(trait, intensity)`` to a mean activation vector of shape ``(hidden_dim,)``.
    """
    bucket: dict[tuple[str, str], list[torch.Tensor]] = {}
    captured: list[torch.Tensor] = []

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().cpu())

    layer = model.model.layers[layer_index]

    for sample in samples:
        captured.clear()
        handle = layer.register_forward_hook(hook)
        inputs = tokenizer(sample.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        last_token = captured[0][0, -1, :]  # (hidden_dim,)
        bucket.setdefault((sample.trait, sample.intensity), []).append(last_token)

    return {k: torch.stack(v).mean(dim=0) for k, v in bucket.items()}


def save_activations(
    activations: dict[tuple[str, str], torch.Tensor],
    out_dir: Path,
) -> None:
    """Save each activation tensor to ``out_dir/<trait>__<intensity>.pt``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (trait, intensity), tensor in activations.items():
        torch.save(tensor, out_dir / f"{trait}__{intensity}.pt")


def load_activations(act_dir: Path) -> dict[tuple[str, str], torch.Tensor]:
    """Load activation tensors saved by :func:`save_activations`.

    Parameters
    ----------
    act_dir : Path
        Directory containing ``<trait>__<intensity>.pt`` files.

    Returns
    -------
    dict[tuple[str, str], torch.Tensor]
        Maps ``(trait, intensity)`` to its activation vector.
    """
    result = {}
    for path in sorted(act_dir.glob("*.pt")):
        trait, intensity = path.stem.split("__")
        result[(trait, intensity)] = torch.load(path, weights_only=True)
    return result

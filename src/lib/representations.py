import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.sentences import Sample, load_accepted


def load_model(model_name: str, device: str, quantization: str | None = None):
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


TOKEN_POOLS = ("avg", "last")


def _reduce(hidden: torch.Tensor, valid: torch.Tensor, token_pooling: str) -> torch.Tensor:
    # hidden: (B, L, D); valid: (B, L) bool content-token mask. Returns (B, D).
    if token_pooling == "avg":
        w = valid.unsqueeze(-1).to(hidden.dtype)
        return (hidden * w).sum(1) / w.sum(1).clamp(min=1)
    if token_pooling == "last":
        rows = torch.arange(hidden.size(0), device=hidden.device)
        last_idx = valid.size(1) - 1 - valid.flip(1).int().argmax(1)
        return hidden[rows, last_idx, :]
    raise ValueError(f"token_pooling must be one of {TOKEN_POOLS}, got {token_pooling!r}")


def extract_activations(
    samples: list[Sample],
    model,
    tokenizer,
    layers: list[int],
    device: str,
    *,
    token_poolings: tuple[str, ...] = ("avg",),
    batch_size: int = 8,
) -> dict[str, dict[tuple[str, str, str, str, int], torch.Tensor]]:
    """Activations per pooling at each requested layer.

    Returns ``{token_pooling: {(trait, intensity, scenario_id, paraphrase_id, layer): vec}}``.
    Activations come from a single forward pass over the prompt — the **prefill phase
    only**; no tokens are generated. Each pooling reduces over the prompt's tokens:
    ``'avg'`` averages all content tokens (excluding BOS/EOS/pad), ``'last'`` takes
    the prompt's final content token. One vector per paraphrase; nothing is averaged here.
    All requested poolings share the same forward pass.
    """
    for p in token_poolings:
        if p not in TOKEN_POOLS:
            raise ValueError(f"token_pooling must be one of {TOKEN_POOLS}, got {p!r}")
    layers = sorted(set(layers))
    special = {
        t
        for t in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        if t is not None
    }

    out: dict[str, dict[tuple[str, str, str, str, int], torch.Tensor]] = {p: {} for p in token_poolings}
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        enc = tokenizer([s.prompt for s in batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hidden = model(**enc, output_hidden_states=True).hidden_states

        mask = enc["attention_mask"].bool()
        valid = mask.clone()
        for sid in special:
            valid &= enc["input_ids"] != sid
        empty = valid.sum(1) == 0
        valid[empty] = mask[empty]

        for layer in layers:
            for p in token_poolings:
                red = _reduce(hidden[layer + 1], valid, p).detach().to("cpu", dtype=torch.float32)
                for i, s in enumerate(batch):
                    key = (s.trait, s.intensity, s.scenario_id, s.paraphrase_id, layer)
                    if key in out[p]:
                        raise ValueError(f"duplicate paraphrase_id encountered: {key}")
                    out[p][key] = red[i]

    return out


def save_representations(
    activations: dict[tuple[str, str, str, str, int], torch.Tensor],
    out_dir: str | Path,
    *,
    meta: dict,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[int, dict[tuple[str, str, str, str], torch.Tensor]] = {}
    for (trait, intensity, scenario_id, paraphrase_id, layer), vec in activations.items():
        by_layer.setdefault(layer, {})[(trait, intensity, scenario_id, paraphrase_id)] = vec
    for layer, vecs in by_layer.items():
        torch.save(vecs, out_dir / f"layer_{layer}.pt")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_representations(rep_dir: str | Path, *, layer: int | None = None):
    """One vector per paraphrase.

    ``layer`` given → ``{(trait, intensity, scenario_id, paraphrase_id): vec}``;
    ``layer`` None → adds the layer to the key.
    """
    rep_dir = Path(rep_dir)
    if layer is not None:
        return torch.load(rep_dir / f"layer_{layer}.pt", weights_only=False)
    out: dict[tuple[str, str, str, str, int], torch.Tensor] = {}
    for path in sorted(rep_dir.glob("layer_*.pt")):
        n = int(path.stem.split("_")[1])
        for key, vec in torch.load(path, weights_only=False).items():
            out[(*key, n)] = vec
    return out


def pool_by_scenario_level(
    activations: dict[tuple[str, str, str, str], torch.Tensor],
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Mean paraphrase vectors per ``(trait, intensity, scenario_id)``.

    Recovers the scenario-centroid view from the paraphrase-level store; the
    analysis helpers in :mod:`lib.analysis` (e.g. ``within_center``) expect this
    3-tuple key shape.
    """
    bucket: dict[tuple[str, str, str], list[torch.Tensor]] = {}
    for (trait, intensity, scenario_id, _paraphrase_id), vec in activations.items():
        bucket.setdefault((trait, intensity, scenario_id), []).append(vec)
    return {k: torch.stack(vs).mean(0) for k, vs in bucket.items()}


def unembedding_covariance(model, chunk: int = 16384) -> torch.Tensor:
    U = model.get_output_embeddings().weight.detach()  # (V, D)
    V, D = U.shape[0], U.shape[1]
    gram = torch.zeros(D, D, dtype=torch.float64)
    col_sum = torch.zeros(D, dtype=torch.float64)
    for s in range(0, V, chunk):
        blk = U[s : s + chunk].to("cpu", torch.float32)
        gram += (blk.T @ blk).double()
        col_sum += blk.sum(0).double()
    mean = col_sum / V
    return gram / V - torch.outer(mean, mean)


def extract_representations(
    dataset: str | Path,
    out_dir: str | Path,
    *,
    model_name: str,
    layers: list[int] | None = None,
    token_poolings: tuple[str, ...] = TOKEN_POOLS,
    batch_size: int = 8,
) -> Path:
    """Extract activations for a sentences_filtered.jsonl dataset and save them.

    Activations are taken from the prompt's **prefill phase only** (no generation), so
    each pooling reduces over the prompt tokens: ``'avg'`` averages content tokens,
    ``'last'`` takes the prompt's final content token. All poolings share one forward
    pass. Per-layer activations are written under ``out_dir/<token_pooling>_token/`` so
    both poolings coexist; the pooling-invariant ``unembeddings_covariance.pt`` is
    written once at ``out_dir/``. ``layers`` defaults to all transformer layers
    ``1..num_hidden_layers``. Extraction is deterministic (no seed involved).
    """
    for p in token_poolings:
        if p not in TOKEN_POOLS:
            raise ValueError(f"token_pooling must be one of {TOKEN_POOLS}, got {p!r}")
    dataset, out_dir = Path(dataset), Path(out_dir)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    samples = load_accepted(dataset)
    print(f"{len(samples)} samples from {dataset}  |  device={device}  |  token_poolings={token_poolings}")

    model, tokenizer = load_model(model_name, device)
    # layer L reads hidden_states[L + 1], so the valid range ends at num_hidden_layers - 1
    layers = layers or list(range(1, model.config.num_hidden_layers))
    per_pooling = extract_activations(
        samples, model, tokenizer, layers, device, token_poolings=token_poolings, batch_size=batch_size
    )

    for p, activations in per_pooling.items():
        pool_dir = out_dir / f"{p}_token"
        meta = {
            "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_dataset": str(dataset),
            "model": model_name,
            "device": device,
            "layers": layers,
            "token_pooling": p,
            "n_samples": len(samples),
            "vectors_per_layer": len(activations) // len(layers),
            "hidden_dim": next(iter(activations.values())).shape[0],
        }
        save_representations(activations, pool_dir, meta=meta)
        print(f"Saved {len(layers)} layers under {pool_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_path = out_dir / "unembeddings_covariance.pt"
    if not cov_path.exists():
        torch.save(unembedding_covariance(model), cov_path)
        print(f"Saved unembeddings_covariance.pt at {out_dir}")
    return out_dir

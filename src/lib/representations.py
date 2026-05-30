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


TOKEN_POOLS = ("mean", "last")


def _reduce(hidden: torch.Tensor, valid: torch.Tensor, pool: str) -> torch.Tensor:
    # hidden: (B, L, D); valid: (B, L) bool content-token mask. Returns (B, D).
    if pool == "mean":
        w = valid.unsqueeze(-1).to(hidden.dtype)
        return (hidden * w).sum(1) / w.sum(1).clamp(min=1)
    if pool == "last":
        rows = torch.arange(hidden.size(0), device=hidden.device)
        last_idx = valid.size(1) - 1 - valid.flip(1).int().argmax(1)
        return hidden[rows, last_idx, :]
    raise ValueError(f"pool must be one of {TOKEN_POOLS}, got {pool!r}")


def extract_activations(
    samples: list[Sample],
    model,
    tokenizer,
    layers: list[int],
    device: str,
    *,
    pool: str = "mean",
    batch_size: int = 8,
) -> dict[tuple[str, str, str, str, int], torch.Tensor]:
    """Activations at each requested layer, keyed by (trait, intensity, scenario_id, paraphrase_id, layer).

    ``pool='mean'`` pools content tokens (excluding BOS/EOS/pad); ``pool='last'`` takes the
    final content token. One vector per paraphrase; nothing is averaged here.
    """
    if pool not in TOKEN_POOLS:
        raise ValueError(f"pool must be one of {TOKEN_POOLS}, got {pool!r}")
    layers = sorted(set(layers))
    special = {
        t
        for t in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        if t is not None
    }

    out: dict[tuple[str, str, str, str, int], torch.Tensor] = {}
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
            red = _reduce(hidden[layer + 1], valid, pool).detach().to("cpu", dtype=torch.float32)
            for i, s in enumerate(batch):
                key = (s.trait, s.intensity, s.scenario_id, s.paraphrase_id, layer)
                if key in out:
                    raise ValueError(f"duplicate paraphrase_id encountered: {key}")
                out[key] = red[i]

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
    model_name: str = "google/gemma-2-2b",
    layers: list[int] | None = None,
    pool: str = "mean",
    batch_size: int = 8,
) -> Path:
    """Extract activations for a sentences_filtered.jsonl dataset and save them.

    ``pool='mean'`` pools content tokens; ``pool='last'`` takes the final content token.
    Per-layer activations are written under ``out_dir/<pool>/`` so both pools can coexist;
    the pool-invariant ``unembed_cov.pt`` is written once at ``out_dir/``.
    """
    if pool not in TOKEN_POOLS:
        raise ValueError(f"pool must be one of {TOKEN_POOLS}, got {pool!r}")
    dataset, out_dir = Path(dataset), Path(out_dir)
    pool_dir = out_dir / pool
    layers = layers or list(range(1, 23))
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    samples = load_accepted(dataset)
    print(f"{len(samples)} samples from {dataset}  |  device={device}  |  pool={pool}")

    model, tokenizer = load_model(model_name, device)
    activations = extract_activations(
        samples, model, tokenizer, layers, device, pool=pool, batch_size=batch_size
    )

    meta = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_dataset": str(dataset),
        "model": model_name,
        "device": device,
        "layers": layers,
        "pool": pool,
        "n_samples": len(samples),
        "vectors_per_layer": len(activations) // len(layers),
        "hidden_dim": next(iter(activations.values())).shape[0],
    }
    save_representations(activations, pool_dir, meta=meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_path = out_dir / "unembed_cov.pt"
    if not cov_path.exists():
        torch.save(unembedding_covariance(model), cov_path)
    print(f"Saved {len(layers)} layers under {pool_dir} (+ unembed_cov.pt at {out_dir})")
    return pool_dir

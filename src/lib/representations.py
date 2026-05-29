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


TOKEN_MODES = ("mean", "last", "first")


def _reduce_all(hidden: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    # hidden: (B, L, D); valid: (B, L) bool content-token mask. Returns each mode as (B, D).
    w = valid.unsqueeze(-1).to(hidden.dtype)
    rows = torch.arange(hidden.size(0), device=hidden.device)
    first_idx = valid.int().argmax(1)
    last_idx = valid.size(1) - 1 - valid.flip(1).int().argmax(1)
    return {
        "mean": (hidden * w).sum(1) / w.sum(1).clamp(min=1),
        "first": hidden[rows, first_idx, :],
        "last": hidden[rows, last_idx, :],
    }


def extract_activations(
    samples: list[Sample],
    model,
    tokenizer,
    layers: list[int],
    device: str,
    *,
    batch_size: int = 8,
) -> dict[tuple[str, str, str, int], dict[str, torch.Tensor]]:
    """Activations at each requested layer, keyed by (trait, intensity, scenario_id, layer).

    Each value is the bundle of per-prompt reductions {"mean", "last", "first"}: "mean" pools
    content tokens (excluding BOS/EOS/pad), "last"/"first" take a single content token. Samples
    sharing a key are averaged per reduction. The analysis side picks which reduction to use.
    """
    layers = sorted(set(layers))
    special = {
        t
        for t in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        if t is not None
    }

    bucket: dict[tuple[str, str, str, int], dict[str, list[torch.Tensor]]] = {}
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
            red = _reduce_all(hidden[layer + 1], valid)
            red = {m: red[m].detach().to("cpu", dtype=torch.float32) for m in TOKEN_MODES}
            for i, s in enumerate(batch):
                key = (s.trait, s.intensity, s.scenario_id, layer)
                d = bucket.setdefault(key, {m: [] for m in TOKEN_MODES})
                for m in TOKEN_MODES:
                    d[m].append(red[m][i])

    return {
        key: {m: torch.stack(vecs[m]).mean(0) for m in TOKEN_MODES}
        for key, vecs in bucket.items()
    }


def save_representations(
    activations: dict[tuple[str, str, str, int], dict[str, torch.Tensor]],
    out_dir: str | Path,
    *,
    meta: dict,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[int, dict[tuple[str, str, str], dict[str, torch.Tensor]]] = {}
    for (trait, intensity, scenario_id, layer), bundle in activations.items():
        by_layer.setdefault(layer, {})[(trait, intensity, scenario_id)] = bundle
    for layer, bundles in by_layer.items():
        torch.save(bundles, out_dir / f"layer_{layer}.pt")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_representations(rep_dir: str | Path, *, layer: int | None = None, token: str = "mean"):
    """One vector per key, selecting `token` ("mean"/"last"/"first") from each saved bundle.

    layer given → {(trait, intensity, scenario_id): vec}; layer None → adds the layer to the key.
    """
    rep_dir = Path(rep_dir)
    if layer is not None:
        bundles = torch.load(rep_dir / f"layer_{layer}.pt", weights_only=False)
        return {key: bundle[token] for key, bundle in bundles.items()}
    out: dict[tuple[str, str, str, int], torch.Tensor] = {}
    for path in sorted(rep_dir.glob("layer_*.pt")):
        n = int(path.stem.split("_")[1])
        for key, bundle in torch.load(path, weights_only=False).items():
            out[(*key, n)] = bundle[token]
    return out


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
    batch_size: int = 8,
) -> Path:
    """Extract activations for a sentences_filtered.jsonl dataset and save them into out_dir.

    Loads the model, extracts at every requested layer (each saved as the {mean, last, first}
    bundle), and writes the layer_<N>.pt bundles, a metadata.json manifest, and unembed_cov.pt.
    The caller chooses out_dir (e.g. data/<timestamp>/representations).
    """
    dataset, out_dir = Path(dataset), Path(out_dir)
    layers = layers or list(range(1, 23))
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    samples = load_accepted(dataset)
    print(f"{len(samples)} samples from {dataset}  |  device={device}")

    model, tokenizer = load_model(model_name, device)
    activations = extract_activations(
        samples, model, tokenizer, layers, device, batch_size=batch_size
    )

    meta = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_dataset": str(dataset),
        "model": model_name,
        "device": device,
        "layers": layers,
        "tokens": list(TOKEN_MODES),
        "n_samples": len(samples),
        "vectors_per_layer": len(activations) // len(layers),
        "hidden_dim": next(iter(activations.values()))["mean"].shape[0],
    }
    save_representations(activations, out_dir, meta=meta)
    torch.save(unembedding_covariance(model), out_dir / "unembed_cov.pt")
    print(f"Saved {len(layers)} layers + unembed_cov.pt under {out_dir}")
    return out_dir

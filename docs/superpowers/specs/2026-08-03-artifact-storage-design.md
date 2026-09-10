# Artifact Storage Design

**Date:** 2026-08-03
**Status:** Approved for planning

## Problem

`.gitattributes` declares `*.pt filter=lfs`, but no `.pt` file in history is an LFS
pointer — `git lfs ls-files` returns 0. Git normally never notices, because the index
stat cache (size + mtime) matches the working tree, so the clean filter is skipped.
When one file's index entry carried the `size: 0` racily-clean marker, Git re-read it,
ran the clean filter, produced a 133-byte pointer, and compared it against a 42 MB
blob — a permanent phantom "modified" that no `checkout` can clear.

That bug is a symptom. The underlying problem is that 291 MB of model activations
(558 MB across all refs) live in Git history at all.

## Scale projection

`src/lib/config.py` declares 7 models; `src/lib/traits.py` declares 5 traits.
Current state is 2 models x 1 trait = 467 MB. Extrapolating from measured per-file
sizes (~648 sentences x d_model x 4 bytes per layer file, two poolings, ~n_layers - 1
layers per model):

| Model | d_model | approx. per trait |
| --- | --- | --- |
| qwen2.5-0.5b | 896 | 0.1 GB |
| qwen2.5-1.5b (and -instruct) | 1536 | 0.2 GB each |
| gemma-2-2b | 2304 | 0.25 GB |
| qwen2.5-3b | 2048 | 0.37 GB |
| llama-3.2-3b | 3072 | 0.43 GB |
| qwen2.5-7b | 3584 | 0.50 GB |

Roughly 2 GB per trait across all 7 models, so **~10 GB for the full 5-trait sweep**.

GitHub Free/Pro includes 10 GiB LFS storage and 10 GiB/month download bandwidth.
Storage counts every version ever pushed; uploads are free, downloads are metered.
At the projected sweep size a single full clone consumes the entire monthly bandwidth
allowance, four collaborators cloning once exceeds it 4x, and one re-extraction
permanently doubles stored bytes. With GitHub's default $0 spending limit the failure
mode is not a bill — LFS stops working repo-wide until the next billing cycle.

Branch switching is *not* a factor: `git lfs` caches fetched objects in
`.git/lfs/objects`, so bandwidth is paid once per unique file per person.

### Correction: the project already pays for this in capability, not bytes

The `~10 GB` figure above is what a *full* layer sweep would cost. The repository
never reaches it, because `multi-model-seeded-analysis` already added a
focal-layer-only commit policy: `.gitignore` keeps `layer_*.pt` out of Git except
one hand-listed focal layer per model. Committed tensors therefore stay under
roughly 1 GB, and the LFS quota wall is never actually hit.

That reframes the decision rather than removing it. The cost has been paid in
scientific capability instead of storage:

- Only `gemma-2-2b` and `qwen2.5-1.5b` under `politeness` have full layer sets.
  Every other model/trait combination has just its focal layer, so any layer
  sweep requires local re-extraction — and the analysis code plots layer sweeps
  (`plot_layer_sweep`, `linearity_metric_sweep`) as a core output.
- `unembeddings_covariance.pt` is downcast to float32 purely because float64 at
  `d_model = 3584` would exceed GitHub's 100 MB per-file limit, then upcast again
  on load.

Both compromises exist to fit Git, not because the science wanted them. Moving
representations to the Hub removes the constraint that forced them, which is the
stronger argument for this design than the quota ceiling.

## Decision

**Git stores what you read. Hugging Face stores what you compute.**

| Artifact | Location | Rationale |
| --- | --- | --- |
| `src/`, configs, `data/**/sentences/**` | Git | small, reviewable, versions with code |
| `results/**/aggregated/` | Git | CSV/tex/json — diffable, so metric changes show up in PR review |
| `results/**/seed_*/` | gitignored | 1,572 files, byte-identical on same-seed re-run |
| `data/**/representations/**.pt` | HF dataset repo (private) | too large to version in Git, cheap to fetch |

Git LFS is retired entirely. No `.pt` file enters Git, which structurally eliminates
the pointer-vs-blob bug class rather than fixing one instance of it.

The HF dataset repo starts **private** (free accounts get 100 GB private storage, with
no "best-effort" caveat) and is flipped to public at submission for citability.

## Rationale

What is standard practice, and what is this project's own choice:

- **Standard:** code in Git, large ML artifacts in object storage or on HF Hub,
  pinned by revision; regenerable outputs gitignored.
- **Standard:** Git LFS — but for bounded binaries that version in lockstep with code.
  Activations that grow with every model and trait are the shape LFS handles badly.
- **Project-specific:** the thin `data_sync` wrapper. Every project writes its own;
  there is no canonical tool for this.

Two things were deliberately *removed* from an earlier draft of this design for being
over-engineered:

- **Per-file sha256 in the lock file.** An HF revision SHA already pins content
  immutably and `huggingface_hub` verifies integrity on download. Redundant.
- **A `pre-commit` framework dependency.** This project keeps tooling minimal (ruff
  only; tests run as plain Python scripts). A ten-line git hook matches that style and
  adds no dependency.

## Components

### 1. HF dataset repo

Layout mirrors the local tree exactly, so `snapshot_download` lands files where the
analysis code already expects them:

```
<run_id>/representations/<model>/<trait>/<pooling>_token/layer_<n>.pt
<run_id>/representations/<model>/unembeddings_covariance.pt
```

Ships with a dataset card (`README.md` in the repo root) describing provenance,
extraction settings, and tensor shapes.

### 2. `src/lib/hub.py`

Lock-file read/write and HF path/pattern construction. Pure functions where possible
so they are testable without network access.

### 3. `src/data_sync.py`

CLI with two subcommands:

```
python src/data_sync.py push --run <run_id>            # after extraction
python src/data_sync.py pull --run <run_id>            # before analysis
python src/data_sync.py pull --run <run_id> --model gemma-2-2b --trait politeness
```

Selective pull matters: someone working on one model+trait fetches ~250 MB, not 10 GB.

### 4. Lock file — `data/<run_id>/representations.lock.json`

Committed to Git. This is what replaces having tensors in Git:

```json
{
  "repo_id": "llm-behaviour-intensity/activations",
  "revision": "<40-char commit sha>",
  "run_id": "20260530_001930",
  "files": ["representations/gemma-2-2b/politeness/avg_token/layer_1.pt", "..."]
}
```

Gives provenance (results trace to exact tensor bytes), pinning (`pull` is
reproducible), and reviewability (a text diff shows when tensors changed).

### 5. Guards

- `.githooks/pre-commit` rejects any staged `.pt`, enabled via
  `git config core.hooksPath .githooks` in setup. The previous failure was a rule that
  was *declared* but never *checked*; this makes the invariant enforced.
- `load_representations()` raises a clear "run data_sync pull" error instead of a
  cryptic `FileNotFoundError` or a silent empty dict.

### 6. Migration (one-time)

Upload current tensors to HF, verify, then `git filter-repo --path-glob '*.pt'
--invert-paths` across all branches, drop the LFS attribute, force-push once with the
team coordinated, everyone re-clones once.

`results/**/seed_*/` needs only `git rm -r --cached` plus a gitignore rule — those are
small text files, so leaving them in history costs nothing and avoids widening the
rewrite.

## Error handling

- Missing tensors -> `FileNotFoundError` naming the exact `data_sync pull` command.
- Missing lock file -> error naming the run id and pointing at `data_sync push`.
- Missing HF auth -> error pointing at `hf auth login` (already README step 4).
- `pull` after a failed partial download: `snapshot_download` resumes and verifies;
  no bespoke retry logic.

## Testing

Following the existing convention (`tests/test_analysis.py`: "Run with plain Python
(no pytest dependency)"), tests are plain Python scripts run with
`.venv/bin/python tests/<name>.py`, asserting inline and printing a summary.

- Lock-file round-trip; missing-lock error message.
- `allow_patterns` construction for every filter combination — a pure function, so no
  network is touched.
- `load_representations` on an empty directory produces the actionable message.
- The pre-commit hook rejects a staged `.pt` (shell-level test).

## Non-goals

- **No precision or layout changes.** fp16 storage and stacked-layer files would cut
  the payload 3-4x, but they alter numerical outputs. Explicitly deferred so results
  stay bit-for-bit identical to what is published today.
- **No DVC or pipeline DAG.** Duplicates the provenance `run_metadata.json` already
  records.
- **No CI.** The repo has none; adding one is out of scope.
- **No pruning of `aggregated/` beyond the seed split.** Further curation is a
  judgement call to make later with the report in hand.

## Open coordination item

PR #3 (`multi-model-seeded-analysis`, 2,222 files) should be merged or explicitly
coordinated before the history rewrite, since the rewrite touches all branches.

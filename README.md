# Internal Representations of Behavioural Trait Intensity in an LLM

_Produced by Francesco Braicovich, Gabriele Bettineschi, Giovanni Berlinghieri, Enrico Adamo, and Andrea Porta in May 2026 for the Machine Learning and Artificial Intelligence course at Bocconi University._

## Setup and development
**Setup**
1. Clone the repo. Model representations are **not** in Git — they live in a Hugging Face dataset repo and are fetched separately (step 7).
2. Install dependencies with uv.
   ```
   uv sync
   ```
3. Login on the HuggingFace website and accept the licenses for [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) and [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B) (the Qwen 2.5 models are ungated).
4. Authenticate yourself on the HuggingFace CLI.
   ```
   hf auth login
   ```
5. Setup nb-clean, one of the many nice tools to make git and jupyter notebooks work better together (if your venv is not in .venv, you should change the first command accordingly).
   ```
   git config --local filter.nb-clean.clean ".venv/bin/nb-clean clean --preserve-cell-outputs --remove-all-notebook-metadata"
   
   echo "*.ipynb filter=nb-clean" >> .git/info/attributes
   ```
6. Enable the commit guard, which stops tensors being committed by accident.
   ```
   git config core.hooksPath .githooks
   ```
7. Fetch the representations for the run you want to analyse.
   ```
   uv run python src/data_sync.py pull --run 20260530_001930
   ```
   Narrow it down if you only need part of the sweep — the full set is large:
   ```
   uv run python src/data_sync.py pull --run 20260530_001930 --model gemma-2-2b --trait politeness
   ```

## Running the analysis

Models are declared in `src/lib/config.py`: `gemma-2-2b`, `llama-3.2-3b` (gated), `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-1.5b-instruct`, `qwen2.5-3b`, `qwen2.5-7b`; each uses a mid-depth focal layer (`num_hidden_layers // 2`). `qwen2.5-1.5b-instruct` is fed the same raw, non-chat-templated sentences as every other model — comparability across the sweep, not realism. Traits are declared in `src/lib/traits.py`: `politeness`, `formality`, `certainty`, `urgency`, `enthusiasm`; every script takes `--trait` and defaults to `politeness`.

1. Generate a trait's sentence dataset (needs `OPENROUTER_API_KEY` in a repo-root `.env`). Output lands in `data/<ts>/sentences/<trait>/`; pass `--data-root` to add a new trait to an existing dataset root so all traits share one timestamp.
   ```
   uv run python src/generate_sentences.py --trait politeness --data-root data/20260530_001930
   ```
2. Extract activations for a model (deterministic — prefill only, no sampling; both token poolings share one forward pass). Output lands in `data/<ts>/representations/<model>/<trait>/`. On a 16GB Apple Silicon Mac, `qwen2.5-3b` may need a smaller `--batch-size` (e.g. `4`); `qwen2.5-7b` does not fit in bf16 on 16GB and needs a cloud GPU (see below).
   ```
   uv run python src/extract_representations.py --model qwen2.5-1.5b --trait politeness
   ```
3. Run each analysis. One full run per master seed (default `0,1,2`); the analysis-stage Monte-Carlo machinery (bootstrap, permutation nulls, CV folds, KMeans init, reliability half-splits) derives per-component streams from the master seed. Per-seed results land in `results/<ts>/<analysis>/<model>/<trait>/<pooling>_token/seed_<k>/` (raw, with `run_metadata.json` provenance) and, with more than one seed, a mean±std aggregate in `aggregated/` next to them.
   ```
   uv run python src/replicate_tigges.py   --model gemma-2-2b --trait politeness --token-pooling avg --seeds 0,1,2
   uv run python src/ordinal_linearity.py  --model gemma-2-2b --trait politeness --token-pooling avg --seeds 0,1,2
   uv run python src/trait_geometry.py     --model gemma-2-2b --trait politeness --token-pooling avg --seeds 0,1,2
   ```
   Or run every model × trait × pooling combo present on disk in one go with `src/run_analyses.py` (a thin wrapper that calls the same three drivers' `main()`; combos with no representations yet are skipped with a warning). Scope it with `--models` / `--traits` / `--poolings` / `--seeds` (each comma-separated, defaults to everything in the registries):
   ```
   uv run python src/run_analyses.py
   ```
4. Aggregation can also be run standalone (e.g. over every analysis at once):
   ```
   uv run python src/aggregate_results.py --discover results/20260530_001930
   ```
5. Collect a cross-combo summary table (apex angle + CI, step cosine, linear-null p, within-scenario Spearman/probe R², markedness reliability, shared-bend R²_cv, one row per model × trait × pooling) into `results/<ts>/summary/summary.csv`:
   ```
   uv run python src/collect_summary.py
   ```

### Adding a trait

Add one entry to `TRAITS` in `src/lib/traits.py`: the rubric guide, the intents to balance scenarios across, the invariant field that must stay fixed across levels, cue families, and a seed example (see the `politeness` entry for the shape). All three levels keep the signed scale — negative / neutral / positive. Then run steps 1–4 with `--trait <name>`; no other code changes are needed.

## Data and results

| Artifact | Where it lives | Why |
| --- | --- | --- |
| Code, configs, sentence datasets | Git | small, reviewable, versions with the code |
| `results/**/aggregated/` | Git | text — metric changes show up in PR diffs |
| `results/**/seed_*/` | not tracked | regenerates byte-identically from the same seed |
| `data/**/representations/*.pt` | [Hugging Face dataset repo](https://huggingface.co/datasets/llm-behaviour-intensity/activations) | too large for Git; fetched with `data_sync` |

Tensors live at [`llm-behaviour-intensity/activations`](https://huggingface.co/datasets/llm-behaviour-intensity/activations). `data/<run_id>/representations.lock.json` is committed and pins the exact Hub revision, so everyone analysing a run reads the same tensor bytes.

After extracting new representations, publish them and commit the updated lock:

```
uv run python src/data_sync.py push --run <run_id>
git add data/<run_id>/representations.lock.json
git commit -m "data: publish representations for <run_id>"
```

This replaces the previous focal-layer-only commit policy. That policy existed only to fit Git's size limits, and it cost real capability: every model/trait except `gemma-2-2b` and `qwen2.5-1.5b` under politeness had just its focal layer available, so layer sweeps needed local re-extraction. With representations on the Hub, full layer sets are kept for every combo.

**Troubleshooting.** `FileNotFoundError: No representations under ...` means you have not fetched the tensors — run the `pull` command it prints. If a commit is rejected with "refusing to commit PyTorch tensors", that is the guard working: push the tensors to the Hub instead.

### Extracting `qwen2.5-7b` on a cloud GPU

`qwen2.5-7b` does not fit in bf16 on a 16GB Mac; extract it on any CUDA box instead (~1 GPU-hour for all 5 traits), then copy the representations back:
```
git clone <repo> && cd <repo>
uv sync
uv run python src/extract_representations.py --model qwen2.5-7b --trait politeness
uv run python src/extract_representations.py --model qwen2.5-7b --trait formality
uv run python src/extract_representations.py --model qwen2.5-7b --trait certainty
uv run python src/extract_representations.py --model qwen2.5-7b --trait urgency
uv run python src/extract_representations.py --model qwen2.5-7b --trait enthusiasm
uv run python src/data_sync.py push --run <ts>
git add data/<ts>/representations.lock.json
git commit -m "data: add qwen2.5-7b representations" && git push
```
The `git commit` on the cloud box is the part that matters: `push` rewrites the lock **on the machine that uploaded**, and `pull` reads the *local* lock to decide which revision to fetch. Skip it and your laptop keeps pinning the pre-upload revision, at which the new tensors do not exist — the pull then succeeds while downloading nothing.

Then locally, take the new lock and fetch against it:
```
git pull
uv run python src/data_sync.py pull --run <ts> --model qwen2.5-7b
```
Everything downstream (analysis drivers, `run_analyses.py`, `collect_summary.py`) is model-agnostic and picks it up once the representations land.

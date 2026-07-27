# Internal Representations of Behavioural Trait Intensity in an LLM

_Produced by Francesco Braicovich, Gabriele Bettineschi, Giovanni Berlinghieri, Enrico Adamo, and Andrea Porta in May 2026 for the Machine Learning and Artificial Intelligence course at Bocconi University._

## Setup and development
**Setup**
1. Before cloning the repo, make sure you have git lfs installed.
   ```
   git lfs install
   ```
If you have already cloned the repo before doing this, you can fix by installing git lfs (command above) and then running:
   ```
   git lfs pull
   ```
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

### Storage: focal-layer-only commits

Only the focal layer's `.pt` per model/trait/pooling is committed (plus `metadata.json` and the model-level `unembeddings_covariance.pt`); every other layer is gitignored (see `.gitignore`). The unembedding covariance is saved as float32 — at 7B hidden sizes (D≈3584) float64 would exceed GitHub's 100MB file limit — and upcast back to float64 on load. The politeness representations for `gemma-2-2b` and `qwen2.5-1.5b`, committed before this policy, keep their full layer sets. A full layer sweep for any other model/trait needs local re-extraction (`extract_representations.py` is deterministic, so it reproduces the same activations).

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
# copy data/<ts>/representations/qwen2.5-7b/ back into the local repo's data/<ts>/
```
Everything downstream (analysis drivers, `run_analyses.py`, `collect_summary.py`) is model-agnostic and picks it up once the representations land.

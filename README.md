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
3. Login on the HuggingFace website and accept the licenses for [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) and [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B) (Qwen 2.5 1.5B is ungated).
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

Models are declared in `src/lib/config.py` (`gemma-2-2b`, `llama-3.2-3b`, `qwen2.5-1.5b`); each uses a mid-depth focal layer (`num_hidden_layers // 2`).

1. Extract activations for a model (deterministic — prefill only, no sampling; both token poolings share one forward pass). Output lands in `data/<ts>/representations/<model>/`.
   ```
   uv run python src/extract_representations.py --model qwen2.5-1.5b
   ```
2. Run each analysis. One full run per master seed (default `0,1,2`); the analysis-stage Monte-Carlo machinery (bootstrap, permutation nulls, CV folds, KMeans init, reliability half-splits) derives per-component streams from the master seed. Per-seed results land in `results/<ts>/<analysis>/<model>/<pooling>_token/seed_<k>/` (raw, with `run_metadata.json` provenance) and, with more than one seed, a mean±std aggregate in `aggregated/` next to them.
   ```
   uv run python src/replicate_tigges.py   --model gemma-2-2b --token-pooling avg --seeds 0,1,2
   uv run python src/ordinal_linearity.py  --model gemma-2-2b --token-pooling avg --seeds 0,1,2
   uv run python src/trait_geometry.py     --model gemma-2-2b --token-pooling avg --seeds 0,1,2
   ```
3. Aggregation can also be run standalone (e.g. over every analysis at once):
   ```
   uv run python src/aggregate_results.py --discover results/20260530_001930
   ```

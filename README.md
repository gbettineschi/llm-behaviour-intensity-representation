# Internal Representations of Behavioural Trait Intensity in an LLM

_Produced by Francesco Braicovich, Gabriele Bettineschi, Giovanni Berlinghieri, Enrico Adamo, and Andrea Porta in May 2026 for the Machine Learning and Artificial Intelligence course at Bocconi University._

## Abstract and introduction

_[placeholder]_


## Setup and development
**Setup**
1. Install dependencies with uv.
   ```
   uv sync
   ```
2. Login on the HuggingFace website and accept the license for [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b).
3. Authenticate yourself on the HuggingFace CLI.
   ```
   hf auth login
   ```
4. Setup nb-clean, one of the many nice tools to make git and jupyter notebooks work better together (if your venv is not in .venv, you should change the first command accordingly).
   ```
   git config --local filter.nb-clean.clean ".venv/bin/nb-clean clean --preserve-cell-outputs --remove-all-notebook-metadata"
   
   echo "*.ipynb filter=nb-clean" >> .git/info/attributes
   ```

**Development**

All code is in `src/`.

Exploration happens in `src/notebooks/`, where notebooks follow the naming convention `{index}_{DDMM}_{surname}` (e.g. `001_0405_bettineschi.ipynb`).

Once a block of code is validated and meant to be reused, it is consolidated in one of the `.py` files that live in `/src` directly. We have one module for loading data, one for handling the model and its internal representations, and one for analysing them.

At the end, we will write the full correct experiment in `scripts/run.py` with configs in `scripts/config.yaml`, for reproducibility.

If we generate data and we want to keep track of it, we put it inside the `/data` folder. Since our data consists of lightweight text prompts, we can keep it on git. 

Output (activations, plots) is written to `output/` (gitignored).

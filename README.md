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
3. Login on the HuggingFace website and accept the license for [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b).
4. Authenticate yourself on the HuggingFace CLI.
   ```
   hf auth login
   ```
5. Setup nb-clean, one of the many nice tools to make git and jupyter notebooks work better together (if your venv is not in .venv, you should change the first command accordingly).
   ```
   git config --local filter.nb-clean.clean ".venv/bin/nb-clean clean --preserve-cell-outputs --remove-all-notebook-metadata"
   
   echo "*.ipynb filter=nb-clean" >> .git/info/attributes
   ```

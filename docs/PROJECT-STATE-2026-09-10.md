# Project State — Politeness / graded-trait representation geometry

**As of 2026-09-10.** Written as a self-contained context hand-off: paste the whole
file into a fresh session. Dense on purpose. No proposed solutions — state only.

---

## 0. One paragraph

We test whether a **graded** high-level attribute (politeness, and four more traits
not yet generated) is encoded in an LLM's residual stream as a single *linearly
graded* direction — i.e. whether the neutral level sits proportionally between the
impolite and polite poles. It does not. Across six models (0.5B–7B, two
architectures, base + instruct, two token poolings) the three level-centroids form
an **ordered but non-collinear** configuration: the neutral centroid juts off the
pole-to-pole axis along a second, reliable direction we call *markedness*. The
effect is decisive against two independent nulls (Fisher-combined p ≈ 3.6·10⁻²³),
reliable (split-half R = 0.89–0.99), and flat across model scale (Spearman ρ =
−0.24, p = 0.46). All evidence is **observational** (geometry of centroids + probe
accuracy + null tests); there is no causal / intervention result. The finding is a
corrective to the linear-representation literature, which almost exclusively tests
**binary** contrasts and therefore structurally cannot observe a bend (two points
are always collinear).

---

## 1. The research question, from first principles

### 1.1 The linear representation hypothesis

A recurring empirical claim in mechanistic interpretability: high-level concepts
are encoded as approximately **linear directions** in the residual stream —
adding/subtracting a fixed vector moves the concept up/down, and a linear probe
recovers it. Evidence has been reported for sentiment (Tigges et al. 2023),
truthfulness (Marks & Tegmark 2024), and continuous world properties like space
and time (Gurnee & Tegmark 2024). Representation-engineering / activation-steering
methods (Zou et al. 2023; Turner et al. 2023; Rimsky et al. 2024) exploit the same
structure.

### 1.2 What "linear" precisely means — and what binary tests actually show

Two clusters (e.g. "polite" vs "impolite" sentence activations) are **collinear in
any geometry**: the difference of their means defines a line, and any two points
lie on a line. So a binary separation demonstrates only that a *direction exists*
— it says nothing about whether **intensity along that direction is linear**,
because you have no third point to check proportionality against.

Concretely, "linear encoding of intensity" is the stronger claim that if you have
three ordered levels L⁻ < L⁰ < L⁺, then

- the level centroids are **collinear** (μ⁰ on the segment μ⁻→μ⁺), and
- μ⁰ sits at the **fraction** of the segment implied by its ordinal position
  (≈ midpoint for an evenly-spaced 3-point scale).

The moment you add an explicit **neutral** level, linearity becomes **falsifiable**.
That is the pivot of the whole project.

### 1.3 The reframe (the paper's actual novelty)

Prior graded-representation work either (a) treats the axis as **bipolar** and
never tests an intermediate level (Tigges 2023 sentiment; Konen et al. 2024
emotion), or (b) tests linear-vs-monotonic but only for **numeric properties that
vary across entities** with an external scalar (Heinzerling & Inui 2024). No one
has asked whether a **graded pragmatic trait** — no external scalar, elicited by
rewriting the *same* propositional content — forms a one-dimensional ordered
ladder.

The claim is therefore not "we found ordinal linearity." It is: **the entire
linear-representation literature has been measured between two categories, and the
picture changes once you add a third.**

### 1.4 Why politeness (and the trait family)

Politeness is: (i) a high-level **pragmatic** attribute (mitigation of face
threat, deference), (ii) **graded by construction** — the same request can be
rude, neutral, or deferential, (iii) **socially constructed**, not anchored to any
physical scalar, (iv) unlike sentiment, only **weakly carried by individual
valenced words** (it lives in framing, indirectness, softeners). This makes it a
hard case for the linear hypothesis and forces strict **content control**.

The four not-yet-generated traits (formality, certainty, urgency, enthusiasm) are
all "**speaker stance**" traits — same structural template, same 10 speech acts.
`enthusiasm` deliberately sits closest to Tigges-style valence as a bridge to
prior work. **Caveat: these five constructs are conceptually correlated** (see §10).

---

## 2. What "the bend" is, operationally

Pool each sentence's activations to one vector (see §4.2); average within
`(level, scenario)` → three centroids per scenario, then pool over scenarios →
`μ⁻` (negative/impolite), `μ⁰` (neutral), `μ⁺` (positive/polite).

Three scalar summaries of the triangle `(μ⁻, μ⁰, μ⁺)`:

| quantity | definition | linear prediction | observed (politeness) |
|---|---|---|---|
| **apex angle** | angle at `μ⁰` between `μ⁰→μ⁻` and `μ⁰→μ⁺` | 180° | 24–77° |
| **step cosine** | `cos(μ⁻→μ⁰, μ⁰→μ⁺)` | +1 | −0.23 to −0.92 |
| **midpoint residual** | ‖μ⁰ − ½(μ⁻+μ⁺)‖ / ‖span‖ | 0 | 0.61–2.13 |

The **shared (valence, markedness) plane** decomposition:

- **valence axis** = the pole-to-pole direction (`μ⁺ − μ⁻`); this is the axis a
  binary probe would find.
- **markedness axis** = the orthogonal direction along which `μ⁰` deviates from
  the valence line. By construction of the decomposition `μ⁻` and `μ⁺` project
  **equally** onto markedness, and `μ⁰`'s markedness coordinate is the bend.

Interpretation of markedness: neutral utterances are not "half-polite" — they are
a **distinct communicative mode** (plain, unmarked). The model represents that
mode off-axis, not as an interpolation. "Markedness" is the linguistics term:
neutral is the unmarked member, both poles are marked.

---

## 3. The dataset

### 3.1 Construction principle — content control

Every scenario fixes a proposition / speech act (`intent_target`: e.g. "send the
latest budget spreadsheet by end of day"). The three level-rewrites and their
paraphrases change **only the trait**; named entities, deadlines, polarity of the
act, imposition, urgency, and **length** are held constant. This is what lets the
geometry be attributed to the trait rather than to lexical or length confounds.

### 3.2 Pipeline (`src/lib/sentences.py`, run via `src/generate_sentences.py`)

```
scenarios → base_sentences → paraphrases → judge_and_filter → export
```

- **Generator**: `openrouter/google/gemini-2.0-flash-001` (temp 0.5).
- **Judge**: `openrouter/deepseek/deepseek-v4-flash` (temp 0.0) — a *different*
  family, to reduce generator/judge leakage.
- `n_scenarios=100`, `paraphrases_per_level=3`, `min_acceptance_score=0.70`.
- **Three gates, in order**, per scenario bundle:
  1. **context**: per-sentence faithfulness to the invariant content (judge
     scores 0–1, must clear 0.70).
  2. **trait_intensity**: judge blind-rates every surviving paraphrase 0–1
     (opaque codes, no level shown); the level means must be **monotone
     increasing** with a min adjacent gap of 0.10.
  3. **length**: longest level's mean word count ≤ 1.15× the shortest.
- Established rule: if a gate rejects heavily, **fix the rubric/judge prompt, do
  not lower the threshold** without explicit approval.

### 3.3 Trait registry (`src/lib/traits.py`)

Pure-data module (no third-party imports — the stdlib-only human-eval app loads it
too). `LEVELS = ["negative","neutral","positive"]`. `DEFAULT_TRAIT="politeness"`.
Per-trait keys: `description, axis, guide` (full generator rubric), `invariant_field`
(`intent_target`), `required_fields`, `cue_families` (paraphrases must spread
across these, no single-marker monoculture), `paraphrase_note`, `seed_example`,
`intents` (the 10 shared speech acts), `generation_constraints`.

**Registered traits**: `politeness` (generated), `formality`, `certainty`,
`urgency`, `enthusiasm` (**registered, NOT generated** — needs
`OPENROUTER_API_KEY` in a repo-root `.env`, which is not present).

**Bug fixed during this work**: the 10 shared intents include a `request`
constraint "Do not change urgency or scope across levels" — harmless for the other
traits, but a **self-contradiction for the `urgency` trait itself**. Fixed via a
minimal `_URGENCY_INTENTS` override + a guard test asserting no trait's intents
pin the trait it is registered under.

### 3.4 The politeness corpus (dataset id `20260530_001930`)

- **633 accepted paraphrases**, **72 / 100 complete scenarios** (all 3 levels
  survive all gates).
- Level counts: negative 212 / neutral 214 / positive 207.
- Mean word count per level: 16.9 / 17.4 / 17.7 (sd ≈ 4.5–4.7) — length balanced.
- Files: `data/20260530_001930/sentences/politeness/{scenarios,sentences_unfiltered,sentences_filtered}.jsonl`,
  `metadata.json`. Field names: `text`, `level`, `trait`, `scenario_id`,
  `paraphrase_id`, `cue_family`, `base_sentence`, `invariant_content`. (Not
  `sentence`/`intensity`.)

### 3.5 Human evaluation — currently a hole

`src/lib/human_eval.py` + `results/human_eval/human_eval_20260529_234251.json`:
**n = 2 answers**, against dataset `20260529_212332` (which **pre-dates** the
current `20260530_001930` corpus). Its agreement stats are all 1.0 — an artifact
of n = 2. There is effectively **no human validation** of the trait labels.

---

## 4. Pipeline architecture (codebase)

```
generate_sentences.py ─▶ extract_representations.py ─▶ {replicate_tigges,
                                                        ordinal_linearity,
                                                        trait_geometry}.py
                                                       ─▶ aggregate_results.py
                                                       ─▶ collect_summary.py
run_analyses.py = orchestrator over models × traits × poolings × the 3 drivers
```

### 4.1 Model registry (`src/lib/config.py`)

`MODELS`: 7 entries, each `{hf_id, focal_layer, params_b}`.

| key | hf_id | layers | focal | params_b | gated |
|---|---|---|---|---|---|
| gemma-2-2b | google/gemma-2-2b | 26 | 13 | 2.0 | yes (granted) |
| llama-3.2-3b | meta-llama/Llama-3.2-3B | 28 | 14 | 3.0 | yes (granted) |
| qwen2.5-0.5b | Qwen/Qwen2.5-0.5B | 24 | 12 | 0.5 | no |
| qwen2.5-1.5b | Qwen/Qwen2.5-1.5B | 28 | 14 | 1.5 | no |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | 28 | 14 | 1.5 | no |
| qwen2.5-3b | Qwen/Qwen2.5-3B | 36 | 18 | 3.0 | no |
| qwen2.5-7b | Qwen/Qwen2.5-7B | 28 | 14 | 7.0 | no |

- **Focal layer rule**: `num_hidden_layers // 2`, declared explicitly so it is
  auditable. Layer counts verified against HF configs.
- `DEFAULT_MODEL="gemma-2-2b"`, `DEFAULT_SEEDS=(0,1,2)`, `DEFAULT_RUN="20260530_001930"`,
  `DEFAULT_DATA_ROOT=Path("data")/DEFAULT_RUN`.
- **Seed derivation**: `child_seed(master, name)` = `SeedSequence([master, crc32(name)])`
  → stable per-component stream (crc32, not `hash()`, for cross-process stability);
  feeds both `default_rng` and sklearn `random_state`.
- `run_metadata(...)` → `run_metadata.json` per seed dir: model, hf_id, trait,
  seed, pooling, focal_layer, dataset, timestamp, **`git_commit`** (HEAD, suffixed
  `-dirty` iff `src/` has uncommitted changes — scoped to `src/` on purpose;
  a whole-tree check would flag every run dirty from its own `results/` output),
  package versions, and **`representations` provenance** (Hub repo_id + revision +
  whether the on-disk tensor digests match the lock — `verified: true/false/null`).
- `instruct` flag is derived from the key name (`"instruct" in model`), not stored.
- `params_b` is nominal (marketing) size, used only as the scale axis in
  `collect_summary`.

### 4.2 Extraction (`src/extract_representations.py`, `src/lib/representations.py`)

- **Prefill phase only** — one forward pass over the prompt, **no generation**,
  `output_hidden_states=True`.
- **bfloat16** model inference (`dtype=torch.bfloat16`); **no quantization**
  (`load_model` supports 4/8-bit via BitsAndBytesConfig but extraction never
  passes it). Activations cast to **float32** on save.
- **Layer convention**: analysis layer `L` reads `hidden_states[L+1]` (index 0 is
  the embedding); valid range `1 … num_hidden_layers − 1`. Default extracts all
  valid layers.
- **Two poolings**, both from the same forward pass, both saved:
  - `avg`: mean over **content tokens** (BOS/EOS/pad masked out).
  - `last`: the final content token's vector.
- Deterministic — **no seed**. But **not bit-identical across hardware** (bf16
  matmul rounding differs MPS vs CUDA). Cross-hardware re-extraction ⇒
  `data_sync verify` reports `mismatched` — expected, not corruption.
- Also writes model-level `unembeddings_covariance.pt` (covariance of the output
  embedding rows), computed in float64, **saved as float32** (float64 at
  `d_model=3584` would exceed GitHub's 100 MB file limit; upcast to float64 on
  load in `inner_products.py`).
- Output layout: `data/<run>/representations/<model>/<trait>/<pooling>_token/layer_<n>.pt`
  + `metadata.json`; `data/<run>/representations/<model>/unembeddings_covariance.pt`.

### 4.3 The three analysis drivers

Each: `_configure(model, trait, pooling, seed)` mutates module globals →
`main(model, trait, pooling, seed)` → writes to
`results/<run>/<analysis>/<model>/<trait>/<pooling>_token/seed_<seed>/`
(figures + `numeric/*.csv|tex|json` + `run_metadata.json`).

**`replicate_tigges.py`** (`ANALYSIS="replication_tigges"`) — replicates Tigges
et al. 2024 §2.2 on each **binary** contrast (pos-vs-neg, pos-vs-neu, neu-vs-neg):
- direction-estimator agreement — cosine between MeanDiff / KMeans / LogReg / PCA
  directions;
- direction-as-classifier **balanced accuracy** under **scenario-grouped CV**
  (no scenario in both train and test);
- projection separation on the MeanDiff axis (AUC, Cohen's d);
- each run twice — raw + **within-scenario fixed-effects centered** (removes the
  per-scenario offset);
- a **layer sweep** of the binary contrast.
This is the "a clean binary axis exists" evidence (§3.1 of the paper).

**`ordinal_linearity.py`** (`ANALYSIS="ordinal_linearity"`) — is the 3-level
ladder *linearly graded*?
- step-vector geometry;
- **inner-product comparison across five whitening spaces** (see §4.4) with
  **Spearman-Brown-disattenuated** cosines;
- PCA views;
- **linearity metrics**: within-scenario vs pooled-naive Spearman & Kendall,
  linear-probe R², midpoint residual, PC1 variance fraction;
- a linearity **layer sweep**;
- a **bag-of-words lexical baseline** (if BoW accuracy ≈ probe accuracy, the trait
  is lexically given away → revisit the rubric, not the gate);
- `steering_axis_sweep` — **observational** (projects activations onto candidate
  axes across strengths; no forward intervention).

**`trait_geometry.py`** (`ANALYSIS="trait_geometry"`) — geometry of the triangle:
- `report_geometry`: apex angle, step cosine, midpoint residual, valence offset,
  markedness, markedness ratio, effective vertices — with **scenario bootstrap
  CIs, N_BOOT = 2000**;
- `report_noise_null`: simulate a **perfectly linear ladder + the dataset's own
  resampled paraphrase noise** (1000 draws); one-sided test — observed step
  cosine significantly **below**, midpoint residual significantly **above**, the
  null;
- `report_subspace`: per-scenario plane rank / participation ratio;
- `report_shared_plane`: the (valence, markedness) decomposition, markedness-axis
  **split-half reliability** (Spearman-Brown), shared-bend cross-validated R²;
- `report_per_intent`: geometry pooled within each of the 10 speech acts;
- `report_steering_direction`: alignment / capture (cos, cos²) of per-scenario
  contrasts vs a global steering axis, principal angles between planes —
  **all observational**, no intervention.

### 4.4 The five whitening spaces (`src/lib/inner_products.py`)

Raw residual-stream cosine is **not coordinate-free** and the space is strongly
**anisotropic** — so the geometry is evaluated under five re-metrizations:

1. **Euclidean (raw)** — identity, reproduces the naive §3.2 geometry.
2. **Anisotropy-corrected** — per-dimension z-score.
3. **Park causal** — inverse-sqrt of the unembedding-row covariance.
4. **Mahalanobis / LDA** — inverse-sqrt of pooled within-level covariance (Ledoit-Wolf).
5. **Within-subjects noise** — inverse-sqrt of the two-way (level + scenario)
   residual covariance; the level mean is removed from the *noise* estimate, not
   the signal.

**Disattenuated cosine**: each off-diagonal cosine ÷ `sqrt(R_i · R_j)`, where `R`
is the Spearman-Brown-corrected reliability from splitting scenarios in half and
recomputing each difference vector independently. Cost driver for large models:
these whiteners require **eigendecomposition of a `d_model × d_model` matrix** —
O(d³), so `d=3584` (7B) is ~64× a single `d=896` (0.5B) decomposition, ×5 spaces,
×3 seeds.

### 4.5 Seeds

Seeds affect **only analysis-stage Monte Carlo**: bootstrap resampling,
permutation nulls, noise-null draws, CV fold assignment, KMeans init, reliability
half-splits. Extraction is deterministic. Point estimates (apex angle, step
cosine, probe R², plane fractions) are essentially seed-invariant; the
seed-varying quantities are the CI bounds and null draws.

### 4.6 Aggregation (`src/aggregate_results.py`)

`aggregate_analysis(base_dir, seeds=None)` folds `seed_<k>/` runs into
`aggregated/`:
- CSVs — key columns must match across seeds; numeric columns identical across
  seeds pass through; varying columns → `<col>_mean` / `<col>_std` (ddof=1) +
  `n_seeds`.
- JSONs — identical leaves kept; varying numeric leaves → `{mean, std, n_seeds}`;
  **provenance keys** (`seed, seeds, timestamp, git_commit, generated_at`) kept
  **per-seed**, never averaged.
- Draw-level dumps (bootstrap draws, null draws, reliability cosines) — **skipped**
  (cross-seed row-wise stats over independent MC draws are meaningless).
- `seeds=` passed by `run_analyses.py` so a **stale** `seed_*` dir from an earlier
  run is not silently folded into the published mean.

### 4.7 Cross-combo summary (`src/collect_summary.py`)

- `summary.csv` — one row per `(model, trait, pooling)`: apex + CI, step cosine,
  midpoint residual, both linear-null p-values, within-scenario Spearman & probe
  R², markedness reliability R, shared-bend R²_cv, `params_b`, `instruct`,
  `n_seeds`. Reads `aggregated/` when present, else the **lowest-numbered** seed
  dir (never a hardcoded `seed_0` — per-seed dirs are gitignored and `--seeds 3,4`
  produces no `seed_0`).
- `meta_analysis.json` — cross-combo synthesis:
  - **Fisher-combined p** across every combo's linear-ladder-null p (the per-combo
    p is already one-sided in the "more bent than chance" direction, so combining
    is coherent);
  - fraction of combos individually significant at α = 0.05;
  - **Spearman correlation** between `params_b` and bend magnitude (`−step_cosine`);
  - per-trait and per-model breakdowns.

---

## 5. Storage architecture (HF Hub, not Git)

### 5.1 Why (design doc: `docs/superpowers/specs/2026-08-03-artifact-storage-design.md`)

`.gitattributes` declared `*.pt filter=lfs` but **no `.pt` in history was ever an
LFS pointer** (`git lfs ls-files` = 0) — LFS was never actually installed when the
files were committed. One racily-clean index entry made Git re-run the clean
filter and produce a permanent phantom "modified" state. Symptom of the real
problem: **~291 MB of activations in Git history** (558 MB across all refs), and a
full 5-trait × 7-model sweep projected to **~10 GB** — past GitHub's free 10 GiB
LFS quota, and a single clone would burn the monthly bandwidth allowance.

### 5.2 The scheme (Gabriele's history rewrite, 2026-08-05 — force-pushed all branches)

**"Git stores what you read. Hugging Face stores what you compute."**

| artifact | where |
|---|---|
| `src/`, configs, `data/**/sentences/**` | Git |
| `results/**/aggregated/` (CSV/tex/json) | Git — diffable in PR review |
| `results/**/seed_*/` (1,572 files, regenerable) | gitignored |
| `data/**/representations/**.pt` | **HF private dataset repo** |

- **HF repo**: `llm-behaviour-intensity/activations` (dataset, private). Layout
  mirrors the local tree so `snapshot_download` lands files where the code
  expects.
- **Lock file** `data/<run>/representations.lock.json` (committed): `repo_id`,
  `revision` (40-char Hub commit sha), `run_id`, `files` (path → **sha256**). The
  sha256 per file turns "these results came from revision X" from a claim into a
  checkable **integrity binding**.
- **`src/lib/hub.py`** — pure functions: `lock_path`, `file_digest`, `write_lock`,
  `read_lock`, `verify_lock`, `allow_patterns` (Hub glob for selective pull).
- **`src/data_sync.py`** — CLI: `push` (upload `data/<run>/representations/**.pt`,
  read the full manifest back off the Hub, rewrite the lock), `pull` (`snapshot_download`
  at the pinned revision; `--model` / `--trait` selective; verifies the filtered
  target actually landed), `verify` (hash on-disk files vs the lock; `missing` is
  fine for a selective pull, `mismatched` is not).
- **`.githooks/pre-commit`** — rejects any staged `.pt`; enabled with
  `git config core.hooksPath .githooks` (a setup step, not committed config).
- `load_representations()` raises a "run `data_sync pull`" error instead of a bare
  `FileNotFoundError` / silent `{}`.

### 5.3 Reconciliation with this session's work

Gabriele's rewrite forked from the "Add formality/…/qwen2.5 …" commit; it had **no
merge base** with the local branch. We **adopted his history** (`git reset --hard
origin/multi-model-seeded-analysis`) and re-applied the one novel piece (the
`collect_summary.py` meta-analysis) on top. Bugs his review caught (all real, all
fixed):

- urgency-trait intent self-contradiction (§3.3);
- `run_analyses.py` skipped combos via `rep_dir.exists()` — but `metadata.json` is
  committed while tensors aren't, so every rep dir exists on a fresh clone; now
  checks `glob("layer_*.pt")`;
- `collect_summary.py` hardcoded `seed_0`;
- `aggregate_analysis` globbed **every** `seed_*` on disk → a stale dir polluted
  the published mean (reproduced: 34.3 ± 56.0 instead of 2.0 ± 1.4);
- `_aggregate_json` averaged RNG **seeds** into a reported statistic;
- `aggregate_results.py --discover` aborted the whole sweep at the first
  single-seed combo.

Design-vs-implementation mismatch (cosmetic, unfixed): the spec doc says per-file
sha256 was "removed as over-engineered"; the shipped `hub.py` **implements it**.

---

## 6. Running on the Bocconi HPC cluster — everything learned

### 6.1 Access

- `ssh 3157425@lnode01-da.hpc.unibocconi.it` — the standard login node (was down
  earlier; `slnode02-da.hpc.unibocconi.it` was a temporary fallback). At one point
  the hostname was **DNS-unresolvable off the campus network / VPN**.
- Pubkey: generated `~/.ssh/bocconi_hpc_ed25519` locally, user added the public
  key to `~/.ssh/authorized_keys` on the cluster. `~/.ssh/config` alias
  `bocconi-hpc` (currently pointed at `slnode02-da`; override with
  `-o HostName=lnode01-da.hpc.unibocconi.it`).
- **Pubkey auth was silently rejected on the temporary node** even with correct
  perms — that node had it disabled during the outage. The standard node works.

### 6.2 Filesystems — HOME IS OVER QUOTA

- `$HOME` (`homesserver:/share/home`): **182 GB used / 180 GB soft / 200 GB hard**.
  Cannot clone, build a venv, or write an HF token cache there. Any job that
  writes to `~/.cache/*` will die on quota.
- `/mnt/beegfsnew` (beegfs, 245 TB, ~93 TB free): the scratch parallel FS.
  - `/mnt/beegfsnew/<uid>/` — you can `mkdir` it but **file writes fail with
    "Remote I/O error"** (no storage targets provisioned for a self-made dir).
  - **`/mnt/beegfsnew/scratch/<uid>/` works** — the shared `scratch/` dir is
    `drwxrwsrwt` (sticky, setgid) and provisions striping correctly. **This is the
    working directory.** Repo lives at
    `/mnt/beegfsnew/scratch/3157425/llm-behaviour-intensity`.
- No `/scratch`, `/work`, `$SCRATCH`, `$WORK`.

### 6.3 Environment — use uv, not conda

- **conda on this cluster is broken for our use**: `module load miniconda3` sets
  `solver: libmamba` as default; `CONDA_NO_PLUGINS=true` disables the libmamba
  plugin → `CondaValueError: non-default solver backend (libmamba) not recognized`;
  and conda's notices cache writes to `~/.cache/conda` → quota error. `conda
  create -p <path>` also failed with `EnvironmentLocationNotFound`.
- **uv works cleanly.** `uv` at `~/.local/bin/uv`. `uv sync` reads `.python-version`
  (3.13) + `uv.lock`, creates `.venv/` in the repo, installs everything.
  `.venv/bin/python` resolves to `/software/miniconda3/bin/python3.13` (a system
  interpreter, present on compute nodes). **torch 2.11.0+cu130** — CUDA works on
  the H200.
- **Redirect every cache to scratch** or the job dies on home quota:
  ```
  export UV_CACHE_DIR=$BASE/uv_cache
  export UV_PYTHON_INSTALL_DIR=$BASE/uv_python
  export HF_HOME=$BASE/hf_home
  export XDG_CACHE_HOME=$BASE/xdg_cache
  ```
  where `BASE=/mnt/beegfsnew/scratch/3157425`.

### 6.4 HF auth on the cluster

- The token file **cannot** live at `~/.cache/huggingface/token` (quota). It is at
  **`$HF_HOME/token`** i.e. `/mnt/beegfsnew/scratch/3157425/hf_home/token`.
- The SLURM scripts `export HF_HOME=$BASE/hf_home`, so jobs authenticate
  automatically.
- The token is the org-scoped one (`llm-behaviour-intensity` read + write). The
  local machine's token is `llm-behaviour-intensity-2` (created 2026-05-05 was the
  *old* one; a re-issued/edited token with org scope is the working one; `hf auth
  login --force` on the local machine set it).
- **Getting org access is a two-step chain that tripped us for ~30 min**: (a)
  Gabriele must add `francescobraicovich` as an org member / repo collaborator on
  HF; (b) *then* the user must edit their fine-grained token to add
  `llm-behaviour-intensity` as a **scoped entity** — org membership does not
  retroactively widen an existing token.
- `google/gemma-2-2b` and `meta-llama/Llama-3.2-3B` are both **`gated=manual` but
  accessible** — the license has been accepted on the `francescobraicovich`
  account.

### 6.5 SLURM

- Template: `~/end-to-end/latent-world-models/run_pipeline.sh` — SBATCH
  conventions, `module load miniconda3`, `--mail-user francesco.braicovich@studbocconi.it`.
- **Partitions**: GPU — `long_gpuh200` (3-day), `gpuh200` (1-day), `medium_gpuh200`
  (6h), `short_gpuh200` (1h), `debug_gpuh200` (15m); also `gpua100`, `gpunew`.
  CPU — `compute` (3-day), `defq` (1-day, default), `medium_cpu` (6h), `short_cpu`
  (1h), `debug_cpu`. **H200 NVL = 143 GB VRAM**, CUDA 13.3, driver 610.
- **Script must**: `mkdir -p out err` before submit (SBATCH `--output=out/%x_%j.out`
  fails otherwise); `export MPLBACKEND=Agg` (headless matplotlib);
  `TOKENIZERS_PARALLELISM=false`; expect **block-buffered stdout** — Python `print`
  to a SLURM pipe does not flush until ~8 KB or process exit, so progress markers
  are invisible mid-run unless you use `python -u` / `PYTHONUNBUFFERED=1`. Monitor
  progress via **result files written**, not stdout.
- `git reset --hard` on the cluster repo does **not** touch gitignored `.pt` or
  untracked result dirs, but **does silently revert tracked `aggregated/` CSVs**
  (this reverted `qwen2.5-1.5b`'s CUDA aggregates to the committed MPS versions).

### 6.6 Timing (H200 / cluster CPU)

- Extraction: **~7–10 min per model** including the HF download (unauthenticated
  downloads are rate-limited and slower).
- Analysis: CPU-bound, **~35–45 min per model** for the full 3-driver × 3-seed
  set. 7B is disproportionately slow (the O(d³) whitening eigendecomps).
- **Job 653891** (first run — one `long_gpuh200` job, 5 Qwen extract + full
  analysis + summary): **3 h 52 m**, COMPLETED. The GPU sat idle for ~3 h of CPU
  analysis — wasteful, hence round 2 is split.
- **Jobs 654335 → 654336** (round 2): `654335` (GPU `medium_gpuh200`) extracts
  `gemma-2-2b` + `llama-3.2-3b` and `data_sync push`es all tensors; `654336`
  (`defq`, `--dependency=afterok:654335`) re-runs the **full 7-model** analysis
  sweep + `collect_summary` on a CPU partition. Status at time of writing: check
  `squeue -u 3157425`.

---

## 7. Results (politeness) — 6 models × 2 poolings × 3 seeds

**All CUDA-extracted, one code version** (job 654336, 2026-09-11). Qwen rows are
byte-identical to the earlier job-653891 run (same tensors, deterministic
analysis); Gemma shifted ≤0.02° from its earlier MPS extraction. Llama-3.2-3B is
**not** included — still HF-gated (403). The table below still shows the job-653891
numbers; they differ from the current committed ones only in Gemma's 3rd decimal.


`apex` = apex angle (°); CI = scenario-bootstrap 95%; `step` = step cosine;
`R` = markedness split-half reliability. Null p ≤ .001 (permutation floor) for
**every** row.

| model | size | pool | apex | 95% CI | step | R |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0.5B | avg | 36.2 | [29, 50] | −0.81 | 0.98 |
| qwen2.5-0.5b | 0.5B | last | 59.0 | [54, 67] | −0.52 | 0.92 |
| qwen2.5-1.5b | 1.5B | avg | 25.8 | [19, 45] | −0.90 | 0.98 |
| qwen2.5-1.5b | 1.5B | last | 58.7 | [54, 66] | −0.52 | 0.92 |
| qwen2.5-1.5b-instruct | 1.5B | avg | 23.6 | [18, 41] | −0.92 | 0.99 |
| qwen2.5-1.5b-instruct | 1.5B | last | 60.5 | [55, 68] | −0.49 | 0.91 |
| qwen2.5-3b | 3B | avg | 35.8 | [31, 44] | −0.81 | 0.98 |
| qwen2.5-3b | 3B | last | 66.8 | [61, 74] | −0.39 | 0.92 |
| qwen2.5-7b | 7B | avg | 31.5 | [19, 82] | −0.85 | 0.91 |
| qwen2.5-7b | 7B | last | 71.3 | [66, 77] | −0.32 | 0.91 |
| gemma-2-2b | 2B | avg | 73.2 | [68, 78] | −0.29 | 0.94 |
| gemma-2-2b | 2B | last | 76.8 | [70, 83] | −0.23 | 0.89 |

Cross-combo (`meta_analysis.json`):

- **Fisher-combined linear-null p = 3.6 × 10⁻²³**; 100% of combos individually
  significant at α = 0.05.
- **Bend vs scale**: Spearman ρ = −0.24, p = 0.46 (n = 12) — no trend.
- **Instruct**: qwen2.5-1.5b 42.3° vs 1.5b-instruct 42.1° mean apex.
- **Pooling**: `avg` bends ~2× harder than `last` in every model (avg mean apex
  ≈ 30°, last ≈ 63°) — **unexplained**.
- Within-scenario ordinality holds: Spearman 0.26–0.90, probe R² 0.79–0.92 — the
  ladder *is* ordered and linearly *decodable*; it is the **geometry** that is
  bent. (Linear decodability ≠ linear geometry.) Note the `avg`-pooling
  within-scenario Spearman is low for the mid Qwen models (0.26–0.40) while probe
  R² stays high — the probe uses all dimensions, the rank correlation is along one.
- `shared_bend_r2_cv` ranges 0.04 (qwen7b avg) – 0.36 (qwen3b avg).

Shared-plane centroids (avg pooling, `(valence, markedness)`) — the literal shape:

```
qwen2.5-1.5b   neg(-6.8, -5.8)  neu(4.5, +11.5)  pos(2.3, -5.8)   apex 25.8°
qwen2.5-7b     neg(-10.8,-6.8)  neu(7.3, +13.5)  pos(3.6, -6.8)   apex 31.5°
gemma-2-2b     neg(-14.3,-5.1)  neu(3.8, +10.2)  pos(10.5,-5.1)   apex 73.2°
```

Neutral's markedness coordinate (+10 to +13.5) vs the poles (≈ −5 to −7) **is**
the bend. Gemma's apex is rounder because its **poles are wider apart in valence**
(−14.3 → +10.5), not because its markedness offset is smaller.

---

## 8. What the multi-model results confirm — and don't

**Confirm** (the bend is not an artifact of):

- a **single model** — 6 models, 2 architectures (Qwen2.5, Gemma-2);
- **model scale** — flat 0.5B → 7B (14×), ρ ≈ 0 with p = 0.46;
- **instruction tuning** — base and instruct identical to ~0.2°;
- **token pooling** — present under both `avg` and `last` (magnitude differs,
  direction of the finding does not);
- **sampling noise** — scenario-bootstrap CIs exclude 180° everywhere; two
  independent nulls (label permutation, linear-ladder + empirical noise) rejected
  at the floor; Fisher p ≈ 10⁻²³;
- **per-scenario scatter** — the markedness axis is a *shared* direction, split-half
  R = 0.89–0.99, neutral on the same markedness side in ~98%+ of scenarios;
- **lexical give-away** — BoW baseline does not match probe accuracy (the trait is
  not just marker words).

**Do not yet confirm**:

- **cross-trait generality** — politeness only; formality / certainty / urgency /
  enthusiasm are registered but **not generated** (no `OPENROUTER_API_KEY`);
- **that the bend is functional / causal** — every metric is observational; the
  "steering alignment / capture" numbers are projections, not interventions;
- **layer generality** — `collect_summary` reports the **focal layer only**; each
  driver computes a layer sweep but those are not aggregated into the summary;
- ~~**cross-hardware consistency**~~ **RESOLVED (job 654336, 2026-09-11)** — all
  six models are now CUDA-extracted and analysed under one code version. The
  MPS→CUDA re-extraction of Gemma shifted its apex angle by **0.01°** and step
  cosine by 0.0002 (bf16 rounding washes out at the geometry level) — this is now
  a *positive* result: cross-hardware reproducibility is excellent;
- **non-pragmatic traits** — all 5 planned traits are speaker-stance / social
  constructs and are conceptually **correlated**; nothing orthogonal (e.g. a
  numeric or perceptual graded property) is tested.

---

## 9. What the paper currently includes (`paper/main.tex`, `paper/AppendixC.tex`)

**Structure**: Introduction · Background and Related Work · Dataset Construction ·
Experiments {3.1 "A clean binary axis exists", 3.2 "The graded ladder is ordered
but bent", 3.3 "No universal direction for steering"} · Conclusion and Limitations
· Appendices {dataset construction details, "Estimation, geometry, and
significance", AppendixC "Full results under both token poolings" (binary axis /
graded-ladder geometry / ordinal structure and the bend / steering geometry and
subspace dimensionality)}.

**Framing as written**: **single trait (politeness)**, reads as a case study. The
stated contributions are (1) "we test graded linearity, not binary separability" —
adding an explicit `NEUTRAL` makes linearity falsifiable; (2) "we probe a
high-level pragmatic trait, not a lexical or externally grounded one." Both are
said to "demand strict content control."

**Related-work position**: Tigges 2023 & Konen 2024 recover single sentiment/emotion
directions but treat them as **bipolar** and never test an intermediate;
Heinzerling & Inui 2024 come closest (linear vs monotonic) but only for
**externally grounded numeric properties that vary across entities**; NLP
politeness work (Danescu-Niculescu-Mizil 2013; controllable rewriters; style
transfer) is **behavioural, not representational**.

**NOT in the paper**: the multi-model scale axis, the instruction-tuning
comparison, any cross-trait evidence, the sharpened corrective framing ("*all*
prior linearity work was measured between two categories"), the cross-combo
meta-analysis (Fisher, scale correlation), and any causal / steering intervention.
**The draft has not been updated to reflect any of this session's multi-model
work.**

---

## 10. The paper's weaknesses (reviewer's eye, from first principles)

1. **No causal evidence — the dominant weakness.** Everything is observational:
   centroid geometry, probe R², null tests. `trait_geometry`'s "steering" section
   projects onto axes; it does not intervene on a forward pass. An interpretability
   reviewer's expected objection: *"You have shown the neutral centroid is
   off-axis and that the off-axis direction is reliable. You have not shown the
   model uses it."* Without one intervention, this is "a careful geometric
   observation," not "a claim about how graded concepts are represented."

2. **"So what" / impact.** The result is corrective / negative — "linearity was
   assumed; it is actually a bend." Unless tied to a downstream consequence (does
   the bend break binary steering vectors? does it imply a better probe?), the
   paper ends at "here is a bend."

3. **Human validation is essentially absent** (§3.5): n = 2, against a superseded
   dataset. No inter-annotator agreement on the trait labels — a dataset-quality
   reviewer will flag this hard.

4. **Trait diversity is narrow.** The 5 planned traits are all speaker-stance /
   social constructs and correlate conceptually. "Generalizes across traits" is
   softer than 5 independent constructs would make it. (`enthusiasm ≈ valence`
   bridges to Tigges — a partial mitigation.)

5. **Novelty delta vs Heinzerling & Inui 2024** needs a sharp statement:
   pragmatic / no external scalar / same-content rewrite **vs** numeric /
   across-entities / externally grounded. The distinction is real but a reviewer
   will probe it.

6. **Layer coverage.** The headline numbers are the focal (mid) layer only. The
   bend could be layer-specific; the computed layer sweeps are not surfaced.

7. **Mixed extraction hardware** (Gemma MPS, Qwen CUDA) in the current results —
   inconsistent provenance a reviewer would notice. Job 654336 fixes it; must be
   done before any figure mixes the two.

8. **`avg` pooling bends ~2× harder than `last`, unexplained.** Not a threat to
   the finding (both bend) but a loose end a reviewer will ask about.

9. **`qwen2.5-7b` `avg` instability**: apex CI [19°, 82°], `shared_bend_r2_cv` =
   0.04. The `last`-pooling 7B estimate is tight; the `avg` one needs more seeds
   or a per-scenario diagnostic before it goes in a table.

10. **Effect-size interpretation.** Apex 24–77° is "very bent," but the paper
    should be explicit about what magnitude of deviation from 180° is
    *substantively* (not just statistically) meaningful, and how the "effective
    number of vertices" / participation-ratio numbers bear on that.

---

## 11. Venue read

- **As-is** (observational, 5 correlated pragmatic traits, no human eval): a
  **strong workshop paper**, and a **borderline main-conference** paper expecting
  split reviews (rigor appreciated; "no causal evidence / incremental / unclear
  impact" rejections likely).
- **Main-track fit**: ICLR ≈ NeurIPS > ICML (ICLR has the strongest
  representation-geometry identity).
- **Workshops** (very strong fit): NeurIPS Mechanistic Interpretability; ICLR
  Re-Align (Representational Alignment) or BGPT; ICML Actionable Interpretability /
  Mech Interp.
- **Not** a NeurIPS D&B paper (dataset is careful but not novel enough to be the
  contribution).
- The single lever discussed for moving it to a clear main-track submission: one
  causal experiment (steer markedness vs valence vs random; **or** show a binary
  `pos − neg` steering vector overshoots neutral *because* of the bend). Steering
  is easy to implement (~30-line forward hook, all direction vectors already
  computed); the cost is the α-coherence sweep and the measurement design.
  Measurement options: representation-recovery (weak), blind LLM judge (the
  existing `deepseek` judge, ~1 day), marker-token logprob shift, or the
  "binary-vector-overshoots" comparison. A first answer is ~3–4 focused days for a
  politeness-only, focal-layer-only version.

---

## 12. Exact engineering state at hand-off

- **Branch**: `multi-model-seeded-analysis`. Local `HEAD = 22085e9`
  ("data: push qwen2.5-0.5b and qwen2.5-1.5b-instruct politeness representations").
- **HF Hub** `llm-behaviour-intensity/activations`: 5 Qwen tensors pushed (CUDA)
  at revision `2e03b15…`; Gemma still MPS from the earlier local push. Job 654335
  re-pushes everything + adds Gemma/Llama CUDA and re-pins.
- **Local repo working tree**: 235 modified + 14 untracked result files from job
  653891's tarball (extracted into `results/20260530_001930/`) — **NOT committed**
  (waiting for job 654336's clean 7-model results). Local
  `representations.lock.json` is **stale** (`11264b3cbeaa…`).
- **`git remote origin`** still `github.com/gbettineschi/llm-behaviour-intensity-representation`
  — the repo moved to `github.com/llm-behaviour-intensity/llm-behaviour-intensity`.
  GitHub's redirect works (all pushes succeed); needs
  `git remote set-url origin https://github.com/llm-behaviour-intensity/llm-behaviour-intensity.git`
  (Claude Code's safety classifier blocks the assistant from running it).
- **Cluster repo**: `/mnt/beegfsnew/scratch/3157425/llm-behaviour-intensity`, on
  `22085e9`, `.venv` built, HF auth working, `out/` `err/` present, scripts
  `run_pipeline.sh` (round 1), `extract_gd.sh` + `analyse_all.sh` (round 2).
- **Jobs**: `653891` COMPLETED (5 Qwen). `654335` COMPLETED (Gemma extracted;
  **Llama-3.2-3B FAILED — 403 `GatedRepoError`**: the Meta license is *not*
  actually approved on the `francescobraicovich` account despite `model_info`
  returning `gated=manual`. Needs a real access request at
  huggingface.co/meta-llama/Llama-3.2-3B + Meta's approval). `654336` COMPLETED
  (4 h 08 m — full 6-model CUDA analysis + summary). **Results integrated into the
  local repo 2026-09-11**; Hub at revision `829a221ace7c` (334 files, 6 models).
- **Coworker brief artifact**: <https://claude.ai/code/artifact/7525652a-6438-41b6-b9f2-6afb2994f237>
  — 6-model numbers; needs updating to 7 once 654336 lands.
- **The 4 trait datasets**: registered in `src/lib/traits.py`; **not generated**
  (`.env` with `OPENROUTER_API_KEY` absent from the repo).
- **Llama-3.2-3B**: license granted; extraction is part of job 654335 (its first
  ever extraction — it was HF-gated for the whole project until now).

---

## 13. File map

| path | role |
|---|---|
| `src/lib/traits.py` | trait registry (rubrics, intents, gates data) |
| `src/lib/config.py` | model registry, path helpers, `child_seed`, `run_metadata`, `git_commit` |
| `src/lib/sentences.py` | dataset generation pipeline (Pipeline, gates, LLM client) |
| `src/lib/representations.py` | extraction, pooling, `load_representations`, unembedding covariance |
| `src/lib/inner_products.py` | 5 whitening spaces, disattenuated cosine |
| `src/lib/analysis.py` | scenario-grouped CV, within-scenario centering, difference vectors |
| `src/lib/directions.py` | MeanDiff/KMeans/LogReg/PCA direction estimators |
| `src/lib/hub.py` | Hub lock read/write, `verify_lock`, `allow_patterns` |
| `src/lib/human_eval.py` | stdlib-only human-eval harness |
| `src/lib/figures.py` `src/lib/exports.py` | plotting style, CSV/tex/json writers |
| `src/generate_sentences.py` | dataset entry point (`--trait`, `--data-root`) |
| `src/extract_representations.py` | extraction entry point (`--model --trait --token-pooling --batch-size`) |
| `src/replicate_tigges.py` | driver — binary-contrast replication |
| `src/ordinal_linearity.py` | driver — graded linearity, 5 spaces, lexical baseline |
| `src/trait_geometry.py` | driver — triangle geometry, nulls, shared plane, per-intent, obs. steering |
| `src/run_analyses.py` | sweep orchestrator (models × traits × poolings × drivers × seeds) |
| `src/aggregate_results.py` | per-seed → `aggregated/` mean±std |
| `src/collect_summary.py` | `summary.csv` + `meta_analysis.json` (Fisher, scale ρ, breakdowns) |
| `src/data_sync.py` | Hub push / pull / verify |
| `paper/main.tex` `paper/AppendixC.tex` | the draft (single-trait framing) |
| `docs/superpowers/specs/2026-08-03-artifact-storage-design.md` | Hub-storage design doc |
| `docs/superpowers/plans/2026-08-03-artifact-storage.md` | its implementation plan |
| `docs/dataset-card.md` | HF dataset card |
| `.githooks/pre-commit` | staged-`.pt` commit guard |
| `tests/test_analysis.py` (16) `tests/test_hub.py` (10) `tests/test_representations_errors.py` (3) `tests/test_pre_commit_hook.sh` (3) | plain-python test suites (no pytest) |

---

## 14. Reproduce / continue commands

```bash
# local: analysis only (tensors from the Hub)
uv run python src/data_sync.py pull --run 20260530_001930
uv run python src/run_analyses.py --traits politeness --seeds 0,1,2
uv run python src/collect_summary.py

# generate a missing trait dataset (needs OPENROUTER_API_KEY in ./.env)
uv run python src/generate_sentences.py --trait formality --data-root data/20260530_001930

# cluster: ssh 3157425@lnode01-da.hpc.unibocconi.it
BASE=/mnt/beegfsnew/scratch/3157425
export UV_CACHE_DIR=$BASE/uv_cache UV_PYTHON_INSTALL_DIR=$BASE/uv_python \
       HF_HOME=$BASE/hf_home XDG_CACHE_HOME=$BASE/xdg_cache MPLBACKEND=Agg
cd $BASE/llm-behaviour-intensity
git fetch origin && git reset --hard origin/multi-model-seeded-analysis   # NB: reverts tracked aggregated/ CSVs
sbatch extract_gd.sh                       # GPU: extract + push
sbatch --dependency=afterok:<jid> analyse_all.sh   # CPU: full sweep + summary
squeue -u 3157425
```

Verification suite: `uv run python tests/test_analysis.py` (+ `test_hub.py`,
`test_representations_errors.py`, `sh tests/test_pre_commit_hook.sh`). Regression
check: a byte-identical rerun of one committed combo's numeric outputs.

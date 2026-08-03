---
license: mit
task_categories:
  - feature-extraction
tags:
  - interpretability
  - activations
  - behavioural-traits
---

# Behavioural Trait Intensity — Model Representations

Hidden-state activations extracted for the study *Internal Representations of
Behavioural Trait Intensity in an LLM* (Bocconi University, 2026).

## Layout

```
<run_id>/representations/<model>/<trait>/<pooling>_token/layer_<n>.pt
<run_id>/representations/<model>/unembeddings_covariance.pt
```

Each `layer_<n>.pt` is a pickled dict keyed by
`(trait, intensity, scenario_id, paraphrase_id)` mapping to a `d_model` tensor —
one vector per paraphrase, taken from the prompt prefill pass. Poolings are
`avg` (mean over content tokens) and `last` (final content token).

`unembeddings_covariance.pt` is model-level, not per-trait: it depends only on
the model's output embeddings.

## Models and traits

Models: `gemma-2-2b`, `llama-3.2-3b`, `qwen2.5-0.5b`, `qwen2.5-1.5b`,
`qwen2.5-1.5b-instruct`, `qwen2.5-3b`, `qwen2.5-7b`.

Traits: `politeness`, `formality`, `certainty`, `urgency`, `enthusiasm` — each
on a signed three-level scale (negative / neutral / positive).

## Provenance

Extraction is deterministic; seeds affect only the analysis stage. Every run is
pinned from the source repository by `data/<run_id>/representations.lock.json`.

Source code: https://github.com/gbettineschi/llm-behaviour-intensity-representation

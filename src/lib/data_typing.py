import json
from pathlib import Path
from typing import NamedTuple

TRAITS = ["politeness", "hedging_confidence"]
LEVELS = ["low", "mid", "high"]


class Sample(NamedTuple):
    prompt: str
    trait: str
    intensity: str
    scenario_id: str = ""


def load_prompts(path: str | Path) -> list[Sample]:
    """Load samples from the canonical export format (data/prompts.json)."""
    with open(path) as f:
        records = json.load(f)
    return [Sample(r["prompt"], r["trait"], r["intensity"]) for r in records]


def load_accepted(path: str | Path) -> list[Sample]:
    """Load samples from a pipeline-output accepted.jsonl file."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                samples.append(Sample(
                    prompt=r["text"],
                    trait=r["trait"],
                    intensity=r["level"],
                    scenario_id=r["scenario_id"],
                ))
    return samples

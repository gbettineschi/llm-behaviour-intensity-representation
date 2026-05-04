import json
from pathlib import Path
from typing import NamedTuple


class Sample(NamedTuple):
    prompt: str
    trait: str
    intensity: str


def load_prompts(path: str | Path) -> list[Sample]:
    """Load prompt records from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to a JSON file containing a list of ``{prompt, trait, intensity}`` objects.

    Returns
    -------
    list[Sample]
        One ``Sample`` per record, preserving file order.
    """
    with open(path) as f:
        records = json.load(f)
    return [Sample(r["prompt"], r["trait"], r["intensity"]) for r in records]

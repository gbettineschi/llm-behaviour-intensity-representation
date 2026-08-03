# Artifact Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move model representations out of Git onto a Hugging Face dataset repo, pinned per run by a committed lock file, and stop committing per-seed analysis dumps.

**Architecture:** Git keeps code, sentence data, and aggregated results. A private HF dataset repo keeps `.pt` tensors, mirroring the local directory layout so `snapshot_download` lands files exactly where the analysis code already reads them. A committed `representations.lock.json` per run pins the HF revision. Analysis code is untouched apart from one actionable error message.

**Tech Stack:** Python 3.13, uv, `huggingface_hub` 1.13.0 (already in `uv.lock` via `transformers`), plain-Python tests (no pytest).

## Global Constraints

- **No pytest.** Tests are plain Python scripts run as `.venv/bin/python tests/<name>.py`, following `tests/test_analysis.py`. Each defines test functions, a `main()` that runs them collecting `AssertionError`s, prints `All N tests PASSED`, and `sys.exit(1)` on failure.
- **Tests reach `src/` via** `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))` followed by imports with `# noqa: E402`.
- **Scripts inside `src/` import siblings as** `from lib.hub import ...` (no `src.` prefix) — Python puts the script's directory on `sys.path`.
- **No precision or tensor-layout changes.** Results must stay bit-for-bit identical.
- **HF repo id default:** `gbettineschi/llm-behaviour-intensity-representations`, `repo_type="dataset"`, `private=True`.
- **Run id in use:** `20260530_001930`.
- **Lint:** `ruff` is the only dev dependency; keep lines under 100 chars.

---

### Task 1: Lock-file helpers

**Files:**
- Create: `src/lib/hub.py`
- Test: `tests/test_hub.py`

**Interfaces:**
- Consumes: nothing
- Produces: `LOCK_NAME: str`, `DEFAULT_REPO_ID: str`, `lock_path(data_root: str | Path) -> Path`, `write_lock(data_root, *, repo_id: str, revision: str, run_id: str, files: list[str]) -> Path`, `read_lock(data_root) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hub.py`:

```python
"""Tests for lib.hub lock files and Hub glob patterns.

Run with plain Python (no pytest dependency)::

    .venv/bin/python tests/test_hub.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib.hub import lock_path, read_lock, write_lock  # noqa: E402


def test_lock_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "20260530_001930"
        files = ["representations/gemma-2-2b/politeness/avg_token/layer_1.pt"]
        written = write_lock(
            root, repo_id="u/r", revision="a" * 40, run_id="20260530_001930", files=files
        )
        assert written == lock_path(root), f"unexpected lock path {written}"
        lock = read_lock(root)
        assert lock["repo_id"] == "u/r", lock
        assert lock["revision"] == "a" * 40, lock
        assert lock["run_id"] == "20260530_001930", lock
        assert lock["files"] == files, lock
    print("  test_lock_roundtrip PASSED")


def test_write_lock_sorts_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        write_lock(root, repo_id="u/r", revision="b" * 40, run_id="run", files=["b.pt", "a.pt"])
        assert read_lock(root)["files"] == ["a.pt", "b.pt"], "files must be sorted"
    print("  test_write_lock_sorts_files PASSED")


def test_read_lock_missing_names_push_command():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "20260530_001930"
        root.mkdir(parents=True)
        try:
            read_lock(root)
        except FileNotFoundError as e:
            assert "data_sync.py push" in str(e), f"unhelpful message: {e}"
            assert "20260530_001930" in str(e), f"message omits run id: {e}"
        else:
            raise AssertionError("read_lock must raise when the lock file is absent")
    print("  test_read_lock_missing_names_push_command PASSED")


def main():
    tests = [
        test_lock_roundtrip,
        test_write_lock_sorts_files,
        test_read_lock_missing_names_push_command,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ASSERTION FAILED: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python tests/test_hub.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.hub'`

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/hub.py`:

```python
"""Hugging Face Hub sync for model representations.

Tensors are not stored in Git. They live in a Hub dataset repo and are pinned
per run by ``data/<run_id>/representations.lock.json``.
"""

import json
from pathlib import Path

LOCK_NAME = "representations.lock.json"
DEFAULT_REPO_ID = "gbettineschi/llm-behaviour-intensity-representations"


def lock_path(data_root: str | Path) -> Path:
    """``data/<run_id>/representations.lock.json``."""
    return Path(data_root) / LOCK_NAME


def write_lock(
    data_root: str | Path,
    *,
    repo_id: str,
    revision: str,
    run_id: str,
    files: list[str],
) -> Path:
    """Record which Hub revision holds this run's tensors. Returns the lock path."""
    path = lock_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "run_id": run_id,
        "files": sorted(files),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def read_lock(data_root: str | Path) -> dict:
    """Load the lock file, or explain how to create it."""
    path = lock_path(data_root)
    if not path.exists():
        run_id = Path(data_root).name
        raise FileNotFoundError(
            f"No representation lock at {path}.\n"
            f"Upload this run's tensors first:\n"
            f"    python src/data_sync.py push --run {run_id}"
        )
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python tests/test_hub.py`
Expected: PASS — `All 3 tests PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/lib/hub.py tests/test_hub.py
git commit -m "feat: add representation lock file helpers"
```

---

### Task 2: Hub glob patterns for selective pull

**Files:**
- Modify: `src/lib/hub.py` (append)
- Test: `tests/test_hub.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1
- Produces: `allow_patterns(run_id: str, *, model: str | None = None, trait: str | None = None) -> list[str]`

**Why the unembedding covariance is special:** it lives at `<run>/representations/<model>/unembeddings_covariance.pt` — model-level, not under a trait directory. A trait filter would otherwise exclude it and break `trait_geometry.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hub.py`, importing `allow_patterns` alongside the existing imports:

```python
def test_allow_patterns_no_filters():
    assert allow_patterns("R") == ["R/representations/*/**"], allow_patterns("R")
    print("  test_allow_patterns_no_filters PASSED")


def test_allow_patterns_model_only():
    got = allow_patterns("R", model="gemma-2-2b")
    assert got == ["R/representations/gemma-2-2b/**"], got
    print("  test_allow_patterns_model_only PASSED")


def test_allow_patterns_trait_only_keeps_unembed_cov():
    got = allow_patterns("R", trait="politeness")
    assert got == [
        "R/representations/*/politeness/**",
        "R/representations/*/unembeddings_covariance.pt",
    ], got
    print("  test_allow_patterns_trait_only_keeps_unembed_cov PASSED")


def test_allow_patterns_model_and_trait():
    got = allow_patterns("R", model="gemma-2-2b", trait="politeness")
    assert got == [
        "R/representations/gemma-2-2b/politeness/**",
        "R/representations/gemma-2-2b/unembeddings_covariance.pt",
    ], got
    print("  test_allow_patterns_model_and_trait PASSED")
```

Add all four to the `tests` list in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python tests/test_hub.py`
Expected: FAIL with `ImportError: cannot import name 'allow_patterns' from 'lib.hub'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/lib/hub.py`:

```python
def allow_patterns(
    run_id: str, *, model: str | None = None, trait: str | None = None
) -> list[str]:
    """Hub glob patterns selecting a subset of one run's representations.

    The unembedding covariance is model-level rather than per-trait, so it stays
    included whenever a trait filter narrows the layer files.
    """
    base = f"{run_id}/representations"
    m = model or "*"
    if trait is None:
        return [f"{base}/{m}/**"]
    return [f"{base}/{m}/{trait}/**", f"{base}/{m}/unembeddings_covariance.pt"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python tests/test_hub.py`
Expected: PASS — `All 7 tests PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/lib/hub.py tests/test_hub.py
git commit -m "feat: add Hub glob patterns for selective representation pull"
```

---

### Task 3: Actionable error when tensors are absent

**Files:**
- Modify: `src/lib/representations.py:118-132` (the `load_representations` body)
- Test: `tests/test_representations_errors.py`

**Interfaces:**
- Consumes: nothing
- Produces: `load_representations` keeps its signature `(rep_dir: str | Path, *, layer: int | None = None)` and raises `FileNotFoundError` naming the pull command when the directory is missing or holds no `layer_*.pt`.

**Why:** with tensors gitignored, a fresh clone previously produced either a bare `FileNotFoundError` on `layer_N.pt` or, worse, a silently empty dict from the glob branch.

- [ ] **Step 1: Write the failing test**

Create `tests/test_representations_errors.py`:

```python
"""The missing-tensor error must tell you how to fetch the tensors.

Run with plain Python (no pytest dependency)::

    .venv/bin/python tests/test_representations_errors.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib.representations import load_representations  # noqa: E402


def _rep_dir(tmp: str) -> Path:
    return Path(tmp) / "data" / "20260530_001930" / "representations" / "m" / "t" / "avg_token"


def test_missing_directory_names_pull_command():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_representations(_rep_dir(tmp))
        except FileNotFoundError as e:
            assert "data_sync.py pull" in str(e), f"unhelpful message: {e}"
            assert "20260530_001930" in str(e), f"message omits run id: {e}"
        else:
            raise AssertionError("must raise when the representation directory is absent")
    print("  test_missing_directory_names_pull_command PASSED")


def test_empty_directory_does_not_return_empty_dict():
    with tempfile.TemporaryDirectory() as tmp:
        d = _rep_dir(tmp)
        d.mkdir(parents=True)
        try:
            load_representations(d)
        except FileNotFoundError as e:
            assert "data_sync.py pull" in str(e), f"unhelpful message: {e}"
        else:
            raise AssertionError("an empty directory must raise, not return {}")
    print("  test_empty_directory_does_not_return_empty_dict PASSED")


def test_explicit_layer_also_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_representations(_rep_dir(tmp), layer=5)
        except FileNotFoundError as e:
            assert "data_sync.py pull" in str(e), f"unhelpful message: {e}"
        else:
            raise AssertionError("the layer= branch must raise too")
    print("  test_explicit_layer_also_raises PASSED")


def main():
    tests = [
        test_missing_directory_names_pull_command,
        test_empty_directory_does_not_return_empty_dict,
        test_explicit_layer_also_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ASSERTION FAILED: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python tests/test_representations_errors.py`
Expected: FAIL — `test_empty_directory_does_not_return_empty_dict` reports "an empty directory must raise, not return {}", and the other two fail on the unhelpful default message.

- [ ] **Step 3: Write minimal implementation**

In `src/lib/representations.py`, add above `load_representations`:

```python
def _run_id_from_rep_dir(rep_dir: Path) -> str:
    """``data/<run_id>/representations/...`` -> ``<run_id>``."""
    for parent in rep_dir.parents:
        if parent.name == "representations":
            return parent.parent.name
    return "<run_id>"
```

Then replace the body of `load_representations` (keeping its docstring) so it begins:

```python
    rep_dir = Path(rep_dir)
    if not rep_dir.is_dir() or not any(rep_dir.glob("layer_*.pt")):
        raise FileNotFoundError(
            f"No representations under {rep_dir}.\n"
            f"Tensors are not stored in Git. Fetch them from the Hugging Face Hub:\n"
            f"    python src/data_sync.py pull --run {_run_id_from_rep_dir(rep_dir)}"
        )
    if layer is not None:
        return torch.load(rep_dir / f"layer_{layer}.pt", weights_only=False)
```

The remainder of the function (the glob loop building `out`) is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python tests/test_representations_errors.py`
Expected: PASS — `All 3 tests PASSED`

Then confirm nothing regressed: `.venv/bin/python tests/test_analysis.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/representations.py tests/test_representations_errors.py
git commit -m "feat: point missing-tensor errors at the data_sync pull command"
```

---

### Task 4: The `data_sync` CLI

**Files:**
- Create: `src/data_sync.py`
- Modify: `pyproject.toml` (add `huggingface-hub` to `dependencies`)

**Interfaces:**
- Consumes: `lib.hub.DEFAULT_REPO_ID`, `lib.hub.allow_patterns`, `lib.hub.read_lock`, `lib.hub.write_lock`
- Produces: CLI only — `python src/data_sync.py {push,pull} --run <run_id> [--model M] [--trait T] [--repo R]`

**Note:** `huggingface_hub` 1.13.0 is already resolved in `uv.lock` transitively via `transformers`. Declaring it explicitly is correct because this module imports it directly. Verified in that version: `CommitInfo.oid` exists, `HfApi.upload_folder` accepts `path_in_repo`, and `snapshot_download` accepts `allow_patterns` and `local_dir`.

This task has no unit test — it is thin I/O glue over the Hub client, and its only non-trivial logic (`allow_patterns`) is already tested in Task 2. Step 5 is a real round-trip against the Hub instead.

- [ ] **Step 1: Declare the dependency**

In `pyproject.toml`, add to the `dependencies` list:

```toml
    "huggingface-hub>=1.13.0",
```

Run: `uv sync`
Expected: succeeds, no version change to `huggingface-hub` in `uv.lock`.

- [ ] **Step 2: Write the CLI**

Create `src/data_sync.py`:

```python
"""Push and pull model representations to a Hugging Face dataset repo.

Tensors are not stored in Git. After extraction, push them; before analysis,
pull them. The committed ``representations.lock.json`` pins the exact revision.

    python src/data_sync.py push --run 20260530_001930
    python src/data_sync.py pull --run 20260530_001930
    python src/data_sync.py pull --run 20260530_001930 --model gemma-2-2b --trait politeness
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from lib.hub import DEFAULT_REPO_ID, allow_patterns, read_lock, write_lock

DATA_ROOT = Path("data")


def cmd_push(args: argparse.Namespace) -> None:
    data_root = DATA_ROOT / args.run
    rep_dir = data_root / "representations"
    if not rep_dir.is_dir():
        raise SystemExit(f"nothing to push: {rep_dir} does not exist")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(rep_dir),
        path_in_repo=f"{args.run}/representations",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"representations for run {args.run}",
    )

    files = sorted(str(p.relative_to(data_root)) for p in rep_dir.rglob("*.pt"))
    path = write_lock(
        data_root, repo_id=args.repo, revision=commit.oid, run_id=args.run, files=files
    )
    print(f"pushed {len(files)} tensors to {args.repo} at {commit.oid}")
    print(f"wrote {path} -- commit it so others pull the same revision")


def cmd_pull(args: argparse.Namespace) -> None:
    data_root = DATA_ROOT / args.run
    lock = read_lock(data_root)
    patterns = allow_patterns(args.run, model=args.model, trait=args.trait)
    snapshot_download(
        repo_id=lock["repo_id"],
        repo_type="dataset",
        revision=lock["revision"],
        allow_patterns=patterns,
        local_dir=str(DATA_ROOT),
    )
    print(f"pulled {patterns} at {lock['revision'][:7]} into {DATA_ROOT}/{args.run}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, needs_filters in (("push", cmd_push, False), ("pull", cmd_pull, True)):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, help="run id, e.g. 20260530_001930")
        p.add_argument("--repo", default=DEFAULT_REPO_ID, help="Hub dataset repo id")
        if needs_filters:
            p.add_argument("--model", default=None, help="fetch only this model")
            p.add_argument("--trait", default=None, help="fetch only this trait")
        p.set_defaults(func=handler)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the CLI parses**

Run: `uv run python src/data_sync.py pull --help`
Expected: help text listing `--run`, `--repo`, `--model`, `--trait`

Run: `uv run python src/data_sync.py push --help`
Expected: help text listing `--run` and `--repo` but **not** `--model`/`--trait`

- [ ] **Step 4: Verify the missing-lock path**

Run: `uv run python src/data_sync.py pull --run nonexistent-run`
Expected: `FileNotFoundError` whose message contains `python src/data_sync.py push --run nonexistent-run`

- [ ] **Step 5: Real round-trip against the Hub**

Requires `hf auth login` to have been run.

```bash
uv run python src/data_sync.py push --run 20260530_001930
```
Expected: prints `pushed 45 tensors to <repo> at <sha>` and writes `data/20260530_001930/representations.lock.json`.

Verify the lock file has a 40-character revision and 45 entries:
```bash
uv run python -c "
import json
lock = json.load(open('data/20260530_001930/representations.lock.json'))
assert len(lock['revision']) == 40, lock['revision']
assert len(lock['files']) == 45, len(lock['files'])
print('lock OK:', lock['repo_id'], lock['revision'][:7], len(lock['files']), 'files')
"
```

Now prove the pull works into a clean location:
```bash
mv data/20260530_001930/representations /tmp/reps-backup
uv run python src/data_sync.py pull --run 20260530_001930 --model gemma-2-2b --trait politeness
ls data/20260530_001930/representations/gemma-2-2b/
```
Expected: `politeness/` and `unembeddings_covariance.pt` present.

Confirm the bytes are identical to what was there before:
```bash
diff <(cd /tmp/reps-backup && find . -name '*.pt' | sort | xargs shasum -a 256 | awk '{print $1}') \
     <(cd data/20260530_001930/representations && find . -name '*.pt' | sort | xargs shasum -a 256 | awk '{print $1}') \
  && echo "BYTES IDENTICAL"
```
Expected: `BYTES IDENTICAL` (after a full `pull` with no filters; restore from `/tmp/reps-backup` if anything is missing).

- [ ] **Step 6: Commit**

```bash
git add src/data_sync.py pyproject.toml uv.lock data/20260530_001930/representations.lock.json
git commit -m "feat: add data_sync push/pull for Hub-hosted representations"
```

---

### Task 5: Repo hygiene — ignore rules, retire LFS, add the guard

**Files:**
- Modify: `.gitignore`
- Delete: `.gitattributes`
- Create: `.githooks/pre-commit`
- Test: `tests/test_pre_commit_hook.sh`

**Interfaces:**
- Consumes: nothing
- Produces: a hook enabled by `git config core.hooksPath .githooks`

**Why a plain hook rather than the `pre-commit` framework:** this project keeps tooling minimal (ruff only, no pytest) and already asks contributors to run a `git config --local` command during setup for nb-clean. A ten-line hook matches that and adds no dependency.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pre_commit_hook.sh`:

```sh
#!/bin/sh
# The pre-commit hook must reject staged .pt files.
#
#   sh tests/test_pre_commit_hook.sh
set -e

HOOK="$(cd "$(dirname "$0")/.." && pwd)/.githooks/pre-commit"
[ -x "$HOOK" ] || { echo "FAIL: $HOOK missing or not executable"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
git init -q .
git config user.email t@t; git config user.name t

# A staged .pt must be rejected.
head -c 100 /dev/urandom > weights.pt
git add -f weights.pt
if "$HOOK" >/dev/null 2>&1; then
    echo "FAIL: hook allowed a staged .pt"; exit 1
fi
"$HOOK" 2>&1 | grep -q "weights.pt" || { echo "FAIL: hook did not name the file"; exit 1; }
echo "  rejects staged .pt PASSED"

# An ordinary file must pass.
git rm -q --cached weights.pt
echo "print(1)" > ok.py
git add ok.py
"$HOOK" >/dev/null 2>&1 || { echo "FAIL: hook rejected a normal commit"; exit 1; }
echo "  allows ordinary files PASSED"

echo
echo "All 2 tests PASSED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sh tests/test_pre_commit_hook.sh`
Expected: FAIL with `FAIL: .../.githooks/pre-commit missing or not executable`

- [ ] **Step 3: Write the hook**

Create `.githooks/pre-commit`:

```sh
#!/bin/sh
# Model tensors live on the Hugging Face Hub, never in Git.
# See docs/superpowers/specs/2026-08-03-artifact-storage-design.md
staged=$(git diff --cached --name-only --diff-filter=AM | grep '\.pt$' || true)
if [ -n "$staged" ]; then
    echo "error: refusing to commit PyTorch tensors:"
    echo "$staged" | sed 's/^/  /'
    echo
    echo "Tensors belong on the Hugging Face Hub:"
    echo "    python src/data_sync.py push --run <run_id>"
    exit 1
fi
```

Make it executable: `chmod +x .githooks/pre-commit`

- [ ] **Step 4: Run test to verify it passes**

Run: `sh tests/test_pre_commit_hook.sh`
Expected: PASS — `All 2 tests PASSED`

- [ ] **Step 5: Update ignore rules and retire LFS**

Append to `.gitignore`:

```
# Model representations - stored on the Hugging Face Hub, fetched with src/data_sync.py
data/**/representations/**/*.pt

# Per-seed analysis dumps - regenerate byte-identically by re-running the same seed
results/**/seed_*/
```

The lock file is `data/<run>/representations.lock.json`, a sibling of the `representations/` directory rather than inside it, so it stays tracked.

Delete the LFS rule — `.gitattributes` contains only `*.pt filter=lfs diff=lfs merge=lfs -text`, so remove the whole file:

```bash
git rm .gitattributes
git lfs uninstall --local
```

Untrack the per-seed dumps without touching history (they are small text files; a rewrite is not warranted):

```bash
git rm -r --cached --quiet results/*/*/*/*/*_token/seed_* 2>/dev/null || true
git status --porcelain | grep -c '^D ' # expect ~1572 on the multi-model branch
```

- [ ] **Step 6: Enable the hook locally and verify end to end**

```bash
git config core.hooksPath .githooks
cp data/20260530_001930/representations/gemma-2-2b/unembeddings_covariance.pt /tmp/probe.pt
cp /tmp/probe.pt ./probe.pt
git add -f probe.pt && git commit -m "should be rejected" ; echo "exit=$?"
rm -f probe.pt && git reset -q
```
Expected: the commit is rejected, `exit=1`, and the message names `probe.pt`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore .githooks/pre-commit tests/test_pre_commit_hook.sh
git commit -m "chore: ignore tensors and per-seed dumps, retire Git LFS, guard commits"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md` (replace setup step 1, add a Data section)
- Create: `docs/dataset-card.md` (uploaded to the Hub repo as its `README.md`)

**Interfaces:**
- Consumes: the CLI from Task 4, the hook from Task 5
- Produces: no code

- [ ] **Step 1: Rewrite the setup section**

In `README.md`, replace setup step 1 (the `git lfs install` block, lines 6-14) with:

```markdown
1. Clone the repo. Model representations are **not** in Git — they live in a
   Hugging Face dataset repo and are fetched separately (step 6).
```

Then add step 6 after the nb-clean step:

```markdown
6. Enable the commit guard, which stops tensors being committed by accident.
   ```
   git config core.hooksPath .githooks
   ```
7. Fetch the representations for the run you want to analyse.
   ```
   python src/data_sync.py pull --run 20260530_001930
   ```
   Narrow it down if you only need part of the sweep — the full set is large:
   ```
   python src/data_sync.py pull --run 20260530_001930 --model gemma-2-2b --trait politeness
   ```
```

- [ ] **Step 2: Add the Data section**

Append to `README.md`:

```markdown
## Data and results

| Artifact | Where it lives | Why |
| --- | --- | --- |
| Code, configs, sentence datasets | Git | small, reviewable, versions with the code |
| `results/**/aggregated/` | Git | text — metric changes show up in PR diffs |
| `results/**/seed_*/` | not tracked | regenerates byte-identically from the same seed |
| `data/**/representations/*.pt` | Hugging Face dataset repo | too large for Git; fetched with `data_sync` |

`data/<run_id>/representations.lock.json` is committed and pins the exact Hub
revision, so everyone analysing a run reads the same tensor bytes.

After extracting new representations, publish them and commit the updated lock:

```
python src/data_sync.py push --run <run_id>
git add data/<run_id>/representations.lock.json
git commit -m "data: publish representations for <run_id>"
```

**Troubleshooting.** `FileNotFoundError: No representations under ...` means you
have not fetched the tensors — run the `pull` command it prints. If a commit is
rejected with "refusing to commit PyTorch tensors", that is the guard working:
push the tensors to the Hub instead.
```

- [ ] **Step 3: Write the dataset card**

Create `docs/dataset-card.md`:

```markdown
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

## Provenance

Extraction is deterministic; seeds affect only the analysis stage. Every run is
pinned from the source repository by `data/<run_id>/representations.lock.json`.

Source code: https://github.com/gbettineschi/llm-behaviour-intensity-representation
```

- [ ] **Step 4: Upload the dataset card**

```bash
uv run python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj='docs/dataset-card.md',
    path_in_repo='README.md',
    repo_id='gbettineschi/llm-behaviour-intensity-representations',
    repo_type='dataset',
)
print('dataset card uploaded')
"
```
Expected: `dataset card uploaded`

- [ ] **Step 5: Commit**

```bash
git add README.md docs/dataset-card.md
git commit -m "docs: document Hub-hosted representations and the commit guard"
```

---

### Task 7: One-time migration

**Files:** none created — this is an operational runbook.

**Interfaces:**
- Consumes: everything from Tasks 1-6, all merged to `master`
- Produces: a repository with no `.pt` in any branch's history

**Do not start this task until PR #3 is merged or the team has explicitly agreed to it**, because the rewrite touches all three branches and everyone re-clones afterwards.

- [ ] **Step 1: Confirm the tensors are safe on the Hub first**

```bash
uv run python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('gbettineschi/llm-behaviour-intensity-representations', repo_type='dataset')
pt = [f for f in files if f.endswith('.pt')]
print(len(pt), 'tensors on the Hub')
assert len(pt) >= 45, 'refusing to proceed: Hub copy incomplete'
"
```
Expected: at least 45 tensors. **Stop if this fails** — the rewrite is not reversible from the Hub if the Hub copy is incomplete.

- [ ] **Step 2: Take a backup bundle**

```bash
git bundle create ../pre-hub-migration-backup.bundle --all
git bundle verify ../pre-hub-migration-backup.bundle
git show-ref > ../pre-hub-migration-refs.txt
```
Expected: `The bundle records a complete history.` Keep both files somewhere durable until the team confirms the new clones work.

- [ ] **Step 3: Install git-filter-repo**

```bash
brew install git-filter-repo
git filter-repo --version
```

- [ ] **Step 4: Strip tensors from every branch**

Create local branches for the remote-only ones so `--all` covers them:

```bash
git branch data-pipeline-hardening origin/data-pipeline-hardening 2>/dev/null || true
git branch multi-model-seeded-analysis origin/multi-model-seeded-analysis 2>/dev/null || true
git filter-repo --path-glob '*.pt' --invert-paths --force
```

- [ ] **Step 5: Verify the rewrite**

```bash
echo "tensors remaining in history: $(git rev-list --all --objects | git cat-file --batch-check='%(objecttype) %(rest)' | awk '$1=="blob" && $2 ~ /\.pt$/' | wc -l)"
git reflog expire --expire=now --all && git gc --prune=now
du -sh .git
```
Expected: `tensors remaining in history: 0`, and `.git` well under 50 MB (it was ~400 MB).

- [ ] **Step 6: Restore the remote and force-push**

`git filter-repo` removes the `origin` remote deliberately:

```bash
git remote add origin git@github.com:gbettineschi/llm-behaviour-intensity-representation.git
git push --force origin master data-pipeline-hardening multi-model-seeded-analysis
```

- [ ] **Step 7: Verify the end state**

```bash
git fetch --all --prune
git status --porcelain           # expect empty
find data -name '*.pt' -newer .gitignore -exec touch {} + ; git status --porcelain
```
Expected: empty both times — touching every tensor no longer dirties the tree, because tensors are no longer tracked at all. This is the regression test for the original bug.

- [ ] **Step 8: Tell the team**

Everyone must re-clone; `git pull` will not work across a rewrite. The message they need:

```
History was rewritten to remove model tensors from Git. Push any unmerged work
to a branch first, then re-clone. After cloning:
    git config core.hooksPath .githooks
    python src/data_sync.py pull --run 20260530_001930
```

---

## Self-Review

**Spec coverage:** HF dataset repo — Task 4 Step 5 / Task 6 Step 4. `src/lib/hub.py` — Tasks 1-2. `src/data_sync.py` — Task 4. Lock file — Task 1, written by Task 4, committed in Task 4 Step 6. Guards — Task 5 (hook) and Task 3 (loader error). Migration — Task 7. Results tiering — Task 5 Step 5. README and dataset card — Task 6. Non-goals respected: no precision changes, no DVC, no CI, no pytest.

**Placeholders:** none — every step carries runnable content.

**Type consistency:** `write_lock`/`read_lock`/`lock_path`/`allow_patterns` signatures match between Tasks 1, 2 and their use in Task 4. `load_representations` keeps its original signature. The `files` list in the lock is relative to `data/<run_id>/` in both writer (Task 4) and test (Task 1).

**Known gap, accepted:** Task 4 has no unit test because it is I/O glue; its logic lives in tested pure functions and Step 5 is a real round-trip with a byte-identity check.

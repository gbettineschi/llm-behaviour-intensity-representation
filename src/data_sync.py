"""Push, pull and verify model representations against a Hugging Face dataset repo.

Tensors are not stored in Git. After extraction, push them; before analysis,
pull them. The committed ``representations.lock.json`` pins the exact revision
*and* a sha256 per file, so ``verify`` can prove the bytes on your disk are the
ones a given commit refers to.

    python src/data_sync.py push   --run 20260530_001930
    python src/data_sync.py pull   --run 20260530_001930
    python src/data_sync.py pull   --run 20260530_001930 --model gemma-2-2b --trait politeness
    python src/data_sync.py verify --run 20260530_001930
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from lib.hub import (
    DEFAULT_REPO_ID,
    allow_patterns,
    file_digest,
    read_lock,
    verify_lock,
    write_lock,
)

DATA_ROOT = Path("data")
COV_NAME = "unembeddings_covariance.pt"


def _tensors_on_disk(data_root: Path, *, model: str | None, trait: str | None) -> list[Path]:
    """Layer tensors under ``data/<run>/representations`` matching the same
    filters ``allow_patterns`` selects. The model-level covariance is excluded:
    it is pulled alongside any trait filter, so it would mask an empty result."""
    root = data_root / "representations"
    if not root.is_dir():
        return []
    hits = []
    for p in root.rglob("*.pt"):
        if p.name == COV_NAME:
            continue
        parts = p.relative_to(root).parts  # (model, trait, <pooling>_token, layer_N.pt)
        if len(parts) < 4:
            continue
        if (model is None or parts[0] == model) and (trait is None or parts[1] == trait):
            hits.append(p)
    return hits


def _describe(lock: dict) -> str:
    """What the pinned revision actually holds, as `model/trait` pairs."""
    combos = sorted(
        {"/".join(f.split("/")[1:3]) for f in lock["files"] if len(f.split("/")) >= 4}
    )
    return ", ".join(combos) or "(nothing)"


def cmd_push(args: argparse.Namespace) -> None:
    data_root = DATA_ROOT / args.run
    rep_dir = data_root / "representations"
    if not rep_dir.is_dir():
        raise SystemExit(f"nothing to push: {rep_dir} does not exist")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    uploaded = sorted(str(p.relative_to(data_root)) for p in rep_dir.rglob("*.pt"))
    commit = api.upload_folder(
        folder_path=str(rep_dir),
        path_in_repo=f"{args.run}/representations",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"representations for run {args.run}",
    )

    # Read the manifest back off the Hub rather than from the local rglob.
    # upload_folder is additive, so the revision holds everything previously
    # pushed too; recording only what this machine happened to have on disk
    # would drop other models from the lock. The Hub also reports the LFS
    # sha256 for each object, so the digests come from the source of truth.
    files: dict[str, str] = {}
    prefix = f"{args.run}/"
    for entry in api.list_repo_tree(
        args.repo, repo_type="dataset", revision=commit.oid, recursive=True, expand=True
    ):
        if not entry.path.endswith(".pt") or not entry.path.startswith(prefix):
            continue
        rel = entry.path[len(prefix):]
        lfs = getattr(entry, "lfs", None)
        digest = getattr(lfs, "sha256", None) if lfs else None
        files[rel] = digest or file_digest(data_root / rel)

    path = write_lock(
        data_root, repo_id=args.repo, revision=commit.oid, run_id=args.run, files=files
    )
    print(f"uploaded {len(uploaded)} tensors from this machine")
    print(f"pinned {len(files)} tensors at {args.repo}@{commit.oid[:12]} with sha256 digests")
    print(f"wrote {path} -- commit it so others pull the same bytes")


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
    # snapshot_download matches nothing without complaining, so a typo'd --model
    # would otherwise report success and leave the absence to surface much later
    # as a FileNotFoundError from an analysis driver. Probe the filtered target
    # specifically: tensors from some *other* model must not count as success.
    got = _tensors_on_disk(data_root, model=args.model, trait=args.trait)
    if not got:
        raise SystemExit(
            f"pulled nothing: {patterns} matched no files at "
            f"{lock['repo_id']}@{lock['revision'][:7]}.\n"
            f"Pinned at this revision: {_describe(lock)}"
        )
    print(f"pulled {len(got)} tensors at {lock['revision'][:7]} into {DATA_ROOT}/{args.run}")


def cmd_verify(args: argparse.Namespace) -> None:
    data_root = DATA_ROOT / args.run
    lock = read_lock(data_root)
    res = verify_lock(data_root, model=args.model, trait=args.trait)
    print(f"lock: {lock['repo_id']}@{lock['revision'][:12]}")
    print(f"  matched    {len(res['matched'])}")
    print(f"  missing    {len(res['missing'])}   (not pulled — fine for a selective pull)")
    print(f"  MISMATCHED {len(res['mismatched'])}")
    if res["mismatched"]:
        for rel in res["mismatched"][:10]:
            print(f"    {rel}")
        raise SystemExit(
            f"\n{len(res['mismatched'])} file(s) on disk differ from the pinned revision.\n"
            "Results computed from them are not reproducible from this commit. Re-pull with:\n"
            f"    python src/data_sync.py pull --run {args.run}\n"
            "or, if the local tensors are the ones you want, publish them:\n"
            f"    python src/data_sync.py push --run {args.run}"
        )
    if not res["matched"]:
        raise SystemExit("nothing to verify: no pinned tensors are on disk")
    # Say what was checked, not just that it passed: "OK" alongside a large
    # missing count reads as "all good" when most of the run is simply absent.
    scope = f"{len(res['matched'])} of {len(res['matched']) + len(res['missing'])} pinned tensors"
    if res["missing"]:
        print(
            f"\nOK — {scope} are on disk and every one matches the pinned revision."
            f"\n{len(res['missing'])} are not on disk. If you expected a full run,"
            f" fetch them:\n    python src/data_sync.py pull --run {args.run}"
        )
    else:
        print(f"\nOK — all {scope} are on disk and match the revision this commit pins.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_help = "run id, e.g. 20260530_001930"

    push = sub.add_parser("push", help="upload a run's tensors and rewrite its lock file")
    push.add_argument("--run", required=True, help=run_help)
    push.add_argument("--repo", default=DEFAULT_REPO_ID, help="Hub dataset repo id to publish to")
    push.set_defaults(func=cmd_push)

    # No --repo on pull: the lock pins repo and revision together, so honouring a
    # repo override here would fetch one repo at another repo's revision.
    pull = sub.add_parser("pull", help="fetch tensors at the revision the lock file pins")
    pull.add_argument("--run", required=True, help=run_help)
    pull.add_argument("--model", default=None, help="fetch only this model")
    pull.add_argument("--trait", default=None, help="fetch only this trait")
    pull.set_defaults(func=cmd_pull)

    verify = sub.add_parser("verify", help="check the tensors on disk against the digests the lock pins")
    verify.add_argument("--run", required=True, help=run_help)
    verify.add_argument("--model", default=None, help="check only this model")
    verify.add_argument("--trait", default=None, help="check only this trait")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

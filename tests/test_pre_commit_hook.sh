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

# A renamed .pt must be rejected too: `git mv` stages it with status R, which an
# A/M-only --diff-filter silently lets through.
git commit -qm "add ok.py"
git add -f weights.pt
git commit -qm "add tensor (the hook is not installed in this scratch repo)"
git mv weights.pt renamed.pt
if "$HOOK" >/dev/null 2>&1; then
    echo "FAIL: hook allowed a renamed .pt"; exit 1
fi
"$HOOK" 2>&1 | grep -q "renamed.pt" || { echo "FAIL: hook did not name the renamed file"; exit 1; }
echo "  rejects renamed .pt PASSED"

echo
echo "All 3 tests PASSED"

#!/usr/bin/env python3
"""Pull the latest Overleaf edits into Mortis/.

Uncommitted local edits are stashed, Overleaf's commits are applied, and the
edits are restored on top. If they clash with Overleaf, git leaves conflict
markers in the affected files and also keeps the edits in `git stash`.

Usage: python3 pull_from_overleaf.py [PAPER]   (default Lingjia -> Mortis_Lingjia/; "" -> Mortis/)
"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = sys.argv[1] if len(sys.argv) > 1 else "Lingjia"  # Mortis/ is a stale backup; pass "" to target it
REPO = ROOT / ("Mortis" + (f"_{PAPER}" if PAPER else ""))
if not (REPO / ".git").is_dir():
    sys.exit(f"error: {REPO} is not its own git repo (no .git); refusing to touch the parent repo.")
KEY_FILE = ROOT / "overleaf_key"
UPSTREAM = "origin/main"

ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", OVERLEAF_KEY=KEY_FILE.read_text().strip())
AUTH = ["-c", 'credential.helper=!f() { test "$1" = get && echo username=git && echo "password=$OVERLEAF_KEY"; }; f']


def git(*args, check=True):
    r = subprocess.run(["git", "-C", str(REPO), *AUTH, *args], env=ENV, text=True, capture_output=True)
    if check and r.returncode:
        sys.exit(f"error: git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r


def main():
    git("fetch", "-q", "origin")
    incoming = git("log", "--format=%h %an %ad %s", "--date=short", f"HEAD..{UPSTREAM}").stdout.strip()
    if not incoming:
        print("Already up to date with Overleaf.")
        return 0
    print("Incoming from Overleaf:\n" + incoming)
    print(git("diff", "--stat", f"HEAD...{UPSTREAM}").stdout)

    stashed = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    if stashed:
        git("stash", "push", "-q", "-m", "pull_from_overleaf: local edits")
    # Replays any unpushed local commits on top of Overleaf (a fast-forward if there are none).
    if git("rebase", "-q", UPSTREAM, check=False).returncode:
        git("rebase", "--abort", check=False)
        if stashed:
            git("stash", "pop", "-q")
        sys.exit("error: local commits conflict with Overleaf; nothing was changed.")
    if stashed and git("stash", "pop", "-q", check=False).returncode:
        conflicts = git("diff", "--name-only", "--diff-filter=U").stdout.split()
        sys.exit("Pulled, but your uncommitted edits conflict with Overleaf in:\n  "
                 + "\n  ".join(conflicts)
                 + "\nResolve the conflict markers there; your edits are also kept in `git stash list`.")
    print("Pulled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

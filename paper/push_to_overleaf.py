#!/usr/bin/env python3
"""Commit everything in Mortis/ and push it to Overleaf.

The commit message is the paper's hash: the first 12 characters of the git tree
hash of Mortis/'s contents, so `git log --format='%T %s'` shows matching
prefixes. Any unpushed local commits are folded into that one commit.

Refuses to push if Overleaf has edits you haven't pulled (run
pull_from_overleaf.py first) or if files still contain conflict markers.

Usage: python3 push_to_overleaf.py [PAPER]   (default Lingjia -> Mortis_Lingjia/; "" -> Mortis/)
"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = sys.argv[1] if len(sys.argv) > 1 else "Lingjia"  # Mortis/ is a stale backup; pass "" to target it
REPO = ROOT / ("Mortis" + (f"_{PAPER}" if PAPER else ""))
if not (REPO / ".git").is_dir():
    sys.exit(f"error: {REPO} is not its own git repo (no .git); refusing to touch the parent repo.")
KEY_FILE = ROOT / "overleaf_key"
BRANCH = "main"
UPSTREAM = f"origin/{BRANCH}"

ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", OVERLEAF_KEY=KEY_FILE.read_text().strip())
AUTH = ["-c", 'credential.helper=!f() { test "$1" = get && echo username=git && echo "password=$OVERLEAF_KEY"; }; f']


def git(*args, check=True):
    r = subprocess.run(["git", "-C", str(REPO), *AUTH, *args], env=ENV, text=True, capture_output=True)
    if check and r.returncode:
        sys.exit(f"error: git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r


def main():
    git("fetch", "-q", "origin")
    if git("rev-list", "--count", f"HEAD..{UPSTREAM}").stdout.strip() != "0":
        sys.exit("error: Overleaf has edits you haven't pulled; run pull_from_overleaf.py first.")
    markers = git("grep", "-InE", "^(<<<<<<<|>>>>>>>)( |$)", check=False).stdout
    if markers:
        sys.exit("error: unresolved conflict markers:\n" + markers)

    git("add", "-A")  # also clears "unmerged" state left by a resolved pull conflict
    git("reset", "-q", "--soft", UPSTREAM)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("Nothing to push; Overleaf is up to date.")
        return 0
    print(git("diff", "--cached", "--stat").stdout)

    paper_hash = git("write-tree").stdout.strip()[:12]
    git("commit", "-q", "-m", paper_hash)
    git("push", "-q", "origin", f"HEAD:{BRANCH}")
    print(f"Pushed {paper_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Sync the minimal compile closure of main.tex into the Overleaf submodule.

Resolves, starting from main.tex, every uncommented \\input, \\includegraphics,
\\bibliography, \\bibliographystyle and locally present \\usepackage file, wipes
the Overleaf directory (except .git) and copies exactly that closure, then
commits and pushes.

Usage: python3 sync_overleaf.py [-m MSG] [--dry-run] [--no-push] [--check]
"""
import argparse, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OVERLEAF = ROOT / "6abe7a4013d356e7fea5e305"
KEY_FILE = ROOT / ".overleaf_key"  # git-ignored; holds the Overleaf git token

CMD = re.compile(r"\\(input|includegraphics|bibliography|bibliographystyle|usepackage)"
                 r"(?:\[[^\]]*\])?\{([^}]+)\}")
COMMENT = re.compile(r"(?<!\\)%.*")


def closure(entry: Path) -> set[Path]:
    needed, queue = {entry}, [entry]
    while queue:
        tex = queue.pop()
        for line in tex.read_text(encoding="utf-8").splitlines():
            for kind, arg in CMD.findall(COMMENT.sub("", line)):
                for name in arg.split(","):
                    name = name.strip()
                    cand = {
                        "input": [name, name + ".tex"],
                        "includegraphics": [name],
                        "bibliography": [name + ".bib"],
                        "bibliographystyle": [name + ".bst"],
                        "usepackage": [name + ".sty"],  # local packages only
                    }[kind]
                    for c in cand:
                        p = ROOT / c
                        if p.is_file() and p not in needed:
                            needed.add(p)
                            if p.suffix == ".tex":
                                queue.append(p)
    return needed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default="Sync paper")
    ap.add_argument("--dry-run", action="store_true", help="print the closure and exit")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--check", action="store_true", help="test-compile in the Overleaf dir before committing")
    args = ap.parse_args()

    files = sorted(p.relative_to(ROOT) for p in closure(ROOT / "main.tex"))
    if args.dry_run:
        print("\n".join(map(str, files)))
        return 0

    for child in OVERLEAF.iterdir():
        if child.name != ".git":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for rel in files:
        dst = OVERLEAF / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    if args.check:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
                       cwd=OVERLEAF, check=True, capture_output=True)
        subprocess.run(["latexmk", "-C"], cwd=OVERLEAF, check=True, capture_output=True)
        (OVERLEAF / "main.bbl").unlink(missing_ok=True)

    git = ["git", "-C", str(OVERLEAF)]
    subprocess.run(git + ["add", "-A"], check=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode == 0:
        print("Overleaf already up to date.")
        return 0
    subprocess.run(git + ["commit", "-m", args.message], check=True)
    if not args.no_push:
        push = git + ["push", "origin", "HEAD"]
        env = os.environ.copy()
        key = env.get("OVERLEAF_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.is_file() else "")
        if key:
            env["OVERLEAF_KEY"] = key
            helper = '!f() { echo username=git; echo "password=$OVERLEAF_KEY"; }; f'
            push = git + ["-c", f"credential.helper={helper}", "push", "origin", "HEAD"]
        subprocess.run(push, check=True, env=env)
    print(f"Synced {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

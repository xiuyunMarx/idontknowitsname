#!/usr/bin/env python3
"""Sync the minimal compile closure of mortis.tex into the Overleaf submodule.

Resolves, starting from mortis.tex, every uncommented \\input, \\includegraphics
(searching \\graphicspath like LaTeX does), \\bibliography, \\bibliographystyle and
locally present \\usepackage file, fetches Overleaf and resets onto it, wipes
the Overleaf directory (except .git) and copies exactly that closure, then
commits and pushes. Local is the source of truth: edits made on Overleaf are
overwritten (with a warning), never merged, so the push is always a fast-forward.

Usage: python3 sync_overleaf.py [-m MSG] [--dry-run] [--no-push] [--check]
"""
import argparse, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OVERLEAF = ROOT / "6abe7a4013d356e7fea5e305"
KEY_FILE = ROOT / ".overleaf_key"  # git-ignored; holds the Overleaf git token

CMD = re.compile(r"\\(input|includegraphics|bibliography|bibliographystyle|usepackage)"
                 r"(?:\[[^\]]*\])?\{([^}]+)\}")
GRAPHICSPATH = re.compile(r"\\graphicspath\{((?:\{[^}]*\})+)\}")
COMMENT = re.compile(r"(?<!\\)%.*")


def closure(entry: Path) -> set[Path]:
    needed, queue = {entry}, [entry]
    graphics, gpath = [], [""]
    while queue:
        tex = queue.pop()
        for line in tex.read_text(encoding="utf-8").splitlines():
            line = COMMENT.sub("", line)
            for dirs in GRAPHICSPATH.findall(line):
                gpath += re.findall(r"\{([^}]*)\}", dirs)
            for kind, arg in CMD.findall(line):
                for name in arg.split(","):
                    name = name.strip()
                    if kind == "includegraphics":
                        graphics.append(name)  # resolved once every \graphicspath is known
                        continue
                    cand = {
                        "input": [name, name + ".tex"],
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
    # Like LaTeX: the name as written first, then each \graphicspath dir in order.
    for name in graphics:
        hits = [ROOT / (d + name) for d in gpath if (ROOT / (d + name)).is_file()]
        if hits:
            needed.add(hits[0])
        else:
            print(f"warning: \\includegraphics{{{name}}} not found, not synced", file=sys.stderr)
    return needed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default="Sync paper")
    ap.add_argument("--dry-run", action="store_true", help="print the closure and exit")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--check", action="store_true", help="test-compile in the Overleaf dir before committing")
    args = ap.parse_args()

    files = sorted(p.relative_to(ROOT) for p in closure(ROOT / "mortis.tex"))
    if args.dry_run:
        print("\n".join(map(str, files)))
        return 0

    git = ["git", "-C", str(OVERLEAF)]
    remote, env = git, os.environ.copy()
    key = env.get("OVERLEAF_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.is_file() else "")
    if key:
        env["OVERLEAF_KEY"] = key
        helper = '!f() { echo username=git; echo "password=$OVERLEAF_KEY"; }; f'
        remote = git + ["-c", f"credential.helper={helper}"]

    # Build on top of whatever Overleaf has; the copy below then overwrites it.
    subprocess.run(remote + ["fetch", "-q", "origin"], check=True, env=env)
    edits = subprocess.run(git + ["diff", "--stat", "HEAD...@{u}"],
                           check=True, capture_output=True, text=True).stdout
    if edits:
        print("warning: overwriting edits made on Overleaf:\n" + edits, file=sys.stderr)
    subprocess.run(git + ["reset", "-q", "@{u}"], check=True)

    for child in OVERLEAF.iterdir():
        if child.name != ".git":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for rel in files:
        dst = OVERLEAF / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    if args.check:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "mortis.tex"],
                       cwd=OVERLEAF, check=True, capture_output=True)
        subprocess.run(["latexmk", "-C"], cwd=OVERLEAF, check=True, capture_output=True)
        (OVERLEAF / "mortis.bbl").unlink(missing_ok=True)

    subprocess.run(git + ["add", "-A"], check=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode == 0:
        print("Overleaf already up to date.")
        return 0
    subprocess.run(git + ["commit", "-m", args.message], check=True)
    if not args.no_push:
        subprocess.run(remote + ["push", "origin", "HEAD"], check=True, env=env)
    print(f"Synced {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

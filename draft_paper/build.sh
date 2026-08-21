#!/usr/bin/env bash
# Build the preprint. Uses latexmk when available, otherwise pdflatex + bibtex.
set -euo pipefail
cd "$(dirname "$0")"

JOB=main

if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -bibtex -halt-on-error -interaction=nonstopmode "$JOB.tex"
else
    pdflatex -halt-on-error -interaction=nonstopmode "$JOB.tex"
    bibtex "$JOB"
    pdflatex -halt-on-error -interaction=nonstopmode "$JOB.tex"
    pdflatex -halt-on-error -interaction=nonstopmode "$JOB.tex"
fi

echo "built: $(pwd)/$JOB.pdf"

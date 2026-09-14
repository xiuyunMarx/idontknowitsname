# Build the paper.
#
#   make            build main.pdf (pdflatex + bibtex, reruns until stable)
#   make quick      single pdflatex pass, no bibtex (fast iteration on prose)
#   make view       build, then open the PDF
#   make check      warn about undefined references / citations
#   make clean      remove build artifacts, keep main.pdf
#   make distclean  remove everything generated, including main.pdf
#
# If latexmk is installed it is used instead of the manual pdflatex/bibtex
# loop. Override the engine with `make LATEX=lualatex`.

MAIN      := main
LATEX     := pdflatex
BIBTEX    := bibtex
LATEXMK   := $(shell command -v latexmk 2>/dev/null)
FLAGS     := -interaction=nonstopmode -halt-on-error -file-line-error
PDFVIEWER ?= xdg-open

SRC := $(MAIN).tex \
       $(wildcard sections/*.tex) \
       $(wildcard style/*.tex) \
       $(wildcard figures/*) $(wildcard figures/*/*) \
       $(wildcard *.bib)

AUX := $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
       $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk \
       $(MAIN).synctex.gz $(MAIN).nav $(MAIN).snm $(MAIN).vrb \
       $(MAIN).loa $(MAIN)-blx.bib $(MAIN).run.xml \
       $(wildcard sections/*.aux) $(wildcard style/*.aux)

.PHONY: all quick view check check-deps clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(SRC) | check-deps
ifneq ($(LATEXMK),)
	$(LATEXMK) -pdf -pdflatex="$(LATEX) $(FLAGS)" $(MAIN).tex
else
	$(LATEX) $(FLAGS) $(MAIN).tex
	@if grep -q '\\citation' $(MAIN).aux; then $(BIBTEX) $(MAIN); fi
	$(LATEX) $(FLAGS) $(MAIN).tex
	@# Rerun while LaTeX still asks for it (cross-refs, citations, longtables).
	@i=0; while grep -qE 'Rerun to get|Rerun LaTeX|undefined references' $(MAIN).log && [ $$i -lt 3 ]; do \
	    i=$$((i+1)); echo "rerun $$i"; $(LATEX) $(FLAGS) $(MAIN).tex; done
endif
	@$(MAKE) --no-print-directory check

quick: | check-deps
	$(LATEX) $(FLAGS) $(MAIN).tex

view: $(MAIN).pdf
	$(PDFVIEWER) $(MAIN).pdf >/dev/null 2>&1 &

check:
	@if [ -f $(MAIN).log ]; then \
	    grep -E 'Citation .* undefined|Reference .* undefined|multiply defined' $(MAIN).log | sort -u | sed 's/^/warning: /' || true; \
	    grep -E '^Overfull \\hbox \([0-9.]{3,}pt' $(MAIN).log | wc -l | sed 's/^/overfull hboxes > 10pt: /'; \
	fi

# Files the paper needs that a stock Debian/Ubuntu TeX Live lacks.
#   acmart.cls, ACM-Reference-Format.bst   -> texlive-publishers
#   algorithm.sty, algpseudocode.sty       -> texlive-science
#   libertine.sty, zi4.sty, newtxmath.sty  -> texlive-fonts-extra (acmart fonts)
DEPS := acmart.cls ACM-Reference-Format.bst algorithm.sty algpseudocode.sty \
        libertine.sty zi4.sty newtxmath.sty

check-deps:
	@missing=""; for f in $(DEPS); do kpsewhich $$f >/dev/null || missing="$$missing $$f"; done; \
	if [ -n "$$missing" ]; then \
	    echo "error: missing TeX files:$$missing"; \
	    echo "  sudo apt install texlive-publishers texlive-science texlive-fonts-extra"; \
	    exit 1; fi

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(MAIN).pdf

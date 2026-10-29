# Build the Mortis paper with the local TeX Live.
#
#   make          compile Mortis/mortis.tex -> mortis.pdf (intermediates in _cache/)
#   make clean    empty _cache/ (keeps mortis.pdf)
PAPER   ?=
SRC_DIR := $(CURDIR)/Mortis$(if $(PAPER),_$(PAPER))
MAIN    := mortis
CACHE   := $(CURDIR)/_cache
TEXBIN  ?= /home/xiaoyu/miniconda3/envs/mortis/texlive/bin/x86_64-linux

export PATH := $(TEXBIN):$(PATH)

FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all clean lingjia

# latexmk tracks dependencies itself (via _cache/$(MAIN).fdb_latexmk), so
# always hand off to it and let it decide whether a rerun is needed.
all:
	latexmk -pdf -cd -auxdir=$(CACHE) -outdir=$(CURDIR) \
	    -pdflatex="pdflatex $(FLAGS) %O %S" $(SRC_DIR)/$(MAIN).tex

clean:
	rm -rf $(CACHE)/*
	rm ./mortis.pdf
	
lingjia:
	$(MAKE) PAPER=Lingjia all

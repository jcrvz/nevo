# Minimal Makefile for Sphinx documentation.
# Run from the docs/ directory.
#
#   make html       — build HTML docs
#   make clean      — remove _build/
#   make livehtml   — rebuild on file changes (requires sphinx-autobuild)

SPHINXBUILD   ?= sphinx-build
SPHINXOPTS    ?=
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo ""
	@echo "Build finished. Open $(BUILDDIR)/html/index.html to view."

clean:
	rm -rf $(BUILDDIR)

livehtml:
	sphinx-autobuild $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html


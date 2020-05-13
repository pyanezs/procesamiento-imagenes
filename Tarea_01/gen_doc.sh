#!/bin/bash

pandoc \
    --from=markdown \
    --to=latex \
    --template=template.latex \
    --filter=pandoc-crossref \
    --filter=pandoc-citeproc \
    --bibliography=sources.bib \
    --csl=bibliography.csl \
    --output=Tarea_1.tex \
    Tarea_1.md


# pandoc \
#     --from=markdown \
#     --to=latex \
#     --template=template.latex \
#     --filter=pandoc-crossref \
#     --filter=pandoc-citeproc \
#     --output=Tarea_1.pdf \
#     Tarea_1.md

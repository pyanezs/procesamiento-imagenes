#!/bin/bash

pandoc \
    --from=markdown \
    --to=latex \
    --template=template.latex \
    --filter=pandoc-crossref \
    --filter=pandoc-citeproc \
    --output=Tarea_1.pdf \
    Tarea_1.md


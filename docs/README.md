# docs/

`architecture.png` — overview of the library's components, embedded in the
top-level README. Rendered from `architecture.tikz` (adapted from the
paper's own Fig. 2 in the paper repository, with class names corrected to
match this repository exactly).

To regenerate after editing `architecture.tikz`:

```bash
cd docs
cat > wrapper.tex <<'TEX'
\documentclass[tikz,border=6pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\begin{document}
\input{architecture.tikz}
\end{document}
TEX
pdflatex -interaction=nonstopmode wrapper.tex
pdftoppm -png -r 300 wrapper.pdf architecture
mv architecture-1.png architecture.png
rm wrapper.tex wrapper.pdf wrapper.aux wrapper.log
```

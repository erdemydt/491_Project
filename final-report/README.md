# final-report

This folder is a self-contained final report package.

## Report

- LaTeX source: `report.tex`
- Bibliography: `refs.bib`
- Built PDF: `report.pdf`

## Regenerating figures

Use the project virtual environment:

```bash
cd /home/erdem/491_Project
./projectEnv/bin/python final-report/scripts/make_stacked_k_vs_tau.py
./projectEnv/bin/python final-report/scripts/make_n_state_sigmaomega_plot.py
./projectEnv/bin/python final-report/scripts/make_two_bit_figures.py
```

Outputs are written into `final-report/images/`.

## Building the PDF

```bash
cd /home/erdem/491_Project/final-report
/usr/local/texlive/2024/bin/x86_64-linux/pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
/usr/local/texlive/2024/bin/x86_64-linux/pdflatex -interaction=nonstopmode -halt-on-error report.tex
/usr/local/texlive/2024/bin/x86_64-linux/pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

## Code

All code used (and any modifications) lives under `final-report/code/`.

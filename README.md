# MEML-75-Axion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23439548.svg)](https://doi.org/10.5281/zenodo.23439548)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Reference implementation and manuscript for model MEML 75.
The paper is under review at *Physical Review D* (Manuscript ID DXR1128).
The preprint is **not** on arXiv; priority is fixed via Zenodo and GitHub.

## Contents
- `MEML_75_final_EN.tex` — full paper source (XeLaTeX, English)
- `MEML_75_final_EN.pdf` — compiled PDF
- `MEML_75_final_RU.tex` — Russian translation
- `MEML_75_final_RU.pdf` — compiled PDF (Russian)
- `MEML_75_complete_EN.py` — all numerical checks (verification mode)
- `README.md` — this file

## Reproduce
```bash
python meml75_complete_EN.py          # verification (~5 s)
python meml75_complete_EN.py --fit    # + reproduce DE-fit (~3 min)

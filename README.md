# Model MEML 75: Verification Scripts and Data

This repository contains the numerical verification scripts and source files for the paper:
**"Model MEML 75: Complete fermion sector, axion-like dark matter, and black hole superradiance from T²/Z₃ geometry"** (submitted to Nuclear Physics B).

## Contents

*   `meml75_complete_EN.py` — The complete Python verification script (v10.9).
*   `MEML_75_final_EN.tex` — The LaTeX source of the manuscript.
*   `MEML_75_final_EN.pdf` — The compiled PDF of the manuscript.

## Requirements

To run the verification script, you need Python 3 and the following standard scientific libraries:
*   `numpy`
*   `scipy` (required for the differential evolution fit, though the default fast check runs without it)

You can install the dependencies via pip:
```bash
pip install numpy scipy
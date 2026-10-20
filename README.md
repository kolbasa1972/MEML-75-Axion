# MEML 75 — Quark and lepton mass matrices from T²/Z₃ geometry

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23439547.svg)](https://doi.org/10.5281/zenodo.23439547)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/kolbasa1972/MEML-75-Axion)](https://github.com/kolbasa1972/MEML-75-Axion/releases)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LaTeX](https://img.shields.io/badge/LaTeX-XeLaTeX-green)

Reference implementation of the model described in:

- **Preprint:** Zenodo (archived with DOI)
- **ORCID:** [0009-0006-5467-7510](https://orcid.org/0009-0006-5467-7510)

This paper presents a **complete phenomenological parametrization of the quark and lepton mass matrices** through the geometry of the six-dimensional orbifold `T²/Z₃` with three independent tori, and predicts an **axion-like particle (ALP)** as dark matter.

## Contents

### Paper (source + compiled)

- `MEML_75_final_EN.tex` — full paper source (XeLaTeX)
- `MEML_75_final_EN.pdf` — compiled English version
- `MEML_75_final_RU.tex` — Russian translation (source)
- `MEML_75_final_RU.pdf` — compiled Russian version

### Verification

- `meml75_complete_EN.py` — all numerical checks for quarks, leptons, and ALP.
- `verification_log.txt` — snapshot of `✅ ALL ASSERTS PASSED` output (archived with the Zenodo release)

## Reproduce

```bash
python meml75_complete_EN7.py
# → "✅ ALL ASSERTS PASSED"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23575072.svg)](https://doi.org/10.5281/zenodo.23575072)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

# Model MEML 75: Quark sector and axion-like dark matter (Version 13.0)

This repository contains the data, Python verification script, and LaTeX source for the manuscript submitted to Nuclear Physics B (ID: **NPB-D-26-02519**).

## Contents

* `MEML75_v13.0_Quark_ALP.tex` — LaTeX source of the manuscript.
* `MEML75_v13.0_Quark_ALP.pdf` — Compiled PDF of the manuscript.
* `meml75_v13.0_verification.py` — Python script reproducing the numerical results.
* `meml75_v13.0_verification.txt` — Output log of the script (shows `ALL ASSERTS PASSED`).
* `LICENSE` — MIT License.

## Key changes in this version (v13.0)

* **Lepton sector removed (negative result):** The attempt to extend the modular construction to leptons using \(3_S\) contractions of \(Y^{(2)}\) and \(Y^{(4)}\) failed to reproduce the PMNS observables simultaneously. Adding \(Y^{(6)}\) yields \(\chi^2=0\) but leaves the modulus \(\tau_l\) undetermined. We report this explicitly as a negative result in Section 8.
* **Quark sector and ALP sector unchanged:** The core results (quark mass matrices, ALP dark matter predictions, black hole superradiance) remain intact.
* **Parameter count:** 11 effective parameters fitting 10 observables (\(N_{\rm dof} = -1\), borderline overfitted).

## Verification

To verify the numerical results of the quark and ALP sectors (§3–7, §10), run:

```bash
python meml75_v13.0_verification.py

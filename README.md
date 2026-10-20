# MEML 75 — Axion-like (ALP) dark matter from T²/Z₃ geometry

Reference implementation of the model described in

- **Manuscript:** Phys. Rev. D (submitted as a Letter), ID **DXR1128**
- **Preprint / priority:** Zenodo, DOI [10.5281/zenodo.23439548](https://doi.org/10.5281/zenodo.23439548)
- **ORCID:** [0009-0006-5467-7510](https://orcid.org/0009-0006-5467-7510)

The paper parametrizes the **quark sector** of the Standard Model
through the geometry of the six-dimensional orbifold `T²/Z₃`
with two independent tori, and predicts an **axion-like particle (ALP)**
as dark matter. The lepton sector is deferred to future work.

## Contents

### Paper (source + compiled)

- `MEML_75_final_EN.tex` — full paper source (XeLaTeX, `fontspec` + `polyglossia`)
- `MEML_75_final_EN.pdf` — compiled English version (the one submitted to PRD)
- `MEML_75_final_RU.tex` — Russian translation (source)
- `MEML_75_final_RU.pdf` — compiled Russian version

### Verification

- `meml75_complete_EN.py` — all numerical checks (`--fit` reproduces the DE fit from scratch)
- `verification_log.txt` — snapshot of `✅ ALL ASSERTS PASSED` output
  (archived with the Zenodo release; optional, regenerate locally with the command below)

### This file

- `README.md`

## Reproduce

```bash
python meml75_complete_EN.py
# → "✅ ALL ASSERTS PASSED"

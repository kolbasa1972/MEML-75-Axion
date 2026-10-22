# kolbasa1972/MEML-75-Complete: Version 13.0

This is the revised version corresponding to the manuscript submitted to the journal Nuclear Physics B (NPB-D-26-02519).

## Contents

* `meml75_complete_EN.py`: Complete Python verification script (v13.0).
* `MEML_75_final_EN.tex`: LaTeX source code of the manuscript.
* `MEML_75_final_EN.pdf`: Compiled PDF file of the manuscript.
* `README.md`: Instructions for running the script.

## Key changes in this version (v13.0)

* **Lepton sector removed (negative result):** The attempt to extend the modular construction to leptons using \(3_S\) contractions failed. Without \(Y^{(6)}\), the fit cannot simultaneously reproduce the PMNS angles and neutrino mass-squared differences (\(\chi^2 \approx 39\)). Adding \(Y^{(6)}\) yields \(\chi^2=0\) but leaves \(\tau_l\) completely undetermined (multiple equivalent solutions). This negative result is documented in §8.
* **Quark sector and ALP remain:** The main results (quark mass matrices, ALP dark matter, and black hole superradiance) are unchanged.
* **Parameter count:** The model now has 11 effective parameters fitting 10 observables (\(N_{\rm dof} = -1\)), reflecting a borderline overfitted status. 

## Verification

Run `python meml75_complete_EN.py` to execute fast checks and confirm the correctness of all numerical data in Sections 3–7 and 10 of the paper. The script explicitly prints a note regarding the negative result in the lepton sector (§8).
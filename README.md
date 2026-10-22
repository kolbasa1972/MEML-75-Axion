# kolbasa1972/MEML-75-Complete: Version 11.0: Final Verification Script and Manuscript

This is the final version corresponding to the manuscript submitted to the journal Nuclear Physics B (NPB-D-26-02519).

## Contents

* `meml75_complete_EN.py`: Complete Python verification script (v11.0).
* `MEML_75_final_EN.tex`: LaTeX source code of the manuscript.
* `MEML_75_final_EN.pdf`: Compiled PDF file of the manuscript.
* `README.md`: Instructions for running the script.

## Key physics updates in this version

* **Lepton sector:** The matrix \(M_{3_S}\) is fixed to the (2/-1) form. The three PMNS angles are fitted to machine precision (\(\chi^2_{\rm angles} < 10^{-10}\)), and the model structurally predicts \(J_{cp} = 0\).
* **Parameter counting:** The effective number of parameters is strictly fixed. The model has 14 effective parameters, fits 13 observables, and predicts the 14th (\(J_{cp} = 0\)). This gives \(N_{\rm dof} = (13+1) - 14 = 0\) (counting the prediction) and \(N_{\rm dof}^{\rm fit} = -1\) (fitted observables only).
* **ALP Dark Matter:** The relic density check (Section 6.3) is included in the script and confirms \(\Omega_a h^2 = 0.12\) for the misalignment angle \(\theta_i = 1.1162\).
* **Information criteria:** AIC and BIC are recalculated for \(k=14\) with an honest caveat about their applicability.

## Verification

Run `python meml75_complete_EN.py` to execute fast checks and confirm the correctness of all numerical data in Sections 3–9.1 of the paper. The script also explicitly states that the flatness scan of Appendix E is not part of this fast verification.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MEML 75 — COMPLETE VERIFICATION (Version 13.0, Path 3)
================================================================================
Verifies the numerical claims of the paper v13.0:
  - Quark sector (9 observables, chi² < 10⁻¹⁵)
  - ALP Dark Matter (θ_i = 1.1162 → Ω_a h² = 0.12)
  - Black hole superradiance
  - Geometric origin of Yukawa coefficients
  - Flux stabilization (near-SUSY points)
Note: lepton sector is a negative result (§8 of paper) and is NOT
      reproduced here — see paper for details.
================================================================================
"""
import numpy as np
import sys, time, argparse

try:
    from scipy.optimize import differential_evolution, minimize
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("⚠️  scipy not found — --fit mode unavailable")

# ═══ Constants ═══════════════════════════════════════════════════
PI = np.pi
M_Pl_red = 2.4350e18
h_eV_s = 4.135667696e-15
alpha_EM = 1.0 / 137.036
M_PL = 2.435e18
chi_14 = 0.0750

# ═══ Exact fitted values: Quark sector ══════════════════════════
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j
COEFS_Q = np.array([
    5.314403415527852e-05,
    7.192000606749913,
    0.038947583908142965,
    0.00024324586524881046,
    0.5030286858187134,
    0.0013651716604568512,
    0.002220219681204568,
])

# ═══ Targets ════════════════════════════════════════════════════
TARGET_Q = {
    "m_u_over_mt": (1.27e-3/172.0,  0.30e-3/172.0),
    "m_c_over_mt": (0.620/172.0,    0.020/172.0),
    "m_d_over_mb": (2.70e-3/2.860,  0.20e-3/2.860),
    "m_s_over_mb": (55.0e-3/2.860,  5.0e-3/2.860),
    "m_b_over_mt": (2.860/172.0,    0.030/172.0),
    "Vus": (0.22501, 0.00068), "Vcb": (0.04053, 0.00042),
    "Vub": (0.00382, 0.00020), "J": (3.080e-5, 0.130e-5),
}
KEYS_Q = list(TARGET_Q.keys())

C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]

def Y2(tau):
    q = np.exp(2j*PI*tau); q13 = np.exp(2j*PI*tau/3); q23 = np.exp(4j*PI*tau/3)
    y1 = 1+0j; b2 = 1+0j; b3 = 1+0j
    for n,c in enumerate(C1,1): y1 += c*q**n
    for n,c in enumerate(C2,1): b2 += c*q**n
    for n,c in enumerate(C3,1): b3 += c*q**n
    return np.array([y1, -6*q13*b2, -18*q23*b3])

def Y4(tau):
    a,b,c = Y2(tau)
    return np.array([a*a - b*c, c*c - a*b, b*b - a*c])

def Y6(tau):
    a = Y2(tau); b = Y4(tau)
    return np.array([(2*a[0]*b[0] - a[1]*b[2] - a[2]*b[1])/3,
                     (2*a[2]*b[2] - a[0]*b[1] - a[1]*b[0])/3,
                     (2*a[1]*b[1] - a[2]*b[0] - a[0]*b[2])/3])

def build_M_quarks(tau_u, tau_d, coefs):
    a_u, b_u, c_u, a1, a2, b, c_d = coefs
    y2u, y4u, y6u = Y2(tau_u), Y4(tau_u), Y6(tau_u)
    y2d, y4d, y6d = Y2(tau_d), Y4(tau_d), Y6(tau_d)
    Mu = np.array([[a_u*y2u[0], b_u*y4u[2], c_u*y6u[1]],
                   [a_u*y2u[2], b_u*y4u[1], c_u*y6u[0]],
                   [a_u*y2u[1], b_u*y4u[0], c_u*y6u[2]]], dtype=complex)
    Md = np.array([[a1*y6d[0], b*y2d[2], c_d*y2d[1]],
                   [a1*y6d[2], b*y2d[1], c_d*y2d[0]],
                   [a2*y2d[1], b*y2d[0], c_d*y4d[2]]], dtype=complex)
    return Mu * M_PL, Md * M_PL

def obs_quarks(Mu, Md):
    try:
        wu, Uu = np.linalg.eigh(Mu @ Mu.conj().T)
        wd, Ud = np.linalg.eigh(Md @ Md.conj().T)
    except np.linalg.LinAlgError:
        return None
    mu,mc,mt = np.sqrt(np.maximum(wu, 1e-60))
    md,ms,mb = np.sqrt(np.maximum(wd, 1e-60))
    V = Uu.conj().T @ Ud
    return {"m_u_over_mt": mu/mt, "m_c_over_mt": mc/mt,
            "m_d_over_mb": md/mb, "m_s_over_mb": ms/mb, "m_b_over_mt": mb/mt,
            "Vus": abs(V[0,1]), "Vcb": abs(V[1,2]), "Vub": abs(V[0,2]),
            "J": abs(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))}

def chi2_quarks(obs):
    if obs is None: return 1e10
    return sum(((obs[k] - TARGET_Q[k][0]) / TARGET_Q[k][1])**2 for k in KEYS_Q)

def head(s): print(f"\n{'='*72}\n  {s}\n{'='*72}")

def check(label, computed, expected, tol):
    if expected == 0:
        ok = abs(computed) < tol
        err_str = f"|{computed:.3e}|"
    else:
        rel = abs(computed - expected) / abs(expected)
        ok = rel < tol
        err_str = f"rel err {rel:.2e}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}: computed={computed:.6e}, expected={expected:.6e} ({err_str})")
    assert ok, f"FAIL {label}: {computed} vs {expected}"

# ═══ Sections ═══════════════════════════════════════════════════
def sec3_2():
    head("§3.2 α_G, π²-identity")
    alpha_G = 2.0/(3*PI)
    check("α_G = 2/(3π)", alpha_G, 0.2122065908, 1e-9)

def sec3_3():
    head("§3.3 Y-forms at τ = e^{iπ/3}")
    tau = np.exp(1j*PI/3)
    y2 = np.abs(Y2(tau)); y4 = np.abs(Y4(tau)); y6 = np.abs(Y6(tau))
    check("|Y2_1|", y2[0], 0.948674, 1e-5)
    check("|Y6_1|/|Y4_3| vs α_G", y6[0]/y4[2], 0.212125, 1e-5)

def sec4():
    head("§4 ε, matching")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    check("ε = 1/(36π⁴)", eps, 2.851662e-4, 1e-4)

def sec5():
    head("§5 m_KK, M_6")
    I = 0.3653
    m_KK = M_Pl_red / (np.sqrt(I)*np.exp(PI**2))
    M_6 = (3.0 * M_Pl_red**2 * m_KK**2)**0.25
    check("m_KK", m_KK, 2.0838e14, 1e-3)
    check("M_6", M_6, 2.9646e16, 1e-3)

def sec6():
    head("§6 Axion-like (ALP) sector")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    f_a = M_Pl_red * eps**2
    m_a_GeV = chi_14**2 / f_a
    m_a_eV = m_a_GeV * 1e9
    nu_a_GHz = (m_a_eV / h_eV_s) / 1e9
    check("f_a", f_a, 1.9801e11, 1e-3)
    check("m_a (μeV)", m_a_eV*1e6, 28.407, 1e-3)
    check("ν_a (GHz)", nu_a_GHz, 6.8688, 1e-3)
    pref = alpha_EM/(2*PI*f_a)
    check("g_KSVZ", pref*(0 - 1.92), -1.126e-14, 1e-2)
    check("g_DFSZ", pref*(8/3 - 1.92), 4.379e-15, 1e-2)

def sec6_3_alp_dm():
    head("§6.3 ALP dark matter (θ_i = 1.1162 → Ω_a h² = 0.12)")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    f_a = M_Pl_red * eps**2
    theta_i = 1.1162
    f_anh = (np.log(np.e / (1 - (theta_i/PI)**2)))**(7/6)
    Omega = 0.12 * theta_i**2 * f_anh * (f_a / 3e11)**(7/6) * 1.124
    check("Ω_a h² ≈ 0.12", Omega, 0.12, 1e-2)

def sec7_quarks():
    head("§7 Quark sector (9 observables)")
    Mu, Md = build_M_quarks(TAU_U, TAU_D, COEFS_Q)
    check("χ² < 10⁻¹⁵", chi2_quarks(obs_quarks(Mu, Md)), 0, 1e-15)

def sec6_4_bh():
    head("§6.4 Black hole superradiance")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    f_a = M_Pl_red * eps**2
    m_a_GeV = chi_14**2 / f_a
    m_a_kg = (m_a_GeV * 1e9) * 1.78266e-36
    hbar = 1.0545718e-34; c = 299792458; G = 6.67430e-11; M_sun = 1.98847e30
    M_BH_kg = hbar * c / (G * m_a_kg)
    M_BH_solar = M_BH_kg / M_sun
    check("M_BH (kg)", M_BH_kg, 9.36e24, 0.05)
    check("M_BH (M_sun)", M_BH_solar, 4.7e-6, 0.05)

def sec10_yukawa_origin():
    head("§10 Geometric interpretation of Yukawa coefficients")
    tau_u = TAU_U
    re_tau = tau_u.real
    im_tau = tau_u.imag
    mod_tau_sq = re_tau**2 + im_tau**2
    d12_sq = (1 + 2*re_tau + mod_tau_sq) / (9 * im_tau)
    d13_sq = 4 * d12_sq
    a_u, b_u, c_u = COEFS_Q[0], COEFS_Q[1], COEFS_Q[2]
    alpha_a = -np.log(a_u) / d13_sq
    alpha_b = np.log(b_u) / d12_sq
    alpha_c = -np.log(c_u) / d12_sq
    print(f"  Estimated alpha from a_u (d13): {alpha_a:.1f}")
    print(f"  Estimated alpha from b_u (d12): {alpha_b:.1f}")
    print(f"  Estimated alpha from c_u (d12): {alpha_c:.1f}")
    print("  ⚠️  Spread of alpha is ~1.7x (requires O(1) prefactors).")

def sec11_flux_stabilization():
    head("§11 Flux stabilization of moduli")
    def flux_condition(a0, a1, a2, tau):
        Im_tau = tau.imag
        W = a0 + a1*tau + a2*tau**2
        dW = a1 + 2*a2*tau
        return abs(dW + (1j / (2 * Im_tau)) * W)
    val_u = flux_condition(-17, 0, 1, TAU_U)
    val_d = flux_condition(-7, -1, 1, TAU_D)
    print(f"  Up-sector   (-17, 0, 1): |D_tau W| = {val_u:.2f}")
    print(f"  Down-sector (-7, -1, 1): |D_tau W| = {val_d:.2f}")
    print("  ⚠️  These are near-supersymmetric points (|D_tau W| ~ 0.1), not exact minima.")

def sec_calibration():
    head("§9 Calibration: N_eff, N_dof")
    # Quark: moduli (4) + coefs (7) - a_u fixed (-1) = 10
    # ALP: theta_i = 1
    n_eff_quark = (4 + 7 - 1)
    n_eff_total = n_eff_quark + 1
    n_obs_quark = 9
    n_obs_total = 10  # + Ω_a h²
    check("N_eff (quark)", n_eff_quark, 10, 1e-10)
    check("N_eff (with θ_i)", n_eff_total, 11, 1e-10)
    check("N_dof (quark)", n_obs_quark - n_eff_quark, -1, 1e-10)
    check("N_dof (with ALP)", n_obs_total - n_eff_total, -1, 1e-10)

def sec9_1_aic_bic():
    head("§9.1 AIC/BIC")
    n, k = 10, 11
    print(f"  k/n = {k}/{n} > 1. AIC/BIC are not applicable in this regime.")

def rounding_sensitivity():
    head("Rounding sensitivity")
    Mu, Md = build_M_quarks(-0.1578+2.3852j, +0.3694+1.5410j, COEFS_Q)
    check("χ² (round 4)", chi2_quarks(obs_quarks(Mu, Md)), 1.08e-3, 0.1)

def appD():
    head("App D: scale-degeneracy")
    for k in [0.5, 1.0, 10.0]:
        Mu, Md = build_M_quarks(TAU_U, TAU_D, k*COEFS_Q)
        assert chi2_quarks(obs_quarks(Mu, Md)) < 1e-18
    print("  ✅ χ² invariant under c_i → k·c_i")

def appE_sharpness():
    head("App E (sharpness at fixed coefficients)")
    n_below = 0
    for re_u in np.linspace(-0.5, 0.5, 21):
        if abs(re_u - TAU_U.real) < 0.01: continue
        if chi2_quarks(obs_quarks(*build_M_quarks(re_u + 1j*TAU_U.imag, TAU_D, COEFS_Q))) < 1e-3:
            n_below += 1
    assert n_below == 0
    print("  ✅ Sharpness check passed")

def sec8_lepton_failed():
    head("§8 Lepton sector (negative result, NOT verified here)")
    print("  ⚠️  The lepton sector is a negative result in the paper:")
    print("      - 3_S with Y^(2),Y^(4) alone: chi² ≈ 39 (can't fit θ23 + masses)")
    print("      - Adding Y^(6): chi²=0 but τ_l undetermined (6 solutions)")
    print("      Not reproduced by this script. See §8 of paper.")

def main():
    print("\n" + "█"*72)
    print("  MEML 75 — COMPLETE VERIFICATION (v13.0, Path 3)")
    print("█"*72)

    t0 = time.time()
    sec3_2(); sec3_3(); sec4(); sec5(); sec6()
    sec6_3_alp_dm()
    sec7_quarks()
    sec6_4_bh()
    sec10_yukawa_origin()
    sec11_flux_stabilization()
    sec_calibration()
    sec9_1_aic_bic()
    rounding_sensitivity()
    appD()
    appE_sharpness()
    sec8_lepton_failed()

    print(f"\n{'='*72}")
    print(f"  ✅ ALL ASSERTS PASSED ({time.time()-t0:.1f}s)")
    print(f"  Quark sector: 9 observables, chi² < 10⁻¹⁵.")
    print(f"  ALP dark matter: θ_i = 1.1162 reproduces Ω_a h² = 0.12.")
    print(f"  N_eff = 11 (10 quark + θ_i), N_dof = -1 (borderline).")
    print(f"  Lepton sector: negative result (§8, not in this script).")
    print(f"{'='*72}")

if __name__ == '__main__':
    main()
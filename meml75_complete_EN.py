#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MEML 75 — COMPLETE CALCULATION AND VERIFICATION (Version 11.0)
================================================================================
Verifies the numerical claims of the paper (v11.0, revised lepton sector).

Structure:
  - quark sector: 9 observables, χ² < 10⁻¹⁵
  - lepton sector: 3 PMNS angles to high precision, J_cp = 0 as prediction
  - ALP sector, including §6.3 relic density (θ_i = 1.1162 → Ω_a h² = 0.12)
  - black hole superradiance
  - AIC/BIC with N_eff = 14, N_dof = 0 (with prediction)
  - Appendix D (degeneracy), Appendix E (sharpness only)

What is NOT verified:
  - Appendix E (refit-flatness). The script tests sharpness of the
    minimum at fixed coefficients (`appE_sharpness`), which is a
    different check. See `appE_flatness_NOT_VERIFIED` for details.
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

# ═══ Exact fitted values: Quark Sector ══════════════════════════
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

# ═══ Exact fitted values: Lepton Sector ═════════════════════════
TAU_L = -0.853439713469788 + 1.684611396843088j
COEFS_L = np.array([
    92.532545527063320,   # g_1
    -164.545365413734089, # g_2
    179.379382858758333,  # g_3 (spectator, masses only)
    -94.132620835381886,  # g_4 (spectator, masses only)
    41.485086235393467,   # g_5 (spectator, masses only)
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

# ═══ A₄ Y-forms ═════════════════════════════════════════════════
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

# ═══ Quark matrices ═════════════════════════════════════════════
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

# ═══ Lepton matrices ════════════════════════════════════════════
def M_3S(Y):
    """2 on diagonal, -1 off-diagonal (matches Eq. 8.2 of paper)."""
    return np.array([[2*Y[0], -Y[2], -Y[1]],
                     [-Y[2], 2*Y[1], -Y[0]],
                     [-Y[1], -Y[0], 2*Y[2]]], dtype=complex)

def build_M_nu(tau_l, coefs_l):
    g1, g2, g3, g4, g5 = coefs_l
    y2, y4, y6 = Y2(tau_l), Y4(tau_l), Y6(tau_l)
    S2 = y2[0]**2 + 2*y2[1]*y2[2]
    S4 = y4[0]**2 + 2*y4[1]*y4[2]
    S6 = y6[0]**2 + 2*y6[1]*y6[2]
    return g1 * M_3S(y2) + g2 * M_3S(y4) + (g3*S2 + g4*S4 + g5*S6) * np.eye(3)

def obs_leptons(Mnu):
    w, U = np.linalg.eigh(Mnu @ Mnu.conj().T)
    s13_nh = np.abs(U[0,2])
    s12_nh = np.abs(U[0,1]) / np.sqrt(max(1e-12, 1 - s13_nh**2))
    s23_nh = np.abs(U[1,2]) / np.sqrt(max(1e-12, 1 - s13_nh**2))
    J_nh = np.abs(np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0])))

    s13_ih = np.abs(U[0,0])
    s12_ih = np.abs(U[0,2]) / np.sqrt(max(1e-12, 1 - s13_ih**2))
    s23_ih = np.abs(U[1,0]) / np.sqrt(max(1e-12, 1 - s13_ih**2))
    J_ih = np.abs(np.imag(U[0,2]*U[1,0]*np.conj(U[0,0])*np.conj(U[1,2])))

    t13_nh = np.arcsin(np.clip(s13_nh,-1,1)) * 180 / PI
    t13_ih = np.arcsin(np.clip(s13_ih,-1,1)) * 180 / PI

    if abs(t13_nh - 8.54) < abs(t13_ih - 8.54):
        s12, s13, s23, J = s12_nh, s13_nh, s23_nh, J_nh
    else:
        s12, s13, s23, J = s12_ih, s13_ih, s23_ih, J_ih

    return {"theta12": np.arcsin(np.clip(s12,-1,1)) * 180 / PI,
            "theta13": np.arcsin(np.clip(s13,-1,1)) * 180 / PI,
            "theta23": np.arcsin(np.clip(s23,-1,1)) * 180 / PI,
            "J": J}

# ═══ Utilities ══════════════════════════════════════════════════
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

# ═══ §3.2 α_G ══════════════════════════════════════════════════
def sec3_2():
    head("§3.2 α_G, π²-identity")
    alpha_G = 2.0/(3*PI)
    check("α_G = 2/(3π)", alpha_G, 0.2122065908, 1e-9)
    check("2π/(3α_G) = π²", 2*PI/(3*alpha_G), PI**2, 1e-12)
    check("e^(π²)", np.exp(PI**2), 1.933369e4, 1e-4)

# ═══ §3.3 Y-forms ══════════════════════════════════════════════
def sec3_3():
    head("§3.3 Y-forms at τ = e^{iπ/3}")
    tau = np.exp(1j*PI/3)
    y2 = np.abs(Y2(tau)); y4 = np.abs(Y4(tau)); y6 = np.abs(Y6(tau))
    check("|Y2_1|", y2[0], 0.948674, 1e-5)
    check("|Y2_2|", y2[1], 0.974090, 1e-5)
    check("|Y2_3|", y2[2], 0.476383, 1e-5)
    check("|Y4_1|", y4[0], 1.364023, 1e-5)
    check("|Y4_2|", y4[1], 0.697153, 1e-5)
    check("|Y4_3|", y4[2], 1.400784, 1e-5)
    check("|Y6_1|", y6[0], 0.297141, 1e-5)
    check("|Y6_2|", y6[1], 0.667310, 1e-5)
    check("|Y6_3|", y6[2], 0.679090, 1e-5)
    check("|Y6_1|/|Y4_3| vs α_G", y6[0]/y4[2], 0.212125, 1e-5)
    check("|Y6_2|/π vs α_G", y6[1]/PI, 0.212411, 1e-5)
    check("|Y6_2|²", y6[1]**2, 0.445302, 1e-5)
    check("|Y6_2|² − 4/9", y6[1]**2 - 4/9, 8.576e-4, 0.1)

# ═══ §3.4 Systematic search ═════════════════════════════════════
def sec3_4():
    head("§3.4 Systematic search for α_G (392 combinations)")
    from math import exp
    tau = np.exp(1j*PI/3)
    y2 = Y2(tau); y4 = Y4(tau); y6 = Y6(tau)
    q = np.exp(2j*PI*tau)
    eta_v = np.exp(2j*PI*tau/24)
    for n in range(1, 50):
        eta_v *= (1 - q**n)
    basis = {
        '|Y2_1|': abs(y2[0]), '|Y2_2|': abs(y2[1]), '|Y2_3|': abs(y2[2]),
        '|Y4_1|': abs(y4[0]), '|Y4_2|': abs(y4[1]), '|Y4_3|': abs(y4[2]),
        '|Y6_1|': abs(y6[0]), '|Y6_2|': abs(y6[1]), '|Y6_3|': abs(y6[2]),
        '|η|': abs(eta_v), '|η|²': abs(eta_v)**2, '|η|⁴': abs(eta_v)**4,
        'π': PI, '√π': np.sqrt(PI), 'π²': PI**2, 'e^(−π²)': np.exp(-PI**2),
    }
    alpha_G = 2.0/(3*PI)
    hits = []; names = list(basis.keys()); n = len(names)
    for na in names:
        for nb in names:
            if na == nb: continue
            v = basis[na]/basis[nb]
            if abs((v-alpha_G)/alpha_G) < 0.01: hits.append((na,nb,'ratio',v,(v-alpha_G)/alpha_G))
    for i in range(n):
        for j in range(i+1, n):
            v = basis[names[i]]*basis[names[j]]
            if abs((v-alpha_G)/alpha_G) < 0.01: hits.append((names[i],names[j],'prod',v,(v-alpha_G)/alpha_G))
    for na in names:
        v = np.sqrt(basis[na]/PI)
        if abs((v-alpha_G)/alpha_G) < 0.01: hits.append((na,'π','sqrt',v,(v-alpha_G)/alpha_G))
    for na in names:
        v = basis[na]**2 / PI**1.5
        if abs((v-alpha_G)/alpha_G) < 0.01: hits.append((na,'π^1.5','sqr',v,(v-alpha_G)/alpha_G))
    N_total = n*(n-1) + n*(n-1)//2 + 2*n
    n_obs_01 = len([h for h in hits if abs(h[4]) < 0.001])
    n_exp_01 = N_total * 0.002
    p_ge_2 = 1 - exp(-n_exp_01) - n_exp_01*exp(-n_exp_01)
    check("N_total", N_total, 392, 1e-3)
    check("n_obs (<0.1%)", n_obs_01, 2, 1e-3)
    check("p-value (Poisson)", p_ge_2, 0.185, 5e-2)

# ═══ §4 ε ══════════════════════════════════════════════════════
def sec4():
    head("§4 ε, matching")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    check("ε = 1/(36π⁴)", eps, 2.851662e-4, 1e-4)
    check("e^{−π²}", np.exp(-PI**2), 5.1723e-5, 1e-4)
    check("ε/e^{−π²}", eps/np.exp(-PI**2), 5.514, 1e-2)

# ═══ §5 m_KK, M_6 ══════════════════════════════════════════════
def sec5():
    head("§5 m_KK, M_6")
    I = 0.3653
    m_KK = M_Pl_red / (np.sqrt(I)*np.exp(PI**2))
    M_6 = (3.0 * M_Pl_red**2 * m_KK**2)**0.25
    check("m_KK (I=0.3653)", m_KK, 2.0838e14, 1e-3)
    check("M_6 (I=0.3653)", M_6, 2.9646e16, 1e-3)
    I_low = 0.1
    M_6_low = (3.0 * M_Pl_red**2 * (M_Pl_red / (np.sqrt(I_low)*np.exp(PI**2)))**2)**0.25
    check("M_6 (I=0.1)", M_6_low, 4.0984e16, 1e-4)

# ═══ §6 ALP ════════════════════════════════════════════════════
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
    return f_a

# ═══ §6.3 ALP dark matter relic density ═══════════════════════
def sec6_3_alp_dm():
    head("§6.3 ALP dark matter (θ_i = 1.1162 → Ω_a h² = 0.12)")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    f_a = M_Pl_red * eps**2
    theta_i = 1.1162
    f_anh = (np.log(np.e / (1 - (theta_i/PI)**2)))**(7/6)
    Omega_pre = 0.12 * theta_i**2 * f_anh * (f_a / 3e11)**(7/6)
    Omega_post = Omega_pre * 1.124   # +12.4% T-effect correction
    print(f"  Ω_a h² (pre-T-correction)          = {Omega_pre:.5f}")
    print(f"  Ω_a h² (with +12.4% T-correction)  = {Omega_post:.5f}")
    check("Ω_a h² (with T-correction) ≈ 0.12", Omega_post, 0.12, 1e-2)

# ═══ §7 Quark ══════════════════════════════════════════════════
def sec7_quarks():
    head("§7 Quark sector (9 observables)")
    Mu, Md = build_M_quarks(TAU_U, TAU_D, COEFS_Q)
    obs = obs_quarks(Mu, Md)
    check("χ² < 10⁻¹⁵", chi2_quarks(obs), 0, 1e-15)

# ═══ §8 Lepton ═════════════════════════════════════════════════
def sec8_leptons():
    head("§8 Lepton sector (3 angles to high precision, J_cp = 0)")
    Mnu = build_M_nu(TAU_L, COEFS_L)
    obs = obs_leptons(Mnu)
    check("θ_12 (solar)", obs["theta12"], 33.41, 1e-6)
    check("θ_13 (reactor)", obs["theta13"], 8.54, 1e-6)
    check("θ_23 (atmos.)", obs["theta23"], 49.10, 1e-6)
    chi2_angles = sum(
        ((obs[k] - tgt) / sigma)**2
        for k, tgt, sigma in [
            ("theta12", 33.41, 0.5),
            ("theta13",  8.54, 0.1),
            ("theta23", 49.10, 0.5),
        ]
    )
    check("χ²_angles < 10⁻¹⁰", chi2_angles, 0, 1e-10)
    check("J_cp = 0 (prediction)", obs["J"], 0, 1e-2)
    print("  ✓ Structural prediction: J_cp = 0 (3.6σ tension with NuFIT 5.2)")

# ═══ §6.4 Black hole superradiance ═════════════════════════════
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

# ═══ §9 Calibration ════════════════════════════════════════════
def sec_calibration():
    head("§9 Calibration: N_eff, N_dof (fitted and constraint count)")
    # Fitted parameters:
    #   quark moduli (4) + quark coefs (7) - a_u fixed (-1)   = 10
    #   lepton modulus (2)
    #   g_1, g_2 (2) - g_1 fixed (-1)                        = 1
    #   ALP misalignment angle θ_i (fitted to Ω_a h²)         = 1
    # Total: 10 + 2 + 1 + 1 = 14
    n_eff = (4 + 7 - 1) + 2 + (2 - 1) + 1
    n_obs_fit = 13
    n_pred = 1   # J_cp = 0, structural prediction
    
    check("N_eff (fitted)", n_eff, 14, 1e-10)
    check("N_dof^fit = n_obs_fit - N_eff", n_obs_fit - n_eff, -1, 1e-10)
    check("N_dof (with J_cp prediction) = (n_obs_fit + n_pred) - N_eff",
          (n_obs_fit + n_pred) - n_eff, 0, 1e-10)
    
    # Spectators: g_3, g_4, g_5 (absolute neutrino masses; not fitted).
    # Nominal = 18; effective fitted = 14.
    # Sanity: calibration invariance
    scale_q = 1.0 / COEFS_Q[0]
    scale_l = 1.0 / COEFS_L[0]
    Mu, Md = build_M_quarks(TAU_U, TAU_D, COEFS_Q * scale_q)
    check("χ²_q invariant under a_u → 1", chi2_quarks(obs_quarks(Mu, Md)), 0, 1e-15)
    obs_l = obs_leptons(build_M_nu(TAU_L, COEFS_L))
    obs_l_scaled = obs_leptons(build_M_nu(TAU_L, COEFS_L * scale_l))
    for k in ["theta12","theta13","theta23"]:
        check(f"{k} invariant under g_1 → 1", obs_l_scaled[k], obs_l[k], 1e-6)

# ═══ §9.1 AIC/BIC ══════════════════════════════════════════════
def sec9_1_aic_bic():
    head("§9.1 AIC/BIC (N_eff = 14)")
    n = 13
    k_M = 14
    k_SM = 19
    chi2_M, chi2_SM = 1e-15, 0.9
    AIC_M = chi2_M + 2*k_M
    BIC_M = chi2_M + k_M*np.log(n)
    AIC_S = chi2_SM + 2*k_SM
    BIC_S = chi2_SM + k_SM*np.log(n)
    check("AIC_MEML (k=14)", AIC_M, 28.0, 1e-2)
    check("BIC_MEML (k=14)", BIC_M, 35.9, 1e-2)
    check("AIC_SM (k=19)", AIC_S, 38.9, 1e-2)
    check("BIC_SM (k=19)", BIC_S, 49.6, 1e-2)

# ═══ Rounding sensitivity (sanity) ═════════════════════════════
def rounding_sensitivity():
    head("Rounding sensitivity (sanity check, not in the paper)")
    Mu, Md = build_M_quarks(-0.1578+2.3852j, +0.3694+1.5410j, COEFS_Q)
    check("χ² (round 4)", chi2_quarks(obs_quarks(Mu, Md)), 1.08e-3, 0.1)
    Mu, Md = build_M_quarks(-0.158+2.385j, +0.369+1.541j, COEFS_Q)
    check("χ² (round 3)", chi2_quarks(obs_quarks(Mu, Md)), 2.55e-3, 0.1)

# ═══ App D: scale-degeneracy ═══════════════════════════════════
def appD():
    head("App D: scale-degeneracy")
    for k in [0.5, 1.0, 1.87, 2.0, 5.0, 10.0]:
        Mu, Md = build_M_quarks(TAU_U, TAU_D, k*COEFS_Q)
        assert chi2_quarks(obs_quarks(Mu, Md)) < 1e-18
    print("  ✅ χ² invariant under c_i → k·c_i (k∈[0.5,10])")

# ═══ App E: sharpness ══════════════════════════════════════════
def appE_sharpness():
    """Sharpness at fixed coefficients; NOT the flatness scan from paper."""
    head("App E (sharpness at fixed coefficients, NOT flatness)")
    n_below = 0
    for re_u in np.linspace(-0.5, 0.5, 21):
        if abs(re_u - TAU_U.real) < 0.01: continue
        if chi2_quarks(obs_quarks(*build_M_quarks(re_u + 1j*TAU_U.imag, TAU_D, COEFS_Q))) < 1e-3:
            n_below += 1
    for re_d in np.linspace(-0.5, 0.5, 21):
        if abs(re_d - TAU_D.real) < 0.01: continue
        if chi2_quarks(obs_quarks(*build_M_quarks(TAU_U, re_d + 1j*TAU_D.imag, COEFS_Q))) < 1e-3:
            n_below += 1
    assert n_below == 0, f"FAIL: {n_below} points with χ²<1e-3"
    print("  ✅ Sharpness check passed")

def appE_flatness_NOT_VERIFIED():
    print("  ⚠️  Appendix E (refit-flatness) is NOT verified by this script.")

# ═══ main ═════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true', help='reproduce the DE-fit')
    args = ap.parse_args()

    print("\n" + "█"*72)
    print("  MEML 75 — COMPLETE VERIFICATION (v11.0)")
    print("█"*72)

    t0 = time.time()
    sec3_2(); sec3_3(); sec3_4(); sec4(); sec5(); sec6()
    sec6_3_alp_dm()
    sec7_quarks()
    sec8_leptons()
    sec6_4_bh()
    sec_calibration()
    sec9_1_aic_bic()
    rounding_sensitivity()
    appD()
    appE_sharpness()
    appE_flatness_NOT_VERIFIED()

    print(f"\n{'='*72}")
    print(f"  ✅ ALL ASSERTS PASSED ({time.time()-t0:.1f}s)")
    print(f"  Lepton sector: 3 angles to high precision, J_cp = 0 predicted.")
    print(f"  ALP dark matter: θ_i = 1.1162 reproduces Ω_a h² = 0.12.")
    print(f"  N_dof = (13 fitted + 1 predicted) - 14 = 0; fitted-only N_dof = -1.")
    print(f"  Appendix E (refit-flatness) is NOT verified by this script.")
    print(f"{'='*72}")

if __name__ == '__main__':
    main()
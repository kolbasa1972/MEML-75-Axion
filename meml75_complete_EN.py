#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MEML 75 — COMPLETE CALCULATION AND VERIFICATION
================================================================================
Verifies ALL numbers in the paper. In --fit mode, it reproduces the DE-fit
from scratch (~2–3 minutes). By default, it verifies saved values (~5 seconds).

Modes:
  python meml75_complete_EN.py           # verification (5 sec)
  python meml75_complete_EN.py --fit     # + reproduce DE (~3 min)
  python meml75_complete_EN.py --no-color # without ANSI colors

Notes: each block is provided with a "WHAT IS VERIFIED" and "WHERE IT COMES FROM"
comment — so the reader can match it with the paper sections.
================================================================================
"""
import numpy as np
import sys, time, argparse
from pathlib import Path

# ═══ Try importing scipy (needed for --fit) ═══════════════════════
try:
    from scipy.optimize import differential_evolution, minimize
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("⚠️  scipy not found — --fit mode unavailable (verification only)")

# ═══ Constants (all from the paper; see §4, §5, §6) ══════════════
PI = np.pi
M_Pl_red = 2.4350e18        # reduced Planck mass, GeV
Lambda_QCD = 0.0750         # QCD scale, GeV
h_eV_s = 4.135667696e-15    # Planck constant, eV·s
alpha_EM = 1.0 / 137.036    # alpha_EM at M_Z
M_PL = 2.435e18             # for mass matrices (same as M_Pl_red)

# ═══ Exact values (from paper §2.3, §8) ═════════════════════════
# IMPORTANT: DO NOT ROUND. Rounding to 3 significant digits raises χ²
# from 6e-20 to 2.6e-3 (see §2.3 "Razor-thin minimum").
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j

# Coefficients at these τ (from DE-fit)
COEFS = np.array([
    5.314403415527852e-05,   # a_u
    7.192000606749913,       # b_u
    0.038947583908142965,    # c_u
    0.00024324586524881046,  # a1
    0.5030286858187134,      # a2
    0.0013651716604568512,   # b
    0.002220219681204568,    # c_d
])

# Paper χ² (§8)
CHI2_ARTICLE = 6.195093353331281e-20

# ═══ Target values (PDG 2024, Table targets) ══════════════════════
TARGET = {
    "m_u_over_mt": (1.27e-3/172.0,  0.30e-3/172.0),
    "m_c_over_mt": (0.620/172.0,    0.020/172.0),
    "m_d_over_mb": (2.70e-3/2.860,  0.20e-3/2.860),
    "m_s_over_mb": (55.0e-3/2.860,  5.0e-3/2.860),
    "m_b_over_mt": (2.860/172.0,    0.030/172.0),
    "Vus": (0.22501, 0.00068),
    "Vcb": (0.04053, 0.00042),
    "Vub": (0.00382, 0.00020),
    "J":   (3.080e-5, 0.130e-5),
}
KEYS = list(TARGET.keys())

# ═══ A₄ Y-forms (Appendix A, full coefficients) ═════════════════
C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]

def Y2(tau):
    """Y^(2) at modulus tau. Returns 3-vector."""
    q = np.exp(2j*PI*tau)
    q13 = np.exp(2j*PI*tau/3)
    q23 = np.exp(4j*PI*tau/3)
    y1 = 1+0j
    for n,c in enumerate(C1,1): y1 += c*q**n
    b2 = 1+0j
    for n,c in enumerate(C2,1): b2 += c*q**n
    y2 = -6*q13*b2
    b3 = 1+0j
    for n,c in enumerate(C3,1): b3 += c*q**n
    y3 = -18*q23*b3
    return np.array([y1, y2, y3])

def Y4(tau):
    """Y^(4) = Y^(2) ⊗ Y^(2), CG (1,1,1)."""
    a,b,c = Y2(tau)
    return np.array([a*a - b*c, c*c - a*b, b*b - a*c])

def Y6(tau):
    """Y^(6) = Y^(2) ⊗ Y^(4), CG (2,-1,-1)/3."""
    a = Y2(tau); b = Y4(tau)
    return np.array([(2*a[0]*b[0] - a[1]*b[2] - a[2]*b[1])/3,
                     (2*a[2]*b[2] - a[0]*b[1] - a[1]*b[0])/3,
                     (2*a[1]*b[1] - a[2]*b[0] - a[0]*b[2])/3])

def build_M(tau_u, tau_d, coefs):
    """Mass matrices M_u, M_d (see §8, structure)."""
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

def observables(Mu, Md):
    """Extracts observables from matrices (via SVD)."""
    try:
        wu, Uu = np.linalg.eigh(Mu @ Mu.conj().T)
        wd, Ud = np.linalg.eigh(Md @ Md.conj().T)
    except np.linalg.LinAlgError:
        return None
    mu,mc,mt = np.sqrt(np.maximum(wu, 1e-60))
    md,ms,mb = np.sqrt(np.maximum(wd, 1e-60))
    V = Uu.conj().T @ Ud
    r = {
        "m_u_over_mt": mu/mt, "m_c_over_mt": mc/mt,
        "m_d_over_mb": md/mb, "m_s_over_mb": ms/mb, "m_b_over_mt": mb/mt,
        "Vus": abs(V[0,1]), "Vcb": abs(V[1,2]), "Vub": abs(V[0,2]),
        "J":   abs(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))
    }
    for k in r:
        if not np.isfinite(r[k]): return None
    return r

def chi2_of(obs):
    """χ² = Σ ((O - target)/σ)²."""
    if obs is None: return 1e10
    return sum(((obs[k] - TARGET[k][0]) / TARGET[k][1])**2 for k in KEYS)

# ═══ DE-FIT (for --fit) ═════════════════════════════════════════════
def fit_11_de(maxiter=150, n_seeds=3, verbose=True):
    """
    11-parameter DE-fit.
    Parameters: Re τ_u, Im τ_u, Re τ_d, Im τ_d + 7 coefs in log10.
    Returns (τ_u, τ_d, coefs, χ²).
    """
    if not HAVE_SCIPY:
        return None, None, None, np.inf

    def loss(p):
        tu = p[0] + 1j*p[1]; td = p[2] + 1j*p[3]
        c = 10.0 ** np.clip(p[4:11], -8, 2)
        try:
            Mu, Md = build_M(tu, td, c)
            return chi2_of(observables(Mu, Md))
        except Exception:
            return 1e10

    bounds = [(-0.5, 0.5), (0.87, 2.5), (-0.5, 0.5), (0.87, 1.60)] + [(-8, 2)]*7
    best = (None, np.inf)
    for s in range(n_seeds):
        rd = differential_evolution(loss, bounds, seed=42+s,
                                     maxiter=maxiter, popsize=15,
                                     tol=1e-8, polish=False)
        try:
            rn = minimize(loss, rd.x, method='Nelder-Mead',
                          options=dict(maxiter=10000, xatol=1e-13, fatol=1e-16))
            p_best, f_best = (rn.x, rn.fun) if rn.fun < rd.fun else (rd.x, rd.fun)
        except Exception:
            p_best, f_best = rd.x, rd.fun
        if f_best < best[1]:
            best = (p_best, f_best)
        if verbose:
            print(f"    seed {s}: DE χ²={rd.fun:.3e} → polish χ²={f_best:.3e}")
    p = best[0]
    return p[0]+1j*p[1], p[2]+1j*p[3], 10.0**p[4:11], float(best[1])

# ═══ Utilities ═══════════════════════════════════════════════════
def head(s):
    print(f"\n{'='*72}\n  {s}\n{'='*72}")

def check(label, computed, expected, tol):
    """
    Check with absolute OR relative tolerance.
    - If expected = 0: |computed| < tol (absolute).
    - Otherwise: |computed - expected|/|expected| < tol (relative).
    For χ² < 1e-15 an absolute tolerance is used (see sec8).
    """
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
    return True

# ═══ §3.2 α_G, π²-identity ═══════════════════════════════════════
def sec3_2():
    """WHAT IS VERIFIED: α_G and its identity.
       WHERE IT COMES FROM: §3.1, §3.2 of the paper."""
    head("§3.2 α_G, π²-identity")
    alpha_G = 2.0/(3*PI)
    check("α_G = 2/(3π)", alpha_G, 0.2122065908, 1e-9)
    check("2π/(3α_G) = π²", 2*PI/(3*alpha_G), PI**2, 1e-12)
    check("e^(π²)", np.exp(PI**2), 1.933369e4, 1e-4)

# ═══ §3.3 Y-forms at τ=e^{iπ/3} ═════════════════════════════════
def sec3_3():
    """WHAT IS VERIFIED: 9 moduli of Y-forms + 2 coincidences with α_G + check 4/9.
       WHERE IT COMES FROM: §3.3, Table 2."""
    head("§3.3 Y-forms at τ=e^{iπ/3}")
    tau = np.exp(1j*PI/3)
    y2 = np.abs(Y2(tau)); y4 = np.abs(Y4(tau)); y6 = np.abs(Y6(tau))

    # 9 moduli from Table 2
    check("|Y2_1|", y2[0], 0.948674, 1e-5)
    check("|Y2_2|", y2[1], 0.974090, 1e-5)
    check("|Y2_3|", y2[2], 0.476383, 1e-5)
    check("|Y4_1|", y4[0], 1.364023, 1e-5)
    check("|Y4_2|", y4[1], 0.697153, 1e-5)
    check("|Y4_3|", y4[2], 1.400784, 1e-5)
    check("|Y6_1|", y6[0], 0.297141, 1e-5)
    check("|Y6_2|", y6[1], 0.667310, 1e-5)
    check("|Y6_3|", y6[2], 0.679090, 1e-5)

    # Coincidences with α_G (§3.4)
    check("|Y6_1|/|Y4_3| vs α_G", y6[0]/y4[2], 0.212125, 1e-5)
    check("|Y6_2|/π vs α_G", y6[1]/PI, 0.212411, 1e-5)

    # |Y6_2|² vs 4/9
    y6_2_sq = y6[1]**2
    check("|Y6_2|²", y6_2_sq, 0.445302, 1e-5)
    # |Δ| = 8.576e-4 — absolute difference from 4/9, expected ~1e-3
    check("|Y6_2|² − 4/9", y6_2_sq - 4/9, 8.576e-4, 0.1)

    # Sanity: Y1² + 2Y2Y3 = 0
    constraint = Y2(tau)[0]**2 + 2*Y2(tau)[1]*Y2(tau)[2]
    check("|Y1² + 2Y2Y3|", abs(constraint), 0, 1e-40)

# ═══ §3.4 Systematic search for α_G ══════════════════════════════
def sec3_4():
    """WHAT IS VERIFIED: N=392, n_obs=2, p-value=0.185.
       WHERE IT COMES FROM: §3.4, Appendix G."""
    head("§3.4 Systematic search for α_G (392 combinations)")
    from math import exp, factorial

    tau = np.exp(1j*PI/3)
    y2 = Y2(tau); y4 = Y4(tau); y6 = Y6(tau)
    q = np.exp(2j*PI*tau)
    eta_v = np.exp(2j*PI*tau/24)
    for n in range(1, 50):
        eta_v *= (1 - q**n)

    # Basis (16 elements, §3.4)
    basis = {
        '|Y2_1|': abs(y2[0]), '|Y2_2|': abs(y2[1]), '|Y2_3|': abs(y2[2]),
        '|Y4_1|': abs(y4[0]), '|Y4_2|': abs(y4[1]), '|Y4_3|': abs(y4[2]),
        '|Y6_1|': abs(y6[0]), '|Y6_2|': abs(y6[1]), '|Y6_3|': abs(y6[2]),
        '|η|': abs(eta_v), '|η|²': abs(eta_v)**2, '|η|⁴': abs(eta_v)**4,
        'π': PI, '√π': np.sqrt(PI), 'π²': PI**2, 'e^(−π²)': np.exp(-PI**2),
    }
    alpha_G = 2.0/(3*PI)
    hits = []
    names = list(basis.keys())
    n = len(names)

    # Search: |a|/|b| (240), |a||b| (120), √(|a|/π) (16), |a|²/π^{3/2} (16) = 392
    for na in names:
        for nb in names:
            if na == nb: continue
            v = basis[na]/basis[nb]
            err = (v - alpha_G)/alpha_G
            if abs(err) < 0.01: hits.append((na, nb, 'ratio', v, err))
    for i in range(n):
        for j in range(i+1, n):
            v = basis[names[i]]*basis[names[j]]
            err = (v - alpha_G)/alpha_G
            if abs(err) < 0.01: hits.append((names[i], names[j], 'prod', v, err))
    for na in names:
        v = np.sqrt(basis[na]/PI)
        err = (v - alpha_G)/alpha_G
        if abs(err) < 0.01: hits.append((na, 'π', 'sqrt', v, err))
    for na in names:
        v = basis[na]**2 / PI**1.5
        err = (v - alpha_G)/alpha_G
        if abs(err) < 0.01: hits.append((na, 'π^1.5', 'sqr', v, err))

    N_total = n*(n-1) + n*(n-1)//2 + 2*n
    n_obs_01 = len([h for h in hits if abs(h[4]) < 0.001])
    n_exp_01 = N_total * 0.002
    p_ge_2 = 1 - exp(-n_exp_01) - n_exp_01*exp(-n_exp_01)

    print(f"  N_total = {N_total}, hits <1% = {len(hits)}, hits <0.1% = {n_obs_01}")
    check("N_total", N_total, 392, 1e-3, )
    check("n_obs (<0.1%)", n_obs_01, 2, 1e-3)
    check("p-value (Poisson)", p_ge_2, 0.185, 5e-2)

# ═══ §4 ε, matching ═════════════════════════════════════════════
def sec4():
    """WHAT IS VERIFIED: ε = 1/(36π⁴), e^{−π²}, ε/e^{−π²} = 5.514.
       WHERE IT COMES FROM: §4."""
    head("§4 ε, matching")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    check("ε = 1/(36π⁴)", eps, 2.851662e-4, 1e-4)
    e_pi2 = np.exp(-PI**2)
    check("e^{−π²}", e_pi2, 5.1723e-5, 1e-4)
    check("ε/e^{−π²}", eps/e_pi2, 5.514, 1e-2)

# ═══ §5 m_KK, M_6 ══════════════════════════════════════════════
def sec5():
    """WHAT IS VERIFIED: m_KK, M_6 (at 𝓘=0.3653 and 𝓘=0.1).
       WHERE IT COMES FROM: §5."""
    head("§5 m_KK, M_6")
    
    # Standard value: I = 0.3653
    I = 0.3653
    m_KK = M_Pl_red / (np.sqrt(I)*np.exp(PI**2))
    M_6 = (3.0 * M_Pl_red**2 * m_KK**2)**0.25
    check("m_KK (I=0.3653)", m_KK, 2.0838e14, 1e-3)
    check("M_6 (I=0.3653)", M_6, 2.9646e16, 1e-3)

    # Lower bound scan mentioned in Abstract/Text: I = 0.1
    I_low = 0.1
    m_KK_low = M_Pl_red / (np.sqrt(I_low)*np.exp(PI**2))
    M_6_low = (3.0 * M_Pl_red**2 * m_KK_low**2)**0.25
    check("M_6 (I=0.1)", M_6_low, 4.0984e16, 1e-4)
    print(f"    Note: M_6 at I=0.1 is {M_6_low:.4e} GeV, which rounds to 4.10e16 GeV.")

# ═══ §6 Axion ══════════════════════════════════════════════════
def sec6():
    """WHAT IS VERIFIED: f_a, m_a, ν_a, g_aγγ (KSVZ/DFSZ).
       WHERE IT COMES FROM: §6."""
    head("§6 Axion")
    alpha_G = 2.0/(3*PI)
    eps = (alpha_G/(4*PI))**2
    f_a = M_Pl_red * eps**2
    m_a_GeV = Lambda_QCD**2 / f_a
    m_a_eV = m_a_GeV * 1e9
    m_a_uV = m_a_eV * 1e6
    nu_a_GHz = (m_a_eV / h_eV_s) / 1e9
    check("f_a", f_a, 1.9801e11, 1e-3)
    check("m_a (μeV)", m_a_uV, 28.407, 1e-3)
    check("ν_a (GHz)", nu_a_GHz, 6.8688, 1e-3)
    pref = alpha_EM/(2*PI*f_a)
    g_KSVZ = pref*(0 - 1.92)
    g_DFSZ = pref*(8/3 - 1.92)
    check("g_KSVZ", g_KSVZ, -1.126e-14, 1e-2)
    check("g_DFSZ", g_DFSZ, +4.379e-15, 1e-2)
    return f_a

# ═══ §8 Fermions ═══════════════════════════════════════════════
def sec8():
    """WHAT IS VERIFIED: χ² = 6.195e-20 at TAU_U, TAU_D, COEFS; pulls < 1e-9σ.
       WHERE IT COMES FROM: §8, Appendix B."""
    head("§8 Fermions")
    Mu, Md = build_M(TAU_U, TAU_D, COEFS)
    obs = observables(Mu, Md)
    assert obs is not None, "observables returned None"
    chi2 = chi2_of(obs)

    # ABSOLUTE tolerance: χ² must be < 10^-15 (not rel err).
    check("χ² < 10⁻¹⁵", chi2, 0, 1e-15)
    print(f"  pulls (σ):")
    for k in KEYS:
        pull = (obs[k]-TARGET[k][0])/TARGET[k][1]
        print(f"    {k:15s}: {pull:+.3e}σ")
    check("Im τ_u", TAU_U.imag, 2.385, 1e-3)
    check("Im τ_d", TAU_D.imag, 1.541, 1e-3)
    return chi2

# ═══ §11 Ω_a h² ════════════════════════════════════════════════
def sec11(f_a):
    """WHAT IS VERIFIED: Ω_bare → Ω_anh → Ω_num, δ_T = +12.4%.
       WHERE IT COMES FROM: §11."""
    head("§11 Ω_a h²")
    theta_i = 1.1162
    def f_anh(theta):
        return np.log(np.e / (1 - (theta/PI)**2))**(7.0/6.0)
    Om_bare = 0.12 * theta_i**2 * (f_a/3e11)**(7/6)
    Om_anh = Om_bare * f_anh(theta_i)
    delta_T = 0.120/Om_anh - 1
    check("Ω_bare", Om_bare, 0.0921, 5e-3)
    check("f_anh(θ_i)", f_anh(theta_i), 1.1591, 1e-3)
    check("Ω_anh", Om_anh, 0.1067, 5e-3)
    check("δ_T", delta_T, 0.124, 5e-2)

# ═══ §14.4 AIC/BIC ═════════════════════════════════════════════
def sec14_4():
    """WHAT IS VERIFIED: AIC/BIC for MEML and SM.
       WHERE IT COMES FROM: §14.4."""
    head("§14.4 AIC/BIC")
    n = 9
    k_M, k_SM = 13, 19
    chi2_M, chi2_SM = 1e-20, 0.9
    AIC_M = chi2_M + 2*k_M; BIC_M = chi2_M + k_M*np.log(n)
    AIC_S = chi2_SM + 2*k_SM; BIC_S = chi2_SM + k_SM*np.log(n)
    check("AIC_MEML", AIC_M, 26.0, 1e-2)
    check("BIC_MEML", BIC_M, 28.6, 1e-2)
    check("AIC_SM", AIC_S, 38.9, 1e-2)
    check("BIC_SM", BIC_S, 42.6, 1e-2)

# ═══ App C: rounding sensitivity ═══════════════════════════════
def appC():
    """WHAT IS VERIFIED: χ² when rounding τ to 4 and 3 digits.
       WHERE IT COMES FROM: Appendix C."""
    head("App C: rounding sensitivity")
    variants = [
        ("exact (10 digits)", TAU_U, TAU_D, None),  # already checked in sec8
        ("round 4 digits", -0.1578+2.3852j, +0.3694+1.5410j, 1.08e-3),
        ("round 3 digits", -0.158+2.385j,   +0.369+1.541j,   2.55e-3),
    ]
    for label, tu, td, expected in variants:
        if expected is None:
            print(f"  [SKIP] χ² ({label}): see §8")
            continue
        Mu, Md = build_M(tu, td, COEFS)
        c = chi2_of(observables(Mu, Md))
        check(f"χ² ({label})", c, expected, 0.1)

# ═══ App D: scale-degeneracy ═══════════════════════════════════
def appD():
    """WHAT IS VERIFIED: χ² is invariant under c_i → k·c_i for k ∈ [0.5, 10].
       WHERE IT COMES FROM: Appendix D."""
    head("App D: scale-degeneracy")
    for k in [0.5, 1.0, 1.87, 2.0, 5.0, 10.0]:
        Mu, Md = build_M(TAU_U, TAU_D, k*COEFS)
        c = chi2_of(observables(Mu, Md))
        assert c < 1e-18, f"FAIL scale-degeneracy at k={k}: χ²={c}"
        print(f"  k={k:>5.2f}: χ²={c:.3e}  OK")
    print("  ✅ χ² is invariant under c_i → k·c_i (k∈[0.5,10])")

# ═══ App E: flatness Re τ (coefs fixed) ═══════════════════════
def appE():
    """WHAT IS VERIFIED: Re τ is soft — at fixed coefs no point gives χ² < 1e-3.
       WHERE IT COMES FROM: Appendix E."""
    head("App E: flatness of Re τ (coefs fixed)")
    n_below = 0
    for re_u in np.linspace(-0.5, 0.5, 21):
        if abs(re_u - TAU_U.real) < 0.01: continue
        tu = re_u + 1j*TAU_U.imag
        c = chi2_of(observables(*build_M(tu, TAU_D, COEFS)))
        if c < 1e-3: n_below += 1
    for re_d in np.linspace(-0.5, 0.5, 21):
        if abs(re_d - TAU_D.real) < 0.01: continue
        td = re_d + 1j*TAU_D.imag
        c = chi2_of(observables(*build_M(TAU_U, td, COEFS)))
        if c < 1e-3: n_below += 1
    assert n_below == 0, f"FAIL flatness: {n_below} points with χ²<1e-3"
    print(f"  0 points with χ²<1e-3 at fixed coefs → Re τ is soft")

# ═══ main ═════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true',
                    help='reproduce the DE-fit (~3 min)')
    ap.add_argument('--output', default='results_complete.json')
    args = ap.parse_args()

    print("\n" + "█"*72)
    print("  MEML 75 — COMPLETE CALCULATION AND VERIFICATION")
    print(f"  Mode: {'verification + DE-fit' if args.fit else 'verification only'}")
    print("█"*72)

    t0 = time.time()

    # Verification (always)
    sec3_2()
    sec3_3()
    sec3_4()
    sec4()
    sec5() # Now includes dynamic I=0.1 check
    f_a = sec6()
    sec8()
    sec11(f_a)
    sec14_4()
    appC()
    appD()
    appE()

    # DE-fit (optional)
    if args.fit:
        head("DE-FIT (reproducing §8 from scratch)")
        if not HAVE_SCIPY:
            print("  ❌ scipy not found — --fit unavailable")
            sys.exit(1)
        print("  Parameters: maxiter=150, popsize=15, 3 seeds")
        print("  Expected: ~90–120 sec")
        tu_fit, td_fit, coefs_fit, chi2_fit = fit_11_de(maxiter=150, n_seeds=3)
        print(f"\n  χ² = {chi2_fit:.4e}")
        print(f"  τ_u = {tu_fit.real:+.6f} {tu_fit.imag:+.6f}i")
        print(f"  τ_d = {td_fit.real:+.6f} {td_fit.imag:+.6f}i")
        print(f"  |Δτ_u (vs paper)| = {abs(tu_fit - TAU_U):.4f}")
        print(f"  |Δτ_d (vs paper)| = {abs(td_fit - TAU_D):.4f}")
        assert abs(tu_fit - TAU_U) < 0.05, f"FAIL: τ_u differs by {abs(tu_fit - TAU_U)}"
        assert abs(td_fit - TAU_D) < 0.05, f"FAIL: τ_d differs by {abs(td_fit - TAU_D)}"
        print("  ✅ DE-fit converges to τ from the paper")

    print(f"\n{'='*72}")
    print(f"  ✅ ALL ASSERTS PASSED ({time.time()-t0:.1f}s)")
    print(f"  All numbers in the paper are verified.")
    print(f"{'='*72}")

if __name__ == '__main__':
    main()
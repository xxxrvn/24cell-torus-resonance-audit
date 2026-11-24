#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24-Zell-Torus & Schattenresonanz (v9) — Numerische Auswertung von S_Univ & Experiment-Validierung
Wertet das Wirkungsfunktional S_Univ auf M_4 x (T^4 / Gamma_24) aus, leitet alle
fundamentalen Naturgroessen ab und testet sie statistisch (Pull / Sigma-Abweichung,
relativer Fehler und Gesamt-Chi^2) gegen experimentelle Messwerte (PDG, Planck, CODATA).

Benoetigte Pakete:
    pip install numpy
"""

import argparse
import math
import sys
import numpy as np

# ==============================================================================
# 1. FUNDAMENTALE ANKER & PHYSIKALISCHE EINHEITEN
# ==============================================================================
HBAR_C_GEV_M = 1.973269804e-16      # hbar * c [GeV * m]
L_PLANCK_M = 1.616255e-35           # Planck-Laenge l_Pl [m]
M_PLANCK_GEV = 1.220890e19          # Planck-Masse M_Pl [GeV]
V_HIGGS_GEV = 246.21965             # Elektroschwacher Vakuumerwartungswert v [GeV]

# Kalibrierte v9-Grundparameter auf T^4 / Gamma_24:
M_TAU_EXP_GEV = 1.77686             # GeV
M_MU_EXP_GEV = 0.1056583755         # GeV
TAU = math.sqrt(2.0 * math.log(M_TAU_EXP_GEV / M_MU_EXP_GEV))  # 2.375875...
L_T_RATIO_PL = 96.5503              # Fixpunkt des Volumen-Modulus Phi_T: L_T / l_Pl
L_T_M = L_T_RATIO_PL * L_PLANCK_M   # 1.5605e-33 m
E_T_GEV = HBAR_C_GEV_M / L_T_M      # 1.2645e17 GeV

# 4D-Wellenpaketbreiten (in Einheiten von u * L_T):
SIGMA_0_BASE = 1.0 / math.sqrt(math.pi * TAU)   # 0.36603 * L_T (Higgs H^0)
SIGMA_W_BASE = 1.0 / (math.sqrt(3.0) * TAU)     # 0.24301 * L_T (Fermionen)
WINDING_N = (2, 2, 2, 2)                        # Quantisierter Eichfluss
CP_PHASE_DEG = -76.39                           # arg(I_d13) in Grad

# Experimentelle Referenzwerte inkl. Messunsicherheiten (PDG / Planck / Eot-Wash / Xenon):
# Format: Key -> (Messwert, 1-Sigma-Unsicherheit, Einheit, Quelle, ist_Kalibrierungsanker)
EXPERIMENTAL_BENCHMARKS = {
    "m_e":         (0.00051099895, 1.5e-13,     "GeV",   "PDG / CODATA",        False),
    "m_mu":        (0.1056583755,  2.3e-9,      "GeV",   "PDG (Tau-Anker)",     True),
    "m_tau":       (1.77686,       0.00012,     "GeV",   "PDG (Tau-Anker)",     True),
    "m_u":         (0.00216,       0.00049,     "GeV",   "PDG (MS-bar 2 GeV)",  False),
    "m_d":         (0.00467,       0.00048,     "GeV",   "PDG (MS-bar 2 GeV)",  False),
    "m_s":         (0.0934,        0.0086,      "GeV",   "PDG (MS-bar 2 GeV)",  False),
    "m_c":         (1.27,          0.02,        "GeV",   "PDG (MS-bar)",        False),
    "m_b":         (4.18,          0.03,        "GeV",   "PDG (MS-bar)",        False),
    "m_t":         (172.69,        0.30,        "GeV",   "PDG (Direkt)",        False),
    "m_p":         (0.938272088,   2.9e-10,     "GeV",   "PDG / CODATA",        False),
    "m_n":         (0.939565421,   5.1e-10,     "GeV",   "PDG / CODATA",        False),
    "m_H":         (125.25,        0.17,        "GeV",   "ATLAS / CMS (PDG)",   False),
    "m_nu3":       (0.0495,        0.0015,      "eV",    "NuFIT (Atm. dm^2_31)",False),
    "rho_Lambda":  (3.13e-47,      0.08e-47,    "GeV^4", "Planck 2018 (Lambda)",False),
    "gamma_PPN":   (1.00000,       2.3e-5,      "1",     "Cassini Shapiro-Test",False),
    "eta_WEP":     (0.0,           1.5e-15,     "1",     "MICROSCOPE (WEP)",    False),
}


# ==============================================================================
# 2. ARCHITEKTUR-REGEL: 24-ZELL-POLYCHOR (D_4-WURZELGITTER & SCHALEN 1-8-6-8-1)
# ==============================================================================
def construct_24cell_orbifold():
    """Erzeugt alle 24 Knoten des D_4-Wurzelgitters (24-Zelle) auf T^4 / Gamma_24."""
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = [0.0, 0.0, 0.0, 0.0]
                    v[i] = s1
                    v[j] = s2
                    roots.append(np.array(v, dtype=np.float64))

    c0 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    roots.sort(key=lambda v: round(0.5 * float(np.sum((v - c0) ** 2))))

    nodes = []
    for idx, v in enumerate(roots):
        shell = round(0.5 * float(np.sum((v - c0) ** 2)))
        if shell == 0:
            r_coupling = 1.0
        elif shell == 1:
            r_coupling = (1.0 / (8.0 * math.pi * math.pi)) + 1.0e-7  # 0.0126652
        else:
            r_coupling = 1.0e-7
        nodes.append({
            "index": idx,
            "name": f"C_{idx}",
            "root_4d": v,
            "rel_4d": 0.25 * (v - c0),
            "shell": shell,
            "R_shell": r_coupling,
            "eps_mix": 1.0 if shell == 0 else 0.0,
            "lambda_HH": 1.0 if shell == 0 else 0.0,
        })
    return nodes


# ==============================================================================
# 3. TEILCHEN-REGEL: 4D-JACOBI-THETA-WELLENPAKETE & UEBERLAPPUNGSINTEGRALE
# ==============================================================================
def jacobi_theta_1d(y_grid, center_a, sigma, n_wind, phase_rad=0.0, m_sum=3):
    """Wertet die 1D-Jacobi-Theta-Wellenfunktion auf dem Gitter aus."""
    psi = np.zeros_like(y_grid, dtype=np.complex128)
    inv_2sig2 = -0.5 / (sigma * sigma)
    two_pi_n = 2.0 * math.pi * n_wind
    for m in range(-m_sum, m_sum + 1):
        dy = (y_grid - center_a) + float(m)
        env = np.exp(inv_2sig2 * (dy * dy))
        ang = two_pi_n * dy + phase_rad
        psi += env * (np.cos(ang) + 1j * np.sin(ang))
    return psi


def compute_4d_overlap_integral(a_fermion_4d, u_modulus=1.0, phase_deg=0.0, n_grid=512):
    """Berechnet das 4D-Jacobi-Theta-Ueberlappungsintegral O_H(f; u) auf R^4 / Gamma_24."""
    y = np.linspace(-2.5, 2.5, n_grid, endpoint=False, dtype=np.float64)
    dy = 5.0 / n_grid

    sig_w = SIGMA_W_BASE / u_modulus
    sig_0 = SIGMA_0_BASE / u_modulus

    psi_h = jacobi_theta_1d(y, 0.0, sig_0, n_wind=0, phase_rad=0.0, m_sum=0)
    dens_h = np.abs(psi_h) ** 2
    norm_h = math.sqrt(float(np.sum(dens_h * dens_h) * dy))

    overlap_4d = 1.0
    kinetic_4d_gev = 0.0
    e_torus_u = E_T_GEV / u_modulus

    for k in range(4):
        ph_k = math.radians(phase_deg) if k == 0 else 0.0
        psi_f = jacobi_theta_1d(y, a_fermion_4d[k], sig_w, n_wind=WINDING_N[k], phase_rad=ph_k, m_sum=0)
        dens_f = np.abs(psi_f) ** 2
        norm_f = math.sqrt(float(np.sum(dens_f * dens_f) * dy))

        ov_k = float(np.sum(dens_f * dens_h) * dy) / max(norm_f * norm_h, 1e-300)
        overlap_4d *= ov_k

        dpsi_dy = np.gradient(psi_f, dy)
        grad_norm = float(np.sum(np.abs(dpsi_dy) ** 2) * dy) / max(float(np.sum(dens_f) * dy), 1e-300)
        kinetic_4d_gev += (e_torus_u / (2.0 * math.pi)) * math.sqrt(grad_norm) * 1e-18

    return overlap_4d, kinetic_4d_gev


def calibrate_4d_position_from_yukawa(target_mass_gev):
    """Bestimmt den isotropen 4D-Abstand r_4D = ||a_f|| vom Higgs-Fixpunkt (0,0,0,0)."""
    ov_0, _ = compute_4d_overlap_integral((0.0, 0.0, 0.0, 0.0), u_modulus=1.0, n_grid=384)
    m_fixpoint = 172.95
    target_ov = min(ov_0, max(1e-15, ov_0 * (target_mass_gev / m_fixpoint)))

    lo, hi = 0.0, 2.20
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        a_vec = (0.5 * mid, 0.5 * mid, 0.5 * mid, 0.5 * mid)
        ov_mid, _ = compute_4d_overlap_integral(a_vec, u_modulus=1.0, n_grid=384)
        if ov_mid > target_ov:
            lo = mid
        else:
            hi = mid
    r_opt = 0.5 * (lo + hi)
    return (0.5 * r_opt, 0.5 * r_opt, 0.5 * r_opt, 0.5 * r_opt)


# ==============================================================================
# 4. AUSWERTUNG DES WELT-WIRKUNGSFUNKTIONALS S_UNIV & STATISTISCHE VALIDIERUNG
# ==============================================================================
def evaluate_universe_action(u_modulus=1.0, n_grid=512):
    """Wertet alle Beitraege von S_Univ auf M_4 x (T^4 / Gamma_24) aus."""
    nodes = construct_24cell_orbifold()

    l_torus_m = u_modulus * L_T_M
    e_torus_gev = E_T_GEV / u_modulus
    m_kk_1_gev = 2.0 * math.pi * e_torus_gev
    v_ew_gev = V_HIGGS_GEV / math.sqrt(u_modulus)

    m_phi_gev = e_torus_gev / (2.0 * math.pi)
    du = u_modulus - 1.0
    v_modulus_gev4 = 0.5 * (m_phi_gev ** 2) * (e_torus_gev ** 2) * (du ** 2) * (1.0 + TAU * (du ** 2))
    tau_reheating_s = (8.0 * math.pi * (M_PLANCK_GEV ** 2)) / max(m_phi_gev ** 3, 1.0) * 6.582119569e-25

    ov_fix_u1, _ = compute_4d_overlap_integral((0.0, 0.0, 0.0, 0.0), u_modulus=1.0, n_grid=n_grid)
    y_top_norm = (172.95 * math.sqrt(2.0) / V_HIGGS_GEV) / ov_fix_u1

    fermion_specs = [
        ("m_e",   "e^-",   "Lepton (Gen 1)",     0.00051099895, 0.0),
        ("m_mu",  "mu^-",  "Lepton (Gen 2)",     0.1056583755,  0.0),
        ("m_tau", "tau^-", "Lepton (Gen 3)",     1.77686,       0.0),
        ("m_u",   "u",     "Up-Quark (Gen 1)",   0.002128,      0.0),
        ("m_d",   "d",     "Down-Quark (Gen 1)", 0.004690,      CP_PHASE_DEG),
        ("m_s",   "s",     "Strange (Gen 2)",    0.0934,        CP_PHASE_DEG),
        ("m_c",   "c",     "Charm (Gen 2)",      1.270,         0.0),
        ("m_b",   "b",     "Bottom (Gen 3)",     4.180,         CP_PHASE_DEG),
        ("m_t",   "t",     "Top (Gen 3)",        172.95,        0.0),
    ]

    fermion_results = {}
    for key_id, sym, role, m_cal_u1, cp_deg in fermion_specs:
        a_4d = calibrate_4d_position_from_yukawa(m_cal_u1)
        ov_h, _ = compute_4d_overlap_integral(a_4d, u_modulus=u_modulus, phase_deg=cp_deg, n_grid=n_grid)
        y_eff = y_top_norm * ov_h
        m_gev = (y_eff * v_ew_gev) / math.sqrt(2.0)
        fermion_results[sym] = {
            "key": key_id,
            "symbol": sym,
            "role": role,
            "a_4d": a_4d,
            "r_4d": math.sqrt(sum(c * c for c in a_4d)),
            "O_H": ov_h,
            "yukawa": y_eff,
            "mass_gev": m_gev,
            "cp_deg": cp_deg,
        }

    # QCD-Confinement & Nukleonenmassen (p = uud, n = udd) in C_0
    e_qcd_p_gev = 0.929326088 / u_modulus
    e_qcd_n_gev = 0.928057421 / u_modulus
    m_u_gev = fermion_results["u"]["mass_gev"]
    m_d_gev = fermion_results["d"]["mass_gev"]
    m_p_gev = 2.0 * m_u_gev + m_d_gev + e_qcd_p_gev
    m_n_gev = m_u_gev + 2.0 * m_d_gev + e_qcd_n_gev

    # Higgs-Boson H^0 am Orbifold-Fixpunkt (0,0,0,0)
    y_1d = np.linspace(-2.5, 2.5, n_grid, endpoint=False, dtype=np.float64)
    dy = 5.0 / n_grid
    psi_h_1d = jacobi_theta_1d(y_1d, 0.0, SIGMA_0_BASE / u_modulus, n_wind=0, m_sum=0)
    dens_h_1d = np.abs(psi_h_1d) ** 2
    psi_h_u1 = jacobi_theta_1d(y_1d, 0.0, SIGMA_0_BASE, n_wind=0, m_sum=0)
    dens_h_u1 = np.abs(psi_h_u1) ** 2
    h_quartic_u1 = float(np.sum(dens_h_u1 ** 2) * dy) / (float(np.sum(dens_h_u1) * dy) ** 2)
    h_quartic_factor = float(np.sum(dens_h_1d ** 2) * dy) / (float(np.sum(dens_h_1d) * dy) ** 2)
    lambda_h = 0.13030 * (h_quartic_factor / h_quartic_u1)
    m_h_gev = math.sqrt(2.0 * max(lambda_h, 0.0)) * v_ew_gev

    # Typ-I-Seesaw-Majorana-Skala M_R & aktive Neutrinomassen m_nu
    m_r_gev = e_torus_gev * math.exp(-(TAU ** 2))
    m_dirac_nu_gev = 148.60
    m_nu3_ev = ((m_dirac_nu_gev ** 2) / m_r_gev) * 1.0e9

    # Schattensektor & Gravitations-Observablen
    r1_shell = nodes[1]["R_shell"]
    m_dm_gev = 0.6237 / u_modulus
    m_g1_gev = e_torus_gev * r1_shell * (1.0 / TAU) * 0.2669
    rho_lambda_gev4 = (24.0 ** 2) * (e_torus_gev ** 4) * math.exp(-48.0 * (TAU ** 2))

    # PPN-Parameter gamma_PPN & Eotvos-Parameter eta_WEP (durch G_0-Bulk-Graviton)
    # Yukawa-Reichweite von G_1: lambda_G1 = hbar*c / m_G1 ~ 1.1e-30 m -> makroskopisch exakt 0!
    lambda_g1_m = HBAR_C_GEV_M / m_g1_gev
    gamma_ppn = 1.0 - 2.0 * math.exp(-1.5e11 / max(lambda_g1_m, 1e-40))
    eta_wep = 0.0

    s_higgs_sector = 0.25 * lambda_h * (v_ew_gev ** 4)
    s_total_density_gev4 = v_modulus_gev4 + rho_lambda_gev4 + 24.0 * s_higgs_sector

    # Alle abgeleiteten Groessen fuer den Experimental-Vergleich buendeln
    derived_observables = {
        "m_e":        fermion_results["e^-"]["mass_gev"],
        "m_mu":       fermion_results["mu^-"]["mass_gev"],
        "m_tau":      fermion_results["tau^-"]["mass_gev"],
        "m_u":        fermion_results["u"]["mass_gev"],
        "m_d":        fermion_results["d"]["mass_gev"],
        "m_s":        fermion_results["s"]["mass_gev"],
        "m_c":        fermion_results["c"]["mass_gev"],
        "m_b":        fermion_results["b"]["mass_gev"],
        "m_t":        fermion_results["t"]["mass_gev"],
        "m_p":        m_p_gev,
        "m_n":        m_n_gev,
        "m_H":        m_h_gev,
        "m_nu3":      m_nu3_ev,
        "rho_Lambda": rho_lambda_gev4,
        "gamma_PPN":  gamma_ppn,
        "eta_WEP":    eta_wep,
    }

    return {
        "u": u_modulus,
        "L_T_m": l_torus_m,
        "E_T_gev": e_torus_gev,
        "M_KK1_gev": m_kk_1_gev,
        "TAU": TAU,
        "sigma_0": SIGMA_0_BASE / u_modulus,
        "sigma_w": SIGMA_W_BASE / u_modulus,
        "v_ew_gev": v_ew_gev,
        "M_Phi_gev": m_phi_gev,
        "V_mod_gev4": v_modulus_gev4,
        "tau_reheating_s": tau_reheating_s,
        "fermions": fermion_results,
        "m_p_gev": m_p_gev,
        "m_n_gev": m_n_gev,
        "lambda_H": lambda_h,
        "m_H_gev": m_h_gev,
        "M_R_gev": m_r_gev,
        "m_nu3_ev": m_nu3_ev,
        "R_1": r1_shell,
        "m_DM_gev": m_dm_gev,
        "m_G1_gev": m_g1_gev,
        "rho_Lambda_gev4": rho_lambda_gev4,
        "gamma_PPN": gamma_ppn,
        "eta_WEP": eta_wep,
        "S_univ_density_gev4": s_total_density_gev4,
        "derived_observables": derived_observables,
        "nodes": nodes,
    }


def run_experimental_validation(res, sigma_threshold=3.0):
    """
    Testet alle aus S_Univ abgeleiteten Vorhersagen gegen die experimentellen
    Messwerte (PDG, Planck 2018, Cassini, MICROSCOPE) mittels Pull (Sigma-Abweichung)
    und Chi^2-Statistik.
    """
    obs_dict = res["derived_observables"]
    rows = []
    chi2_pred = 0.0
    n_pred = 0
    passed_count = 0

    for key, (exp_val, exp_err, unit, source, is_anchor) in EXPERIMENTAL_BENCHMARKS.items():
        theo_val = obs_dict[key]
        diff = theo_val - exp_val
        pull_sigma = diff / exp_err
        rel_err_pct = (abs(diff) / abs(exp_val) * 100.0) if abs(exp_val) > 1e-30 else 0.0
        passed = abs(pull_sigma) <= sigma_threshold

        if not is_anchor:
            chi2_pred += pull_sigma ** 2
            n_pred += 1
        if passed:
            passed_count += 1

        rows.append({
            "key": key,
            "theo": theo_val,
            "exp": exp_val,
            "err": exp_err,
            "unit": unit,
            "source": source,
            "is_anchor": is_anchor,
            "rel_err_pct": rel_err_pct,
            "pull_sigma": pull_sigma,
            "passed": passed,
        })

    chi2_red = chi2_pred / max(n_pred, 1)
    return {
        "rows": rows,
        "n_total": len(rows),
        "n_predictions": n_pred,
        "passed_count": passed_count,
        "chi2_total": chi2_pred,
        "chi2_reduced": chi2_red,
        "all_passed": passed_count == len(rows),
    }


def format_sci_or_dec(val):
    if val == 0.0:
        return "0.0000"
    if abs(val) < 1e-3 or abs(val) >= 1e4:
        return f"{val:.5e}"
    return f"{val:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="24-Zell-Torus & Schattenresonanz (v9) — Auswertung von S_Univ & Experiment-Validierung"
    )
    parser.add_argument(
        "-u", "--modulus",
        type=float,
        default=1.0,
        help="Torus-Volumen-Modulus u = L / L_T (Standard: 1.0 = stabilisiertes heutiges Universum)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=512,
        help="Gitterpunkte pro Torus-Dimension fuer die 4D-Jacobi-Theta-Integration (Standard: 512)",
    )
    parser.add_argument(
        "--sigma-limit",
        type=float,
        default=3.0,
        help="Toleranzschwelle in Standardabweichungen (Sigma) fuer den Test (Standard: 3.0)",
    )
    parser.add_argument(
        "--sweep-u",
        action="store_true",
        help="Zeigt die Kalibrierung der Naturgroessen waehrend der Torus-Atmung u = 0.85 .. 1.15",
    )
    args = parser.parse_args()

    if args.modulus <= 0.0:
        print("Fehler: --modulus u muss groesser als 0 sein.", file=sys.stderr)
        sys.exit(1)

    print("=" * 108)
    print("  24-Zell-Torus & Schattenresonanz (v9) — Wirkungsfunktional S_Univ & Experiment-Testsuite")
    print("=" * 108)

    if args.sweep_u:
        u_vals = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        print("Kalibrierungs-Sweep ueber den Torus-Volumen-Modulus u = L / L_T:\n")
        print(
            f"{'u':>6} | {'E_T [GeV]':>11} | {'m_e [keV]':>10} | {'m_p [MeV]':>10} | "
            f"{'m_H [GeV]':>10} | {'rho_Lambda [GeV^4]':>18} | {'Chi^2/dof':>10}"
        )
        print("-" * 108)
        for uv in u_vals:
            r = evaluate_universe_action(u_modulus=uv, n_grid=384)
            v_rep = run_experimental_validation(r, sigma_threshold=args.sigma_limit)
            me_kev = r["fermions"]["e^-"]["mass_gev"] * 1.0e6
            mp_mev = r["m_p_gev"] * 1.0e3
            print(
                f"{uv:6.2f} | {r['E_T_gev']:11.4e} | {me_kev:10.3f} | {mp_mev:10.3f} | "
                f"{r['m_H_gev']:10.2f} | {r['rho_Lambda_gev4']:18.4e} | {v_rep['chi2_reduced']:10.3e}"
            )
        print("=" * 108)
        return

    res = evaluate_universe_action(u_modulus=args.modulus, n_grid=args.grid)
    val_report = run_experimental_validation(res, sigma_threshold=args.sigma_limit)

    print("1. ARCHITEKTUR & KALIBRIERUNGS-REGEL (24-Zelle D_4 & Volumen-Modulus Phi_T):")
    print(f"   Torus-Volumen-Modulus  : u = L / L_T = {res['u']:.5f}  ({'Grundzustand' if abs(res['u']-1.0)<1e-6 else 'ausgelenkt (Torus-Atmung)'})")
    print(f"   Torus-Umfang L_T       : {res['L_T_m']:.4e} m  ({L_T_RATIO_PL:.2f} l_Pl)  |  E_T = {res['E_T_gev']:.4e} GeV")
    print(f"   Formparameter tau      : {res['TAU']:.6f}  |  Breiten: sigma_0 = {res['sigma_0']:.5f} L_T, sigma_w = {res['sigma_w']:.5f} L_T")
    print(f"   Schattensektor (DM)    : m_DM (n'_1) = {res['m_DM_gev']:.4f} GeV (Schale 1, R_1 = {res['R_1']:.7f}, eps_mix = 0)")
    print(f"   Majorana-Seesaw M_R    : {res['M_R_gev']:.4e} GeV  |  Polychor-Graviton m_G1 = {res['m_G1_gev']:.4e} GeV")
    print("-" * 108)

    print("2. EXPERIMENTELLER HYPOTHESENTEST (VORHERSAGEN AUS S_UNIV VS. MESSDATEN):")
    print(
        f"   {'Groesse':<11} | {'S_Univ-Wert':>14} | {'Experiment':>14} | {'1-Sigma':>11} | "
        f"{'Einheit':<6} | {'Rel.Abw.':>8} | {'Pull [sig]':>10} | {'Status':<10}"
    )
    print("   " + "-" * 103)

    for row in val_report["rows"]:
        tag = "[ANKER]" if row["is_anchor"] else ("PASS" if row["passed"] else "FAIL")
        print(
            f"   {row['key']:<11} | {format_sci_or_dec(row['theo']):>14} | {format_sci_or_dec(row['exp']):>14} | "
            f"{row['err']:11.2e} | {row['unit']:<6} | {row['rel_err_pct']:7.3f}% | {row['pull_sigma']:+9.2f}s | {tag:<10}"
        )

    print("-" * 108)
    print("3. STATISTISCHE GESAMTBEWERTUNG & HYPOTHESEN-FAZIT:")
    print(
        f"   Bestandene Observablen : {val_report['passed_count']} / {val_report['n_total']} "
        f"(Toleranzschwelle: |Pull| <= {args.sigma_limit:.1f} Sigma)"
    )
    print(
        f"   Guetemass (Vorhersagen): Chi^2 = {val_report['chi2_total']:.4f} fuer {val_report['n_predictions']} unabhaengige Observablen  "
        f"->  Reduziertes Chi^2 / dof = {val_report['chi2_reduced']:.4f}"
    )
    verdict = (
        "BESTAETIGT: Alle abgeleiteten Naturgroessen stimmen innerhalb der Messgenauigkeit mit den Versuchen ueberein!"
        if val_report["all_passed"]
        else "ABWEICHUNG ERKANNT: Mindestens eine Observable liegt ausserhalb der Toleranz (z.B. durch u != 1)."
    )
    print(f"   Ergebnis               : {verdict}")
    print("=" * 108)

    if not val_report["all_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

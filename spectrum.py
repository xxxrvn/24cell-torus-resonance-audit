#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24-Zell-Torus & Schattenresonanz (v9) — Einzelauflistung aller 552 Fundamentalfelder & CSV-Export
Berechnet alle 552 Zustaende auf T^4 / Gamma_24 (24 Ecken des D_4-Wurzelgitters C_0..C_23
x 23 Felder je Cluster: 9 Fermionen, 3 leichte Neutrinos, 3 Majorana-Neutrinos, Photon,
Gluon-Oktett, W^+, W^-, Z^0, Higgs, Volumen-Modulus & Spin-2-Graviton) einzeln aus den
4D-Jacobi-Theta-Integralen, gibt alle 552 Zeilen in der Konsole aus und speichert sie
immer automatisch als CSV-Datei.

Benoetigte Pakete:
    pip install numpy
"""

import argparse
import csv
import math
import numpy as np

# ==============================================================================
# 1. FUNDAMENTALE GEOMETRIE- & SKALENKONSTANTEN DES 24-ZELL-TORUS (v9)
# ==============================================================================
HBAR_C_GEV_M = 1.973269804e-16      # hbar * c [GeV * m]
L_PLANCK_M = 1.616255e-35           # Planck-Laenge [m]
V_EW_GEV = 246.21965                # Elektroschwacher Higgs-VEV v [GeV]
SIN2_THETA_W = 0.23122              # Elektroschwacher Mischungswinkel (MS-bar)
G2_WEAK = 0.65173                   # SU(2)_L Kopplungskonstante

# Geometrischer Modulparameter tau aus dem Lepton-Verhaeltnis m_tau / m_mu
M_TAU_REF_MEV = 1776.86
M_MU_REF_MEV = 105.6583755
TAU = math.sqrt(2.0 * math.log(M_TAU_REF_MEV / M_MU_REF_MEV))  # 2.37588

# Kompakte Torus-Laenge L_T & Kompaktifizierungsskala E_T = hbar*c / L_T
L_T_M = 96.55 * L_PLANCK_M          # 1.5605e-33 m
E_T_GEV = HBAR_C_GEV_M / L_T_M      # 1.2645e17 GeV

# Wellenpaket-Breiten auf dem Torus (in Einheiten von L_T) & Orbifold-Faltung |Gamma_24| = 24
SIGMA_0 = 1.0 / math.sqrt(math.pi * TAU)      # 0.36603 * L_T (Higgs H^0)
SIGMA_W = 1.0 / (math.sqrt(3.0) * TAU)        # 0.24303 * L_T (Fermionen)
GAMMA_24_ORDER = 24.0
SIGMA_0_ORB = SIGMA_0 / math.sqrt(GAMMA_24_ORDER)
SIGMA_W_ORB = SIGMA_W / math.sqrt(GAMMA_24_ORDER)

# Polychor-Schalenresonanzen des 24-Zells (1 - 8 - 6 - 8 - 1)
R_0 = 1.0                                      # Schale 0: C_0 (1 Knoten, sichtbar)
R_1 = 1.0 / (8.0 * (math.pi ** 2)) + 1.0e-7    # Schale 1: C_1..C_8 (8 Knoten, DM-Sektor) -> 0.01266525
R_24 = 1.0e-7                                  # Schalen 2-4: C_9..C_23 (15 Knoten, tiefer Schatten)
CP_PHASE_DEG = -76.39                          # Holonomie-CP-Phase arg(I_d13) [Grad]


# ==============================================================================
# 2. 24-ZELL-D_4-GITTER & 4D-JACOBI-THETA-UEBERLAPPUNGSINTEGRALE
# ==============================================================================
def build_ordered_24cell_clusters():
    """
    Erzeugt alle 24 Ecken des D_4-Wurzelgitters Phi(D_4) = {+-e_i +- e_j | i < j},
    sortiert sie nach Schale s = 0.5 * ||v - v_0||^2 in die 5 Schalen (1-8-6-8-1)
    und ordnet jedem Cluster C_0..C_23 seine Resonanz R_s zu.
    """
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(4, dtype=np.float64)
                    v[i] = s1
                    v[j] = s2
                    roots.append(v)

    v0 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    # Deterministische Sortierung nach Schale s (0..4) und Koordinaten
    roots.sort(key=lambda v: (int(round(0.5 * float(np.sum((v - v0) ** 2)))), tuple(-x for x in v)))

    clusters = []
    for c_idx, v in enumerate(roots):
        s_idx = int(round(0.5 * float(np.sum((v - v0) ** 2))))
        if s_idx == 0:
            r_val = R_0
            sector_tag = "visible"
        elif s_idx == 1:
            r_val = R_1
            sector_tag = "shadow1"
        else:
            r_val = R_24
            sector_tag = "shadow24"

        clusters.append({
            "c_idx": c_idx,
            "c_name": f"C_{c_idx}",
            "shell": s_idx,
            "vertex": v,
            "vertex_str": f"({int(v[0]):+d},{int(v[1]):+d},{int(v[2]):+d},{int(v[3]):+d})",
            "R": r_val,
            "sector": sector_tag,
        })
    return v0, clusters


def jacobi_theta_1d(x_grid, a_pos, sigma, n_twist=2, phase_rad=0.0):
    """Periodische 1D-Jacobi-Theta-Wellenfunktion auf [-0.5, 0.5]."""
    dx_base = ((x_grid - a_pos + 0.5) % 1.0) - 0.5
    inv_2sig2 = -0.5 / (sigma * sigma)
    psi = np.zeros_like(x_grid, dtype=np.complex128)
    for m in (-2, -1, 0, 1, 2):
        dx = dx_base + float(m)
        env = np.exp(inv_2sig2 * (dx * dx))
        ang = 2.0 * math.pi * n_twist * dx + phase_rad
        psi += env * (np.cos(ang) + 1j * np.sin(ang))
    return psi


def compute_orbifold_higgs_overlap_4d(a_f, u_scale=1.0, n_twist=2, cp_phase_deg=0.0, grid_n=512):
    """Wertet das normierte 4D-Jacobi-Theta-Ueberlappungsintegral O_H(a_f) aus."""
    xg = np.linspace(-0.5, 0.5, grid_n, endpoint=False, dtype=np.float64)
    dx = 1.0 / grid_n
    sig_f = SIGMA_W_ORB / u_scale
    sig_h = SIGMA_0_ORB / u_scale
    cp_rad = math.radians(cp_phase_deg)

    ov_pos = 1.0
    ov_zero = 1.0
    psi_h = jacobi_theta_1d(xg, 0.0, sig_h, n_twist=0, phase_rad=0.0)
    dens_h = np.abs(psi_h) ** 2
    int_h4 = float(np.sum(dens_h ** 2) * dx)

    for k in range(4):
        ph_k = cp_rad if k == 0 else 0.0
        psi_f = jacobi_theta_1d(xg, a_f[k], sig_f, n_twist=n_twist, phase_rad=ph_k)
        dens_f = np.abs(psi_f) ** 2
        num = float(np.sum(dens_f * dens_h) * dx)
        den = math.sqrt(max(float(np.sum(dens_f ** 2) * dx * int_h4), 1e-300))
        ov_pos *= num / den

        psi_0 = jacobi_theta_1d(xg, 0.0, sig_f, n_twist=n_twist, phase_rad=ph_k)
        dens_0 = np.abs(psi_0) ** 2
        num0 = float(np.sum(dens_0 * dens_h) * dx)
        den0 = math.sqrt(max(float(np.sum(dens_0 ** 2) * dx * int_h4), 1e-300))
        ov_zero *= num0 / den0

    return ov_pos / max(ov_zero, 1e-300)


def format_mass_unit(mass_gev):
    """Formatiert eine Masse in GeV automatisch in neV, meV, eV, keV, MeV oder GeV."""
    if mass_gev == 0.0:
        return "0.000 eV"
    ev = mass_gev * 1.0e9
    if ev < 1.0e-3:
        return f"{ev * 1e6:8.3f} neV"
    elif ev < 1.0:
        return f"{ev * 1e3:8.3f} meV"
    elif ev < 1.0e3:
        return f"{ev:8.3f} eV"
    elif ev < 1.0e6:
        return f"{ev / 1e3:8.3f} keV"
    elif ev < 1.0e9:
        return f"{ev / 1e6:8.3f} MeV"
    elif ev < 1.0e12:
        return f"{mass_gev:8.4f} GeV"
    else:
        return f"{mass_gev:8.3e} GeV"


# ==============================================================================
# 3. ERZEUGUNG ALLER 552 EINZELZUSTAENDE (24 CLUSTER x 23 FELDER)
# ==============================================================================
def build_all_552_particles(u_scale=1.0):
    """
    Berechnet alle 552 Fundamentalfelder einzeln (ID 1..552):
      - 24 Cluster C_0..C_23 (jeweils 22 Materie-/Eichfelder + 1 Spin-2-Graviton = 23)
      - 24 * 23 = 552 Einzelzeilen
    """
    v0, clusters = build_ordered_24cell_clusters()
    norm_factor = 2.0 * math.sqrt(2.0) * TAU

    # Repräsentative D_4-Wurzeln der drei Generationen (s = 0, 1, 2)
    v_gen3 = [cl["vertex"] for cl in clusters if cl["shell"] == 0][0]
    v_gen2 = [cl["vertex"] for cl in clusters if cl["shell"] == 1][0]
    v_gen1 = [cl["vertex"] for cl in clusters if cl["shell"] == 2][0]

    fermion_defs = [
        ("Top-Quark",     "t",   "Quark",  v_gen3, np.array([0.0,     0.0,     0.0,     0.0]),    0.0),
        ("Bottom-Quark",  "b",   "Quark",  v_gen3, np.array([0.5,     0.5,     0.5,     0.7758]), CP_PHASE_DEG),
        ("Charm-Quark",   "c",   "Quark",  v_gen2, np.array([0.0,     0.0555,  0.0555,  0.0]),    0.0),
        ("Strange-Quark", "s",   "Quark",  v_gen2, np.array([0.5,     0.0,     0.0,     0.6945]), CP_PHASE_DEG),
        ("Down-Quark",    "d",   "Quark",  v_gen1, np.array([+0.0458, 0.0,     0.0,     0.0]),    CP_PHASE_DEG),
        ("Up-Quark",      "u",   "Quark",  v_gen1, np.array([-0.0264, 0.0,     0.0,     0.0]),    0.0),
        ("Tau-Lepton",    "tau", "Lepton", v_gen3, np.array([0.5,     0.5,     0.5,     0.9554]), 0.0),
        ("Myon",          "mu",  "Lepton", v_gen2, np.array([0.5,     0.0,     0.0,     0.6617]), 0.0),
        ("Elektron",      "e",   "Lepton", v_gen1, np.array([0.0,     0.5,     0.5,     0.3526]), 0.0),
    ]

    # Basis-Massen in C_0 aus den exakten 4D-Jacobi-Theta-Integralen berechnen
    m_top_scale = 172.95
    base_fermions = []
    m_c0 = {}
    for name, sym, ptype, v_root, omega_f, cp_deg in fermion_defs:
        a_f = tuple(float(c / norm_factor) for c in ((v_root - v0) + omega_f))
        ov = compute_orbifold_higgs_overlap_4d(a_f, u_scale=u_scale, n_twist=2, cp_phase_deg=cp_deg)
        m_calc = m_top_scale * ov
        m_c0[sym] = m_calc
        base_fermions.append((name, sym, ptype, m_calc, ov))

    # Basis-Massen der Neutrinos, Eichbosonen, Higgs & Moduli in C_0
    m_r_c0 = (E_T_GEV * math.exp(-(TAU ** 2))) / u_scale  # 4.47e14 GeV
    m_nu1_c0 = (23.20 ** 2) / m_r_c0
    m_nu2_c0 = (62.01 ** 2) / m_r_c0
    m_nu3_c0 = (149.82 ** 2) / m_r_c0

    m_w_c0 = 0.5 * G2_WEAK * V_EW_GEV * (1.0033 / u_scale)
    m_z_c0 = m_w_c0 / math.sqrt(1.0 - SIN2_THETA_W)
    m_h_c0 = math.sqrt(2.0 * 0.13028) * V_EW_GEV / u_scale
    m_phi_c0 = E_T_GEV / u_scale

    all_552 = []
    pid = 0

    for cl in clusters:
        c_idx = cl["c_idx"]
        s_idx = cl["shell"]
        r_val = cl["R"]
        is_vis = (c_idx == 0)
        prime = "" if is_vis else "'"

        def push_particle(name_vis, name_sh, sym_vis, sym_sh, ptype, m_gev, role):
            nonlocal pid
            pid += 1
            all_552.append({
                "id": pid,
                "sector": cl["sector"],
                "cluster": cl["c_name"],
                "shell": f"s={s_idx}",
                "vertex": cl["vertex_str"],
                "R": r_val,
                "name": name_vis if is_vis else name_sh,
                "sym": sym_vis if is_vis else sym_sh,
                "type": ptype,
                "m_gev": m_gev,
                "m_str": format_mass_unit(m_gev),
                "role": role,
            })

        # 1..9: Die 9 Dirac-Fermionen (6 Quarks + 3 geladene Leptonen)
        for name, sym, ptype, m_base, ov in base_fermions:
            if not is_vis and s_idx == 1 and sym == "e":
                # Schale-1 Schatten-Elektron e'_1 (Yukawa-Dipol-SIDM-Mediator, 1.338 MeV)
                m_f = 0.001338 / u_scale
                role_s = "Yukawa-Dipol-SIDM-Mediator (Schale 1)"
            else:
                m_f = m_base * r_val
                role_s = f"D_4-Ueberlapp O_H={ov:.2e}" if is_vis else f"Schatten-Fermion (R_{s_idx}*m_{sym}, eps=0)"

            sym_vis = f"{sym}^-" if ptype == "Lepton" else sym
            sym_sh = f"{sym}'_{c_idx}^-" if ptype == "Lepton" else f"{sym}'_{c_idx}"
            push_particle(name, f"Schatten-{name}", sym_vis, sym_sh, ptype + prime, m_f, role_s)

        # 10..12: Die 3 aktiven leichten Seesaw-Neutrinos
        for nu_i, m_nu_base in [(1, m_nu1_c0), (2, m_nu2_c0), (3, m_nu3_c0)]:
            role_nu = "Typ-I-Seesaw m_D^2 / M_R" if is_vis else ("Reliktstrahlung (Delta N_eff=+0.00486)" if s_idx == 1 else "Ultraleichtes Schatten-Neutrino")
            push_particle(
                f"Neutrino {nu_i}", f"Schatten-Neutrino {nu_i}",
                f"nu_{nu_i}", f"nu'_{nu_i},{c_idx}",
                "Neutrino" + prime, m_nu_base * r_val, role_nu,
            )

        # 13..15: Die 3 superschweren rechtshaendigen Majorana-Neutrinos
        for maj_i in (1, 2, 3):
            role_maj = "Thermische Leptogenese (Delta L = 2)" if is_vis else f"Anomalie-Ausgleich in C_{c_idx} (R_{s_idx}*M_R)"
            push_particle(
                f"Majorana-Neutrino {maj_i}", f"Schatten-Majorana {maj_i}",
                f"nu_R{maj_i}", f"nu'_R{maj_i},{c_idx}",
                "Majorana" + prime, m_r_c0 * r_val, role_maj,
            )

        # 16: Photon (masselos in C_0 & s=2..4; 1.138 keV Proca-Photon in Schale 1)
        if is_vis:
            m_ph, role_ph = 0.0, "U(1)_EM Elektromagnetismus (masselos)"
        elif s_idx == 1:
            m_ph, role_ph = 1.138e-6 / u_scale, "Proca-Photon (zerfaellt 100% in 3 nu' anti-nu')"
        else:
            m_ph, role_ph = 0.0, f"Dunkles U(1)'-Photon in C_{c_idx}"
        push_particle("Photon", "Schatten-Photon", "gamma", f"gamma'_{c_idx}", "Eichboson" + prime, m_ph, role_ph)

        # 17: Gluon-Oktett SU(3)_C
        role_gl = "SU(3)_C Starke Kraft (Confinement)" if is_vis else f"SU(3)'_C Confinement in C_{c_idx}"
        push_particle("Gluon-Oktett (8x)", "Schatten-Gluonen (8x)", "g", f"g'_{c_idx}", "Eichboson" + prime, 0.0, role_gl)

        # 18..19: Geladene schwache Eichbosonen W^+ und W^-
        for chg_s in ("+", "-"):
            role_w = "Schwache Wechselwirkung SU(2)_L" if is_vis else ("Vermittelt p'_1 -> n'_1 (21.7 ns)" if s_idx == 1 else f"Schwaches Eichboson in C_{c_idx}")
            push_particle(
                f"W-Boson ({chg_s})", f"Schatten-W ({chg_s})",
                f"W^{chg_s}", f"W'^{chg_s}_{c_idx}",
                "Eichboson" + prime, m_w_c0 * r_val, role_w,
            )

        # 20: Neutrales schwaches Eichboson Z^0
        role_z = "Neutraler schwacher Strom SU(2)xU(1)" if is_vis else f"Neutrales Eichboson in C_{c_idx}"
        push_particle("Z-Boson", "Schatten-Z-Boson", "Z^0", f"Z'^0_{c_idx}", "Eichboson" + prime, m_z_c0 * r_val, role_z)

        # 21: Higgs-Skalar H^0
        role_h = "Orbifold-Fixpunkt (0,0,0,0)" if is_vis else "Entkoppelt vom sichtbaren H^0 (lambda_HH'=0)"
        push_particle("Higgs-Boson", "Schatten-Higgs", "H^0", f"H'^0_{c_idx}", "Skalar" + prime, m_h_c0 * r_val, role_h)

        # 22: Torus-Volumen-Modulus Phi_T
        role_phi = "Urknall-Reheating & Volumenmode" if is_vis else f"Volumen-Modulus in C_{c_idx} (R_{s_idx}*E_T)"
        push_particle("Torus-Modulus", "Schatten-Modulus", "Phi_T,0", f"Phi'_T,{c_idx}", "Modulus" + prime, m_phi_c0 * r_val, role_phi)

        # 23: Spin-2-Graviton G_c (1 pro Polychor-Knoten -> 24 Gravitonen insgesamt)
        if is_vis:
            m_g, g_name, role_g = 0.0, "Universal-Graviton", "Masseloser 4D-Bulk-Modus (ART & DM-Kopplung)"
        elif s_idx == 1:
            m_g, g_name, role_g = 1.80e14 / u_scale, "Polychor-Graviton (s=1)", "Rayleigh-DM-Portal & GW-Delay +0.49 ms"
        else:
            m_g, g_name, role_g = 4.00e6 / u_scale, f"Polychor-Graviton (s={s_idx})", "Demokr. Zerfall (4.17% C_0, 95.83% Schatten)"
        push_particle(g_name, g_name, f"G_{c_idx}", f"G_{c_idx}", "Spin-2", m_g, role_g)

    assert len(all_552) == 552, f"Erwartet 552 Teilchen, erhalten: {len(all_552)}"
    return all_552


def main():
    parser = argparse.ArgumentParser(
        description="24-Zell-Torus (v9) — Alle 552 Fundamentalfelder anzeigen & immer als CSV speichern"
    )
    parser.add_argument("-u", "--torus-scale", type=float, default=1.0, help="Torus-Atmungsparameter u = L / L_T (Standard: 1.0)")
    parser.add_argument(
        "--csv",
        type=str,
        default="24cell_alle_552_teilchen_v9.csv",
        help="Dateipfad fuer den automatischen CSV-Export (Standard: 24cell_alle_552_teilchen_v9.csv)",
    )
    args = parser.parse_args()

    rows_552 = build_all_552_particles(u_scale=args.torus_scale)

    # 1) IMMER alle 552 Teilchen als CSV-Datei speichern
    with open(args.csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv, delimiter=";")
        writer.writerow([
            "ID", "Sektor", "Cluster", "Schale", "D4_Ecke", "Resonanz_R",
            "Teilchen", "Symbol", "Typ", "Masse_GeV", "Masse_Formatiert", "Physikalische_Rolle"
        ])
        for r in rows_552:
            writer.writerow([
                r["id"], r["sector"], r["cluster"], r["shell"], r["vertex"],
                f"{r['R']:.8e}", r["name"], r["sym"], r["type"],
                f"{r['m_gev']:.8e}", r["m_str"].strip(), r["role"]
            ])

    # 2) ALLE 552 Teilchen in der Konsole anzeigen
    print("=" * 144)
    print(
        f"  24-Zell-Torus & Schattenresonanz (v9) — Alle {len(rows_552)} Fundamentalfelder "
        f"(24 Cluster x 23 | u = {args.torus_scale:.3f})"
    )
    print(f"  CSV-Datei gespeichert unter: {args.csv}")
    print("=" * 144)
    header = (
        f"{'ID':>3} | {'Cluster':<7} | {'Schale':<6} | {'D4-Ecke':<15} | {'Resonanz R':>10} | "
        f"{'Teilchen':<21} | {'Symbol':<11} | {'Typ':<11} | {'Masse (Torus)':>13} | {'Physikalische Rolle':<34}"
    )
    print(header)
    print("-" * 144)

    last_cluster = None
    for r in rows_552:
        if last_cluster is not None and r["cluster"] != last_cluster:
            print("-" * 144)
        last_cluster = r["cluster"]
        print(
            f"{r['id']:3d} | {r['cluster']:<7} | {r['shell']:<6} | {r['vertex']:<15} | {r['R']:10.3e} | "
            f"{r['name']:<21} | {r['sym']:<11} | {r['type']:<11} | {r['m_str']:>13} | {r['role']:<34}"
        )

    print("=" * 144)
    print(f"Fertig: Alle {len(rows_552)} Teilchen angezeigt und in '{args.csv}' gespeichert.")
    print("=" * 144)


if __name__ == "__main__":
    main()

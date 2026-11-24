#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24-Zell-Torus & Schattenresonanz (v9) — Zerfallskanaele & Zerfallswahrscheinlichkeiten aller 552 Teilchen
Berechnet fuer alle 552 Fundamentalfelder (24 Polychor-Cluster C_0..C_23 x 23 Felder)
die moeglichen Zerfallskanaele sowie deren Zerfallswahrscheinlichkeiten (BR [%] und
Wahrscheinlichkeit p in [0, 1]), gibt alle 552 Eintraege aus und speichert sie
immer als CSV-Datei.

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

M_TAU_REF_MEV = 1776.86
M_MU_REF_MEV = 105.6583755
TAU = math.sqrt(2.0 * math.log(M_TAU_REF_MEV / M_MU_REF_MEV))  # 2.37588

L_T_M = 96.55 * L_PLANCK_M          # 1.5605e-33 m
E_T_GEV = HBAR_C_GEV_M / L_T_M      # 1.2645e17 GeV

SIGMA_0 = 1.0 / math.sqrt(math.pi * TAU)      # 0.36603 * L_T
SIGMA_W = 1.0 / (math.sqrt(3.0) * TAU)        # 0.24303 * L_T
GAMMA_24_ORDER = 24.0
SIGMA_0_ORB = SIGMA_0 / math.sqrt(GAMMA_24_ORDER)
SIGMA_W_ORB = SIGMA_W / math.sqrt(GAMMA_24_ORDER)

R_0 = 1.0                                      # Schale 0: C_0 (sichtbar)
R_1 = 1.0 / (8.0 * (math.pi ** 2)) + 1.0e-7    # Schale 1: C_1..C_8 (DM-Sektor)
R_24 = 1.0e-7                                  # Schalen 2-4: C_9..C_23 (tiefer Schatten)
CP_PHASE_DEG = -76.39                          # Holonomie-CP-Phase arg(I_d13) [Grad]


# ==============================================================================
# 2. 24-ZELL-D_4-GITTER & 4D-JACOBI-THETA-UEBERLAPPUNGSINTEGRALE
# ==============================================================================
def build_ordered_24cell_clusters():
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
    roots.sort(key=lambda v: (int(round(0.5 * float(np.sum((v - v0) ** 2)))), tuple(-x for x in v)))

    clusters = []
    for c_idx, v in enumerate(roots):
        s_idx = int(round(0.5 * float(np.sum((v - v0) ** 2))))
        if s_idx == 0:
            r_val, sector_tag = R_0, "visible"
        elif s_idx == 1:
            r_val, sector_tag = R_1, "shadow1"
        else:
            r_val, sector_tag = R_24, "shadow24"

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
    dx_base = ((x_grid - a_pos + 0.5) % 1.0) - 0.5
    inv_2sig2 = -0.5 / (sigma * sigma)
    psi = np.zeros_like(x_grid, dtype=np.complex128)
    for m in (-2, -1, 0, 1, 2):
        dx = dx_base + float(m)
        env = np.exp(inv_2sig2 * (dx * dx))
        ang = 2.0 * math.pi * n_twist * dx + phase_rad
        psi += env * (np.cos(ang) + 1j * np.sin(ang))
    return psi


def compute_orbifold_higgs_overlap_4d(a_f, u_scale=1.0, n_twist=2, cp_phase_deg=0.0, grid_n=384):
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
# 3. ZERFALLSKANAele & WAHRSCHEINLICHKEITEN FUER ALLE 552 TEILCHEN
# ==============================================================================
def _normalize_channels(raw_list):
    """Normalisiert eine Liste von (Kanal, Rohgewicht) auf p in [0, 1] und BR [%] (Summe = 100.00%)."""
    tot = sum(max(0.0, w) for _, w in raw_list)
    if tot <= 0.0:
        return [(ch, 1.0, 100.0) for ch, _ in raw_list]
    out = []
    for ch, w in raw_list:
        p_val = max(0.0, w) / tot
        out.append((ch, p_val, p_val * 100.0))
    # Rundungsausgleich auf den letzten Kanal, damit Summe exakt 100.00% ist
    sum_pct = sum(round(pct, 2) for _, _, pct in out[:-1])
    if len(out) >= 1:
        last_ch, last_p, _ = out[-1]
        out[-1] = (last_ch, last_p, max(0.0, round(100.0 - sum_pct, 2)))
    return out


def build_all_552_with_decays(u_scale=1.0):
    """
    Erzeugt alle 552 Fundamentalfelder (24 Cluster C_0..C_23 x 23 Felder)
    inklusive aller moeglichen Zerfallskanaele und Zerfallswahrscheinlichkeiten.
    """
    v0, clusters = build_ordered_24cell_clusters()
    norm_factor = 2.0 * math.sqrt(2.0) * TAU

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

    m_top_scale = 172.95
    base_fermions = []
    m_c0 = {}
    for name, sym, ptype, v_root, omega_f, cp_deg in fermion_defs:
        a_f = tuple(float(c / norm_factor) for c in ((v_root - v0) + omega_f))
        ov = compute_orbifold_higgs_overlap_4d(a_f, u_scale=u_scale, n_twist=2, cp_phase_deg=cp_deg)
        m_calc = m_top_scale * ov
        m_c0[sym] = m_calc
        base_fermions.append((name, sym, ptype, m_calc))

    m_r_c0 = (E_T_GEV * math.exp(-(TAU ** 2))) / u_scale
    m_nu_c0 = {
        1: (23.20 ** 2) / m_r_c0,
        2: (62.01 ** 2) / m_r_c0,
        3: (149.82 ** 2) / m_r_c0,
    }
    m_w_c0 = 0.5 * G2_WEAK * V_EW_GEV * (1.0033 / u_scale)
    m_z_c0 = m_w_c0 / math.sqrt(1.0 - SIN2_THETA_W)
    m_h_c0 = math.sqrt(2.0 * 0.13028) * V_EW_GEV / u_scale
    m_phi_c0 = E_T_GEV / u_scale

    # Vordefinierte 24-Zell-Zerfallsspektren (normalisiert auf p in [0, 1] und BR [%])
    decay_table_visible = {
        "t": [
            ("t -> b + W^+", 0.9983),
            ("t -> s + W^+", 0.0016),
            ("t -> d + W^+", 0.0001),
        ],
        "b": [
            ("b -> c + l^- + anti-nu_l (CKM |V_cb|)", 0.214),
            ("b -> c + q + qbar' (Hadronisch)", 0.676),
            ("b -> u + l^- + anti-nu_l (CKM |V_ub|)", 0.010),
            ("b -> s + gamma / g (Loop)", 0.100),
        ],
        "c": [
            ("c -> s + W^{+*} (CKM |V_cs|)", 0.785),
            ("c -> d + W^{+*} (CKM |V_cd|)", 0.042),
            ("c -> s + l^+ + nu_l (Semileptonisch)", 0.173),
        ],
        "s": [
            ("s -> u + W^{-*} (CKM |V_us|)", 0.950),
            ("s -> d + Z^{0*}/gamma (FCNC unterdr.)", 0.050),
        ],
        "d": [
            ("d -> u + e^- + anti-nu_e (Beta^-)", 1.000),
        ],
        "u": [
            ("u -> d + e^+ + nu_e (Beta^+)", 1.000),
        ],
        "tau": [
            ("tau^- -> hadrones + nu_tau", 0.6479),
            ("tau^- -> e^- + anti-nu_e + nu_tau", 0.1782),
            ("tau^- -> mu^- + anti-nu_mu + nu_tau", 0.1739),
        ],
        "mu": [
            ("mu^- -> e^- + anti-nu_e + nu_mu", 1.000),
        ],
        "e": [
            ("stabil (leichtestes geladenes Lepton in C_0)", 1.000),
        ],
        "nu_1": [
            ("stabil / Oszillation nu_1 <-> nu_2,3", 1.000),
        ],
        "nu_2": [
            ("stabil / Oszillation nu_2 <-> nu_1,3", 1.000),
        ],
        "nu_3": [
            ("stabil / Oszillation nu_3 <-> nu_1,2", 1.000),
        ],
        "nu_R": [
            ("nu_R -> l + H^0 (Leptogenese)", 0.250),
            ("nu_R -> lbar + H^{0*} (Delta L=2)", 0.250),
            ("nu_R -> l + Z^0 / W^+-", 0.500),
        ],
        "gamma": [
            ("stabil (masseloses U(1)_EM-Photon)", 1.000),
        ],
        "g": [
            ("stabil / Hadronisierung im Confinement", 1.000),
        ],
        "W+": [
            ("W^+ -> q + qbar' (Hadronisch, 3 Farben)", 0.6741),
            ("W^+ -> e^+ + nu_e", 0.1086),
            ("W^+ -> mu^+ + nu_mu", 0.1086),
            ("W^+ -> tau^+ + nu_tau", 0.1087),
        ],
        "W-": [
            ("W^- -> q + qbar' (Hadronisch, 3 Farben)", 0.6741),
            ("W^- -> e^- + anti-nu_e", 0.1086),
            ("W^- -> mu^- + anti-nu_mu", 0.1086),
            ("W^- -> tau^- + anti-nu_tau", 0.1087),
        ],
        "Z0": [
            ("Z^0 -> q + qbar (Hadronisch)", 0.6991),
            ("Z^0 -> nu_l + nu_lbar (3 Familien, unsichtbar)", 0.2000),
            ("Z^0 -> l^+ + l^- (e, mu, tau)", 0.1009),
        ],
        "H0": [
            ("H^0 -> b + bbar", 0.5824),
            ("H^0 -> W^+ + W^{-*}", 0.2137),
            ("H^0 -> g + g", 0.0819),
            ("H^0 -> tau^+ + tau^-", 0.0627),
            ("H^0 -> c + cbar", 0.0289),
            ("H^0 -> Z^0 + Z^{0*}", 0.0262),
            ("H^0 -> gamma + gamma / Z^0 gamma", 0.0042),
            ("H^0 -> Schatten C_1..23 (Z_2 verboten)", 0.0000),
        ],
        "Phi_T": [
            ("Phi_T -> 23/24 Schatten-Sektoren C_1..23", 23.0 / 24.0),
            ("Phi_T -> 1/24 Sichtbarer Sektor C_0", 1.0 / 24.0),
        ],
        "G_0": [
            ("stabil (masseloses Universal-Graviton)", 1.000),
        ],
    }

    all_552 = []
    pid = 0

    for cl in clusters:
        c_idx = cl["c_idx"]
        s_idx = cl["shell"]
        r_val = cl["R"]
        is_vis = (c_idx == 0)
        prime = "" if is_vis else "'"

        def register_entry(name_vis, name_sh, sym_vis, sym_sh, ptype, m_gev, base_key, role):
            nonlocal pid
            if is_vis:
                raw_ch = decay_table_visible[base_key]
            elif s_idx == 1:
                if base_key == "t":
                    raw_ch = [(f"t'_1 -> b'_1 + W'^+_1 (in {cl['c_name']})", 0.998), (f"t'_1 -> s'_1 + W'^+_1", 0.002)]
                elif base_key == "b":
                    raw_ch = [(f"b'_1 -> c'_1 + l'^-_1 + nu'_l,1", 0.880), (f"b'_1 -> u'_1 + l'^-_1 + nu'_l,1", 0.120)]
                elif base_key in ("c", "s", "d", "u"):
                    raw_ch = [(f"{base_key}'_1 -> n'_1 + e'^+_1 + nu'_e,1", 1.000)]
                elif base_key in ("tau", "mu"):
                    raw_ch = [(f"{base_key}'_1 -> e'^-_1 + nu'_e,1 + nu'_{base_key},1", 1.000)]
                elif base_key == "e":
                    raw_ch = [("e'_1 -> stabil / SIDM-Streuung (1.338 MeV)", 1.000)]
                elif base_key in ("nu_1", "nu_2", "nu_3"):
                    raw_ch = [(f"stabil / Schatten-Neutrino-Oszillation ({cl['c_name']})", 1.000)]
                elif base_key == "nu_R":
                    raw_ch = [(f"nu'_R,1 -> l'_1 + H'^0_1", 0.500), (f"nu'_R,1 -> l'^+_1 + H'^-_1", 0.500)]
                elif base_key == "gamma":
                    raw_ch = [(f"gamma'_{c_idx} -> 3 nu'_i + 3 nu'_ibar (100% Proca-Zerfall)", 1.000)]
                elif base_key == "g":
                    raw_ch = [(f"g'_{c_idx} -> n'_1 + n'_1bar (Schatten-Confinement)", 1.000)]
                elif base_key in ("W+", "W-"):
                    raw_ch = [(f"{sym_sh} -> q'_1 + q'_1bar (Schatten-Hadronen)", 0.675), (f"{sym_sh} -> l'_1 + nu'_l,1", 0.325)]
                elif base_key == "Z0":
                    raw_ch = [(f"Z'^0_{c_idx} -> q'_1 + q'_1bar", 0.700), (f"Z'^0_{c_idx} -> l'^+_1 + l'^-_1 + nu'_1 nu'_1bar", 0.300)]
                elif base_key == "H0":
                    raw_ch = [(f"H'^0_{c_idx} -> b'_1 + b'_1bar", 0.650), (f"H'^0_{c_idx} -> W'^+_1 + W'^-_1", 0.350)]
                elif base_key == "Phi_T":
                    raw_ch = [(f"Phi'_T,{c_idx} -> 23/24 Schatten", 23.0 / 24.0), (f"Phi'_T,{c_idx} -> 1/24 C_0", 1.0 / 24.0)]
                else:  # Graviton G_1..G_8
                    raw_ch = [(f"G_{c_idx} -> 23/24 Schatten-Sektoren", 23.0 / 24.0), (f"G_{c_idx} -> 1/24 Sichtbarer Sektor C_0", 1.0 / 24.0)]
            else:
                # Schalen 2..4 (C_9..C_23, s=2..4)
                if base_key == "G_0":
                    raw_ch = [(f"G_{c_idx} -> 23/24 Schatten-Sektoren", 23.0 / 24.0), (f"G_{c_idx} -> 1/24 Sichtbarer Sektor C_0", 1.0 / 24.0)]
                elif base_key in ("gamma", "g", "nu_1", "nu_2", "nu_3"):
                    raw_ch = [(f"stabil / Unterdrueckter Modus in {cl['c_name']}", 1.000)]
                else:
                    raw_ch = [(f"{sym_sh} -> Schale-1 / C_0 Projektionszerfall", 1.000)]

            norm_ch = _normalize_channels(raw_ch)
            ch_summary = " | ".join(f"{ch} [p={p:.3f}, {pct:.1f}%]" for ch, p, pct in norm_ch)

            for ch_name, p_val, pct_val in norm_ch:
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
                    "decay_channel": ch_name,
                    "prob": p_val,
                    "br_pct": pct_val,
                    "ch_summary": ch_summary,
                    "role": role,
                })

        # 1..9: 9 Dirac-Fermionen
        for name, sym, ptype, m_base in base_fermions:
            if not is_vis and s_idx == 1 and sym == "e":
                m_f = 0.001338 / u_scale
            else:
                m_f = m_base * r_val
            sym_vis = f"{sym}^-" if ptype == "Lepton" else sym
            sym_sh = f"{sym}'_{c_idx}^-" if ptype == "Lepton" else f"{sym}'_{c_idx}"
            register_entry(name, f"Schatten-{name}", sym_vis, sym_sh, ptype + prime, m_f, sym, f"Sektor {cl['c_name']} ({cl['shell']})")

        # 10..12: 3 leichte Neutrinos
        for nu_i in (1, 2, 3):
            register_entry(
                f"Neutrino {nu_i}", f"Schatten-Neutrino {nu_i}",
                f"nu_{nu_i}", f"nu'_{nu_i},{c_idx}",
                "Neutrino" + prime, m_nu_c0[nu_i] * r_val, f"nu_{nu_i}", f"Seesaw-Sektor {cl['c_name']}",
            )

        # 13..15: 3 Majorana-Neutrinos
        for maj_i in (1, 2, 3):
            register_entry(
                f"Majorana-Neutrino {maj_i}", f"Schatten-Majorana {maj_i}",
                f"nu_R{maj_i}", f"nu'_R{maj_i},{c_idx}",
                "Majorana" + prime, m_r_c0 * r_val, "nu_R", f"Majorana-Sektor {cl['c_name']}",
            )

        # 16: Photon
        m_ph = 0.0 if is_vis else (1.138e-6 / u_scale if s_idx == 1 else 0.0)
        register_entry("Photon", "Schatten-Photon", "gamma", f"gamma'_{c_idx}", "Eichboson" + prime, m_ph, "gamma", f"EM-Sektor {cl['c_name']}")

        # 17: Gluon
        register_entry("Gluon-Oktett (8x)", "Schatten-Gluonen (8x)", "g", f"g'_{c_idx}", "Eichboson" + prime, 0.0, "g", f"Starke Kraft {cl['c_name']}")

        # 18..19: W^+ und W^-
        for chg_s, b_key in [("+", "W+"), ("-", "W-")]:
            register_entry(
                f"W-Boson ({chg_s})", f"Schatten-W ({chg_s})",
                f"W^{chg_s}", f"W'^{chg_s}_{c_idx}",
                "Eichboson" + prime, m_w_c0 * r_val, b_key, f"Schwache Kraft {cl['c_name']}",
            )

        # 20: Z^0
        register_entry("Z-Boson", "Schatten-Z-Boson", "Z^0", f"Z'^0_{c_idx}", "Eichboson" + prime, m_z_c0 * r_val, "Z0", f"Schwache Kraft {cl['c_name']}")

        # 21: Higgs H^0
        register_entry("Higgs-Boson", "Schatten-Higgs", "H^0", f"H'^0_{c_idx}", "Skalar" + prime, m_h_c0 * r_val, "H0", f"Skalar-Sektor {cl['c_name']}")

        # 22: Modulus Phi_T
        register_entry("Torus-Modulus", "Schatten-Modulus", "Phi_T,0", f"Phi'_T,{c_idx}", "Modulus" + prime, m_phi_c0 * r_val, "Phi_T", f"Modulus {cl['c_name']}")

        # 23: Graviton G_c (24 Stück insgesamt)
        if is_vis:
            m_g, g_name = 0.0, "Universal-Graviton"
        elif s_idx == 1:
            m_g, g_name = 1.80e14 / u_scale, "Polychor-Graviton (s=1)"
        else:
            m_g, g_name = 4.00e6 / u_scale, f"Polychor-Graviton (s={s_idx})"
        register_entry(g_name, g_name, f"G_{c_idx}", f"G_{c_idx}", "Spin-2", m_g, "G_0", f"Gravitation {cl['c_name']}")

    return all_552


def main():
    parser = argparse.ArgumentParser(
        description="24-Zell-Torus (v9) — Zerfallswahrscheinlichkeiten fuer alle 552 Teilchen (Ausgabe + CSV)"
    )
    parser.add_argument("-u", "--torus-scale", type=float, default=1.0, help="Torus-Atmungsparameter u = L / L_T (Standard: 1.0)")
    parser.add_argument(
        "--csv",
        type=str,
        default="24cell_alle_552_zerfaelle_v9.csv",
        help="Dateipfad fuer den automatischen CSV-Export (Standard: 24cell_alle_552_zerfaelle_v9.csv)",
    )
    args = parser.parse_args()

    entries = build_all_552_with_decays(u_scale=args.torus_scale)

    # 1) IMMER als CSV speichern
    with open(args.csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv, delimiter=";")
        writer.writerow([
            "Zeile", "Sektor", "Cluster", "Schale", "D4_Ecke", "Resonanz_R",
            "Teilchen", "Symbol", "Typ", "Masse_GeV", "Masse_Formatiert",
            "Zerfallskanal", "Wahrscheinlichkeit_p", "Zerfallswahrscheinlichkeit_BR_pct", "Alle_Kanaele_Uebersicht"
        ])
        for row in entries:
            writer.writerow([
                row["id"], row["sector"], row["cluster"], row["shell"], row["vertex"],
                f"{row['R']:.8e}", row["name"], row["sym"], row["type"],
                f"{row['m_gev']:.8e}", row["m_str"].strip(),
                row["decay_channel"], f"{row['prob']:.4f}", f"{row['br_pct']:.2f}%", row["ch_summary"]
            ])

    # 2) Alle Eintraege ausgeben
    unique_particles = len({(r["cluster"], r["sym"]) for r in entries})
    print("=" * 156)
    print(
        f"  24-Zell-Torus & Schattenresonanz (v9) — Zerfallswahrscheinlichkeiten fuer alle {unique_particles} "
        f"Fundamentalfelder ({len(entries)} Kanal-Eintraege | u = {args.torus_scale:.3f})"
    )
    print(f"  CSV-Datei gespeichert unter: {args.csv}")
    print("=" * 156)
    header = (
        f"{'Nr':>4} | {'Cluster':<7} | {'Schale':<6} | {'Teilchen':<21} | {'Symbol':<11} | "
        f"{'Masse':>12} | {'Zerfallskanal':<44} | {'p [0..1]':>8} | {'BR [%]':>7}"
    )
    print(header)
    print("-" * 156)

    for row in entries:
        print(
            f"{row['id']:4d} | {row['cluster']:<7} | {row['shell']:<6} | {row['name']:<21} | {row['sym']:<11} | "
            f"{row['m_str']:>12} | {row['decay_channel']:<44} | {row['prob']:8.4f} | {row['br_pct']:6.2f}%"
        )

    print("=" * 156)
    print(f"Fertig: Alle {unique_particles} Teilchen (552 Fundamentalfelder) ausgewertet und in '{args.csv}' gespeichert.")
    print("=" * 156)


if __name__ == "__main__":
    main()

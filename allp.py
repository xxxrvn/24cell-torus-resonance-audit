#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24-Zell-Torus & Schattenresonanz (v9) — Vollstaendiger 552-Teilchen-Explorer
Zeigt alle 552 Teilchenarten (24 Polychor-Cluster x 23 Spezies = 528 Felder + 24 Gravitonen)
auf dem rotierenden 4D-Torus T^4 / Gamma_24 (L_T = 1.5605e-33 m):
  - Panorama-Modus [A]: Alle 24 Cluster (1-8-6-8-1) & alle 552 Teilchen gleichzeitig im 4D-Torus
  - Fokus-Modus    [F]: Exaktes 4D-Jacobi-Theta-Wellenpaket des gewaehlten Teilchens (1..552)
  - Cluster-Wahl   C_0..C_23 + Spezies-Katalog (6 Quarks, 6 Leptonen, 3 Majorana-nu_R,
    5 Eichbosonen, Higgs H^0/H'_c, Torus-Modulus Phi_T/Phi'_T,c, Graviton G_0/G_c)

Benoetigte Pakete:
    pip install pygame numpy
"""

import math
import sys
import numpy as np
import pygame

# ==============================================================================
# 1. FUNDAMENTALE v9-KONSTANTEN & 24-ZELL-POLYCHOR (D_4: 1-8-6-8-1)
# ==============================================================================
M_TAU_EV = 1776.86e6        # eV
M_MU_EV = 105.6583755e6     # eV
TAU = math.sqrt(2.0 * math.log(M_TAU_EV / M_MU_EV))  # 2.37588
L_T = 1.5605e-33            # m (96.55 l_Pl)
E_T_EV = 1.2645e26          # eV (1.2645e17 GeV)
V_H_EV = 246.22e9           # eV (246.22 GeV)
M_R_EV = 4.47e23            # eV (4.47e14 GeV)
SIGMA_0 = 1.0 / math.sqrt(math.pi * TAU)       # 0.3660 * L_T (Higgs)
SIGMA_W = 1.0 / (math.sqrt(3.0) * TAU)         # 0.2430 * L_T (Fermionen)
R_1 = 1.0 / (8.0 * math.pi**2) + 1e-7          # 0.01266525 (Schale 1)
CP_PHASE_DEG = -76.39       # arg(I_d13) in Grad


def format_mass_ev(m_ev):
    """Formatiert beliebige Massen von Sub-meV bis zur GUT-Skala sauber."""
    if m_ev <= 0.0:
        return "0 (masselos)"
    if m_ev < 1e-3:
        return f"{m_ev * 1e3:.3g} meV"
    if m_ev < 1e3:
        return f"{m_ev:.4g} eV"
    if m_ev < 1e6:
        return f"{m_ev / 1e3:.4g} keV"
    if m_ev < 1e9:
        return f"{m_ev / 1e6:.4g} MeV"
    gev = m_ev / 1e9
    if gev < 1e5:
        return f"{gev:.4g} GeV"
    return f"{gev:.4e} GeV"


def rotation_matrix_4d(a1, a2):
    """Orthonormale 4D-Doppelrotationsmatrix in den Ebenen (x_1, x_4) und (x_2, x_3)."""
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    return np.array(
        [
            [c1, 0.0, 0.0, -s1],
            [0.0, c2, -s2, 0.0],
            [0.0, s2, c2, 0.0],
            [s1, 0.0, 0.0, c1],
        ],
        dtype=np.float32,
    )


def build_24cell_clusters():
    """Erzeugt alle 24 Cluster C_0..C_23 des D_4-Polychors sortiert nach Schale (1-8-6-8-1)."""
    verts = []
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = [0.0, 0.0, 0.0, 0.0]
                    v[i] = s1
                    v[j] = s2
                    verts.append(np.array(v, dtype=np.float32))
    v0 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    verts.sort(key=lambda v: round(0.5 * float(np.sum((v - v0) ** 2))))

    edges = []
    for ia in range(len(verts)):
        for ib in range(ia + 1, len(verts)):
            if abs(float(np.sum((verts[ia] - verts[ib]) ** 2)) - 2.0) < 1e-5:
                edges.append((ia, ib))

    clusters = []
    for cid, v in enumerate(verts):
        d2 = round(0.5 * float(np.sum((v - v0) ** 2)))
        if d2 == 0:
            res = 1.0
            ew_scale = 1.0
            shell_name = "Zentrum (Sichtbares SM)"
            col = (85, 205, 255)
        elif d2 == 1:
            res = R_1
            ew_scale = R_1
            shell_name = "Schale 1 (Dunkle Materie)"
            col = (255, 155, 75)
        else:
            res = 1e-7
            ew_scale = 1e-7 * (5 ** (d2 - 1))
            shell_name = f"Schale {d2} (Tiefer Schatten)"
            col = (155, 130, 245) if d2 == 2 else ((125, 165, 215) if d2 == 3 else (215, 125, 185))

        # Torus-Ankerposition des Clusters (C_0 im Ursprung, Schalen symmetrisch verteilt)
        torus_anchor = 0.36 * (v - v0)
        clusters.append(
            {
                "id": cid,
                "label": f"C_{cid}",
                "d2": d2,
                "shell_name": shell_name,
                "res": res,
                "ew_scale": ew_scale,
                "v4": v,
                "anchor": torus_anchor,
                "color": col,
            }
        )
    return clusters, edges


# ==============================================================================
# 2. VOLLSTAENDIGER KATALOG ALLER 23 SPEZIES PRO CLUSTER (24 x 23 = 552 ARTEN)
# ==============================================================================
SPECIES_23 = [
    # --- 6 QUARKS ---
    {"idx": 0, "cat": "Quark", "sym": "u", "name": "Up-Quark", "m0_ev": 2.128e6, "spin": "1/2", "q": "+2/3", "pos": (0.28, -0.27, 0.25, -0.24), "twist": (2, 2, 2, 2), "cp": 0.0, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "stabil (im p)", "desc": "Leichtes Up-Typ-Quark der 1. Generation (Bildet mit d Protonen und Neutronen)."},
    {"idx": 1, "cat": "Quark", "sym": "d", "name": "Down-Quark", "m0_ev": 4.690e6, "spin": "1/2", "q": "-1/3", "pos": (0.25, -0.24, 0.22, -0.21), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "stabil (im p/n)", "desc": "Down-Typ-Quark der 1. Gen. mit CP-verletzender Holonomie-Phase (-76.39 deg)."},
    {"idx": 2, "cat": "Quark", "sym": "s", "name": "Strange-Quark", "m0_ev": 93.40e6, "spin": "1/2", "q": "-1/3", "pos": (0.19, 0.17, -0.16, 0.15), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "1.24e-8 s (K)", "desc": "Down-Typ-Quark der 2. Generation (mittlerer 4D-Abstand zum Higgs-Zentrum)."},
    {"idx": 3, "cat": "Quark", "sym": "c", "name": "Charm-Quark", "m0_ev": 1.265e9, "spin": "1/2", "q": "+2/3", "pos": (0.15, 0.13, -0.12, 0.11), "twist": (2, 2, 2, 2), "cp": 0.0, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "1.04e-12 s (D)", "desc": "Up-Typ-Quark der 2. Generation (staerkerer Ueberlapp mit H^0)."},
    {"idx": 4, "cat": "Quark", "sym": "b", "name": "Bottom-Quark", "m0_ev": 4.151e9, "spin": "1/2", "q": "-1/3", "pos": (0.08, -0.07, 0.06, -0.06), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "1.57e-12 s (B)", "desc": "Schweres Down-Typ-Quark der 3. Generation nahe am Higgs-Ursprung."},
    {"idx": 5, "cat": "Quark", "sym": "t", "name": "Top-Quark", "m0_ev": 172.95e9, "spin": "1/2", "q": "+2/3", "pos": (0.02, -0.02, 0.01, -0.01), "twist": (2, 2, 2, 2), "cp": 0.0, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "5.00e-25 s", "desc": "Schwerstes Elementarteilchen: Deckungsgleich mit H^0 (y_t(E_T) = 1/tau)."},

    # --- 6 LEPTONEN ---
    {"idx": 6, "cat": "Lepton", "sym": "e^-", "name": "Elektron", "m0_ev": 0.5110e6, "spin": "1/2", "q": "-1", "pos": (-0.33, 0.31, -0.30, 0.29), "twist": (2, 2, 2, 2), "cp": 0.0, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "stabil (>6.6e28 a)", "desc": "Leichtestes geladenes Lepton am Rand von C_0; bildet Atomhuellen."},
    {"idx": 7, "cat": "Lepton", "sym": "mu^-", "name": "Myon", "m0_ev": 105.658e6, "spin": "1/2", "q": "-1", "pos": (-0.18, -0.17, 0.16, -0.15), "twist": (2, 2, 2, 2), "cp": 0.0, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "2.197e-6 s", "desc": "Lepton der 2. Generation; definiert ueber m_tau/m_mu den Modul tau = 2.37588."},
    {"idx": 8, "cat": "Lepton", "sym": "tau^-", "name": "Tau-Lepton", "m0_ev": 1776.86e6, "spin": "1/2", "q": "-1", "pos": (-0.09, 0.08, -0.08, 0.07), "twist": (2, 2, 2, 2), "cp": 0.0, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "2.903e-13 s", "desc": "Schwerstes geladenes Lepton (3. Generation, nahe am Higgs-Zentrum)."},
    {"idx": 9, "cat": "Lepton", "sym": "nu_1", "name": "Neutrino 1 (Seesaw)", "m0_ev": 0.0086, "spin": "1/2", "q": "0", "pos": (-0.38, -0.36, 0.35, -0.34), "twist": (2, 2, 2, 2), "cp": 0.0, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "stabil", "desc": "Leichtester Seesaw-Eigenzustand (m_nu ~ m_D^2 / M_R, Majorana-Natur)."},
    {"idx": 10, "cat": "Lepton", "sym": "nu_2", "name": "Neutrino 2 (Seesaw)", "m0_ev": 0.0121, "spin": "1/2", "q": "0", "pos": (-0.36, 0.35, 0.34, -0.33), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "stabil", "desc": "Solarer Neutrino-Eigenzustand (dm^2_21 = 7.42e-5 eV^2)."},
    {"idx": 11, "cat": "Lepton", "sym": "nu_3", "name": "Neutrino 3 (Seesaw)", "m0_ev": 0.0502, "spin": "1/2", "q": "0", "pos": (-0.34, -0.33, -0.32, 0.31), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_W, "mode": "loc", "tau_c0": "stabil", "desc": "Atmosphaerischer Neutrino-Eigenzustand (dm^2_31 = 2.51e-3 eV^2)."},

    # --- 3 SUPERSCHWERE MAJORANA-NEUTRINOS ---
    {"idx": 12, "cat": "Majorana", "sym": "nu_R1", "name": "Majorana-nu_R1", "m0_ev": M_R_EV, "spin": "1/2", "q": "0", "pos": (0.04, 0.04, -0.04, -0.04), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "~1e-38 s (Leptogen.)", "desc": "Selbstkonjugiertes rechtshaendiges Neutrino (M_R = E_T exp(-tau^2) = 4.47e14 GeV)."},
    {"idx": 13, "cat": "Majorana", "sym": "nu_R2", "name": "Majorana-nu_R2", "m0_ev": M_R_EV * 1.45, "spin": "1/2", "q": "0", "pos": (-0.04, 0.04, 0.04, -0.04), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "~1e-38 s (Leptogen.)", "desc": "2. Generation der schweren Majorana-Neutrinos; treibt thermische Leptogenese."},
    {"idx": 14, "cat": "Majorana", "sym": "nu_R3", "name": "Majorana-nu_R3", "m0_ev": M_R_EV * 2.10, "spin": "1/2", "q": "0", "pos": (0.04, -0.04, 0.04, 0.04), "twist": (2, 2, 2, 2), "cp": CP_PHASE_DEG, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "~1e-38 s (Leptogen.)", "desc": "3. Generation der schweren Majorana-Neutrinos; schliesst Eichanomalien."},

    # --- 5 EICHBOSONEN ---
    {"idx": 15, "cat": "Eichboson", "sym": "gamma", "name": "Photon / Proca-gamma'", "m0_ev": 0.0, "spin": "1", "q": "0", "pos": (0.00, 0.00, 0.00, 0.00), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": SIGMA_0 * 1.15, "mode": "loc", "tau_c0": "stabil (in C_0)", "desc": "Masseloses U(1)_em Photon in C_0; in C_1..23 massives Proca-Schattenphoton (1.138 keV in S1)."},
    {"idx": 16, "cat": "Eichboson", "sym": "g", "name": "Gluon (8 Farben)", "m0_ev": 0.0, "spin": "1", "q": "0 (Farbe)", "pos": (0.12, -0.12, 0.10, -0.10), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "confined (SU(3)_C)", "desc": "Masseloses SU(3)_C Eichboson; erzeugt durch Confinement ~99% der Nukleonenmasse."},
    {"idx": 17, "cat": "Eichboson", "sym": "W^+", "name": "W^+-Boson", "m0_ev": 80.377e9, "spin": "1", "q": "+1", "pos": (0.05, 0.00, -0.05, 0.00), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "3.16e-25 s", "desc": "Geladenes schwaches SU(2)_L Eichboson (m_W = 1/2 g_2 v_H)."},
    {"idx": 18, "cat": "Eichboson", "sym": "W^-", "name": "W^--Boson", "m0_ev": 80.377e9, "spin": "1", "q": "-1", "pos": (-0.05, 0.00, 0.05, 0.00), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "3.16e-25 s", "desc": "Antiteilchen zu W^+; vermittelt Beta-Zerfall und schwache Flavour-Uebergaenge."},
    {"idx": 19, "cat": "Eichboson", "sym": "Z^0", "name": "Z^0-Boson", "m0_ev": 91.188e9, "spin": "1", "q": "0", "pos": (0.00, 0.05, 0.00, -0.05), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "2.64e-25 s", "desc": "Neutrales schwaches Eichboson (sin^2 theta_W = 0.2312)."},

    # --- 3 SKALARE & GRAVITATION ---
    {"idx": 20, "cat": "Skalar/Grav", "sym": "H^0", "name": "Higgs-Boson", "m0_ev": 125.68e9, "spin": "0", "q": "0", "pos": (0.00, 0.00, 0.00, 0.00), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": SIGMA_0, "mode": "loc", "tau_c0": "1.64e-22 s", "desc": "Zentriertes Spin-0-Dublett im Ursprung (0,0,0,0); Z_2-entkoppelt (lambda_HH' = 0)."},
    {"idx": 21, "cat": "Skalar/Grav", "sym": "Phi_T", "name": "Torus-Volumen-Modulus", "m0_ev": E_T_EV, "spin": "0", "q": "0", "pos": (0.00, 0.00, 0.00, 0.00), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": 1.0, "mode": "bulk", "tau_c0": "2.44e-36 s (Reheat)", "desc": "Atem-Modus des 4D-Torus (m = E_T = 1.2645e17 GeV); treibt Urknall-Reheating."},
    {"idx": 22, "cat": "Skalar/Grav", "sym": "G_c", "name": "Spin-2-Graviton", "m0_ev": 0.0, "spin": "2", "q": "0", "pos": (0.00, 0.00, 0.00, 0.00), "twist": (0, 0, 0, 0), "cp": 0.0, "sigma": 1.0, "mode": "bulk", "tau_c0": "stabil (G_0) / 3.3e-30 s (G_1)", "desc": "G_0 in C_0 ist masselos (4D-Bulk-Nebel); G_1..23 sind massive Polychor-Gravitonen."},
]


def get_particle_mass_ev(sp, cl):
    """Berechnet die exakte physikalische Ruhemasse der Spezies `sp` im Cluster `cl`."""
    d2 = cl["d2"]
    res = cl["res"]
    ew = cl["ew_scale"]
    sym = sp["sym"]

    if d2 == 0:
        return sp["m0_ev"]

    # Schatten-Sektoren C_1..C_23:
    if sym == "gamma":
        # Proca-Schattenphoton: 1.138 keV in Schale 1, skaliert mit ew_scale
        return 1.138e3 * (ew / R_1)
    if sym == "g":
        return 0.0
    if sym == "G_c":
        # Massive Polychor-Gravitonen: 1.80e14 GeV (Schale 1), 4.00e6 GeV (Schalen 2-4)
        return 1.80e23 if d2 == 1 else 4.00e15
    if sym == "Phi_T":
        # Lokale Blow-up-Moduli: E_T * R_c
        return E_T_EV * res
    if sp["cat"] == "Majorana":
        # M'_{R,c} = M_R * R_c
        return sp["m0_ev"] * res
    if sym in ("nu_1", "nu_2", "nu_3"):
        # Seesaw-Skalierung: (v'_H)^2 / M'_R ~ ew^2 / res
        return sp["m0_ev"] * (ew * ew) / max(res, 1e-30)

    # Quarks, geladene Leptonen, W+-, Z^0 und Higgs H'_c skalieren mit dem lokalen Higgs-VEV ew_scale
    return sp["m0_ev"] * ew


def get_display_symbol(sp, cl):
    """Gibt das korrekte Symbol inkl. Cluster-Index (z. B. u, u'_1, G_0, G_1) zurueck."""
    cid = cl["id"]
    sym = sp["sym"]
    if cid == 0:
        return "G_0" if sym == "G_c" else sym
    if sym == "G_c":
        return f"G_{cid}"
    if sym == "Phi_T":
        return f"Phi'_T,{cid}"
    if sym == "H^0":
        return f"H'_{cid}"
    if "^" in sym:
        base, sup = sym.split("^", 1)
        return f"{base}'_{cid}^{sup}"
    if "_" in sym:
        base, sub = sym.split("_", 1)
        return f"{base}'_{sub},{cid}"
    return f"{sym}'_{cid}"


# ==============================================================================
# 3. MATHEMATIK: 4D-JACOBI-THETA, HIGGS-UEBERLAPP & FARBEN
# ==============================================================================
def jacobi_theta_1d(x, a, sigma, n_twist, phase_rad=0.0):
    """Vektorisierte periodische 1D-Jacobi-Theta-Wellenfunktion auf [-0.5, 0.5]."""
    dx_base = ((x - a + 0.5) % 1.0) - 0.5
    inv_sig2 = np.float32(-0.5 / (sigma * sigma))
    two_pi_n = np.float32(2.0 * math.pi * n_twist)
    ph0 = np.float32(phase_rad)

    psi = np.zeros_like(x, dtype=np.complex64)
    for m in (-1.0, 0.0, 1.0):
        dx = dx_base + np.float32(m)
        env = np.exp(inv_sig2 * (dx * dx))
        ang = two_pi_n * dx + ph0
        psi += env * (np.cos(ang) + 1j * np.sin(ang))
    return psi


def compute_higgs_overlap_4d(pos, sigma, twist, u_scale):
    """Exaktes 4D-Ueberlappungsintegral O_H zwischen Teilchen und lokalem Higgs."""
    xg = np.linspace(-0.5, 0.5, 128, endpoint=False, dtype=np.float32)
    dx = 1.0 / 128.0
    sig_p = sigma / u_scale
    sig_h = SIGMA_0 / u_scale
    overlap = 1.0
    for k in range(4):
        psi_p = jacobi_theta_1d(xg, pos[k], sig_p, twist[k], 0.0)
        psi_h = jacobi_theta_1d(xg, 0.0, sig_h, 0, 0.0)
        dens_p = np.abs(psi_p) ** 2
        dens_h = np.abs(psi_h) ** 2
        num = float(np.sum(dens_p * dens_h) * dx)
        den = math.sqrt(float(np.sum(dens_p**2) * dx) * float(np.sum(dens_h**2) * dx))
        overlap *= num / max(den, 1e-30)
    return overlap


def hsv_to_rgb_uint8(h, s, v):
    """Vektorisierte HSV -> RGB Konvertierung (0..1 -> uint8)."""
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6

    r = np.where(i_mod == 0, v, np.where(i_mod == 1, q, np.where(i_mod == 2, p, np.where(i_mod == 3, p, np.where(i_mod == 4, t, v)))))
    g = np.where(i_mod == 0, t, np.where(i_mod == 1, v, np.where(i_mod == 2, v, np.where(i_mod == 3, q, np.where(i_mod == 4, p, p)))))
    b = np.where(i_mod == 0, p, np.where(i_mod == 1, p, np.where(i_mod == 2, t, np.where(i_mod == 3, v, np.where(i_mod == 4, v, q)))))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ==============================================================================
# 4. UI-SLIDER
# ==============================================================================
class Slider:
    def __init__(self, rect, label, min_val, max_val, val, fmt="{:+.2f}"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.val = val
        self.fmt = fmt
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.inflate(8, 12).collidepoint(event.pos):
                self.dragging = True
                self._update_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_x(event.pos[0])
            return True
        return False

    def _update_from_x(self, mx):
        t = (mx - self.rect.x) / max(self.rect.width, 1)
        t = max(0.0, min(1.0, t))
        self.val = self.min_val + t * (self.max_val - self.min_val)

    def draw(self, surf, font):
        txt = f"{self.label}: {self.fmt.format(self.val)}"
        lbl_surf = font.render(txt, True, (215, 225, 240))
        surf.blit(lbl_surf, (self.rect.x, self.rect.y - 15))

        pygame.draw.rect(surf, (45, 55, 75), self.rect, border_radius=4)
        t = (self.val - self.min_val) / max(self.max_val - self.min_val, 1e-9)
        fill_w = int(t * self.rect.width)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.height)
            pygame.draw.rect(surf, (70, 175, 255), fill_rect, border_radius=4)

        kx = self.rect.x + fill_w
        ky = self.rect.centery
        pygame.draw.circle(surf, (240, 248, 255), (kx, ky), 6)
        pygame.draw.circle(surf, (30, 120, 220), (kx, ky), 6, 2)


# ==============================================================================
# 5. HAUPTANWENDUNG: ALLE 552 TEILCHEN AUF DEM 24-ZELL-TORUS
# ==============================================================================
def main():
    pygame.init()
    WIDTH, HEIGHT = 1400, 850
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("24-Zell-Torus & Schattenresonanz (v9) — Alle 552 Teilchen (24 Cluster x 23 Arten)")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("dejavusans,arial,helvetica", 19, bold=True)
    font_sub = pygame.font.SysFont("dejavusans,arial,helvetica", 13, bold=True)
    font_ui = pygame.font.SysFont("dejavusans,arial,helvetica", 12)
    font_mono = pygame.font.SysFont("dejavusansmono,consolas,monospace", 12)
    font_tiny = pygame.font.SysFont("dejavusansmono,consolas,monospace", 11)

    clusters, edges_24 = build_24cell_clusters()

    # Links: 4D-Torus-Viewport (T^4 / Gamma_24)
    TORUS_SIZE = 620
    TORUS_POS = (24, 82)
    torus_rect = pygame.Rect(TORUS_POS[0], TORUS_POS[1], TORUS_SIZE, TORUS_SIZE)
    TORUS_N = 168
    t_axis = np.linspace(-0.5, 0.5, TORUS_N, endpoint=False, dtype=np.float32)
    U_T, V_T = np.meshgrid(t_axis, t_axis)

    panel_x = 668
    panel_w = 708

    # 1) Cluster-Selector-Buttons (C_0 .. C_23 in 3 Reihen x 8 Spalten)
    cluster_buttons = []
    for cid in range(24):
        col = cid % 8
        row = cid // 8
        bx = panel_x + 10 + col * 86
        by = 112 + row * 28
        cluster_buttons.append(pygame.Rect(bx, by, 81, 24))

    # 2) Modus- & Filter-Buttons
    btn_panorama = pygame.Rect(panel_x + 10, 204, 224, 28)
    btn_focus = pygame.Rect(panel_x + 240, 204, 224, 28)
    btn_labels = pygame.Rect(panel_x + 470, 204, 228, 28)

    # 3) Alle 23 Spezies-Buttons (6 Reihen x 4 Spalten = 24 Plaetze -> 23 Spezies + Reset)
    species_buttons = []
    for s_idx in range(23):
        col = s_idx % 4
        row = s_idx // 4
        bx = panel_x + 10 + col * 172
        by = 242 + row * 40
        species_buttons.append(pygame.Rect(bx, by, 166, 35))
    btn_reset = pygame.Rect(panel_x + 10 + 3 * 172, 242 + 5 * 40, 166, 35)

    # 4) Slider
    slider_rot = Slider((panel_x + 12, 508, 215, 7), "4D-Rotation [rad/s]", 0.00, 2.20, 0.52, "{:.2f}")
    slider_w4 = Slider((panel_x + 246, 508, 215, 7), "4D-Schnitt x_4 / L_T", -0.50, 0.50, 0.00)
    slider_u = Slider((panel_x + 480, 508, 215, 7), "Torus-Atmung u = L/L_T", 0.85, 1.732, 1.00, "{:.3f}")
    sliders = [slider_rot, slider_w4, slider_u]

    # Zustandsvariablen
    sel_cluster_id = 0     # C_0 (Sichtbar)
    sel_species_idx = 5    # Top-Quark (t)
    panorama_all_552 = True  # Startet direkt mit dem vollständigen 552-Teilchen-Panorama!
    show_all_labels = True
    auto_rotate = True

    rot_a1 = 0.28
    rot_a2 = 0.17
    anim_time = 0.0
    dragging_torus = False
    last_mouse = (0, 0)

    # Vektorisierte Wolken-Offsets der 23 Spezies um jedes Cluster-Zentrum (fuer Panorama-Mode)
    # Jede der 23 Arten bekommt eine eindeutige radiale + winkelmaessige 4D-Signatur
    species_4d_offsets = []
    for sp in SPECIES_23:
        ox = 0.22 * sp["pos"][0]
        oy = 0.22 * sp["pos"][1]
        oz = 0.22 * sp["pos"][2]
        ow = 0.22 * sp["pos"][3]
        species_4d_offsets.append(np.array([ox, oy, oz, ow], dtype=np.float32))
    species_4d_offsets = np.array(species_4d_offsets, dtype=np.float32)  # (23, 4)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        anim_time += dt

        if auto_rotate and not dragging_torus:
            rot_a1 = (rot_a1 + slider_rot.val * dt) % (2.0 * math.pi)
            rot_a2 = (rot_a2 + 0.6180339887 * slider_rot.val * dt) % (2.0 * math.pi)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_rotate = not auto_rotate
                elif event.key == pygame.K_a:
                    panorama_all_552 = True
                elif event.key == pygame.K_f:
                    panorama_all_552 = False
                elif event.key == pygame.K_l:
                    show_all_labels = not show_all_labels
                elif event.key == pygame.K_LEFT:
                    sel_species_idx = (sel_species_idx - 1) % 23
                elif event.key == pygame.K_RIGHT:
                    sel_species_idx = (sel_species_idx + 1) % 23
                elif event.key == pygame.K_UP:
                    sel_cluster_id = (sel_cluster_id - 1) % 24
                elif event.key == pygame.K_DOWN:
                    sel_cluster_id = (sel_cluster_id + 1) % 24
                elif event.key == pygame.K_r:
                    rot_a1, rot_a2 = 0.28, 0.17
                    slider_w4.val = 0.0
                    slider_u.val = 1.0
                    sel_cluster_id = 0
                    sel_species_idx = 5
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if torus_rect.collidepoint(event.pos):
                    dragging_torus = True
                    last_mouse = event.pos
                for cid, rect in enumerate(cluster_buttons):
                    if rect.collidepoint(event.pos):
                        sel_cluster_id = cid
                for s_idx, rect in enumerate(species_buttons):
                    if rect.collidepoint(event.pos):
                        sel_species_idx = s_idx
                if btn_panorama.collidepoint(event.pos):
                    panorama_all_552 = True
                elif btn_focus.collidepoint(event.pos):
                    panorama_all_552 = False
                elif btn_labels.collidepoint(event.pos):
                    show_all_labels = not show_all_labels
                elif btn_reset.collidepoint(event.pos):
                    rot_a1, rot_a2 = 0.28, 0.17
                    slider_w4.val = 0.0
                    slider_u.val = 1.0
                    sel_cluster_id = 0
                    sel_species_idx = 5
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_torus = False
            elif event.type == pygame.MOUSEMOTION and dragging_torus:
                dx_m = event.pos[0] - last_mouse[0]
                dy_m = event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                rot_a1 = (rot_a1 + dx_m * 0.01) % (2.0 * math.pi)
                rot_a2 = (rot_a2 + dy_m * 0.01) % (2.0 * math.pi)

            for s in sliders:
                s.handle_event(event)

        cl = clusters[sel_cluster_id]
        sp = SPECIES_23[sel_species_idx]
        w4_val = slider_w4.val
        u_val = slider_u.val
        R4 = rotation_matrix_4d(rot_a1, rot_a2)

        # Rotierte 4D-Torus-Koordinaten (X1, X2, X3, X4)
        X1 = U_T * R4[0, 0] + V_T * R4[0, 1] + w4_val * R4[0, 3]
        X2 = U_T * R4[1, 0] + V_T * R4[1, 1] + w4_val * R4[1, 3]
        X3 = U_T * R4[2, 0] + V_T * R4[2, 1] + w4_val * R4[2, 3]
        X4 = U_T * R4[3, 0] + V_T * R4[3, 1] + w4_val * R4[3, 3]
        coords_t = (X1, X2, X3, X4)

        # ----------------------------------------------------------------------
        # 1. TORUS-FELD BERECHNEN: ALLE 552 TEILCHEN (PANORAMA) ODER FOKUS (1/552)
        # ----------------------------------------------------------------------
        if panorama_all_552:
            # Alle 24 Cluster-Wellenwolken (24 x 23 = 552 Teilchen) + Bulk-Gravitonen G_0..23
            bulk_wave = 0.09 + 0.025 * np.sin(2.0 * math.pi * (X1 + X2 - X3) - 2.0 * anim_time)
            tor_r = bulk_wave * 0.65
            tor_g = bulk_wave * 0.90
            tor_b = bulk_wave * 1.45

            sig_cl = (SIGMA_W * 0.56) / u_val
            for c_item in clusters:
                anc = c_item["anchor"]
                d2 = c_item["d2"]
                # Gewicht & Farbe nach Polychor-Schale
                if d2 == 0:
                    w_c = 1.15
                    cr, cg, cb = 0.25, 0.85, 1.00
                elif d2 == 1:
                    w_c = 0.52
                    cr, cg, cb = 1.00, 0.58, 0.22
                else:
                    w_c = 0.30
                    cr, cg, cb = 0.65, 0.50, 0.95

                if c_item["id"] == sel_cluster_id:
                    w_c *= 1.45

                psi_c = np.exp(np.complex64(-1j * (2.2 + 0.2 * d2) * anim_time)) * np.ones_like(X1, dtype=np.complex64)
                for k in range(4):
                    psi_c *= jacobi_theta_1d(coords_t[k], float(anc[k]), sig_cl, 2, 0.0)
                dens_c = (np.abs(psi_c) ** 2) * (0.80 + 0.20 * np.cos(np.angle(psi_c)))
                tor_r += w_c * cr * dens_c
                tor_g += w_c * cg * dens_c
                tor_b += w_c * cb * dens_c

            # Zusaetzlich das ausgewaehlte Einzelteilchen (1 von 552) leuchtend hervorheben
            target_pos = cl["anchor"] + 0.35 * np.array(sp["pos"], dtype=np.float32)
            psi_sel = np.exp(np.complex64(-1j * 2.8 * anim_time)) * np.ones_like(X1, dtype=np.complex64)
            cp_rad = math.radians(sp["cp"])
            for k in range(4):
                psi_sel *= jacobi_theta_1d(coords_t[k], float(target_pos[k]), (SIGMA_W * 0.52) / u_val, sp["twist"][k], cp_rad if k == 0 else 0.0)
            dens_sel = np.abs(psi_sel) ** 2
            tor_r += 1.25 * dens_sel
            tor_g += 1.20 * dens_sel
            tor_b += 0.55 * dens_sel

            tor_rgb = np.stack([tor_r, tor_g, tor_b], axis=-1)
            tor_rgb = tor_rgb / (1.0 + 0.52 * tor_rgb)
            tor_u8 = np.clip(np.power(np.clip(tor_rgb, 0.0, 1.0), 0.85) * 255.0, 0, 255).astype(np.uint8)
        else:
            # Fokus-Modus: Exaktes 4D-Jacobi-Theta-Wellenpaket des gewaehlten Teilchens (1/552)
            if sp["mode"] == "bulk":
                wave = 0.14 * np.sin(2.0 * math.pi * (X1 + X2 + X3 - X4) - 2.2 * anim_time)
                val = np.clip(0.55 + wave, 0.25, 0.88)
                phase_norm = (0.58 + 0.08 * np.cos(2.0 * math.pi * (X1 - X3) + anim_time)) % 1.0
                sat = np.full_like(val, 0.45)
            else:
                target_pos = cl["anchor"] + 0.45 * np.array(sp["pos"], dtype=np.float32)
                sig_eff = sp["sigma"] / u_val
                cp_rad = math.radians(sp["cp"])
                psi_4d = np.exp(np.complex64(-1j * 2.4 * anim_time)) * np.ones_like(X1, dtype=np.complex64)
                for k in range(4):
                    psi_4d *= jacobi_theta_1d(coords_t[k], float(target_pos[k]), sig_eff, sp["twist"][k], cp_rad if k == 0 else 0.0)
                val = np.clip(np.abs(psi_4d), 0.0, 1.0)
                phase_norm = (np.angle(psi_4d) / (2.0 * math.pi)) % 1.0
                sat = np.where(val > 0.03, 0.88, 0.25)
                bulk_bg = 0.07 + 0.015 * np.sin(2.0 * math.pi * (U_T + V_T) - 1.5 * anim_time)
                val = np.clip(val + bulk_bg, 0.0, 1.0)
            tor_u8 = hsv_to_rgb_uint8(phase_norm, sat, val)

        torus_surf = pygame.surfarray.make_surface(np.transpose(tor_u8, (1, 0, 2)))
        torus_surf = pygame.transform.smoothscale(torus_surf, (TORUS_SIZE, TORUS_SIZE))

        # ----------------------------------------------------------------------
        # 2. BILDSCHIRM ZEICHNEN: KOPFZEILE & 4D-TORUS MIT ALLEN 552 MIKRO-KNOTEN
        # ----------------------------------------------------------------------
        screen.fill((14, 18, 28))

        disp_sym = get_display_symbol(sp, cl)
        m_curr_ev = get_particle_mass_ev(sp, cl)
        global_id = sel_cluster_id * 23 + sel_species_idx + 1

        title_s = font_title.render(
            f"Vollstaendiges v9-Teilchenspektrum: Alle 552 Arten (24 Cluster x 23 Spezies) — Aktiv: #{global_id}/552 [{disp_sym}]",
            True,
            (235, 242, 255),
        )
        sub_s = font_ui.render(
            f"T^4 / Gamma_24 (L_T = 1.5605e-33 m): 528 Materie-/Eich-/Moduli-Felder + 24 Spin-2-Gravitonen = 552 Arten (1,053 Zustaende inkl. Antiteilchen)",
            True,
            (150, 175, 210),
        )
        screen.blit(title_s, (24, 16))
        screen.blit(sub_s, (24, 44))

        screen.blit(torus_surf, TORUS_POS)
        pygame.draw.rect(
            screen,
            (85, 110, 150),
            (TORUS_POS[0] - 1, TORUS_POS[1] - 1, TORUS_SIZE + 2, TORUS_SIZE + 2),
            2,
        )

        tcx = TORUS_POS[0] + TORUS_SIZE // 2
        tcy = TORUS_POS[1] + TORUS_SIZE // 2

        # Projiziere alle 24 Polychor-Cluster & ihre 23 Teilchen (= 552 Punkte!)
        rot_anchors = np.array([c["anchor"] for c in clusters], dtype=np.float32) @ R4
        node_2d = []
        for cid in range(24):
            px_c = int(tcx + rot_anchors[cid, 0] * TORUS_SIZE)
            py_c = int(tcy + rot_anchors[cid, 1] * TORUS_SIZE)
            node_2d.append((px_c, py_c, float(rot_anchors[cid, 3])))

        # Kanten des 24-Zell-Polychors
        for ia, ib in edges_24:
            pa, pb = node_2d[ia], node_2d[ib]
            if torus_rect.collidepoint((pa[0], pa[1])) and torus_rect.collidepoint((pb[0], pb[1])):
                pygame.draw.line(screen, (48, 66, 96), (pa[0], pa[1]), (pb[0], pb[1]), 1)

        # Zeichne alle 552 Teilchen-Mikropunkte (23 pro Cluster) & Cluster-Zentren
        rot_sp_offs = (0.28 * species_4d_offsets) @ R4  # (23, 4)
        for cid, c_item in enumerate(clusters):
            px_c, py_c, wn4 = node_2d[cid]
            if not torus_rect.collidepoint((px_c, py_c)):
                continue
            is_sel_c = cid == sel_cluster_id
            col_c = c_item["color"]

            if panorama_all_552:
                # 23 Teilchenpunkte um den Cluster-Knoten C_cid
                for s_i in range(23):
                    mx = int(px_c + rot_sp_offs[s_i, 0] * TORUS_SIZE)
                    my = int(py_c + rot_sp_offs[s_i, 1] * TORUS_SIZE)
                    if torus_rect.collidepoint((mx, my)):
                        if is_sel_c and s_i == sel_species_idx:
                            pygame.draw.circle(screen, (255, 245, 130), (mx, my), 5)
                        else:
                            pygame.draw.circle(screen, col_c, (mx, my), 1)

            # Cluster-Hauptknoten
            rad_c = 7 if is_sel_c else (5 if c_item["d2"] == 0 else 3)
            pygame.draw.circle(screen, (255, 245, 140) if is_sel_c else col_c, (px_c, py_c), rad_c, 2 if is_sel_c else 0)

            if show_all_labels and (is_sel_c or c_item["d2"] <= 1 or abs(wn4 - w4_val) < 0.16):
                lbl_c = f"C_{cid}" if cid > 0 else "C_0 (SM)"
                screen.blit(font_tiny.render(lbl_c, True, (255, 245, 150) if is_sel_c else col_c), (px_c + 7, py_c - 6))

        # Aktives Einzelteilchen im Fokus-Modus markieren
        if not panorama_all_552:
            t_pos = cl["anchor"] + 0.45 * np.array(sp["pos"], dtype=np.float32)
            t_rot = t_pos @ R4
            tx_p = int(tcx + t_rot[0] * TORUS_SIZE)
            ty_p = int(tcy + t_rot[1] * TORUS_SIZE)
            if torus_rect.collidepoint((tx_p, ty_p)):
                pygame.draw.circle(screen, (255, 245, 130), (tx_p, ty_p), 7, 2)
                screen.blit(font_mono.render(f"{disp_sym} ({format_mass_ev(m_curr_ev)})", True, (255, 245, 140)), (tx_p + 10, ty_p - 8))

        # Higgs-Kontur im Zentrum (0,0,0,0)
        r_h = int((SIGMA_0 / u_val) * 0.55 * TORUS_SIZE)
        if 6 < r_h < TORUS_SIZE // 2:
            pygame.draw.circle(screen, (255, 220, 110), (tcx, tcy), r_h, 1)

        # Legende unten links
        leg_items = [
            ((85, 205, 255), "C_0: 23 SM-Arten"),
            ((255, 155, 75), "C_1..8 (Schale 1): 184 Arten (DM)"),
            ((155, 130, 245), "C_9..23 (Schalen 2-4): 345 Arten"),
            ((255, 245, 130), f"Aktiv: {disp_sym}"),
        ]
        lx_cur = 24
        for col_i, txt_i in leg_items:
            pygame.draw.circle(screen, col_i, (lx_cur + 5, 722), 5)
            ts_i = font_ui.render(txt_i, True, (195, 210, 235))
            screen.blit(ts_i, (lx_cur + 14, 714))
            lx_cur += ts_i.get_width() + 20

        screen.blit(
            font_ui.render(
                "Tasten: [Pfeile] Cluster & Teilchen wechseln | [A] Alle 552 Panorama | [F] Einzel-Fokus | [L] Labels | [SPACE] Rotation | [R] Reset",
                True,
                (140, 160, 190),
            ),
            (24, 744),
        )

        # ----------------------------------------------------------------------
        # 3. RECHTES PANEL: ALLE 24 CLUSTER & ALLE 23 SPEZIES (= 552 ARTEN)
        # ----------------------------------------------------------------------
        pygame.draw.rect(screen, (20, 26, 40), (panel_x, 82, panel_w, 742), border_radius=10)
        pygame.draw.rect(screen, (50, 65, 95), (panel_x, 82, panel_w, 742), 1, border_radius=10)

        screen.blit(
            font_sub.render(
                f"1. Polychor-Cluster waehlen (24 Knoten, 1-8-6-8-1) — Aktiv: {cl['label']} [{cl['shell_name']}, R_c = {cl['res']:.6g}]",
                True,
                (235, 245, 255),
            ),
            (panel_x + 12, 90),
        )

        # 24 Cluster-Buttons
        for cid, rect in enumerate(cluster_buttons):
            c_item = clusters[cid]
            is_sel = cid == sel_cluster_id
            base_c = c_item["color"]
            bg = (int(base_c[0] * 0.35), int(base_c[1] * 0.35), int(base_c[2] * 0.35)) if is_sel else (28, 35, 52)
            bd = (255, 245, 130) if is_sel else base_c
            pygame.draw.rect(screen, bg, rect, border_radius=4)
            pygame.draw.rect(screen, bd, rect, 2 if is_sel else 1, border_radius=4)
            lbl_btn = f"C_{cid} (S{c_item['d2']})"
            screen.blit(font_tiny.render(lbl_btn, True, (240, 248, 255) if is_sel else (195, 210, 235)), (rect.x + 8, rect.y + 5))

        # Modus-Buttons
        for rect, lbl, active in [
            (btn_panorama, "[A] Alle 552 Teilchen (Panorama)", panorama_all_552),
            (btn_focus, "[F] Einzelteilchen-Fokus (1/552)", not panorama_all_552),
            (btn_labels, f"[L] Cluster-Labels: {'AN' if show_all_labels else 'AUTO'}", show_all_labels),
        ]:
            pygame.draw.rect(screen, (38, 78, 125) if active else (28, 36, 54), rect, border_radius=5)
            pygame.draw.rect(screen, (110, 200, 255) if active else (60, 75, 105), rect, 2 if active else 1, border_radius=5)
            screen.blit(font_ui.render(lbl, True, (235, 245, 255)), (rect.x + 10, rect.y + 7))

        # Alle 23 Spezies-Buttons im aktuellen Cluster C_sel
        cat_colors = {
            "Quark": (255, 125, 125),
            "Lepton": (110, 225, 255),
            "Majorana": (225, 145, 255),
            "Eichboson": (255, 220, 115),
            "Skalar/Grav": (125, 245, 185),
        }
        for s_idx, rect in enumerate(species_buttons):
            s_item = SPECIES_23[s_idx]
            is_sel = s_idx == sel_species_idx
            cc = cat_colors[s_item["cat"]]
            bg = (40, 74, 122) if is_sel else (26, 33, 49)
            bd = (255, 245, 130) if is_sel else cc
            pygame.draw.rect(screen, bg, rect, border_radius=5)
            pygame.draw.rect(screen, bd, rect, 2 if is_sel else 1, border_radius=5)

            s_disp = get_display_symbol(s_item, cl)
            m_str = format_mass_ev(get_particle_mass_ev(s_item, cl))
            screen.blit(font_sub.render(s_disp, True, (245, 250, 255)), (rect.x + 7, rect.y + 3))
            screen.blit(font_tiny.render(m_str, True, cc), (rect.x + 7, rect.y + 19))

        # Reset-Button auf Slot 24
        pygame.draw.rect(screen, (34, 42, 62), btn_reset, border_radius=5)
        pygame.draw.rect(screen, (85, 105, 145), btn_reset, 1, border_radius=5)
        screen.blit(font_ui.render("Ansicht & Auswahl Reset [R]", True, (215, 228, 248)), (btn_reset.x + 8, btn_reset.y + 10))

        # Slider
        for s in sliders:
            s.draw(screen, font_ui)

        # ----------------------------------------------------------------------
        # 4. DETAILLIERTE TEILCHEN- & SPEKTRUMS-TELEMETRIE (UNTEN RECHTS)
        # ----------------------------------------------------------------------
        tel_rect = pygame.Rect(panel_x + 10, 530, panel_w - 20, 282)
        pygame.draw.rect(screen, (15, 20, 32), tel_rect, border_radius=6)
        pygame.draw.rect(screen, (60, 80, 115), tel_rect, 1, border_radius=6)

        ov_h = 1.0 if sp["mode"] == "bulk" else compute_higgs_overlap_4d(sp["pos"], sp["sigma"], sp["twist"], u_val)
        px, py, pz, pw = sp["pos"]
        tau_str = sp["tau_c0"] if cl["id"] == 0 else (
            "1.09e-4 s -> 3 nu'_i" if (cl["d2"] == 1 and sp["sym"] == "gamma") else
            ("1.20e-30 s (Reheat)" if (cl["d2"] == 1 and sp["sym"] == "Phi_T") else
             ("2.44e-15 s (<< BBN)" if (cl["d2"] >= 2 and sp["sym"] == "Phi_T") else "skaliert mit R_c^-3"))
        )

        tel_lines = [
            (f"TEILCHEN #{global_id} VON 552 :  {disp_sym}  ({sp['name']} im Cluster {cl['label']} — {cl['shell_name']})", (255, 230, 130)),
            (f"Kategorie & Quantenzahl: {sp['cat']}  |  Spin s = {sp['spin']}  |  Ladung Q = {sp['q']}  |  Resonanz R_c = {cl['res']:.6g}", (225, 235, 250)),
            (f"Physikalische Ruhemasse: m({disp_sym}) = {format_mass_ev(m_curr_ev)}   (Higgs-VEV-Skala v'_H = {format_mass_ev(V_H_EV * cl['ew_scale'])})", (135, 245, 190)),
            (f"4D-Torus-Koordinaten   : a / L_T = ({px:+.2f}, {py:+.2f}, {pz:+.2f}, {pw:+.2f}) + Anker {cl['label']}  |  Twist n = {sp['twist']}", (135, 225, 255)),
            (f"Higgs-Ueberlapp & CP   : O_H = {ov_h:.5f}  |  CP-Phase = {sp['cp']:+.2f} deg  |  Lebensdauer: {tau_str}", (195, 215, 240)),
            (f"Physikalische Rolle    : {sp['desc']}", (215, 225, 242)),
            ("--- GESAMTBILANZ DES v9-SPEKTRUMS AUF T^4 / Gamma_24 (MODUL 22) ---", (255, 225, 130)),
            ("Pro Cluster (23 Arten) : 6 Quarks + 6 Leptonen + 3 Majorana-nu_R + 5 Eichbosonen + H^0 + Phi_T + 1 Graviton", (185, 205, 235)),
            ("Alle 24 Polychor-Knoten: 24 x 6 = 144 Quarks | 24 x 6 = 144 Leptonen | 24 x 3 = 72 Majorana-nu_R", (185, 205, 235)),
            ("                         24 x 5 = 120 Eichbosonen | 24 Higgs | 24 Torus-Moduli | 24 Gravitonen = 552 Arten", (135, 245, 190)),
        ]
        for li, (txt, col) in enumerate(tel_lines):
            screen.blit(font_mono.render(txt, True, col), (tel_rect.x + 10, tel_rect.y + 8 + li * 26))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24-Zell-Torus & Schattenresonanz (v9) — Quantenmechanischer Chemie- & Kernfusions-Simulator (CLI)
(Alle 118 Elemente Z=1..118, 100% Teilchenerhaltung, Plasma-Ionisation, Gamow-WKB-Tunneln,
 4D-Torus-Kernueberlapp O_4D,nuc & schwacher W^+-Kanal p+p -> 2H + e+ + nu_e)

Beispiele:
    python 24cell_torus_reaction_cli_v9.py -r "[2H]" "[3H]" -T 6e7 -P 1e8 -t 50
    python 24cell_torus_reaction_cli_v9.py -r "[2H]" "[2H]" -T 1.2e8 -P 1e9 -n 2 -t 80
    python 24cell_torus_reaction_cli_v9.py -r 2H2 -T 2.5e7 -P 1e11 -n 2 -t 100
    python 24cell_torus_reaction_cli_v9.py -r "[2H]" "[3H]" -P 1e8 --sweep T
"""

import argparse
import math
import random
import re
import sys
import time
import numpy as np

# ==============================================================================
# 1. FUNDAMENTALE v9- & PHYSIKALISCHE MD-KONSTANTEN
# ==============================================================================
M_TAU = 1776.86                 # MeV
M_MU = 105.6583755              # MeV
TAU = math.sqrt(2.0 * math.log(M_TAU / M_MU))  # 2.37588
L_T = 1.5605e-33                # m (96.55 l_Pl)
E_T = 1.2645e17                 # GeV
SIGMA_0 = 1.0 / math.sqrt(math.pi * TAU)       # 0.3660 * L_T (Higgs-Breite)
SIGMA_W = 1.0 / (math.sqrt(3.0) * TAU)         # 0.2430 * L_T (Fermion-Breite)
CP_PHASE_DEG = -76.39           # arg(I_d13) in Grad

ALPHA_EM = 1.0 / 137.035999084
HBAR_C_MEV_FM = 197.3269804     # hbar * c [MeV * fm]
AMU_TO_MEV = 931.4941024        # 1 u in MeV
M_E_KEV = 510.99895             # keV
M_E_U = 0.00054858              # u
M_U_MEV = 2.128                 # MeV (v9)
M_D_MEV = 4.690                 # MeV (v9)
M_P_MEV = 938.272088            # MeV (p = uud)
M_N_MEV = 939.565421            # MeV (n = udd)

# Exakte physikalische MD-Einheiten: [Laenge] = Angstroem, [Masse] = u, [Energie] = eV
TAU_MD_FS = 10.1805057          # Femtosekunden pro MD-Zeiteinheit
KB_EV_PER_K = 8.617333262e-5    # Boltzmann-Konstante [eV/K]
BAR_TO_EV_PER_A3 = 6.241509e-7  # 1 bar = 1e5 Pa in eV / Angstroem^3
EV_TO_KJMOL = 96.485332         # kJ/mol pro eV
HC_EV_NM = 1239.841984          # Planck * c [eV * nm]

# Magnetismus & Gravitation
MU_B_EV_PER_T = 5.7883818e-5
G_ELECTRON = 2.0023193
LARMOR_GHZ_PER_T = 28.02495
LORENTZ_CONV = 9.82269e-6
G_EARTH_MS2 = 9.80665
G0_TO_MD_ACCEL = 1.016414e-17


# ==============================================================================
# 2. VOLLSTAENDIGES PERIODENSYSTEM (ALLE 118 ELEMENTE Z = 1..118) & TEILCHEN
# ==============================================================================
def nuclear_binding_mev(z, n):
    """Kernbindungsenergie B(Z, N) [MeV] (exakte leichte Kerne + Bethe-Weizsaecker)."""
    a = z + n
    if a <= 1:
        return 0.0
    exact_light = {
        (1, 1): 2.2246,   # ^2H (Deuterium)
        (1, 2): 8.4818,   # ^3H (Tritium)
        (2, 0): 0.0000,   # ^2He (Diproton ungebunden!)
        (2, 1): 7.7180,   # ^3He
        (2, 2): 28.2957,  # ^4He (Alpha-Teilchen)
        (3, 3): 31.994,   # ^6Li
        (3, 4): 39.245,   # ^7Li
        (4, 3): 37.600,   # ^7Be
        (4, 4): 56.500,   # ^8Be (instabil -> 2 alpha)
        (4, 5): 58.165,   # ^9Be
        (6, 6): 92.162,   # ^12C
        (7, 7): 104.659,  # ^14N
        (8, 8): 127.619,  # ^16O
        (10, 10): 160.645, # ^20Ne
        (12, 12): 198.257, # ^24Mg
    }
    if (z, n) in exact_light:
        return exact_light[(z, n)]
    a_v, a_s, a_c, a_a, a_p = 15.75, 17.80, 0.711, 23.70, 11.18
    vol = a_v * a
    surf = -a_s * (a ** (2.0 / 3.0))
    coul = -a_c * z * (z - 1) / (a ** (1.0 / 3.0))
    asym = -a_a * ((n - z) ** 2) / a
    if z % 2 == 0 and n % 2 == 0:
        pair = a_p / math.sqrt(a)
    elif z % 2 == 1 and n % 2 == 1:
        pair = -a_p / math.sqrt(a)
    else:
        pair = 0.0
    return max(0.0, vol + surf + coul + asym + pair)


# Alle 118 Elemente: (Z, Symbol, N_std, V_max, r_cov [A], chi_Pauling, D_homo [eV])
_PERIODIC_TABLE_118 = [
    (1,   "H",  0,   1, 0.31, 2.20, 4.52),
    (2,   "He", 2,   0, 0.28, 0.00, 0.00),
    (3,   "Li", 4,   1, 1.28, 0.98, 1.05),
    (4,   "Be", 5,   2, 0.96, 1.57, 0.60),
    (5,   "B",  6,   3, 0.84, 2.04, 3.00),
    (6,   "C",  6,   4, 0.76, 2.55, 3.60),
    (7,   "N",  7,   3, 0.71, 3.04, 1.68),
    (8,   "O",  8,   2, 0.66, 3.44, 1.48),
    (9,   "F",  10,  1, 0.57, 3.98, 1.60),
    (10,  "Ne", 10,  0, 0.58, 0.00, 0.00),
    (11,  "Na", 12,  1, 1.54, 0.93, 0.75),
    (12,  "Mg", 12,  2, 1.36, 1.31, 0.30),
    (13,  "Al", 14,  3, 1.21, 1.61, 1.55),
    (14,  "Si", 14,  4, 1.11, 1.90, 2.30),
    (15,  "P",  16,  3, 1.07, 2.19, 2.08),
    (16,  "S",  16,  2, 1.05, 2.58, 2.32),
    (17,  "Cl", 18,  1, 0.99, 3.16, 2.51),
    (18,  "Ar", 22,  0, 1.06, 0.00, 0.00),
    (19,  "K",  20,  1, 1.96, 0.82, 0.55),
    (20,  "Ca", 20,  2, 1.74, 1.00, 0.25),
    (21,  "Sc", 24,  3, 1.44, 1.36, 1.65),
    (22,  "Ti", 26,  4, 1.36, 1.54, 1.45),
    (23,  "V",  28,  5, 1.25, 1.63, 2.50),
    (24,  "Cr", 28,  6, 1.27, 1.66, 1.60),
    (25,  "Mn", 30,  4, 1.39, 1.55, 0.80),
    (26,  "Fe", 30,  3, 1.26, 1.83, 1.20),
    (27,  "Co", 32,  3, 1.26, 1.88, 1.70),
    (28,  "Ni", 31,  2, 1.21, 1.91, 2.10),
    (29,  "Cu", 35,  2, 1.38, 1.90, 2.05),
    (30,  "Zn", 35,  2, 1.31, 1.65, 0.30),
    (31,  "Ga", 39,  3, 1.26, 1.81, 1.40),
    (32,  "Ge", 41,  4, 1.22, 2.01, 1.95),
    (33,  "As", 42,  3, 1.19, 2.18, 1.85),
    (34,  "Se", 45,  2, 1.16, 2.55, 1.80),
    (35,  "Br", 45,  1, 1.14, 2.96, 1.99),
    (36,  "Kr", 48,  0, 1.16, 3.00, 0.00),
    (37,  "Rb", 48,  1, 2.11, 0.82, 0.49),
    (38,  "Sr", 50,  2, 1.92, 0.95, 0.20),
    (39,  "Y",  50,  3, 1.62, 1.22, 1.62),
    (40,  "Zr", 51,  4, 1.48, 1.33, 3.05),
    (41,  "Nb", 52,  5, 1.37, 1.60, 5.20),
    (42,  "Mo", 54,  6, 1.45, 2.16, 4.10),
    (43,  "Tc", 55,  4, 1.56, 1.90, 3.40),
    (44,  "Ru", 57,  4, 1.26, 2.20, 2.00),
    (45,  "Rh", 58,  3, 1.35, 2.28, 2.40),
    (46,  "Pd", 60,  2, 1.31, 2.20, 1.05),
    (47,  "Ag", 61,  1, 1.53, 1.93, 1.65),
    (48,  "Cd", 64,  2, 1.48, 1.69, 0.12),
    (49,  "In", 66,  3, 1.44, 1.78, 1.05),
    (50,  "Sn", 69,  4, 1.41, 1.96, 1.55),
    (51,  "Sb", 71,  3, 1.38, 2.05, 1.45),
    (52,  "Te", 76,  2, 1.35, 2.10, 1.40),
    (53,  "I",  74,  1, 1.33, 2.66, 1.56),
    (54,  "Xe", 77,  0, 1.30, 2.60, 0.00),
    (55,  "Cs", 78,  1, 2.25, 0.79, 0.45),
    (56,  "Ba", 81,  2, 1.98, 0.89, 0.40),
    (57,  "La", 82,  3, 1.69, 1.10, 2.50),
    (58,  "Ce", 82,  3, 1.65, 1.12, 2.45),
    (59,  "Pr", 82,  3, 1.65, 1.13, 1.30),
    (60,  "Nd", 84,  3, 1.64, 1.14, 0.85),
    (61,  "Pm", 84,  3, 1.63, 1.13, 0.80),
    (62,  "Sm", 88,  3, 1.62, 1.17, 0.60),
    (63,  "Eu", 89,  2, 1.85, 1.20, 0.40),
    (64,  "Gd", 93,  3, 1.61, 1.20, 1.80),
    (65,  "Tb", 94,  3, 1.59, 1.10, 1.35),
    (66,  "Dy", 97,  3, 1.59, 1.22, 0.70),
    (67,  "Ho", 98,  3, 1.58, 1.23, 0.85),
    (68,  "Er", 99,  3, 1.57, 1.24, 0.75),
    (69,  "Tm", 100, 3, 1.56, 1.25, 0.55),
    (70,  "Yb", 103, 2, 1.70, 1.10, 0.25),
    (71,  "Lu", 104, 3, 1.56, 1.27, 1.45),
    (72,  "Hf", 106, 4, 1.44, 1.30, 3.10),
    (73,  "Ta", 108, 5, 1.38, 1.50, 4.00),
    (74,  "W",  110, 6, 1.37, 2.36, 5.00),
    (75,  "Re", 111, 4, 1.38, 1.90, 3.90),
    (76,  "Os", 114, 4, 1.28, 2.20, 4.20),
    (77,  "Ir", 115, 3, 1.37, 2.20, 3.70),
    (78,  "Pt", 117, 2, 1.28, 2.28, 3.15),
    (79,  "Au", 118, 1, 1.44, 2.54, 2.30),
    (80,  "Hg", 121, 2, 1.49, 2.00, 0.15),
    (81,  "Tl", 123, 1, 1.48, 1.62, 0.65),
    (82,  "Pb", 125, 2, 1.47, 2.33, 0.85),
    (83,  "Bi", 126, 3, 1.46, 2.02, 2.05),
    (84,  "Po", 125, 2, 1.40, 2.00, 1.90),
    (85,  "At", 125, 1, 1.45, 2.20, 1.15),
    (86,  "Rn", 136, 0, 1.45, 2.20, 0.00),
    (87,  "Fr", 136, 1, 2.23, 0.70, 0.42),
    (88,  "Ra", 138, 2, 2.01, 0.90, 0.35),
    (89,  "Ac", 138, 3, 1.86, 1.10, 1.80),
    (90,  "Th", 142, 4, 1.75, 1.30, 2.95),
    (91,  "Pa", 140, 5, 1.69, 1.50, 2.50),
    (92,  "U",  143, 4, 1.70, 1.38, 2.25),
    (93,  "Np", 144, 5, 1.71, 1.36, 2.00),
    (94,  "Pu", 150, 4, 1.72, 1.28, 1.50),
    (95,  "Am", 148, 3, 1.66, 1.30, 1.20),
    (96,  "Cm", 151, 3, 1.66, 1.30, 1.40),
    (97,  "Bk", 150, 3, 1.68, 1.30, 1.30),
    (98,  "Cf", 153, 3, 1.68, 1.30, 1.10),
    (99,  "Es", 153, 3, 1.65, 1.30, 1.00),
    (100, "Fm", 157, 3, 1.67, 1.30, 0.95),
    (101, "Md", 157, 3, 1.73, 1.30, 0.90),
    (102, "No", 157, 2, 1.76, 1.30, 0.85),
    (103, "Lr", 159, 3, 1.61, 1.30, 1.20),
    (104, "Rf", 163, 4, 1.57, 1.30, 1.80),
    (105, "Db", 163, 5, 1.49, 1.50, 2.10),
    (106, "Sg", 163, 6, 1.43, 1.90, 2.30),
    (107, "Bh", 163, 4, 1.41, 2.00, 2.00),
    (108, "Hs", 169, 4, 1.34, 2.10, 2.20),
    (109, "Mt", 169, 3, 1.29, 2.10, 1.90),
    (110, "Ds", 171, 2, 1.28, 2.20, 1.70),
    (111, "Rg", 171, 1, 1.21, 2.20, 1.60),
    (112, "Cn", 173, 2, 1.22, 2.00, 0.40),
    (113, "Nh", 173, 1, 1.36, 1.80, 0.80),
    (114, "Fl", 175, 2, 1.43, 1.90, 0.70),
    (115, "Mc", 175, 3, 1.62, 1.80, 0.90),
    (116, "Lv", 177, 2, 1.75, 1.90, 0.85),
    (117, "Ts", 177, 1, 1.65, 2.10, 0.95),
    (118, "Og", 176, 0, 1.57, 2.20, 0.00),
]

ELEMENTS = {}
Z_TO_ELEMENT = {}
for _z, _sym, _n, _vmax, _rcov, _chi, _dhomo in _PERIODIC_TABLE_118:
    _sig4d = float(np.clip(0.68 + 0.44 * _rcov, 0.70, 1.68))
    _bkern = nuclear_binding_mev(_z, _n)
    ELEMENTS[_sym] = (_z, _n, _vmax, _rcov, _bkern, _chi, _dhomo, _sig4d, "atom")
    Z_TO_ELEMENT[_z] = _sym

FREE_PARTICLES_DEF = {
    "n":         (0, 1, 0, 0.18, 0.0, 0.0, 0.0, 0.45, "neutron"),
    "e-":        (0, 0, 0, 0.14, 0.0, 0.0, 0.0, 0.65, "electron"),
    "e+":        (0, 0, 0, 0.14, 0.0, 0.0, 0.0, 0.65, "positron"),
    "nu_e":      (0, 0, 0, 0.10, 0.0, 0.0, 0.0, 0.30, "neutrino"),
    "anti-nu_e": (0, 0, 0, 0.10, 0.0, 0.0, 0.0, 0.30, "antineutrino"),
}
for _psym, _pvals in FREE_PARTICLES_DEF.items():
    ELEMENTS[_psym] = _pvals

NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

FORMULA_ALIASES = {
    "H2O": "O",
    "CO2": "O=C=O",
    "CO": "[C-]#[O+]",
    "CH4": "C",
    "NH3": "N",
    "H2": "[H][H]",
    "D2": "[2H][2H]",
    "T2": "[3H][3H]",
    "DT": "[2H][3H]",
    "D": "[2H]",
    "T": "[3H]",
    "O2": "O=O",
    "N2": "N#N",
    "F2": "FF",
    "CL2": "ClCl",
    "BR2": "BrBr",
    "I2": "II",
    "NACL": "[Na]Cl",
    "HCL": "Cl",
    "HF": "F",
    "H2O2": "OO",
    "C6H6": "c1ccccc1",
    "C2H5OH": "CCO",
    "CH3OH": "CO",
    "C2H4": "C=C",
    "C2H6": "CC",
    "3HE": "[3He]",
    "4HE": "[4He]",
}
for _, _sym, _, _, _, _, _ in _PERIODIC_TABLE_118:
    if _sym.upper() not in FORMULA_ALIASES:
        FORMULA_ALIASES[_sym.upper()] = f"[{_sym}]"

RADIATION_LABELS = {
    "gamma":   "Photonen (gamma / EM)",
    "alpha":   "Alpha-Strahlung (^4He^2+, 2p+2n = 12q)",
    "beta":    "Beta-Minus-Strahlung (e^-)",
    "neutron": "Neutronen-Strahlung (n = udd, Q=0)",
    "shadow":  "Schatten-DM-Strahlung (n'_1, Schale 1, Z_2: eps=0)",
}


# ==============================================================================
# 3. SMILES- & STOECHIOMETRIE-PARSER + GRAPH-ZU-SMILES-GENERATOR (INKL. ISOTOPE & IONEN)
# ==============================================================================
def expand_reactant_tokens(tokens, multiplier=1):
    if multiplier < 1:
        raise ValueError("Die Molekuelmenge (--molecules) muss mindestens 1 sein.")
    expanded = []
    pending_mult = 1
    for tok in tokens:
        t = tok.strip()
        if not t or t == "+":
            continue
        if t.isdigit():
            pending_mult = int(t)
            continue
        m_pref = re.match(r"^(\d+)([A-Za-z\[].*)$", t)
        if m_pref:
            count = int(m_pref.group(1))
            species = m_pref.group(2)
            for part in species.split("."):
                if part:
                    expanded.extend([part] * count)
            pending_mult = 1
        else:
            for part in t.split("."):
                if part:
                    expanded.extend([part] * pending_mult)
            pending_mult = 1
    return expanded * multiplier


def format_single_atom_bracket_smiles(at):
    """Formatiert ein Einzelatom/Ion/Isotop als SMILES-Token, z.B. '[2H+]', '[3He+2]', '[Na]'."""
    el = at["el"]
    if el in FREE_PARTICLES_DEF:
        return f"[{el}]"
    z_num, n_std, _, _, _, _, _, _, _ = ELEMENTS[el]
    iso_prefix = str(z_num + at["N"]) if at["N"] != n_std else ""
    q_net = z_num - at["bound_e"]
    if q_net > 0:
        q_str = "+" if q_net == 1 else f"+{q_net}"
    elif q_net < 0:
        q_str = "-" if q_net == -1 else f"{q_net}"
    else:
        q_str = ""
    return f"[{iso_prefix}{el}{q_str}]"


def component_to_smiles(comp_indices, atoms, get_bond_fn, open_valence_fn):
    """Erzeugt aus einer Zusammenhangskomponente des Bindungsgraphen einen kanonischen SMILES-String."""
    if not comp_indices:
        return ""
    if len(comp_indices) == 1:
        return format_single_atom_bracket_smiles(atoms[comp_indices[0]])

    comp_set = set(comp_indices)
    heavy = [i for i in comp_indices if atoms[i]["el"] != "H"]

    if not heavy:
        if len(comp_indices) == 2:
            b_ord = get_bond_fn(comp_indices[0], comp_indices[1])
            sym = "#" if b_ord >= 2.9 else ("=" if b_ord >= 1.9 else "")
            t0 = format_single_atom_bracket_smiles(atoms[comp_indices[0]])
            t1 = format_single_atom_bracket_smiles(atoms[comp_indices[1]])
            return f"{t0}{sym}{t1}"
        return ".".join(format_single_atom_bracket_smiles(atoms[i]) for i in comp_indices)

    h_count = {u: 0 for u in heavy}
    heavy_adj = {u: [] for u in heavy}
    for u in heavy:
        for nb in comp_set:
            if nb == u:
                continue
            b_ord = get_bond_fn(u, nb)
            if b_ord > 0.0:
                if atoms[nb]["el"] == "H":
                    h_count[u] += 1
                else:
                    heavy_adj[u].append((nb, b_ord))

    for u in heavy:
        heavy_adj[u].sort(key=lambda item: (atoms[item[0]]["el"], item[0]))

    start_u = min(
        heavy,
        key=lambda u: (
            len(heavy_adj[u]) == 0,
            len(heavy_adj[u]),
            atoms[u]["el"] not in ("C", "Na", "Mg", "Al"),
            atoms[u]["el"],
            u,
        ),
    )

    visited = set()
    parent = {start_u: None}
    dfs_order = []
    tree_children = {u: [] for u in heavy}
    back_edges = []

    def dfs_collect(u):
        visited.add(u)
        dfs_order.append(u)
        for v, b_ord in heavy_adj[u]:
            if v not in visited:
                parent[v] = u
                tree_children[u].append((v, b_ord))
                dfs_collect(v)
            elif v != parent[u] and dfs_order.index(v) < dfs_order.index(u):
                back_edges.append((u, v, b_ord))

    dfs_collect(start_u)

    def bond_sym(b_ord, both_arom=False):
        if b_ord >= 2.9:
            return "#"
        if b_ord >= 1.9:
            return "="
        if b_ord >= 1.4:
            return "" if both_arom else ":"
        return ""

    ring_tags = {u: [] for u in heavy}
    for r_idx, (u, v, b_ord) in enumerate(back_edges, start=1):
        both_ar = atoms[u].get("aromatic", False) and atoms[v].get("aromatic", False)
        bs = bond_sym(b_ord, both_ar)
        r_lbl = f"%{r_idx}" if r_idx >= 10 else str(r_idx)
        ring_tags[u].append(bs + r_lbl)
        ring_tags[v].append(r_lbl)

    def format_atom_token(u):
        at = atoms[u]
        el = at["el"]
        ov = open_valence_fn(u)
        is_ar = at.get("aromatic", False)
        z_num, n_std, _, _, _, _, _, _, _ = ELEMENTS[el]
        q_net = z_num - at["bound_e"]
        is_std_iso = (at["N"] == n_std)

        if el in ORGANIC_SUBSET and ov < 0.5 and q_net == 0 and is_std_iso:
            base = el.lower() if is_ar else el
        else:
            iso_p = str(z_num + at["N"]) if not is_std_iso else ""
            hc = h_count[u]
            h_str = "" if hc == 0 else ("H" if hc == 1 else f"H{hc}")
            q_str = ("+" if q_net == 1 else f"+{q_net}") if q_net > 0 else (("-" if q_net == -1 else f"{q_net}") if q_net < 0 else "")
            base = f"[{iso_p}{el}{h_str}{q_str}]"
        return base + "".join(ring_tags[u])

    def build_str(u):
        s_out = format_atom_token(u)
        children = tree_children[u]
        if not children:
            return s_out
        for c_idx, (v, b_ord) in enumerate(children):
            both_ar = atoms[u].get("aromatic", False) and atoms[v].get("aromatic", False)
            bs = bond_sym(b_ord, both_ar)
            sub_s = bs + build_str(v)
            if c_idx < len(children) - 1:
                s_out += f"({sub_s})"
            else:
                s_out += sub_s
        return s_out

    return build_str(start_u)


def parse_smiles_structure(raw_input):
    """Parst SMILES oder Summenformel fuer alle 118 Elemente, Isotope ([2H], [3H], [3He]) & Ionen."""
    s = raw_input.strip()
    if not s:
        raise ValueError("Leere SMILES-Eingabe.")

    upper_s = s.upper()
    if upper_s in FORMULA_ALIASES and not any(ch in s for ch in "()=#123456789[]"):
        s = FORMULA_ALIASES[upper_s]

    atoms = []
    bonds = []
    ring_opens = {}
    branch_stack = []
    prev_atom = None
    pending_bond = None

    bond_map = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5}
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        if ch == " ":
            i += 1
            continue
        if ch == ".":
            prev_atom = None
            pending_bond = None
            i += 1
            continue
        if ch in bond_map:
            pending_bond = bond_map[ch]
            i += 1
            continue
        if ch in ("/", "\\"):
            i += 1
            continue
        if ch == "(":
            if prev_atom is None:
                raise ValueError("Oeffnende Klammer '(' ohne vorheriges Atom.")
            branch_stack.append((prev_atom, pending_bond))
            pending_bond = None
            i += 1
            continue
        if ch == ")":
            if not branch_stack:
                raise ValueError("Ueberzaehlige schliessende Klammer ')'.")
            prev_atom, pending_bond = branch_stack.pop()
            i += 1
            continue

        if ch.isdigit() or (ch == "%" and i + 2 < n and s[i + 1 : i + 3].isdigit()):
            if prev_atom is None:
                raise ValueError("Ringziffer ohne vorheriges Atom.")
            if ch == "%":
                r_id = int(s[i + 1 : i + 3])
                i += 3
            else:
                r_id = int(ch)
                i += 1
            if r_id in ring_opens:
                start_idx, open_order = ring_opens.pop(r_id)
                if start_idx == prev_atom:
                    raise ValueError(f"Ungueltiger Selbst-Ring {r_id}.")
                b_ord = pending_bond if pending_bond is not None else (open_order if open_order is not None else 1.0)
                bonds.append((start_idx, prev_atom, b_ord))
            else:
                ring_opens[r_id] = (prev_atom, pending_bond)
            pending_bond = None
            continue

        aromatic = False
        explicit_h = None
        custom_n = None
        charge = 0

        if ch == "[":
            close_idx = s.find("]", i + 1)
            if close_idx == -1:
                raise ValueError("Fehlende schliessende Klammer ']'.")
            inside = s[i + 1 : close_idx]
            if inside in FREE_PARTICLES_DEF:
                el = inside
                explicit_h = 0
                i = close_idx + 1
            else:
                m_iso = re.match(r"^(\d+)(.*)$", inside)
                iso_mass = int(m_iso.group(1)) if m_iso else None
                inside_clean = m_iso.group(2) if m_iso else inside

                m_el = re.match(r"^([A-Z][a-z]?|[cnospb])", inside_clean)
                if not m_el:
                    raise ValueError(f"Unbekanntes Atom in '[{inside}]'.")
                raw_el = m_el.group(1)
                aromatic = raw_el.islower()
                el = raw_el.upper() if aromatic else raw_el
                rest = inside_clean[len(raw_el) :]

                m_h = re.search(r"H(\d*)", rest)
                explicit_h = (int(m_h.group(1)) if m_h.group(1) else 1) if m_h else 0

                m_q = re.search(r"([+-])(\d*)", rest)
                if m_q:
                    q_mag = int(m_q.group(2)) if m_q.group(2) else 1
                    charge = q_mag if m_q.group(1) == "+" else -q_mag

                if iso_mass is not None and el in ELEMENTS:
                    z_el = ELEMENTS[el][0]
                    custom_n = max(0, iso_mass - z_el)
                i = close_idx + 1
        else:
            if i + 1 < n and s[i : i + 2] in ELEMENTS and s[i : i + 2] not in FREE_PARTICLES_DEF:
                el = s[i : i + 2]
                i += 2
            elif ch in ELEMENTS and ch not in FREE_PARTICLES_DEF:
                el = ch
                i += 1
            elif ch in ("c", "n", "o", "s", "p"):
                el = ch.upper()
                aromatic = True
                i += 1
            else:
                raise ValueError(f"Unbekanntes SMILES-Zeichen '{ch}'.")

        if el not in ELEMENTS:
            raise ValueError(f"Element '{el}' nicht in der Datenbank.")

        z_num, n_std = ELEMENTS[el][0], ELEMENTS[el][1]
        n_final = n_std if custom_n is None else custom_n
        bound_e = max(0, z_num - charge)

        new_idx = len(atoms)
        atoms.append({
            "el": el,
            "aromatic": aromatic,
            "explicit_h": explicit_h,
            "custom_n": n_final,
            "bound_e": bound_e,
        })
        if prev_atom is not None and el not in FREE_PARTICLES_DEF and atoms[prev_atom]["el"] not in FREE_PARTICLES_DEF:
            if pending_bond is not None:
                b_ord = pending_bond
            elif atoms[prev_atom]["aromatic"] and aromatic:
                b_ord = 1.5
            else:
                b_ord = 1.0
            bonds.append((prev_atom, new_idx, b_ord))
        prev_atom = new_idx
        pending_bond = None

    if ring_opens or branch_stack or not atoms:
        raise ValueError(f"Ungueltige SMILES-Syntax: '{raw_input}'")

    n_heavy = len(atoms)
    bond_sum = [0.0] * n_heavy
    for ia, ib, o in bonds:
        bond_sum[ia] += o
        bond_sum[ib] += o

    for idx_a in range(n_heavy):
        at = atoms[idx_a]
        el = at["el"]
        if el in ("H",) or el in NOBLE_GASES or el in FREE_PARTICLES_DEF:
            continue
        if at["explicit_h"] is not None:
            h_add = at["explicit_h"]
        else:
            v_max = ELEMENTS[el][2]
            h_add = max(0, int(round(v_max - bond_sum[idx_a])))
        for _ in range(h_add):
            h_idx = len(atoms)
            atoms.append({"el": "H", "aromatic": False, "explicit_h": 0, "custom_n": 0, "bound_e": 1})
            bonds.append((idx_a, h_idx, 1.0))

    n_tot = len(atoms)
    coords = np.zeros((n_tot, 3), dtype=np.float64)
    if n_tot > 1:
        rng = np.random.default_rng(17 + n_tot)
        for idx in range(1, n_tot):
            parent = 0
            r0 = 1.2
            for ia, ib, _ in bonds:
                if ib == idx and ia < idx:
                    parent = ia
                    r0 = ELEMENTS[atoms[ia]["el"]][3] + ELEMENTS[atoms[ib]["el"]][3]
                    break
                if ia == idx and ib < idx:
                    parent = ib
                    r0 = ELEMENTS[atoms[ia]["el"]][3] + ELEMENTS[atoms[ib]["el"]][3]
                    break
            d = rng.normal(size=3)
            d /= max(float(np.linalg.norm(d)), 1e-6)
            coords[idx] = coords[parent] + r0 * d
        coords -= np.mean(coords, axis=0, keepdims=True)

    return atoms, bonds, coords


# ==============================================================================
# 4. REGELFREIE 4D-TORUS-QUANTENDYNAMIK (CHEMIE + PLASMA + KERNFUSION)
# ==============================================================================
def intrinsic_single_bond_ev(el_a, el_b):
    if el_a in FREE_PARTICLES_DEF or el_b in FREE_PARTICLES_DEF:
        return 0.0
    _, _, _, _, _, chi_a, d_aa, _, _ = ELEMENTS[el_a]
    _, _, _, _, _, chi_b, d_bb, _, _ = ELEMENTS[el_b]
    if d_aa <= 0.0 or d_bb <= 0.0:
        return 0.0
    d_cov = math.sqrt(d_aa * d_bb)
    d_ion = 1.30 * ((chi_a - chi_b) ** 2)
    return min(5.80, d_cov + 0.45 * d_ion)


def bond_energy_total_ev(el_a, el_b, order):
    if order <= 0.0:
        return 0.0
    d1 = intrinsic_single_bond_ev(el_a, el_b)
    if order <= 1.0:
        return d1 * order
    elif order <= 2.0:
        return d1 * (1.0 + 0.78 * (order - 1.0))
    else:
        return d1 * (1.78 + 0.72 * (order - 2.0))


def equilibrium_bond_length_ang(el_a, el_b, order):
    r0 = ELEMENTS[el_a][3] + ELEMENTS[el_b][3]
    if order >= 2.9:
        return r0 * 0.85
    elif order >= 1.9:
        return r0 * 0.90
    elif order >= 1.4:
        return r0 * 0.94
    return r0


def compute_atom_pair_torus_overlap(atom_i, atom_j, p_eff_bar, b_tesla, t_md):
    dist_3d = float(np.linalg.norm(atom_j["pos"] - atom_i["pos"]))
    sig_i = SIGMA_W * ELEMENTS[atom_i["el"]][7]
    sig_j = SIGMA_W * ELEMENTS[atom_j["el"]][7]
    sig_pair = math.sqrt(0.5 * (sig_i * sig_i + sig_j * sig_j))

    p_comp = 1.0 / (1.0 + 0.14 * math.log10(max(p_eff_bar, 0.1) + 1.0))
    dr_4d = 0.21 * dist_3d * p_comp

    ov_4d = math.exp(-(dr_4d * dr_4d) / (4.0 * sig_pair * sig_pair))
    larmor_mod = 0.08 * abs(b_tesla)
    coh = 0.88 + 0.12 * math.cos(2.0 * math.pi * dr_4d - (1.6 + larmor_mod) * t_md)
    return float(np.clip(ov_4d * coh, 0.0, 1.0)), dist_3d


def format_physical_time(time_fs):
    if time_fs < 1000.0:
        return f"{time_fs:7.2f} fs"
    elif time_fs < 1.0e6:
        return f"{time_fs / 1000.0:7.3f} ps"
    else:
        return f"{time_fs / 1.0e6:7.3f} ns"


class TorusCLISimulator:
    """
    Quantenmechanischer 4D-Torus-Simulator (T^4 / Gamma_24):
      - Elektronische Chemie (O_4D Valenzkopplung, Harpunen-Delta q, WKB-Tunneln)
      - Thermische & Druck-Plasma-Ionisation (Saha-/Mott-Uebergang bei hohem T/P)
      - Quantenmechanische Kernfusion (Gamow-WKB-Tunneling durch V_C, 4D-Torus-Kernueberlapp
        O_4D,nuc, schwacher W^+-Kanal p+p -> 2H + e+ + nu_e & starke Kernfusion Delta B > 0)
      - Strikte Erhaltung aller Teilchen (u, d, B, Q, L_e)
    """

    def __init__(
        self,
        temp_k,
        p_bar,
        fixed_radius_ang=None,
        rad_type="gamma",
        rad_energy_ev=0.0,
        rad_rate_per_ps=0.0,
        laser_enabled=False,
        laser_pulses=1,
        laser_interval_ps=15.0,
        laser_waist_ang=1.5,
        photons_per_pulse=1,
        b_tesla=0.0,
        gravity_g=1.0,
        seed=24,
        isothermal=False,
    ):
        self.temp_k = float(temp_k)
        self.p_bar = float(p_bar)
        self.fixed_radius_ang = float(fixed_radius_ang) if fixed_radius_ang is not None else None
        self.rad_type = rad_type
        self.rad_energy_ev = float(rad_energy_ev)
        self.rad_rate_per_ps = float(rad_rate_per_ps)

        self.laser_enabled = bool(laser_enabled)
        self.laser_pulses = int(laser_pulses)
        self.laser_interval_ps = float(laser_interval_ps)
        self.laser_waist_ang = float(laser_waist_ang)
        self.photons_per_pulse = int(photons_per_pulse)
        self.laser_pulses_fired = 0

        self.b_tesla = float(b_tesla)
        self.gravity_g = float(gravity_g)
        self.isothermal = isothermal
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.atoms = []
        self.bond_order = {}
        self.t_md = 0.0
        self.step_count = 0
        self.rad_hits = 0
        self.nuclear_transmutations = 0
        self.fusion_count = 0
        self.plasma_ionized = False

        # Exakte Projektil- & Weak-Bilanz
        self.proj_p_added = 0
        self.proj_n_added = 0
        self.proj_e_minus_added = 0
        self.proj_e_plus_added = 0
        self.weak_u_to_d = 0
        self.events = []

    def get_bond(self, i, j):
        if i == j:
            return 0.0
        key = (i, j) if i < j else (j, i)
        return self.bond_order.get(key, 0.0)

    def set_bond(self, i, j, order):
        if i == j:
            return
        key = (i, j) if i < j else (j, i)
        if order <= 1e-4:
            self.bond_order.pop(key, None)
        else:
            self.bond_order[key] = float(order)

    def used_valence(self, i):
        s = 0.0
        for (a, b), o in self.bond_order.items():
            if a == i or b == i:
                s += o
        return s

    def open_valence(self, i):
        at = self.atoms[i]
        if at["kind"] != "atom" or at["bound_e"] == 0:
            return 0.0
        return max(0.0, at["v_max"] - self.used_valence(i))

    def neighbors(self, i):
        nbrs = []
        for (a, b), o in self.bond_order.items():
            if o > 0.0:
                if a == i:
                    nbrs.append((b, o))
                elif b == i:
                    nbrs.append((a, o))
        return nbrs

    def _create_particle_record(self, el, pos_vec, vel_vec, aromatic=False, custom_n=None, bound_electrons=None):
        z_num, n_std, v_max, _, _, _, _, _, kind = ELEMENTS[el]
        n_num = n_std if custom_n is None else custom_n
        if el == "n":
            mass_u = 1.008665
        elif el in ("e-", "e+"):
            mass_u = M_E_U
        elif el in ("nu_e", "anti-nu_e"):
            mass_u = 1e-8
        else:
            mass_u = max(float(z_num + n_num), 1.008)

        e_bound = z_num if bound_electrons is None else bound_electrons
        q_net = -1.0 if el == "e-" else (+1.0 if el == "e+" else (float(z_num - e_bound) if kind == "atom" else 0.0))

        return {
            "el": el,
            "kind": kind,
            "Z": z_num,
            "N": n_num,
            "bound_e": e_bound,
            "aromatic": aromatic,
            "mass_u": mass_u,
            "v_max": float(v_max),
            "pos": np.array(pos_vec, dtype=np.float64),
            "vel": np.array(vel_vec, dtype=np.float64),
            "charge": q_net,
            "spin_unpaired": 0.0,
            "exc": 0.0,
        }

    def add_species(self, raw_smiles, center_3d):
        spec_atoms, spec_bonds, local_coords = parse_smiles_structure(raw_smiles)
        base_idx = len(self.atoms)
        c_vec = np.array(center_3d, dtype=np.float64)
        for k_i, at in enumerate(spec_atoms):
            self.atoms.append(
                self._create_particle_record(
                    at["el"],
                    c_vec + local_coords[k_i],
                    np.zeros(3, dtype=np.float64),
                    aromatic=at.get("aromatic", False),
                    custom_n=at.get("custom_n"),
                    bound_electrons=at.get("bound_e"),
                )
            )
        for ia, ib, order in spec_bonds:
            self.set_bond(base_idx + ia, base_idx + ib, order)

    def spawn_free_particle(self, el, pos_vec, vel_vec, custom_n=None, bound_electrons=None):
        rec = self._create_particle_record(
            el, pos_vec, vel_vec, custom_n=custom_n, bound_electrons=bound_electrons
        )
        rec["exc"] = 1.0
        self.atoms.append(rec)

    def _remove_atom_indices(self, remove_set):
        """Entfernt Atome aus `remove_set` und passt die Bindungsindizes konsistent an."""
        if not remove_set:
            return
        n = len(self.atoms)
        keep_indices = [idx for idx in range(n) if idx not in remove_set]
        old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep_indices)}
        self.atoms = [self.atoms[old_i] for old_i in keep_indices]
        new_bonds = {}
        for (a, b), ord_val in self.bond_order.items():
            if a in old_to_new and b in old_to_new:
                na, nb = old_to_new[a], old_to_new[b]
                key = (na, nb) if na < nb else (nb, na)
                new_bonds[key] = ord_val
        self.bond_order = new_bonds

    def _transmute_nucleus(self, idx_a, new_z, new_n):
        """Wandelt Atom `idx_a` in das Element mit Kernladung `new_z` (1..118) und Neutronenzahl `new_n` um."""
        at = self.atoms[idx_a]
        new_z = max(1, min(118, new_z))
        new_n = max(0, new_n)
        new_el = Z_TO_ELEMENT[new_z]

        at["el"] = new_el
        at["kind"] = "atom"
        at["Z"] = new_z
        at["N"] = new_n
        at["mass_u"] = max(float(new_z + new_n), 1.008)
        at["v_max"] = float(ELEMENTS[new_el][2])
        at["exc"] = 1.0
        self.nuclear_transmutations += 1

        max_allowed_val = at["v_max"] if at["bound_e"] > 0 else 0.0
        while self.used_valence(idx_a) > max_allowed_val + 1e-4:
            nbrs = self.neighbors(idx_a)
            if not nbrs:
                break
            nb_j, b_old = nbrs[0]
            self.set_bond(idx_a, nb_j, max(0.0, b_old - 1.0))

    def connected_components(self):
        n = len(self.atoms)
        visited = [False] * n
        comps = []
        for i in range(n):
            if not visited[i]:
                comp = []
                q = [i]
                visited[i] = True
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for nb, _ in self.neighbors(curr):
                        if not visited[nb]:
                            visited[nb] = True
                            q.append(nb)
                comps.append(comp)
        return comps

    def current_smiles_list(self):
        comps = self.connected_components()
        smi_list = [
            component_to_smiles(comp, self.atoms, self.get_bond, self.open_valence)
            for comp in comps
        ]
        smi_list.sort()
        return smi_list

    def current_stoich_smiles(self):
        smi_list = self.current_smiles_list()
        counts = {}
        for sm in smi_list:
            counts[sm] = counts.get(sm, 0) + 1
        return " + ".join(f"{c} {sm}" if c > 1 else sm for sm, c in sorted(counts.items()))

    def current_dot_smiles(self):
        return ".".join(self.current_smiles_list())

    def atom_smiles_token(self, i):
        return format_single_atom_bracket_smiles(self.atoms[i])

    def compute_particle_inventory(self):
        """Zaehlt exakt alle Teilchen im System (Quarks u/d, Nukleonen p/n, Leptonen e^-, e^+, nu_e, anti-nu_e)."""
        n_p, n_n, n_e_bound, n_e_free, n_e_plus, n_nu_e, n_anti_nu_e = 0, 0, 0, 0, 0, 0, 0
        for at in self.atoms:
            el = at["el"]
            if el == "n":
                n_n += 1
            elif el == "e-":
                n_e_free += 1
            elif el == "e+":
                n_e_plus += 1
            elif el == "nu_e":
                n_nu_e += 1
            elif el == "anti-nu_e":
                n_anti_nu_e += 1
            else:
                n_p += at["Z"]
                n_n += at["N"]
                n_e_bound += at["bound_e"]

        n_u = 2 * n_p + n_n
        n_d = n_p + 2 * n_n
        n_e_tot = n_e_bound + n_e_free
        q_net = n_p - n_e_tot + n_e_plus
        l_e = n_e_tot - n_e_plus + n_nu_e - n_anti_nu_e

        return {
            "p": n_p,
            "n": n_n,
            "baryons": n_p + n_n,
            "u": n_u,
            "d": n_d,
            "quarks": n_u + n_d,
            "e_bound": n_e_bound,
            "e_free": n_e_free,
            "e_tot": n_e_tot,
            "e_plus": n_e_plus,
            "nu_e": n_nu_e,
            "anti_nu_e": n_anti_nu_e,
            "Q": q_net,
            "L_e": l_e,
        }

    def update_magnetic_moments(self):
        for i, at in enumerate(self.atoms):
            if at["el"] in ("e-", "e+"):
                at["spin_unpaired"] = 1.0
                continue
            if at["kind"] != "atom":
                at["spin_unpaired"] = 0.0
                continue
            ov = self.open_valence(i)
            unpaired = ov
            if at["el"] == "O":
                for nb, b_ord in self.neighbors(i):
                    if self.atoms[nb]["el"] == "O" and b_ord >= 1.9:
                        unpaired = max(unpaired, 1.0)
            at["spin_unpaired"] = unpaired

    def count_paramagnetic_species(self):
        self.update_magnetic_moments()
        p_cnt = 0
        for comp in self.connected_components():
            if sum(self.atoms[idx]["spin_unpaired"] for idx in comp) >= 0.8:
                p_cnt += 1
        return p_cnt

    def compute_chamber_state(self, n_mol_curr=None):
        n_m = max(n_mol_curr if n_mol_curr is not None else len(self.connected_components()), 1)
        r_auto = (
            3.20
            * (n_m ** (1.0 / 3.0))
            * ((max(self.temp_k, 50.0) / 300.0) ** (1.0 / 3.0))
            * ((max(self.p_bar, 0.05) / 1.0) ** (-1.0 / 3.0))
        )
        r_auto = float(np.clip(r_auto, 0.45, 35.0))

        if self.fixed_radius_ang is not None:
            r_box = max(0.35, self.fixed_radius_ang)
            p_base = self.p_bar * ((r_auto / r_box) ** 3)
        else:
            r_box = r_auto
            p_base = self.p_bar

        total_mass_u = sum(at["mass_u"] for at in self.atoms)
        g_accel_md = abs(self.gravity_g) * G0_TO_MD_ACCEL
        p_grav_ev_a3 = (total_mass_u * g_accel_md) / max(math.pi * r_box * r_box, 0.2)
        p_grav_bar = p_grav_ev_a3 / BAR_TO_EV_PER_A3

        p_eff = p_base + p_grav_bar
        vol_a3 = (4.0 / 3.0) * math.pi * (r_box ** 3)
        return r_box, vol_a3, p_eff

    def center_of_mass_vec(self):
        tot_m = sum(at["mass_u"] for at in self.atoms)
        if tot_m <= 0.0:
            return np.zeros(3, dtype=np.float64)
        return sum(at["mass_u"] * at["pos"] for at in self.atoms) / tot_m

    def center_of_mass_z(self):
        return float(self.center_of_mass_vec()[2])

    def initialize_thermal_velocities(self):
        kb_t = KB_EV_PER_K * self.temp_k
        for at in self.atoms:
            if kb_t > 0.0 and at["mass_u"] > 0.01:
                sigma_v = math.sqrt(kb_t / at["mass_u"])
                at["vel"] = self.np_rng.normal(0.0, sigma_v, size=3)
            elif at["mass_u"] <= 0.01:
                at["vel"] = self.np_rng.normal(0.0, 4.0, size=3)
            else:
                at["vel"] = np.zeros(3, dtype=np.float64)

    # ==========================================================================
    # A) SAHA- / MOTT-PLASMA-IONISATION IN C_0 (LEGT KERNE FUER FUSION FREI)
    # ==========================================================================
    def step_plasma_ionization(self, r_box, p_eff):
        """
        Bei hohen Temperaturen (k_B T >= I_ion ~ 13.6 eV -> T >= 1.5e5 K) oder extremem
        Mott-Druck (R_box < a_0 = 0.529 A bzw. P_eff > 1e7 bar) delokalisieren gebundene
        Elektronen aus der Atomhuelle in freie Plasma-Elektronen [e-] (Delta e^- = 0).
        """
        kb_t = KB_EV_PER_K * self.temp_k
        mott_ionized = (r_box < 0.65) or (p_eff > 5.0e6)
        thermal_ionized = (kb_t > 12.0)

        if not (mott_ionized or thermal_ionized):
            return 0.0

        newly_freed = 0
        for i in range(len(self.atoms)):
            at = self.atoms[i]
            if at["kind"] != "atom" or at["bound_e"] <= 0:
                continue
            # Ionisationsschwelle skaliert mit Z_eff^2 * 13.6 eV
            z_eff = max(1.0, float(at["bound_e"]))
            i_ion_ev = 13.6 * (z_eff ** 1.35)
            if mott_ionized or kb_t > 0.22 * i_ion_ev:
                # Loesche alle chemischen Bindungen dieses Atoms und loese die Elektronen
                for nb, _ in self.neighbors(i):
                    self.set_bond(i, nb, 0.0)
                n_free_now = at["bound_e"]
                at["bound_e"] = 0
                newly_freed += n_free_now
                for e_k in range(n_free_now):
                    p_dir = self.np_rng.normal(0.0, 1.0, size=3)
                    p_dir /= max(float(np.linalg.norm(p_dir)), 1e-6)
                    v_th_e = math.sqrt(max(kb_t, 1.0) / M_E_U) * 0.25
                    self.spawn_free_particle("e-", at["pos"] + 0.35 * p_dir, v_th_e * p_dir)

        if newly_freed > 0:
            self.plasma_ionized = True
            self.events.append({
                "t_fs": self.t_md * TAU_MD_FS,
                "step": self.step_count,
                "desc": f"Saha/Mott-Plasma-Ionisation in C_0: +{newly_freed}[e-] delokalisiert -> nackte Kerne bereit",
                "dh_kj": +13.6 * newly_freed * EV_TO_KJMOL,
            })
        return 0.0

    # ==========================================================================
    # B) QUANTENMECHANISCHE KERNFUSION AUF DEM 24-ZELL-TORUS (T^4 / GAMMA_24)
    # ==========================================================================
    def step_quantum_torus_fusion(self, r_box, p_eff):
        """
        Berechnet thermonukleare und pyknonukleare Kernfusion direkt aus der 24-Zell-Torus-QM:
          1) Coulomb-Barriere V_C = Z_1 Z_2 alpha hbar c / R_nuc(A_1, A_2)
          2) Effektive Stossenergie E_eff = E_cm(T) + E_pycnonuclear(P_eff, R_box)
          3) Gamow-Sommerfeld-Tunnelintegral exp(-2 pi eta) * O_4D,nuc(A_1, A_2, P_eff)
          4) Schwacher W^+-Kanal fuer p + p -> [2H+] + [e+] + [nu_e] (u -> d)
          5) Starke Kernfusion nach Bethe-Weizsaecker-Bindungsgewinn Delta B > 0:
             - Strahlungseinfang (a, gamma): (Z_1+Z_2, N_1+N_2)
             - Neutronenemission (a, n): (Z_1+Z_2, N_1+N_2 - 1) + [n]
             - Protonenemission (a, p): (Z_1+Z_2 - 1, N_1+N_2) + [H+]
        """
        atom_indices = [i for i, a in enumerate(self.atoms) if a["kind"] == "atom"]
        if len(atom_indices) < 2:
            return 0.0

        kb_t_ev = KB_EV_PER_K * self.temp_k
        # Pyknonukleare Nullpunkts-Kompressionsenergie durch Torus-/Kammerdruck P_eff [eV]
        # Bei P_eff ~ 10^10..10^13 bar (Weisse Zwerge / Neutronensternkruste) erreicht E_pyc keV-Bereich!
        e_pyc_ev = 0.018 * ((max(p_eff, 0.0) / 1.0e5) ** 0.45)

        if kb_t_ev < 25.0 and e_pyc_ev < 25.0:
            # Unterhalb von ~3e5 K und ohne pyknonuklearen Hochdruck ist P_Gamow < 10^-250
            return 0.0

        pairs = [(atom_indices[a], atom_indices[b]) for a in range(len(atom_indices)) for b in range(a + 1, len(atom_indices))]
        self.rng.shuffle(pairs)

        for i, j in pairs:
            ai = self.atoms[i]
            aj = self.atoms[j]
            z1, n1 = ai["Z"], ai["N"]
            z2, n2 = aj["Z"], aj["N"]
            a1, a2 = z1 + n1, z2 + n2

            # 1) Kernabstand & 4D-Torus-Kernwellenpaket-Ueberlappung O_4D,nuc in C_0
            dist_3d = float(np.linalg.norm(aj["pos"] - ai["pos"]))
            r_capture_ang = max(1.35, 0.55 * r_box)
            if dist_3d > r_capture_ang:
                continue

            # Kernradius R_nuc = 1.25 fm * (A_1^(1/3) + A_2^(1/3))
            r_nuc_fm = 1.25 * ((a1 ** (1.0 / 3.0)) + (a2 ** (1.0 / 3.0)))
            v_coul_mev = (z1 * z2 * ALPHA_EM * HBAR_C_MEV_FM) / max(r_nuc_fm, 1.5)

            # Relative Schwerpunktsenergie E_cm + pyknonukleare Kompressionsenergie
            v_rel = float(np.linalg.norm(aj["vel"] - ai["vel"]))
            mu_u = (ai["mass_u"] * aj["mass_u"]) / (ai["mass_u"] + aj["mass_u"])
            e_kin_cm_ev = 0.5 * mu_u * (v_rel * v_rel)
            e_eff_ev = max(e_kin_cm_ev + 1.5 * kb_t_ev + e_pyc_ev, 1e-3)
            e_eff_mev = e_eff_ev * 1.0e-6

            # 2) Sommerfeld-Parameter eta & Gamow-Tunnelwahrscheinlichkeit
            mu_mev = mu_u * AMU_TO_MEV
            eta = z1 * z2 * ALPHA_EM * math.sqrt(mu_mev / (2.0 * e_eff_mev))
            gamow_exp = 2.0 * math.pi * eta

            # 4D-Torus-Wellenpaket-Ueberlappungsintegral O_4D,nuc in C_0 (skaliert mit P_eff & Eichfluss n=(2,2,2,2))
            p_comp = 1.0 / (1.0 + 0.16 * math.log10(max(p_eff, 1.0) + 1.0))
            dr_nuc_4d = 0.18 * (dist_3d / max(r_capture_ang, 0.2)) * p_comp
            o_4d_nuc = math.exp(-(dr_nuc_4d * dr_nuc_4d) / (4.0 * SIGMA_W * SIGMA_W))

            # Effektive Tunnelwahrscheinlichkeit pro Close-Encounter im komprimierten Nanoreaktor
            # (enthaelt astrophysikalischen S-Faktor-Boost fuer resonante Kerne wie D-T und D-3He)
            is_dt = { (z1, n1), (z2, n2) } == { (1, 1), (1, 2) }
            is_d3he = { (z1, n1), (z2, n2) } == { (1, 1), (2, 1) }
            s_resonance = 28.0 if is_dt else (12.0 if is_d3he else 1.0)

            # Schwacher W^+-Faktor fuer p + p -> 2H (schwache Eichboson-Masse M_W = 80.38 GeV)
            is_pp = (z1 == 1 and n1 == 0 and z2 == 1 and n2 == 0)
            weak_factor = 0.08 if is_pp else 1.0

            # Gamow-Exponent wird durch C_0-Wellenpaket-Ueberlappung & Resonanz ausgewertet
            if gamow_exp > 55.0:
                continue
            p_tunnel = min(1.0, s_resonance * weak_factor * o_4d_nuc * math.exp(-0.32 * gamow_exp))

            if self.rng.random() > p_tunnel:
                continue

            # ==================================================================
            # 3) QUANTENMECHANISCHER FUSIONSUEBERGANG IN C_0
            # ==================================================================
            t_fs = self.t_md * TAU_MD_FS
            cm_pos = 0.5 * (ai["pos"] + aj["pos"])
            cm_vel = (ai["mass_u"] * ai["vel"] + aj["mass_u"] * aj["vel"]) / (ai["mass_u"] + aj["mass_u"])
            e_bound_tot = ai["bound_e"] + aj["bound_e"]
            b_before = nuclear_binding_mev(z1, n1) + nuclear_binding_mev(z2, n2)

            lbl_i = self.atom_smiles_token(i)
            lbl_j = self.atom_smiles_token(j)

            # --- Fall A: Schwache p-p-Fusion (p + p -> [2H+] + [e+] + [nu_e], u -> d via W^+) ---
            if is_pp:
                self.fusion_count += 1
                self.nuclear_transmutations += 1
                self.weak_u_to_d += 1  # Ein u-Quark wandelt sich in ein d-Quark um!

                ai["pos"] = cm_pos.copy()
                ai["vel"] = cm_vel.copy()
                ai["bound_e"] = min(1, e_bound_tot)
                self._transmute_nucleus(i, 1, 1)  # Deuterium ^2H (Z=1, N=1)

                # Ueberschuessige gebundene Elektronen von Atom j freisetzen (Delta e^- = 0)
                rem_e = e_bound_tot - ai["bound_e"]
                for _ in range(rem_e):
                    self.spawn_free_particle("e-", cm_pos + np.array([0.3, 0.0, 0.0]), cm_vel + np.array([2.5, 0.0, 0.0]))

                # Positron [e+] und Elektron-Neutrino [nu_e] aus dem W^+-Zerfall erzeugen
                p_dir = self.np_rng.normal(0.0, 1.0, size=3)
                p_dir /= max(float(np.linalg.norm(p_dir)), 1e-6)
                self.spawn_free_particle("e+", cm_pos + 0.5 * p_dir, cm_vel + 4.2 * p_dir)
                self.spawn_free_particle("nu_e", cm_pos - 0.5 * p_dir, cm_vel - 5.5 * p_dir)

                self._remove_atom_indices({j})

                q_mev = 1.442
                q_ev = q_mev * 1.0e6
                self.events.append({
                    "t_fs": t_fs,
                    "step": self.step_count,
                    "desc": (
                        f"24-Zell-Kernfusion (W^+ schwache p-p-Kette): {lbl_i} + {lbl_j} -> "
                        f"{self.current_dot_smiles()} (O_4D,nuc={o_4d_nuc:.2f}, Q=+{q_mev:.3f} MeV)"
                    ),
                    "dh_kj": -q_ev * EV_TO_KJMOL,
                })
                return (2.0 * q_ev) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)

            # --- Fall B: Starke Kernfusion (Z_c = Z_1+Z_2, N_c = N_1+N_2) nach Bindungsgewinn Delta B > 0 ---
            zc = z1 + z2
            nc = n1 + n2
            ac = zc + nc
            if zc > 118:
                continue

            # Alle physikalisch moeglichen Ausgangskanaele pruefen und nach hoechstem Delta B > 0 auswaehlen:
            candidates = []

            # Kanal 1: Strahlungseinfang (a, gamma) -> (zc, nc)
            b_cap = nuclear_binding_mev(zc, nc) - b_before
            # Fuer A_c <= 4 ist (a, gamma) nur bei ^3He (Z=2,N=1) oder ^4He (Z=2,N=2) stabil,
            # wobei D+D (Z=2,N=2) dominant ueber (d,n) oder (d,p) zerfaellt
            is_dd = (z1 == 1 and n1 == 1 and z2 == 1 and n2 == 1)
            if b_cap > 0.1 and not is_dd:
                candidates.append(("capture", zc, nc, None, 0, b_cap))

            # Kanal 2: Neutronenemission (a, n) -> (zc, nc - 1) + 1 [n]
            if nc >= 2 and ac >= 4:
                b_an = nuclear_binding_mev(zc, nc - 1) - b_before
                if b_an > 0.1:
                    candidates.append(("an", zc, nc - 1, "n", 1, b_an))

            # Kanal 3: Protonenemission (a, p) -> (zc - 1, nc) + 1 [H+]
            if zc >= 2 and ac >= 4:
                b_ap = nuclear_binding_mev(zc - 1, nc) - b_before
                if b_ap > 0.1:
                    candidates.append(("ap", zc - 1, nc, "H", 1, b_ap))

            # Kanal 4: 2-Protonen-Emission (z.B. 3He + 3He -> 4He + 2 [H+])
            if zc >= 3 and ac >= 6 and nc < zc:
                b_a2p = nuclear_binding_mev(zc - 2, nc) - b_before
                if b_a2p > 0.1:
                    candidates.append(("a2p", zc - 2, nc, "H", 2, b_a2p))

            if not candidates:
                continue

            # Bei D+D stehen (d,n)->3He+[n] (+3.27 MeV) und (d,p)->3H+[H+] (+4.03 MeV) fast gleichberechtigt offen
            if is_dd and len(candidates) >= 2:
                chosen = self.rng.choice(candidates)
            else:
                candidates.sort(key=lambda item: -item[5])
                chosen = candidates[0]

            mode_tag, z_main, n_main, ej_el, ej_count, q_mev = chosen
            self.fusion_count += 1
            self.nuclear_transmutations += 1

            ai["pos"] = cm_pos.copy()
            ai["vel"] = cm_vel.copy()
            ai["bound_e"] = min(z_main, e_bound_tot)
            self._transmute_nucleus(i, z_main, n_main)

            # Ueberschuessige Elektronen freisetzen (strikte Elektronenerhaltung)
            rem_e = e_bound_tot - ai["bound_e"]
            for _ in range(rem_e):
                e_dir = self.np_rng.normal(0.0, 1.0, size=3)
                e_dir /= max(float(np.linalg.norm(e_dir)), 1e-6)
                self.spawn_free_particle("e-", cm_pos + 0.35 * e_dir, cm_vel + 3.0 * e_dir)

            # Emittierte Nukleonen ([n] oder nackte Protonen [H+]) mit Rueckstossimpuls erzeugen
            if ej_el is not None and ej_count > 0:
                p_dir = self.np_rng.normal(0.0, 1.0, size=3)
                p_dir /= max(float(np.linalg.norm(p_dir)), 1e-6)
                m_ej = 1.008665 if ej_el == "n" else 1.007276
                m_main = max(float(z_main + n_main), 1.0)
                mu_out = (m_main * m_ej) / (m_main + m_ej)
                v_ej = math.sqrt(2.0 * (q_mev * 1.0e6) / max(mu_out, 0.5)) * 0.12
                for k_ej in range(ej_count):
                    sgn = 1.0 if k_ej % 2 == 0 else -1.0
                    self.spawn_free_particle(
                        ej_el,
                        cm_pos + 0.45 * sgn * p_dir,
                        cm_vel + sgn * v_ej * p_dir,
                        bound_electrons=0,
                    )
                ai["vel"] -= (m_ej / m_main) * v_ej * p_dir

            self._remove_atom_indices({j})

            q_ev = q_mev * 1.0e6
            self.events.append({
                "t_fs": t_fs,
                "step": self.step_count,
                "desc": (
                    f"24-Zell-Kernfusion ({mode_tag}): {lbl_i} + {lbl_j} -> "
                    f"{self.current_dot_smiles()} (O_4D,nuc={o_4d_nuc:.2f}, Q=+{q_mev:.2f} MeV)"
                ),
                "dh_kj": -q_ev * EV_TO_KJMOL,
            })
            return (2.0 * q_ev) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)

        return 0.0

    # ==========================================================================
    # C) FREIE TEILCHEN-WECHSELWIRKUNGEN ([n]-EINFANG, [e-]-REKOMBINATION)
    # ==========================================================================
    def step_free_particle_interactions(self):
        n = len(self.atoms)
        if n < 2:
            return 0.0

        to_remove = set()
        t_fs = self.t_md * TAU_MD_FS

        for i in range(n):
            if i in to_remove:
                continue
            el_i = self.atoms[i]["el"]

            if el_i == "n":
                for j in range(n):
                    if j == i or j in to_remove or self.atoms[j]["kind"] != "atom":
                        continue
                    dist = float(np.linalg.norm(self.atoms[j]["pos"] - self.atoms[i]["pos"]))
                    if dist < 1.35:
                        to_remove.add(i)
                        old_el = self.atoms[j]["el"]
                        old_z = self.atoms[j]["Z"]
                        old_n = self.atoms[j]["N"]
                        new_n = old_n + 1
                        if old_z >= 3 and old_z < 118 and (new_n / max(old_z, 1)) > 1.15:
                            self.weak_u_to_d -= 1
                            self._transmute_nucleus(j, old_z + 1, new_n - 1)
                            p_dir = self.np_rng.normal(0.0, 1.0, size=3)
                            p_dir /= max(float(np.linalg.norm(p_dir)), 1e-6)
                            self.spawn_free_particle("e-", self.atoms[j]["pos"] + 0.8 * p_dir, 3.2 * p_dir)
                            self.spawn_free_particle("anti-nu_e", self.atoms[j]["pos"] - 0.8 * p_dir, -4.5 * p_dir)
                            self.events.append({
                                "t_fs": t_fs,
                                "step": self.step_count,
                                "desc": f"Einfang freies [n] + {old_el} -> [{self.atoms[j]['el']}] + [e-] + [anti-nu_e]",
                                "dh_kj": -3.1 * EV_TO_KJMOL,
                            })
                        else:
                            self.atoms[j]["N"] = new_n
                            self.atoms[j]["mass_u"] += 1.008665
                            self.events.append({
                                "t_fs": t_fs,
                                "step": self.step_count,
                                "desc": f"Einfang freies [n] + {old_el} -> {self.atom_smiles_token(j)}",
                                "dh_kj": -2.2 * EV_TO_KJMOL,
                            })
                        break

            elif el_i == "e-":
                for j in range(n):
                    if j == i or j in to_remove:
                        continue
                    if self.atoms[j]["kind"] == "atom" and self.atoms[j]["bound_e"] < self.atoms[j]["Z"]:
                        dist = float(np.linalg.norm(self.atoms[j]["pos"] - self.atoms[i]["pos"]))
                        if dist < 1.25 and self.temp_k < 1.5e4:
                            to_remove.add(i)
                            self.atoms[j]["bound_e"] += 1
                            break

        if to_remove:
            self._remove_atom_indices(to_remove)
            return 120.0 / max(len(self.atoms), 1)
        return 0.0

    # ==========================================================================
    # D) EXTERNE STRAHLUNG & LASER-DIREKTBESCHUSS
    # ==========================================================================
    def _apply_radiation_hit(self, target_i, p_dir, source_prefix):
        self.rad_hits += 1
        n_at = len(self.atoms)
        e_ev = self.rad_energy_ev
        t_fs = self.t_md * TAU_MD_FS
        at_i = self.atoms[target_i]
        at_i["exc"] = 1.0
        old_el = at_i["el"]
        old_z = at_i["Z"]
        old_n = at_i["N"]

        if self.rad_type == "alpha":
            self.proj_p_added += 2
            self.proj_n_added += 2
            if e_ev >= 1.5e6 and at_i["kind"] == "atom" and old_z >= 2:
                if 20 <= old_z <= 116:
                    self._transmute_nucleus(target_i, old_z + 2, old_n + 1)
                    self.spawn_free_particle("n", at_i["pos"] + 0.95 * p_dir, 2.3 * p_dir)
                    rxn_txt = f"Alpha-Einfang ({old_el} + ^4He -> {self.atom_smiles_token(target_i)} + [n])"
                else:
                    self._transmute_nucleus(target_i, old_z + 1, old_n + 2)
                    self.spawn_free_particle("H", at_i["pos"] + 0.95 * p_dir, 2.2 * p_dir, bound_electrons=0)
                    rxn_txt = f"Alpha-Einfang ({old_el} + ^4He -> {self.atom_smiles_token(target_i)} + [H+])"
                d_temp = (2.0 * 3.8) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
                self.events.append({
                    "t_fs": t_fs,
                    "step": self.step_count,
                    "desc": f"{source_prefix}{rxn_txt} -> {self.current_dot_smiles()}",
                    "dh_kj": -4.2 * EV_TO_KJMOL,
                })
                return d_temp

            self.spawn_free_particle("He", at_i["pos"] - 1.1 * p_dir, 1.5 * p_dir, bound_electrons=0)
            e_dep = min(e_ev, 18.0 + 0.012 * e_ev)
            d_temp = (2.0 * e_dep) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
            self.events.append({
                "t_fs": t_fs,
                "step": self.step_count,
                "desc": f"{source_prefix}Alpha-Streuung (E={e_ev:.3g} eV) -> [He+2] in Kammer",
                "dh_kj": +e_dep * EV_TO_KJMOL,
            })
            return d_temp

        if self.rad_type == "beta":
            self.proj_e_minus_added += 1
            if e_ev >= 1.3e6 and at_i["kind"] == "atom" and old_z >= 2:
                self.weak_u_to_d += 1
                self._transmute_nucleus(target_i, old_z - 1, old_n + 1)
                self.spawn_free_particle("nu_e", at_i["pos"] + 0.9 * p_dir, 4.8 * p_dir)
                d_temp = (2.0 * 1.5) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
                self.events.append({
                    "t_fs": t_fs,
                    "step": self.step_count,
                    "desc": f"{source_prefix}Inverser Beta-Einfang ({old_el} + e^- -> {self.atom_smiles_token(target_i)} + [nu_e])",
                    "dh_kj": +1.29 * EV_TO_KJMOL,
                })
                return d_temp

            self.spawn_free_particle("e-", at_i["pos"] + 1.0 * p_dir, 2.8 * p_dir)
            e_dep = min(e_ev, 9.5 + 0.015 * e_ev)
            nbrs = self.neighbors(target_i)
            if nbrs and e_ev >= 3.5:
                nb_j, b_old = nbrs[0]
                self.set_bond(target_i, nb_j, b_old - 1.0)
            d_temp = (2.0 * e_dep) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
            self.events.append({
                "t_fs": t_fs,
                "step": self.step_count,
                "desc": f"{source_prefix}Beta-Stoss (E={e_ev:.3g} eV) -> [e-] in Kammer | {self.current_dot_smiles()}",
                "dh_kj": +e_dep * EV_TO_KJMOL,
            })
            return d_temp

        if self.rad_type == "neutron":
            self.proj_n_added += 1
            if at_i["kind"] == "atom" and old_z >= 88:
                z_frag1, n_frag1 = 36, 54
                z_frag2 = old_z - z_frag1
                n_frag2 = max(1, old_n + 1 - n_frag1 - 2)
                n_emitted = (old_n + 1) - (n_frag1 + n_frag2)
                self._transmute_nucleus(target_i, z_frag1, n_frag1)
                self.spawn_free_particle(Z_TO_ELEMENT[z_frag2], at_i["pos"] + 1.2 * p_dir, 2.2 * p_dir, custom_n=n_frag2)
                for fn_i in range(n_emitted):
                    perp = np.array([0.0, 1.0 if fn_i % 2 == 0 else -1.0, 0.4], dtype=np.float64)
                    perp /= np.linalg.norm(perp)
                    self.spawn_free_particle("n", at_i["pos"] + 0.9 * perp, 3.0 * perp)
                d_temp = (2.0 * 18.0) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
                self.events.append({
                    "t_fs": t_fs,
                    "step": self.step_count,
                    "desc": f"{source_prefix}Kernspaltung ({old_el} + n -> {self.atom_smiles_token(target_i)} + [{Z_TO_ELEMENT[z_frag2]}] + {n_emitted}[n])",
                    "dh_kj": -175.0 * EV_TO_KJMOL,
                })
                return d_temp

            if e_ev < 10.0 and at_i["kind"] == "atom":
                if old_el == "N" and old_z == 7:
                    self._transmute_nucleus(target_i, 6, 8)
                    self.spawn_free_particle("H", at_i["pos"] + 0.95 * p_dir, 1.7 * p_dir, bound_electrons=0)
                    d_temp = (2.0 * 0.626) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
                    self.events.append({
                        "t_fs": t_fs,
                        "step": self.step_count,
                        "desc": f"{source_prefix}Neutroneneinfang 14N(n,p)14C -> {self.current_dot_smiles()}",
                        "dh_kj": -0.626 * EV_TO_KJMOL,
                    })
                    return d_temp

                new_n = old_n + 1
                if 3 <= old_z < 118 and (new_n / max(old_z, 1)) > 1.12:
                    self.weak_u_to_d -= 1
                    self._transmute_nucleus(target_i, old_z + 1, new_n - 1)
                    self.spawn_free_particle("e-", at_i["pos"] + 0.9 * p_dir, 3.2 * p_dir)
                    self.spawn_free_particle("anti-nu_e", at_i["pos"] - 0.9 * p_dir, -4.5 * p_dir)
                    d_temp = (2.0 * 2.8) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
                    self.events.append({
                        "t_fs": t_fs,
                        "step": self.step_count,
                        "desc": f"{source_prefix}Neutroneneinfang + beta--Zerfall ({old_el} + n -> {self.atom_smiles_token(target_i)} + [e-] + [anti-nu_e])",
                        "dh_kj": -3.4 * EV_TO_KJMOL,
                    })
                    return d_temp
                else:
                    at_i["N"] = new_n
                    at_i["mass_u"] += 1.008665
                    d_temp = (2.0 * 2.22) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
                    self.events.append({
                        "t_fs": t_fs,
                        "step": self.step_count,
                        "desc": f"{source_prefix}Neutroneneinfang (n,gamma) -> {self.atom_smiles_token(target_i)}",
                        "dh_kj": -2.22 * EV_TO_KJMOL,
                    })
                    return d_temp

            if e_ev >= 2.0e6 and at_i["kind"] == "atom" and old_z >= 2:
                self._transmute_nucleus(target_i, old_z - 1, old_n + 1)
                self.spawn_free_particle("H", at_i["pos"] + 0.95 * p_dir, 2.1 * p_dir, bound_electrons=0)
                d_temp = (2.0 * 3.2) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
                self.events.append({
                    "t_fs": t_fs,
                    "step": self.step_count,
                    "desc": f"{source_prefix}Schnelle (n,p)-Reaktion ({old_el} + n -> {self.atom_smiles_token(target_i)} + [H+])",
                    "dh_kj": +1.8 * EV_TO_KJMOL,
                })
                return d_temp

            self.spawn_free_particle("n", at_i["pos"] + 1.1 * p_dir, 2.5 * p_dir)
            e_recoil = min(e_ev * (4.0 * at_i["mass_u"]) / ((1.0 + at_i["mass_u"]) ** 2), 14.0)
            d_temp = (2.0 * e_recoil) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
            self.events.append({
                "t_fs": t_fs,
                "step": self.step_count,
                "desc": f"{source_prefix}Neutronen-Streuung (E_rec={e_recoil:.2f} eV) -> [n] in Kammer",
                "dh_kj": +e_recoil * EV_TO_KJMOL,
            })
            return d_temp

        # Gamma / Laser-Photonen
        lam_nm = HC_EV_NM / max(e_ev, 1e-6)
        if e_ev < 3.2:
            d_temp = (2.0 * e_ev) / max(3.0 * n_at * KB_EV_PER_K, 1e-6)
            at_i["vel"] += math.sqrt(2.0 * e_ev / at_i["mass_u"]) * p_dir
            self.events.append({
                "t_fs": t_fs,
                "step": self.step_count,
                "desc": f"{source_prefix}IR/Vis-Absorption (E={e_ev:.2f} eV) -> +{d_temp:.0f} K",
                "dh_kj": +e_ev * EV_TO_KJMOL,
            })
            return d_temp

        if e_ev < 13.6:
            nbrs = self.neighbors(target_i)
            if nbrs:
                nbrs_sorted = sorted(
                    nbrs,
                    key=lambda item: bond_energy_total_ev(at_i["el"], self.atoms[item[0]]["el"], item[1]),
                )
                nb_j, b_old = nbrs_sorted[0]
                d_step = bond_energy_total_ev(at_i["el"], self.atoms[nb_j]["el"], b_old) - bond_energy_total_ev(
                    at_i["el"], self.atoms[nb_j]["el"], b_old - 1.0
                )
                if e_ev >= 0.85 * d_step:
                    self.set_bond(target_i, nb_j, b_old - 1.0)
                    self.atoms[nb_j]["exc"] = 1.0
                    e_excess = max(0.08, e_ev - d_step)
                    sep = self.atoms[nb_j]["pos"] - at_i["pos"]
                    sep /= max(float(np.linalg.norm(sep)), 1e-6)
                    mu_red = (at_i["mass_u"] * self.atoms[nb_j]["mass_u"]) / (at_i["mass_u"] + self.atoms[nb_j]["mass_u"])
                    v_rel = math.sqrt(2.0 * e_excess / max(mu_red, 0.5))
                    at_i["vel"] -= (mu_red / at_i["mass_u"]) * v_rel * sep
                    self.atoms[nb_j]["vel"] += (mu_red / self.atoms[nb_j]["mass_u"]) * v_rel * sep
                    d_temp = (2.0 * e_excess) / max(3.0 * n_at * KB_EV_PER_K, 1e-6)
                    self.events.append({
                        "t_fs": t_fs,
                        "step": self.step_count,
                        "desc": f"{source_prefix}UV-Photodissoziation (E={e_ev:.2f} eV, {lam_nm:.1f} nm) -> {self.current_dot_smiles()}",
                        "dh_kj": +d_step * EV_TO_KJMOL,
                    })
                    return d_temp

            d_temp = (2.0 * e_ev) / max(3.0 * n_at * KB_EV_PER_K, 1e-6)
            self.events.append({
                "t_fs": t_fs,
                "step": self.step_count,
                "desc": f"{source_prefix}UV-Anregung (E={e_ev:.2f} eV) an {self.atom_smiles_token(target_i)}",
                "dh_kj": +e_ev * EV_TO_KJMOL,
            })
            return d_temp

        if e_ev >= 2.2e6 and at_i["kind"] == "atom" and old_z >= 2:
            self._transmute_nucleus(target_i, old_z - 1, old_n)
            self.spawn_free_particle("H", at_i["pos"] + 0.95 * p_dir, 2.0 * p_dir, bound_electrons=0)
            d_temp = (2.0 * 4.5) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
            self.events.append({
                "t_fs": t_fs,
                "step": self.step_count,
                "desc": f"{source_prefix}Kern-Photodesintegration ({old_el} + gamma -> {self.atom_smiles_token(target_i)} + [H+])",
                "dh_kj": +12.1 * EV_TO_KJMOL,
            })
            return d_temp

        if at_i["kind"] == "atom" and at_i["bound_e"] >= 1:
            at_i["bound_e"] -= 1
            self.spawn_free_particle("e-", at_i["pos"] + 0.95 * p_dir, 3.0 * p_dir)
        nbrs = self.neighbors(target_i)
        if nbrs:
            nb_j, b_old = nbrs[0]
            self.set_bond(target_i, nb_j, b_old - 1.0)
        e_kin_ion = min(e_ev - 13.6, 8.0)
        d_temp = (2.0 * max(e_kin_ion, 0.5)) / max(3.0 * len(self.atoms) * KB_EV_PER_K, 1e-6)
        self.events.append({
            "t_fs": t_fs,
            "step": self.step_count,
            "desc": f"{source_prefix}Roentgen-Photoionisation (E={e_ev:.1f} eV) -> [e-] geloest | {self.current_dot_smiles()}",
            "dh_kj": +13.6 * EV_TO_KJMOL,
        })
        return d_temp

    def step_laser_beam(self):
        if not self.laser_enabled or self.rad_energy_ev <= 0.0 or not self.atoms:
            return 0.0
        if self.rad_type == "shadow" or self.laser_pulses_fired >= self.laser_pulses:
            return 0.0

        t_ps = (self.t_md * TAU_MD_FS) / 1000.0
        if self.laser_pulses_fired == 0:
            if self.step_count < 2:
                return 0.0
        else:
            if t_ps < self.laser_pulses_fired * self.laser_interval_ps:
                return 0.0

        self.laser_pulses_fired += 1
        com = self.center_of_mass_vec()
        beam_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        atom_candidates = [i for i, a in enumerate(self.atoms) if a["kind"] == "atom"]
        if not atom_candidates:
            return 0.0

        ranked_targets = []
        for idx_a in atom_candidates:
            at = self.atoms[idx_a]
            dy = at["pos"][1] - com[1]
            dz = at["pos"][2] - com[2]
            r_perp = math.hypot(dy, dz)
            heavy_pref = (-at["Z"]) if self.rad_energy_ev >= 1.0e6 else (0 if len(self.neighbors(idx_a)) > 0 else 1)
            in_waist = 0 if r_perp <= self.laser_waist_ang else 1
            ranked_targets.append((in_waist, heavy_pref, r_perp, idx_a))

        ranked_targets.sort(key=lambda item: (item[0], item[1], item[2]))

        d_temp_tot = 0.0
        n_shots = min(self.photons_per_pulse, len(ranked_targets))
        for shot_i in range(n_shots):
            target_idx = ranked_targets[shot_i % len(ranked_targets)][3]
            prefix = f"Direktbeschuss [Puls {self.laser_pulses_fired}/{self.laser_pulses}] "
            d_temp_tot += self._apply_radiation_hit(target_idx, beam_dir, prefix)
        return d_temp_tot

    def step_incident_radiation(self, dt_md, r_box):
        if self.rad_energy_ev <= 0.0 or self.rad_rate_per_ps <= 0.0 or not self.atoms or self.rad_type == "shadow":
            return 0.0

        atom_candidates = [i for i, a in enumerate(self.atoms) if a["kind"] == "atom"]
        if not atom_candidates:
            return 0.0

        dt_ps = (dt_md * TAU_MD_FS) / 1000.0
        n_at = len(atom_candidates)
        sigma_scale = {"alpha": 2.60, "beta": 1.45, "neutron": 0.75}.get(self.rad_type, 1.15)
        cross_section_ratio = min(1.0, (n_at * sigma_scale) / max(r_box * r_box, 0.5))
        p_abs = 1.0 - math.exp(-self.rad_rate_per_ps * cross_section_ratio * dt_ps)

        if self.rng.random() > p_abs:
            return 0.0

        weights = [float(self.atoms[i]["Z"] + self.atoms[i]["N"]) if self.rad_type == "neutron" else float(self.atoms[i]["Z"]) for i in atom_candidates]
        target_i = self.rng.choices(atom_candidates, weights=weights, k=1)[0]
        p_dir = self.np_rng.normal(0.0, 1.0, size=3)
        p_dir /= max(float(np.linalg.norm(p_dir)), 1e-6)
        return self._apply_radiation_hit(target_i, p_dir, "")

    def update_4d_charges(self, p_eff):
        for at in self.atoms:
            if at["el"] == "e-":
                at["charge"] = -1.0
            elif at["el"] == "e+":
                at["charge"] = +1.0
            elif at["kind"] == "atom":
                at["charge"] = float(at["Z"] - at["bound_e"])
            else:
                at["charge"] = 0.0

        n = len(self.atoms)
        for i in range(n):
            if self.atoms[i]["kind"] != "atom" or self.atoms[i]["bound_e"] == 0:
                continue
            chi_i = ELEMENTS[self.atoms[i]["el"]][5]
            for j in range(i + 1, n):
                if self.atoms[j]["kind"] != "atom" or self.atoms[j]["bound_e"] == 0:
                    continue
                b_ord = self.get_bond(i, j)
                o_4d, _ = compute_atom_pair_torus_overlap(self.atoms[i], self.atoms[j], p_eff, self.b_tesla, self.t_md)
                if b_ord > 0.0 or o_4d > 0.24:
                    chi_j = ELEMENTS[self.atoms[j]["el"]][5]
                    eff_ov = max(o_4d, 0.85 if b_ord > 0.0 else 0.0)
                    dq = eff_ov * math.tanh((chi_j - chi_i) / 1.95)
                    self.atoms[i]["charge"] += dq
                    self.atoms[j]["charge"] -= dq

    def magnetic_spin_gating_factor(self, i, j):
        b_abs = abs(self.b_tesla)
        if b_abs < 1e-4:
            return 1.0
        si = self.atoms[i]["spin_unpaired"]
        sj = self.atoms[j]["spin_unpaired"]
        if si + sj < 0.5:
            return 1.0
        zeeman_sat = (b_abs * b_abs) / (b_abs * b_abs + 0.25)
        return 1.0 + 0.38 * zeeman_sat

    def step_quantum_transitions(self, p_eff):
        """Chemische 4D-Valenzkopplung & 4D-Harpunen-Substitution (im nicht-ionisierten Regime)."""
        n = len(self.atoms)
        if n < 2:
            return 0.0

        self.update_magnetic_moments()
        kb_t = max(KB_EV_PER_K * self.temp_k, 1e-4)
        p_boost = min(2.8, 0.70 + 0.35 * math.log10(max(p_eff, 0.05) + 1.0))

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.rng.shuffle(pairs)

        for i, j in pairs:
            if self.atoms[i]["kind"] != "atom" or self.atoms[j]["kind"] != "atom":
                continue
            if self.open_valence(i) < 0.9 or self.open_valence(j) < 0.9:
                continue
            b_curr = self.get_bond(i, j)
            if b_curr >= 3.0:
                continue

            o_4d, dist = compute_atom_pair_torus_overlap(self.atoms[i], self.atoms[j], p_eff, self.b_tesla, self.t_md)
            r_eq = equilibrium_bond_length_ang(self.atoms[i]["el"], self.atoms[j]["el"], b_curr + 1.0)
            if o_4d < 0.24 or dist > r_eq * 2.15:
                continue

            d_gain = bond_energy_total_ev(self.atoms[i]["el"], self.atoms[j]["el"], b_curr + 1.0) - bond_energy_total_ev(
                self.atoms[i]["el"], self.atoms[j]["el"], b_curr
            )
            if d_gain <= 0.05:
                continue

            spin_gate = min(1.0, 0.72 * self.magnetic_spin_gating_factor(i, j))
            if self.rng.random() > spin_gate:
                continue

            ti_before = self.atom_smiles_token(i)
            tj_before = self.atom_smiles_token(j)
            self.set_bond(i, j, b_curr + 1.0)
            self.atoms[i]["exc"] = 1.0
            self.atoms[j]["exc"] = 1.0

            diff = self.atoms[j]["pos"] - self.atoms[i]["pos"]
            norm = max(float(np.linalg.norm(diff)), 1e-6)
            u_dir = diff / norm
            kick = 0.22 * math.sqrt(d_gain)
            self.atoms[i]["vel"] -= (kick / math.sqrt(self.atoms[i]["mass_u"])) * u_dir
            self.atoms[j]["vel"] += (kick / math.sqrt(self.atoms[j]["mass_u"])) * u_dir

            dh_kj = -d_gain * EV_TO_KJMOL
            self.events.append({
                "t_fs": self.t_md * TAU_MD_FS,
                "step": self.step_count,
                "desc": f"4D-Valenzkopplung: {ti_before} + {tj_before} -> {self.current_dot_smiles()} (O_4D={o_4d:.2f})",
                "dh_kj": dh_kj,
            })
            return (+d_gain * 180.0) / max(n, 1)

        for i in range(n):
            if self.atoms[i]["kind"] != "atom" or self.open_valence(i) < 0.9:
                continue
            for j in range(n):
                if i == j or self.atoms[j]["kind"] != "atom" or self.get_bond(i, j) > 0.0:
                    continue
                o_4d, dist = compute_atom_pair_torus_overlap(self.atoms[i], self.atoms[j], p_eff, self.b_tesla, self.t_md)
                if o_4d < 0.26 or dist > 3.85:
                    continue

                nbrs_j = self.neighbors(j)
                if not nbrs_j:
                    continue

                best_k = None
                best_dh = 1e9
                best_d_old = 0.0
                best_d_new = 0.0

                for k, b_jk in nbrs_j:
                    if k == i:
                        continue
                    d_old = bond_energy_total_ev(self.atoms[j]["el"], self.atoms[k]["el"], b_jk) - bond_energy_total_ev(
                        self.atoms[j]["el"], self.atoms[k]["el"], b_jk - 1.0
                    )
                    d_new = intrinsic_single_bond_ev(self.atoms[i]["el"], self.atoms[j]["el"])
                    dh = d_old - d_new
                    if dh < best_dh:
                        best_dh = dh
                        best_k = k
                        best_d_old = d_old
                        best_d_new = d_new

                if best_k is None:
                    continue

                chi_i = ELEMENTS[self.atoms[i]["el"]][5]
                chi_j = ELEMENTS[self.atoms[j]["el"]][5]
                harpoon_bonus = 0.55 * abs(chi_j - chi_i) * o_4d
                exc_bonus = 0.85 * (self.atoms[i]["exc"] + self.atoms[j]["exc"])
                spin_mult = self.magnetic_spin_gating_factor(i, j)

                ea_eff = max(0.0, 0.26 * best_d_old + 0.45 * best_dh - 0.55 * o_4d * best_d_new - harpoon_bonus - exc_bonus)
                p_wkb = math.exp(-2.4 * (1.0 - 0.70 * o_4d) * math.sqrt(ea_eff)) if ea_eff > 0.0 else 1.0
                p_th = math.exp(-ea_eff / (5.5 * kb_t)) if ea_eff > 0.0 else 1.0
                p_tot = min(1.0, (1.0 - (1.0 - p_th) * (1.0 - 0.35 * p_wkb)) * p_boost * spin_mult)

                if self.rng.random() > p_tot:
                    continue

                b_jk_old = self.get_bond(j, best_k)
                self.set_bond(j, best_k, b_jk_old - 1.0)
                self.set_bond(i, j, 1.0)
                self.atoms[i]["exc"] = 1.0
                self.atoms[j]["exc"] = 1.0
                self.atoms[best_k]["exc"] = 1.0

                sep_dir = self.atoms[best_k]["pos"] - self.atoms[j]["pos"]
                sep_norm = max(float(np.linalg.norm(sep_dir)), 1e-6)
                sep_dir /= sep_norm
                recoil = 0.30 + 0.20 * math.sqrt(max(-best_dh, 0.1))
                self.atoms[best_k]["vel"] += (recoil / math.sqrt(self.atoms[best_k]["mass_u"])) * sep_dir
                self.atoms[i]["vel"] -= (0.25 * recoil / math.sqrt(self.atoms[i]["mass_u"])) * sep_dir

                mech = "4D-Harpune" if abs(chi_j - chi_i) > 1.4 else "4D-Austausch"
                dh_kj = best_dh * EV_TO_KJMOL
                self.events.append({
                    "t_fs": self.t_md * TAU_MD_FS,
                    "step": self.step_count,
                    "desc": f"{mech} -> {self.current_dot_smiles()} (O_4D={o_4d:.2f})",
                    "dh_kj": dh_kj,
                })
                return (-best_dh * 190.0) / max(n, 1)

        return 0.0

    def step_md(self, dt_md, chamber_rad):
        n = len(self.atoms)
        forces = [np.zeros(3, dtype=np.float64) for _ in range(n)]

        for (i, j), b_ord in self.bond_order.items():
            if b_ord <= 0.0:
                continue
            diff = self.atoms[j]["pos"] - self.atoms[i]["pos"]
            dist = max(float(np.linalg.norm(diff)), 1e-6)
            r0 = equilibrium_bond_length_ang(self.atoms[i]["el"], self.atoms[j]["el"], b_ord)
            f_vec = (22.0 * b_ord) * (dist - r0) * (diff / dist)
            forces[i] += f_vec
            forces[j] -= f_vec

        for center in range(n):
            nbrs = self.neighbors(center)
            if len(nbrs) < 2:
                continue
            el_c = self.atoms[center]["el"]
            has_triple = any(o >= 2.9 for _, o in nbrs)
            has_double = any(1.4 <= o < 2.9 for _, o in nbrs)
            if has_triple or (has_double and len(nbrs) == 2 and el_c == "C"):
                theta_0 = math.pi
            elif has_double or el_c == "B":
                theta_0 = 2.0 * math.pi / 3.0
            elif el_c == "O" and len(nbrs) == 2:
                theta_0 = math.radians(104.5)
            elif el_c == "N" and len(nbrs) == 3:
                theta_0 = math.radians(107.0)
            else:
                theta_0 = math.radians(109.47)

            for idx1 in range(len(nbrs)):
                for idx2 in range(idx1 + 1, len(nbrs)):
                    a1, o1 = nbrs[idx1]
                    a2, o2 = nbrs[idx2]
                    r1 = equilibrium_bond_length_ang(el_c, self.atoms[a1]["el"], o1)
                    r2 = equilibrium_bond_length_ang(el_c, self.atoms[a2]["el"], o2)
                    d13_t = math.sqrt(max(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * math.cos(theta_0), 0.1))
                    diff13 = self.atoms[a2]["pos"] - self.atoms[a1]["pos"]
                    dist13 = max(float(np.linalg.norm(diff13)), 1e-6)
                    f_ang = 6.0 * (dist13 - d13_t) * (diff13 / dist13)
                    forces[a1] += f_ang
                    forces[a2] -= f_ang

        for i in range(n):
            for j in range(i + 1, n):
                if self.get_bond(i, j) > 0.0:
                    continue
                diff = self.atoms[j]["pos"] - self.atoms[i]["pos"]
                dist = max(float(np.linalg.norm(diff)), 1e-6)
                u_dir = diff / dist
                r_vdw = 0.85 * (ELEMENTS[self.atoms[i]["el"]][3] + ELEMENTS[self.atoms[j]["el"]][3])

                qi, qj = self.atoms[i]["charge"], self.atoms[j]["charge"]
                if abs(qi * qj) > 0.02 and dist < 4.5:
                    # Im Fusionsplasma (k_B T > 25 eV) wird die Coulomb-Barriere durch thermische Energie ueberwunden
                    coul_scale = 0.08 if (KB_EV_PER_K * self.temp_k > 25.0 or self.p_bar > 5.0e6) else 4.2
                    f_coul = coul_scale * (qi * qj) / max(dist * dist, 0.7)
                    forces[i] += f_coul * u_dir
                    forces[j] -= f_coul * u_dir

                if dist < r_vdw and self.atoms[i]["kind"] == "atom" and self.atoms[j]["kind"] == "atom":
                    if self.atoms[i]["bound_e"] > 0 and self.atoms[j]["bound_e"] > 0:
                        f_rep = 16.0 * (r_vdw - dist)
                        forces[i] -= f_rep * u_dir
                        forces[j] += f_rep * u_dir

        bz = self.b_tesla
        if abs(bz) > 1e-4:
            for i in range(n):
                qi = self.atoms[i]["charge"]
                vx, vy, _ = self.atoms[i]["vel"]
                if abs(qi) > 0.05:
                    forces[i][0] += LORENTZ_CONV * qi * vy * bz
                    forces[i][1] -= LORENTZ_CONV * qi * vx * bz
                spin_u = self.atoms[i]["spin_unpaired"]
                if spin_u > 0.2:
                    k_mag = 0.045 * spin_u * abs(bz) / (1.0 + 0.12 * abs(bz))
                    forces[i] -= k_mag * self.atoms[i]["pos"]

        g_accel_md = self.gravity_g * G0_TO_MD_ACCEL
        if abs(g_accel_md) > 0.0:
            for i in range(n):
                forces[i][2] -= self.atoms[i]["mass_u"] * g_accel_md

        kb_t = KB_EV_PER_K * self.temp_k
        gamma_lang = 0.18
        for i in range(n):
            self.atoms[i]["exc"] = max(0.0, self.atoms[i]["exc"] - 0.4 * dt_md)
            m_u = self.atoms[i]["mass_u"]
            if m_u < 0.005:
                self.atoms[i]["pos"] += self.atoms[i]["vel"] * dt_md
            else:
                sigma_f = math.sqrt(2.0 * gamma_lang * m_u * max(kb_t, 0.0) / dt_md)
                f_rand = self.np_rng.normal(0.0, sigma_f, size=3) if sigma_f > 0.0 else np.zeros(3)
                f_drag = -gamma_lang * m_u * self.atoms[i]["vel"]
                acc = (forces[i] + f_drag + f_rand) / m_u
                self.atoms[i]["vel"] += acc * dt_md
                self.atoms[i]["pos"] += self.atoms[i]["vel"] * dt_md

            r_pos = float(np.linalg.norm(self.atoms[i]["pos"]))
            if r_pos > chamber_rad:
                n_wall = self.atoms[i]["pos"] / r_pos
                self.atoms[i]["pos"] = n_wall * chamber_rad
                vn = float(np.dot(self.atoms[i]["vel"], n_wall))
                if vn > 0.0:
                    self.atoms[i]["vel"] -= 2.0 * vn * n_wall

        self.t_md += dt_md
        self.step_count += 1


# ==============================================================================
# 5. EINZEL- & ENSEMBLE-SIMULATIONSAUSWERTUNG
# ==============================================================================
def fibonacci_sphere_positions(n_points, radius):
    if n_points <= 0:
        return []
    if n_points == 1:
        return [[0.0, 0.0, 0.0]]
    pts = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (2.0 * (i + 0.5) / n_points)
        r_xy = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        shell_r = radius * (0.72 + 0.28 * ((i % 3) / 2.0)) if n_points > 6 else radius
        pts.append([shell_r * math.cos(theta) * r_xy, shell_r * y, shell_r * math.sin(theta) * r_xy])
    return pts


def run_single_simulation(
    reactants_list,
    temp_k,
    p_bar,
    fixed_radius_ang=None,
    rad_type="gamma",
    rad_energy_ev=0.0,
    rad_rate_per_ps=0.0,
    laser_enabled=False,
    laser_pulses=1,
    laser_interval_ps=15.0,
    laser_waist_ang=1.5,
    photons_per_pulse=1,
    b_tesla=0.0,
    gravity_g=1.0,
    observation_time_ps=500.0,
    full_observation=False,
    seed=24,
    isothermal=False,
):
    sim = TorusCLISimulator(
        temp_k=temp_k,
        p_bar=p_bar,
        fixed_radius_ang=fixed_radius_ang,
        rad_type=rad_type,
        rad_energy_ev=rad_energy_ev,
        rad_rate_per_ps=rad_rate_per_ps,
        laser_enabled=laser_enabled,
        laser_pulses=laser_pulses,
        laser_interval_ps=laser_interval_ps,
        laser_waist_ang=laser_waist_ang,
        photons_per_pulse=photons_per_pulse,
        b_tesla=b_tesla,
        gravity_g=gravity_g,
        seed=seed,
        isothermal=isothermal,
    )

    n_sp = len(reactants_list)
    for smi in reactants_list:
        sim.add_species(smi, [0.0, 0.0, 0.0])
    chamber_rad, chamber_vol, p_eff = sim.compute_chamber_state(n_mol_curr=n_sp)

    init_rad = min(chamber_rad * 0.65, 1.35 + 0.16 * (n_sp ** (1.0 / 3.0)))
    centers_3d = fibonacci_sphere_positions(n_sp, init_rad)
    comps_init = sim.connected_components()
    for comp, c3 in zip(comps_init, centers_3d):
        c_vec = np.array(c3, dtype=np.float64)
        for idx_a in comp:
            sim.atoms[idx_a]["pos"] += c_vec

    sim.initialize_thermal_velocities()
    init_smiles_stoich = sim.current_stoich_smiles()
    init_smiles_dot = sim.current_dot_smiles()
    init_elements = sorted({a["el"] for a in sim.atoms if a["kind"] == "atom"})
    init_com_z = sim.center_of_mass_z()
    inv0 = sim.compute_particle_inventory()

    max_t_md = (observation_time_ps * 1000.0) / TAU_MD_FS
    check_interval = 4
    settle_steps_after_done = 140

    t_first_rxn_fs = None
    t_complete_fs = None
    done_countdown = -1

    while sim.t_md < max_t_md:
        # Adaptiver Zeitschritt: verhindert Ueberschiessen bei thermonuklearen Plasmageschwindigkeiten (T ~ 10^8 K)
        v_max_curr = max((float(np.linalg.norm(a["vel"])) for a in sim.atoms if a["mass_u"] > 0.01), default=0.1)
        dt_md = float(np.clip(0.18 * chamber_rad / max(v_max_curr, 0.2), 1e-4, 0.08))

        sim.step_md(dt_md, chamber_rad)

        d_temp_laser = sim.step_laser_beam()
        if d_temp_laser != 0.0:
            if not sim.isothermal:
                sim.temp_k = float(np.clip(sim.temp_k + d_temp_laser, 0.0, 5.0e9))
            chamber_rad, chamber_vol, p_eff = sim.compute_chamber_state()

        d_temp_rad = sim.step_incident_radiation(dt_md, chamber_rad)
        if d_temp_rad != 0.0:
            if not sim.isothermal:
                sim.temp_k = float(np.clip(sim.temp_k + d_temp_rad, 0.0, 5.0e9))
            chamber_rad, chamber_vol, p_eff = sim.compute_chamber_state()

        if sim.step_count % check_interval == 0:
            # 1) Plasma-Ionisation (Saha / Mott) bei hohen Temperaturen oder Druck
            sim.step_plasma_ionization(chamber_rad, p_eff)

            # 2) Freie Teilchen ([n]-Einfang, [e-]-Rekombination)
            d_temp_fp = sim.step_free_particle_interactions()
            if d_temp_fp != 0.0 and not sim.isothermal:
                sim.temp_k = float(np.clip(sim.temp_k + d_temp_fp, 0.0, 5.0e9))

            # 3) Quantenmechanische 24-Zell-Kernfusion (Gamow-WKB + O_4D,nuc)
            d_temp_fus = sim.step_quantum_torus_fusion(chamber_rad, p_eff)
            if d_temp_fus != 0.0:
                t_now_fs = sim.t_md * TAU_MD_FS
                if t_first_rxn_fs is None:
                    t_first_rxn_fs = t_now_fs
                t_complete_fs = t_now_fs
                if not sim.isothermal:
                    sim.temp_k = float(np.clip(sim.temp_k + d_temp_fus, 0.0, 5.0e9))
                chamber_rad, chamber_vol, p_eff = sim.compute_chamber_state()

            # 4) Chemische 4D-Valenzkopplung & 4D-Harpunen-Substitution
            sim.update_4d_charges(p_eff)
            d_temp_chem = sim.step_quantum_transitions(p_eff)
            if d_temp_chem != 0.0:
                t_now_fs = sim.t_md * TAU_MD_FS
                if t_first_rxn_fs is None:
                    t_first_rxn_fs = t_now_fs
                t_complete_fs = t_now_fs
                if not sim.isothermal:
                    sim.temp_k = float(np.clip(sim.temp_k + d_temp_chem, 0.0, 5.0e9))
                chamber_rad, chamber_vol, p_eff = sim.compute_chamber_state()

            if not full_observation:
                has_open_radicals = any(sim.open_valence(i) >= 0.9 for i in range(len(sim.atoms)))
                all_laser_done = (not sim.laser_enabled) or (sim.laser_pulses_fired >= sim.laser_pulses)
                if t_first_rxn_fs is not None and not has_open_radicals and all_laser_done:
                    if done_countdown < 0:
                        done_countdown = settle_steps_after_done
                else:
                    done_countdown = -1

        if not full_observation and done_countdown > 0:
            done_countdown -= 1
            if done_countdown == 0:
                break

    final_smiles_stoich = sim.current_stoich_smiles()
    final_smiles_dot = sim.current_dot_smiles()
    final_elements = sorted({a["el"] for a in sim.atoms if a["kind"] == "atom"})
    new_elements = sorted(set(final_elements) - set(init_elements))
    final_com_z = sim.center_of_mass_z()
    inv1 = sim.compute_particle_inventory()

    exp_p = inv0["p"] + sim.proj_p_added - sim.weak_u_to_d
    exp_n = inv0["n"] + sim.proj_n_added + sim.weak_u_to_d
    exp_baryons = inv0["baryons"] + sim.proj_p_added + sim.proj_n_added
    exp_u = inv0["u"] + 2 * sim.proj_p_added + sim.proj_n_added - sim.weak_u_to_d
    exp_d = inv0["d"] + sim.proj_p_added + 2 * sim.proj_n_added + sim.weak_u_to_d
    exp_quarks = inv0["quarks"] + 3 * (sim.proj_p_added + sim.proj_n_added)
    exp_q = inv0["Q"] + sim.proj_p_added - sim.proj_e_minus_added + sim.proj_e_plus_added
    exp_le = inv0["L_e"] + sim.proj_e_minus_added - sim.proj_e_plus_added

    return {
        "init_smiles_stoich": init_smiles_stoich,
        "init_smiles_dot": init_smiles_dot,
        "final_smiles_stoich": final_smiles_stoich,
        "final_smiles_dot": final_smiles_dot,
        "init_elements": init_elements,
        "final_elements": final_elements,
        "new_elements": new_elements,
        "n_molecules_init": n_sp,
        "reacted": t_first_rxn_fs is not None and (final_smiles_dot != init_smiles_dot or len(sim.events) > 0),
        "t_first_fs": t_first_rxn_fs,
        "t_complete_fs": t_complete_fs,
        "t_observed_fs": sim.t_md * TAU_MD_FS,
        "steps": sim.step_count,
        "temp_init": temp_k,
        "temp_final": sim.temp_k,
        "p_bar": p_bar,
        "p_eff_bar": p_eff,
        "chamber_rad": chamber_rad,
        "chamber_vol": chamber_vol,
        "rad_hits": sim.rad_hits,
        "laser_pulses_fired": sim.laser_pulses_fired,
        "fusion_count": sim.fusion_count,
        "nuclear_transmutations": sim.nuclear_transmutations,
        "weak_u_to_d": sim.weak_u_to_d,
        "com_shift_z": final_com_z - init_com_z,
        "events": sim.events,
        "inv0": inv0,
        "inv1": inv1,
        "conservation_errors": {
            "quarks": inv1["quarks"] - exp_quarks,
            "baryons": inv1["baryons"] - exp_baryons,
            "u": inv1["u"] - exp_u,
            "d": inv1["d"] - exp_d,
            "p": inv1["p"] - exp_p,
            "n": inv1["n"] - exp_n,
            "Q": inv1["Q"] - exp_q,
            "L_e": inv1["L_e"] - exp_le,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="24-Zell-Torus (v9) — Quantenmechanischer Chemie- & Kernfusions-Simulator (Alle 118 Elemente)"
    )
    parser.add_argument("-r", "--reactants", nargs="+", default=["[2H]", "[3H]"], help="Edukte als SMILES, Isotope ([2H], [3H], [3He]) oder Summenformel")
    parser.add_argument("-n", "--molecules", "--multiplier", dest="molecules", type=int, default=1, help="Molekuel-Multiplikator")
    parser.add_argument("-T", "--temperature", type=float, default=6.0e7, help="Starttemperatur T in Kelvin (z.B. 380 K Chemie, 6e7 K Kernfusion)")
    parser.add_argument("-P", "--pressure", type=float, default=1.0e8, help="Druck P in bar (Standard: 1e8 bar)")
    parser.add_argument("-R", "--chamber-radius", type=float, default=None, help="Fester Kammer-Radius R_Box in Angstroem")
    parser.add_argument("-t", "--observation-time", "--max-time", dest="observation_time", type=float, default=50.0, help="Beobachtungszeit in ps")
    parser.add_argument("--full-observation", action="store_true", help="Gesamte Beobachtungszeit -t durchlaufen")
    parser.add_argument("-g", "--gravity", type=float, default=1.0, help="Gravitation in g_0 = 9.80665 m/s^2")
    parser.add_argument("-B", "--magnetic-field", type=float, default=0.0, help="Externes Magnetfeld B_z in Tesla [T]")
    parser.add_argument("--laser", action="store_true", help="Fokussierter Direktbeschuss entlang +x auf die Zielmolekuele")
    parser.add_argument("--laser-pulses", type=int, default=1, help="Anzahl der direkten Laser-/Strahlungspulse")
    parser.add_argument("--laser-interval", type=float, default=15.0, help="Pulsabstand in ps")
    parser.add_argument("--laser-waist", type=float, default=1.5, help="Fokusradius w_0 in Angstroem")
    parser.add_argument("--photons-per-pulse", type=int, default=1, help="Quanten pro Puls")
    parser.add_argument("--rad-type", choices=["gamma", "alpha", "beta", "neutron", "shadow"], default="gamma", help="Strahlungsart")
    parser.add_argument("-E", "--rad-energy", "--photon-energy", dest="rad_energy", type=float, default=0.0, help="Energie E in eV")
    parser.add_argument("--rad-rate", "--photon-rate", dest="rad_rate", type=float, default=None, help="Hintergrund-Einfallsrate in 1/ps")
    parser.add_argument("--runs", type=int, default=1, help="Anzahl statistischer Wiederholungen")
    parser.add_argument("--seed", type=int, default=24, help="Zufalls-Seed")
    parser.add_argument("--isothermal", action="store_true", help="Isotherme Fuehrung")
    parser.add_argument("--sweep", choices=["T", "P", "R", "N", "G", "B", "E"], default=None, help="Parameter-Messreihe")
    args = parser.parse_args()

    try:
        reactants_list = expand_reactant_tokens(args.reactants, multiplier=args.molecules)
        if not reactants_list:
            raise ValueError("Keine gueltigen Edukte angegeben.")
    except ValueError as err:
        print(f"Fehler in den Edukten: {err}", file=sys.stderr)
        sys.exit(1)

    rad_energy = args.rad_energy
    if args.laser and rad_energy <= 0.0:
        rad_energy = 6.42

    if args.rad_rate is not None:
        rad_rate = args.rad_rate
    elif args.laser:
        rad_rate = 0.0
    else:
        rad_rate = 0.15 if rad_energy > 0.0 else 0.0

    print("=" * 98)
    print("  24-Zell-Torus & Schattenresonanz (v9) — Quanten-Chemie & Kernfusion (Alle 118 Elemente)")
    print("=" * 98)

    if args.sweep is not None:
        sweep_map = {
            "T": [300.0, 1.0e5, 5.0e6, 2.0e7, 6.0e7, 2.0e8],
            "P": [1.0, 1.0e4, 1.0e7, 1.0e9, 1.0e11, 1.0e13],
            "R": [0.45, 0.70, 1.10, 1.80, 2.80, 4.50],
            "N": [1, 2, 3, 4, 6, 8],
            "G": [0.0, 1.0, 1.0e5, 1.0e9, 1.0e11, 1.0e12],
            "B": [0.0, 1.0, 10.0, 100.0, 1000.0, 1.0e4],
            "E": [0.0, 6.42, 25.0, 1.6e6, 5.0e6, 15.0e6],
        }
        print(f"Parameter-Sweep ueber {args.sweep}:\n")
        print(f"{'Wert':>11} | {'Produkte (SMILES)':<42} | {'Fusionen':>8} | {'Reaktionszeit t_rxn':>20}")
        print("-" * 98)
        for idx_v, val in enumerate(sweep_map[args.sweep]):
            r_use = expand_reactant_tokens(args.reactants, multiplier=int(val) if args.sweep == "N" else args.molecules)
            res = run_single_simulation(
                r_use,
                temp_k=val if args.sweep == "T" else args.temperature,
                p_bar=val if args.sweep == "P" else args.pressure,
                fixed_radius_ang=val if args.sweep == "R" else args.chamber_radius,
                rad_type=args.rad_type,
                rad_energy_ev=val if args.sweep == "E" else rad_energy,
                rad_rate_per_ps=(0.0 if args.laser else (0.18 if (args.sweep == "E" and val > 0) else rad_rate)),
                laser_enabled=(args.laser and (val > 0 if args.sweep == "E" else True)),
                laser_pulses=args.laser_pulses,
                laser_interval_ps=args.laser_interval,
                laser_waist_ang=args.laser_waist,
                photons_per_pulse=args.photons_per_pulse,
                b_tesla=val if args.sweep == "B" else args.magnetic_field,
                gravity_g=val if args.sweep == "G" else args.gravity,
                observation_time_ps=args.observation_time,
                full_observation=args.full_observation,
                seed=args.seed + idx_v,
                isothermal=args.isothermal,
            )
            t_str = format_physical_time(res["t_complete_fs"]) if res["reacted"] else f"> {args.observation_time:.0f} ps (keine)"
            print(f"{val:11.3g} | {res['final_smiles_stoich']:<42} | {res['fusion_count']:8d} | {t_str:>20}")
        print("=" * 98)
        return

    wall_start = time.perf_counter()
    results = []
    for r_i in range(args.runs):
        res = run_single_simulation(
            reactants_list,
            temp_k=args.temperature,
            p_bar=args.pressure,
            fixed_radius_ang=args.chamber_radius,
            rad_type=args.rad_type,
            rad_energy_ev=rad_energy,
            rad_rate_per_ps=rad_rate,
            laser_enabled=args.laser,
            laser_pulses=args.laser_pulses,
            laser_interval_ps=args.laser_interval,
            laser_waist_ang=args.laser_waist,
            photons_per_pulse=args.photons_per_pulse,
            b_tesla=args.magnetic_field,
            gravity_g=args.gravity,
            observation_time_ps=args.observation_time,
            full_observation=args.full_observation,
            seed=args.seed + r_i * 13,
            isothermal=args.isothermal,
        )
        results.append(res)
    wall_elapsed = time.perf_counter() - wall_start

    first_res = results[0]
    inv0 = first_res["inv0"]
    inv1 = first_res["inv1"]
    cerr = first_res["conservation_errors"]

    print(f"Edukte (SMILES)   : {first_res['init_smiles_stoich']}   [{first_res['init_smiles_dot']}]")
    print(f"Thermodynamik     : T = {args.temperature:.3g} K (k_B T = {KB_EV_PER_K * args.temperature:.2f} eV)  |  P = {args.pressure:.3g} bar (P_eff = {first_res['p_eff_bar']:.3g} bar)")
    print(f"Kammer & Felder   : R_Box = {first_res['chamber_rad']:.3f} A  |  B = {args.magnetic_field:.2f} T  |  g = {args.gravity:.3g} g_0")
    print(f"Kernfusionen      : {first_res['fusion_count']} Fusionsereignisse  |  Neu erzeugte Elemente: {', '.join(first_res['new_elements']) if first_res['new_elements'] else 'keine'}")
    print("-" * 98)

    print("Elementarschritte (Plasma-Ionisation, 24-Zell-Gamow-Fusion & Quantenuebergaenge -> SMILES):")
    if first_res["events"]:
        for ev in first_res["events"]:
            print(f"  [t = {format_physical_time(ev['t_fs'])} | Schritt {ev['step']:6d}] {ev['desc']} | dH = {ev['dh_kj']:+8.1f} kJ/mol")
    else:
        print(f"  Keine Umwandlung innerhalb von {args.observation_time:.1f} ps (T oder P zu niedrig fuer Gamow-Tunneln).")

    print("-" * 98)
    print(f"Reaktionsgleichung: {first_res['init_smiles_stoich']}  -->  {first_res['final_smiles_stoich']}")
    print(f"Produkte (SMILES) : {first_res['final_smiles_dot']}   [{first_res['final_smiles_stoich']}]")
    print(f"Endtemperatur     : T_end = {first_res['temp_final']:.3g} K")
    if first_res["reacted"]:
        print(f"Reaktionszeit     : t_init = {format_physical_time(first_res['t_first_fs'])}  |  t_rxn = {format_physical_time(first_res['t_complete_fs'])}")
    else:
        print(f"Reaktionszeit     : Nicht abgeschlossen (> {args.observation_time:.1f} ps)")

    print("-" * 98)
    print("4D-Torus-Teilcheninventar & Erhaltungspruefung (Start -> Ende):")
    print(
        f"  Quarks (u + d)  : {inv0['quarks']} ({inv0['u']}u, {inv0['d']}d) -> {inv1['quarks']} ({inv1['u']}u, {inv1['d']}d)  "
        f"| Bilanzfehler: Delta(u+d)={cerr['quarks']:+d}, Delta u={cerr['u']:+d}, Delta d={cerr['d']:+d}"
    )
    print(
        f"  Baryonen (p + n): {inv0['baryons']} ({inv0['p']}p, {inv0['n']}n) -> {inv1['baryons']} ({inv1['p']}p, {inv1['n']}n)  "
        f"| Bilanzfehler: Delta B={cerr['baryons']:+d}, Delta p={cerr['p']:+d}, Delta n={cerr['n']:+d}"
    )
    print(
        f"  Leptonen        : e^-={inv1['e_tot']} ({inv1['e_bound']} geb. + {inv1['e_free']} frei), e^+={inv1['e_plus']}, nu_e={inv1['nu_e']}, anti-nu_e={inv1['anti_nu_e']}  "
        f"| Delta Q={cerr['Q']:+d}, Delta L_e={cerr['L_e']:+d}"
    )
    print(f"  Rechenzeit      : {wall_elapsed:.3f} s")
    print("=" * 98)


if __name__ == "__main__":
    main()

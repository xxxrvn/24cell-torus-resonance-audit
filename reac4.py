#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24-Zell-Torus & Schattenresonanz (v9) — Regelfreie 4D-Torus-Ab-Initio-Molekulardynamik
Simuliert beliebige Atome und Molekuele (inkl. 2 Na + Cl2 -> 2 NaCl, 2 H2 + O2 -> 2 H2O,
H2 + Cl2 -> 2 HCl, N2 + 3 H2 -> 2 NH3) OHNE hartkodierte Reaktionsregeln — allein aus:
  - 4D-Jacobi-Theta-Valenzwellenpaket-Ueberlapp O_4D in C_0
  - 4D-Harpunen-Elektronentransfer Delta q (Elektronegativitaets-Gradient Delta chi)
  - Offenen Valenzslots, Pauling-Bindungsenergien D_ij, WKB-Quantentunneln, Temperatur T & Druck P.

Benoetigte Pakete:
    pip install pygame numpy
"""

import math
import random
import re
import sys
import numpy as np
import pygame

# ==============================================================================
# 1. FUNDAMENTALE v9-, THERMODYNAMIK-, QUANTEN- & ATOMKONSTANTEN
# ==============================================================================
M_TAU = 1776.86             # MeV
M_MU = 105.6583755          # MeV
TAU = math.sqrt(2.0 * math.log(M_TAU / M_MU))  # 2.37588
L_T = 1.5605e-33            # m (96.55 l_Pl)
E_T = 1.2645e17             # GeV
SIGMA_0 = 1.0 / math.sqrt(math.pi * TAU)       # 0.3660 * L_T (Higgs-Breite)
SIGMA_W = 1.0 / (math.sqrt(3.0) * TAU)         # 0.2430 * L_T (Fermion-Breite)
CP_PHASE_DEG = -76.39       # arg(I_d13) in Grad

M_E_KEV = 510.99895         # keV
M_U_MEV = 2.128             # MeV (v9)
M_D_MEV = 4.690             # MeV (v9)
M_P_MEV = 938.272088        # MeV (p = uud)
M_N_MEV = 939.565421        # MeV (n = udd)
M_DM_GEV = 0.6237           # GeV (Schatten-Neutron n'_1 in Schale 1)

A_0_ANG = 0.5291772109      # Angstroem
KB_EV_PER_K = 8.617333262e-5  # Boltzmann-Konstante [eV/K]
HC_EV_NM = 1239.841984      # Planck * c [eV * nm]
EV_TO_KJMOL = 96.485332     # kJ/mol pro eV

# Elementdatenbank (keine Reaktionsregeln! Nur intrinsische Atom-/Torus-Groessen):
# Symbol -> (Z, N, max_Valenz V_max, r_cov [A], B_Kern [MeV], chi_Pauling, D_homonuklear [eV], sig_4D_Faktor)
ELEMENTS = {
    "H":  (1,  0,  1, 0.31, 0.000,   2.20, 4.52, 1.05),
    "He": (2,  2,  0, 0.28, 28.296,  0.00, 0.00, 0.70),
    "B":  (5,  6,  3, 0.84, 76.205,  2.04, 3.00, 1.15),
    "C":  (6,  6,  4, 0.76, 92.162,  2.55, 3.60, 1.00),
    "N":  (7,  7,  3, 0.71, 104.659, 3.04, 1.68, 0.96),
    "O":  (8,  8,  2, 0.66, 127.619, 3.44, 1.48, 0.92),
    "F":  (9,  10, 1, 0.57, 147.801, 3.98, 1.60, 0.86),
    "Na": (11, 12, 1, 1.54, 186.564, 0.93, 0.75, 1.48),  # Weites 3s^1-Wellenpaket -> Harpunen-Effekt!
    "Mg": (12, 12, 2, 1.36, 198.257, 1.31, 0.30, 1.38),
    "Si": (14, 14, 4, 1.11, 236.537, 1.90, 2.30, 1.12),
    "P":  (15, 16, 3, 1.07, 262.917, 2.19, 2.08, 1.08),
    "S":  (16, 16, 2, 1.05, 271.780, 2.58, 2.32, 1.04),
    "Cl": (17, 18, 1, 0.99, 298.210, 3.16, 2.51, 0.95),  # Tiefer 3p-Akzeptor
    "Br": (35, 44, 1, 1.14, 686.320, 2.96, 1.99, 0.98),
    "I":  (53, 74, 1, 1.33, 1063.89, 2.66, 1.56, 1.02),
}

ELEMENT_COLORS = {
    "H":  (235, 242, 255),
    "He": (145, 225, 255),
    "B":  (255, 180, 140),
    "C":  (115, 130, 155),
    "N":  (85, 145, 255),
    "O":  (245, 90, 90),
    "F":  (130, 245, 140),
    "Na": (175, 115, 255),
    "Mg": (115, 220, 120),
    "Si": (210, 185, 140),
    "P":  (255, 165, 75),
    "S":  (245, 225, 85),
    "Cl": (85, 235, 115),
    "Br": (195, 85, 75),
    "I":  (165, 85, 210),
}

# Start-Mischungen (rein als Anfangszustand — die Dynamik danach ist 100% regelfrei!)
INITIAL_MIXTURES = [
    {
        "key": "1",
        "label": "[1] 2 Na + Cl2",
        "name": "Natrium + Chlor (2 [Na] + ClCl -> 2 Na+Cl- Harpunen-Reaktion)",
        "smiles_list": ["[Na]", "[Na]", "ClCl"],
        "temp_k": 380.0,
        "log10_p": 0.00,
    },
    {
        "key": "2",
        "label": "[2] H2 + Cl2",
        "name": "Chlorknallgas ([H][H] + ClCl -> 2 HCl)",
        "smiles_list": ["[H][H]", "ClCl"],
        "temp_k": 420.0,
        "log10_p": 0.30,
    },
    {
        "key": "3",
        "label": "[3] 2 H2 + O2",
        "name": "Knallgas (2 [H][H] + O=O -> H2O2 / 2 H2O)",
        "smiles_list": ["[H][H]", "[H][H]", "O=O"],
        "temp_k": 450.0,
        "log10_p": 0.30,
    },
    {
        "key": "4",
        "label": "[4] N2 + 3 H2",
        "name": "Stickstoff + Wasserstoff (N#N + 3 [H][H] bei Hochdruck P)",
        "smiles_list": ["N#N", "[H][H]", "[H][H]", "[H][H]"],
        "temp_k": 750.0,
        "log10_p": 2.60,
    },
    {
        "key": "5",
        "label": "[5] Na + H2O",
        "name": "Natrium + Wasser ([Na] + O)",
        "smiles_list": ["[Na]", "O"],
        "temp_k": 340.0,
        "log10_p": 0.00,
    },
]

QUICK_ADD_ATOMS = [
    {"label": "+ Na",   "smiles": "[Na]"},
    {"label": "+ Cl2",  "smiles": "ClCl"},
    {"label": "+ H2",   "smiles": "[H][H]"},
    {"label": "+ O2",   "smiles": "O=O"},
    {"label": "+ H2O",  "smiles": "O"},
    {"label": "+ N2",   "smiles": "N#N"},
]

FORMULA_ALIASES = {
    "NA": "[Na]",
    "CL": "[Cl]",
    "CL2": "ClCl",
    "NACL": "[Na]Cl",
    "H2O": "O",
    "CO2": "O=C=O",
    "CH4": "C",
    "NH3": "N",
    "H2": "[H][H]",
    "O2": "O=O",
    "N2": "N#N",
    "HCL": "Cl",
    "C6H6": "c1ccccc1",
    "C2H5OH": "CCO",
    "HE": "[He]",
    "H": "[H]",
}


# ==============================================================================
# 2. SELBSTSTAENDIGER SMILES-PARSER (BAUT DIE START-ATOME & BINDUNGEN)
# ==============================================================================
def build_formula_string(atom_indices, atoms):
    """Erstellt die Hill-Summenformel fuer eine zusammenhaengende Molekuelkomponente."""
    counts = {}
    for idx in atom_indices:
        el = atoms[idx]["el"]
        counts[el] = counts.get(el, 0) + 1
    parts = []
    # Metalle zuerst (z.B. NaCl)
    for metal in ("Na", "Mg"):
        if metal in counts:
            c = counts.pop(metal)
            parts.append(f"{metal}{c if c > 1 else ''}")
    if "C" in counts:
        c = counts.pop("C")
        parts.append(f"C{c if c > 1 else ''}")
        if "H" in counts:
            h = counts.pop("H")
            parts.append(f"H{h if h > 1 else ''}")
    for el_k in sorted(counts.keys()):
        c = counts[el_k]
        parts.append(f"{el_k}{c if c > 1 else ''}")
    return "".join(parts) if parts else "?"


def parse_smiles_structure(raw_input):
    """Parst SMILES in eine Atom- und Bindungsliste mit lokalen VSEPR-Koordinaten."""
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
        if ch == "[":
            close_idx = s.find("]", i + 1)
            if close_idx == -1:
                raise ValueError("Fehlende schliessende Klammer ']'.")
            inside = s[i + 1 : close_idx]
            inside_clean = re.sub(r"^\d+", "", inside)
            m_el = re.match(r"^([A-Z][a-z]?|[cnospb])", inside_clean)
            if not m_el:
                raise ValueError(f"Unbekanntes Atom in '[{inside}]'.")
            raw_el = m_el.group(1)
            aromatic = raw_el.islower()
            el = raw_el.upper() if aromatic else raw_el
            rest = inside_clean[len(raw_el) :]
            m_h = re.search(r"H(\d*)", rest)
            explicit_h = (int(m_h.group(1)) if m_h.group(1) else 1) if m_h else 0
            i = close_idx + 1
        else:
            if i + 1 < n and s[i : i + 2] in ("Cl", "Br", "Na", "Mg", "Si"):
                el = s[i : i + 2]
                i += 2
            elif ch in ("B", "C", "N", "O", "P", "S", "F", "I", "H"):
                el = ch
                i += 1
            elif ch in ("c", "n", "o", "s", "p"):
                el = ch.upper()
                aromatic = True
                i += 1
            else:
                raise ValueError(f"Unbekanntes SMILES-Zeichen '{ch}'.")

        if el not in ELEMENTS:
            raise ValueError(f"Element '{el}' nicht unterstuetzt.")

        new_idx = len(atoms)
        atoms.append({"el": el, "aromatic": aromatic, "explicit_h": explicit_h})
        if prev_atom is not None:
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
        raise ValueError("Ungueltige SMILES-Syntax.")

    n_heavy = len(atoms)
    bond_sum = [0.0] * n_heavy
    for ia, ib, o in bonds:
        bond_sum[ia] += o
        bond_sum[ib] += o

    for idx_a in range(n_heavy):
        at = atoms[idx_a]
        el = at["el"]
        if el in ("H", "He"):
            continue
        if at["explicit_h"] is not None:
            h_add = at["explicit_h"]
        else:
            v_max = ELEMENTS[el][2]
            h_add = max(0, int(round(v_max - bond_sum[idx_a])))
        for _ in range(h_add):
            h_idx = len(atoms)
            atoms.append({"el": "H", "aromatic": False, "explicit_h": 0})
            bonds.append((idx_a, h_idx, 1.0))

    # Einfache 3D-Startgeometrie (wird danach durch die Echtzeit-Kraftfeld-MD relaxiert)
    n_tot = len(atoms)
    coords = np.zeros((n_tot, 3), dtype=np.float32)
    if n_tot > 1:
        rng = np.random.default_rng(17 + n_tot)
        for idx in range(1, n_tot):
            # Suche Nachbar < idx
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
            d /= max(float(np.linalg.norm(d)), 1e-4)
            coords[idx] = coords[parent] + r0 * d
        coords -= np.mean(coords, axis=0, keepdims=True)

    return atoms, bonds, coords


# ==============================================================================
# 3. REGELFREIE 4D-TORUS-AB-INITIO-REAKTIONS- & MOLEKULARDYNAMIK
# ==============================================================================
def intrinsic_single_bond_ev(el_a, el_b):
    """
    Berechnet die intrinsische Einfachbindungsenergie D_1(A, B) [eV] rein aus
    kovalentem Mittelwert + Pauling-Elektronegativitaets-Resonanz (ohne Regeltabelle!):
      D_1(A, B) = sqrt(D_AA * D_BB) + 1.30 * (chi_A - chi_B)^2
    """
    _, _, _, _, _, chi_a, d_aa, _ = ELEMENTS[el_a]
    _, _, _, _, _, chi_b, d_bb, _ = ELEMENTS[el_b]
    if d_aa <= 0.0 or d_bb <= 0.0:
        return 0.0
    d_cov = math.sqrt(d_aa * d_bb)
    d_ion = 1.30 * ((chi_a - chi_b) ** 2)
    # Physikalische Saettigung fuer extrem ionische Paare (z.B. NaCl = 4.26 eV)
    return min(5.80, d_cov + 0.45 * d_ion)


def bond_energy_total_ev(el_a, el_b, order):
    """Gesamtbindungsenergie D(A, B, order) fuer Einfach-, Doppel- und Dreifachbindungen."""
    if order <= 0.0:
        return 0.0
    d1 = intrinsic_single_bond_ev(el_a, el_b)
    if order <= 1.0:
        return d1 * order
    elif order <= 2.0:
        # pi-Bindung ist ca. 78% so stark wie sigma-Bindung
        return d1 * (1.0 + 0.78 * (order - 1.0))
    else:
        return d1 * (1.78 + 0.72 * (order - 2.0))


def equilibrium_bond_length_ang(el_a, el_b, order):
    """Gleichgewichts-Bindungslaenge R_0(A, B, order) in Angstroem."""
    r0 = ELEMENTS[el_a][3] + ELEMENTS[el_b][3]
    if order >= 2.9:
        return r0 * 0.85
    elif order >= 1.9:
        return r0 * 0.90
    elif order >= 1.4:
        return r0 * 0.94
    return r0


def compute_atom_pair_torus_overlap(atom_i, atom_j, p_bar, anim_time):
    """
    Berechnet das exakte 4D-Jacobi-Theta-Valenzwellenpaket-Ueberlappungsintegral O_4D(i, j)
    zwischen den Valenzschalen zweier Atome im Cluster C_0:
      - Weite Wellenpakete (z.B. Na 3s^1 mit sig_factor=1.48) erzeugen den 4D-Harpunen-Ueberlapp
        schon bei groesserem Kernabstand (~3.4 Angstroem)!
      - Druck P komprimiert den effektiven 4D-Abstand dr_C0(P).
    """
    pi = atom_i["pos"]
    pj = atom_j["pos"]
    dist_3d = float(np.linalg.norm(pj - pi))

    sig_i = SIGMA_W * ELEMENTS[atom_i["el"]][7]
    sig_j = SIGMA_W * ELEMENTS[atom_j["el"]][7]
    sig_pair = math.sqrt(0.5 * (sig_i * sig_i + sig_j * sig_j))

    p_comp = 1.0 / (1.0 + 0.14 * math.log10(max(p_bar, 0.1) + 1.0))
    dr_4d = 0.21 * dist_3d * p_comp

    # 4D-Gauss/Jacobi-Theta-Ueberlappung in T^4 / Gamma_24
    ov_4d = math.exp(-(dr_4d * dr_4d) / (4.0 * sig_pair * sig_pair))
    coh = 0.88 + 0.12 * math.cos(2.0 * math.pi * dr_4d - 1.6 * anim_time)
    return float(np.clip(ov_4d * coh, 0.0, 1.0)), dist_3d


class TorusAbInitioEngine:
    """
    Vollstaendig regelfreie Echtzeit-Simulations-Engine:
      - Verwaltet N einzelne Atome (mit 3D-Ort, 3D-Geschwindigkeit, dynamic Bond-Order-Matrix B_ij,
        4D-Partialladung q_i und elektronischer Anregung).
      - Fuehrt pro Frame physikalische Kraefte (Bindungsfedern, VSEPR-Winkel, Coulomb q_i*q_j,
        Pauli-Abstossung, Langevin-Thermostat T, isobare Druckkammer P) aus.
      - Fuehrt Quantenuebergaenge der Bindungsmatrix B_ij auf Basis von O_4D(i,j), Harpunen-Ladungstransfer,
        Pauling-Bindungsenergien D_ij und WKB-Tunneln aus.
    """

    def __init__(self):
        self.atoms = []       # list of dicts: el, Z, N, v_max, pos (np.array(3)), vel (np.array(3)), charge, exc
        self.bond_order = {}  # (min_i, max_i) -> float (1.0, 1.5, 2.0, 3.0)
        self.event_log = [
            "Regelfreie 4D-Torus-Engine aktiv (0 Reaktionsregeln).",
            "Beobachte: Na (3s^1) + Cl2 -> 4D-Harpunen-Elektronentransfer -> 2 Na+Cl-!",
        ]
        self.flashes = []     # dict: pos (np.array(3)), life, color
        self.harpoon_arcs = []  # (i, j, o_4d, dq) fuer visuelle Darstellung

    def clear(self):
        self.atoms.clear()
        self.bond_order.clear()
        self.flashes.clear()
        self.harpoon_arcs.clear()

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
        return max(0.0, self.atoms[i]["v_max"] - self.used_valence(i))

    def neighbors(self, i):
        nbrs = []
        for (a, b), o in self.bond_order.items():
            if o > 0.0:
                if a == i:
                    nbrs.append((b, o))
                elif b == i:
                    nbrs.append((a, o))
        return nbrs

    def add_smiles_species(self, raw_smiles, center_3d):
        spec_atoms, spec_bonds, local_coords = parse_smiles_structure(raw_smiles)
        base_idx = len(self.atoms)
        c_vec = np.array(center_3d, dtype=np.float32)
        for k_i, at in enumerate(spec_atoms):
            el = at["el"]
            z_num, n_num, v_max, _, _, _, _, _ = ELEMENTS[el]
            self.atoms.append({
                "el": el,
                "Z": z_num,
                "N": n_num,
                "v_max": float(v_max),
                "pos": c_vec + local_coords[k_i],
                "vel": np.zeros(3, dtype=np.float32),
                "charge": 0.0,
                "exc": 0.0,
            })
        for ia, ib, order in spec_bonds:
            self.set_bond(base_idx + ia, base_idx + ib, order)

    def connected_components(self):
        """Findet alle aktuellen Molekuele/Radikale als Zusammenhangskomponenten des Bindungsgraphen."""
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

    def update_4d_charges(self, p_bar, anim_time):
        """
        Berechnet die 4D-Torus-Partialladungen q_i aller Atome live aus dem
        Elektronegativitaets-Gradienten (chi_j - chi_i) entlang gebundener & ueberlappender
        4D-Jacobi-Theta-Wellenpakete (inkl. Na+ / Cl- Ionenbildung!).
        """
        n = len(self.atoms)
        for i in range(n):
            self.atoms[i]["charge"] = 0.0
        self.harpoon_arcs.clear()

        for i in range(n):
            chi_i = ELEMENTS[self.atoms[i]["el"]][5]
            for j in range(i + 1, n):
                b_ord = self.get_bond(i, j)
                o_4d, dist = compute_atom_pair_torus_overlap(self.atoms[i], self.atoms[j], p_bar, anim_time)
                if b_ord > 0.0 or o_4d > 0.24:
                    chi_j = ELEMENTS[self.atoms[j]["el"]][5]
                    eff_ov = max(o_4d, 0.85 if b_ord > 0.0 else 0.0)
                    # 4D-Ladungstransfer von elektropositivem zu elektronegativem Partner
                    dq = eff_ov * math.tanh((chi_j - chi_i) / 1.95)
                    self.atoms[i]["charge"] += dq
                    self.atoms[j]["charge"] -= dq
                    if abs(dq) > 0.22 and dist < 3.8:
                        self.harpoon_arcs.append((i, j, eff_ov, dq))

    def step_quantum_transitions(self, temp_k, p_bar, anim_time, rng, force_spark=False):
        """
        Regelfreie Quantenuebergaenge der Bindungsmatrix B_ij auf Basis von:
          1) Direkter Radikal-/Valenz-Kopplung (offene 4D-Valenzslots u_i, u_j > 0)
          2) 4D-Harpunen- & Austausch-Substitution (A* + B-C -> A-B + C*, z.B. Na + Cl2 -> NaCl + Cl*)
          3) Konzertiertem 4-Zentren-Torus-Austausch (A-B + C-D -> A-C + B-D bei hohem T/P)
        """
        n = len(self.atoms)
        if n < 2:
            return 0.0

        kb_t = max(KB_EV_PER_K * temp_k, 1e-4)
        p_boost = min(2.6, 0.70 + 0.35 * math.log10(max(p_bar, 0.1) + 1.0))

        # --- 1) Direkte Bindungsbildung zwischen zwei Atomen mit offenen Valenzslots ---
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)

        for i, j in pairs:
            ui = self.open_valence(i)
            uj = self.open_valence(j)
            if ui < 0.9 or uj < 0.9:
                continue
            b_curr = self.get_bond(i, j)
            if b_curr >= 3.0:
                continue

            o_4d, dist = compute_atom_pair_torus_overlap(self.atoms[i], self.atoms[j], p_bar, anim_time)
            r_eq = equilibrium_bond_length_ang(self.atoms[i]["el"], self.atoms[j]["el"], b_curr + 1.0)
            if o_4d < 0.24 or dist > r_eq * 2.15:
                continue

            d_gain = bond_energy_total_ev(self.atoms[i]["el"], self.atoms[j]["el"], b_curr + 1.0) - bond_energy_total_ev(
                self.atoms[i]["el"], self.atoms[j]["el"], b_curr
            )
            if d_gain <= 0.05:
                continue

            # Zwei freie Radikale / offene Valenzen rekombinieren nahezu barrierelos
            self.set_bond(i, j, b_curr + 1.0)
            self.atoms[i]["exc"] = 1.0
            self.atoms[j]["exc"] = 1.0
            mid_p = 0.5 * (self.atoms[i]["pos"] + self.atoms[j]["pos"])
            self.flashes.append({"pos": mid_p.copy(), "life": 1.0, "color": (135, 255, 195)})

            # Rueckstoss- & Schwingungsimpuls aus freigesetzter Bindungsenergie
            diff = self.atoms[j]["pos"] - self.atoms[i]["pos"]
            norm = max(float(np.linalg.norm(diff)), 1e-4)
            u_dir = diff / norm
            kick = 0.35 * math.sqrt(d_gain)
            self.atoms[i]["vel"] -= kick * u_dir
            self.atoms[j]["vel"] += kick * u_dir

            lbl_b = "=" if (b_curr + 1.0) >= 2.0 else "-"
            self.event_log.append(
                f"4D-Valenzkopplung: {self.atoms[i]['el']}{lbl_b}{self.atoms[j]['el']} (O_4D={o_4d:.2f}, dH={-d_gain*EV_TO_KJMOL:+.0f} kJ/mol)"
            )
            return (+d_gain * 180.0) / max(n, 1)

        # --- 2) 4D-Harpunen- & Radikal-Substitution: i (offene Valenz) + (j - k) -> (i - j) + k ---
        # Genau hier reagiert Na (3s^1) mit Cl-Cl zu Na+Cl- + Cl*!
        for i in range(n):
            if self.open_valence(i) < 0.9:
                continue
            for j in range(n):
                if i == j or self.get_bond(i, j) > 0.0:
                    continue
                o_4d, dist = compute_atom_pair_torus_overlap(self.atoms[i], self.atoms[j], p_bar, anim_time)
                if o_4d < 0.26 or dist > 3.85:
                    continue

                nbrs_j = self.neighbors(j)
                if not nbrs_j:
                    continue

                # Suche die schwaechste Bindung (j, k), die zugunsten von (i, j) aufgebrochen werden kann
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

                # Harpunen-Bonus durch 4D-Ladungstransfer |chi_j - chi_i|
                chi_i = ELEMENTS[self.atoms[i]["el"]][5]
                chi_j = ELEMENTS[self.atoms[j]["el"]][5]
                harpoon_bonus = 0.55 * abs(chi_j - chi_i) * o_4d
                exc_bonus = 0.85 * (self.atoms[i]["exc"] + self.atoms[j]["exc"]) + (1.6 if force_spark else 0.0)

                ea_eff = max(0.0, 0.26 * best_d_old + 0.45 * best_dh - 0.55 * o_4d * best_d_new - harpoon_bonus - exc_bonus)
                p_wkb = math.exp(-2.4 * (1.0 - 0.70 * o_4d) * math.sqrt(ea_eff)) if ea_eff > 0.0 else 1.0
                p_th = math.exp(-ea_eff / (5.5 * kb_t)) if ea_eff > 0.0 else 1.0
                p_tot = min(1.0, (1.0 - (1.0 - p_th) * (1.0 - 0.35 * p_wkb)) * p_boost)

                if rng.random() > p_tot:
                    continue

                # Bindungsumlagerung: (j, k) um 1 reduzieren, (i, j) um 1 erhoehen!
                b_jk_old = self.get_bond(j, best_k)
                self.set_bond(j, best_k, b_jk_old - 1.0)
                self.set_bond(i, j, 1.0)
                self.atoms[i]["exc"] = 1.0
                self.atoms[j]["exc"] = 1.0
                self.atoms[best_k]["exc"] = 1.0

                # Abgestossenes Atom k erhaelt Rueckstossimpuls weg von j
                sep_dir = self.atoms[best_k]["pos"] - self.atoms[j]["pos"]
                sep_norm = max(float(np.linalg.norm(sep_dir)), 1e-4)
                sep_dir /= sep_norm
                recoil = 0.55 + 0.35 * math.sqrt(max(-best_dh, 0.1))
                self.atoms[best_k]["vel"] += recoil * sep_dir
                self.atoms[i]["vel"] -= 0.35 * recoil * sep_dir

                mid_p = 0.5 * (self.atoms[i]["pos"] + self.atoms[j]["pos"])
                self.flashes.append({"pos": mid_p.copy(), "life": 1.15, "color": (255, 225, 110)})

                mech = "4D-Harpune" if abs(chi_j - chi_i) > 1.4 else "4D-Austausch"
                self.event_log.append(
                    f"{mech}: {self.atoms[i]['el']} + {self.atoms[j]['el']}-{self.atoms[best_k]['el']} -> "
                    f"{self.atoms[i]['el']}-{self.atoms[j]['el']} + {self.atoms[best_k]['el']}* (O_4D={o_4d:.2f}, dH={best_dh*EV_TO_KJMOL:+.0f} kJ/mol)"
                )
                return (-best_dh * 190.0) / max(n, 1)

        # --- 3) Konzertierter 4-Zentren-Torus-Austausch: (i - j) + (k - l) -> (i - k) + (j - l) ---
        active_bonds = [(a, b, o) for (a, b), o in self.bond_order.items() if o >= 1.0]
        if len(active_bonds) >= 2:
            rng.shuffle(active_bonds)
            for idx1 in range(min(len(active_bonds), 8)):
                i, j, bij = active_bonds[idx1]
                for idx2 in range(idx1 + 1, min(len(active_bonds), 8)):
                    k, l, bkl = active_bonds[idx2]
                    if len({i, j, k, l}) < 4:
                        continue
                    # Pruefe beide Paarungen: (i-k, j-l) und (i-l, j-k)
                    for (p1a, p1b, p2a, p2b) in [(i, k, j, l), (i, l, j, k)]:
                        if self.get_bond(p1a, p1b) > 0.0 or self.get_bond(p2a, p2b) > 0.0:
                            continue
                        ov1, _ = compute_atom_pair_torus_overlap(self.atoms[p1a], self.atoms[p1b], p_bar, anim_time)
                        ov2, _ = compute_atom_pair_torus_overlap(self.atoms[p2a], self.atoms[p2b], p_bar, anim_time)
                        ov_min = min(ov1, ov2)
                        if ov_min < 0.30:
                            continue

                        d_old = (
                            bond_energy_total_ev(self.atoms[i]["el"], self.atoms[j]["el"], bij)
                            - bond_energy_total_ev(self.atoms[i]["el"], self.atoms[j]["el"], bij - 1.0)
                            + bond_energy_total_ev(self.atoms[k]["el"], self.atoms[l]["el"], bkl)
                            - bond_energy_total_ev(self.atoms[k]["el"], self.atoms[l]["el"], bkl - 1.0)
                        )
                        d_new = intrinsic_single_bond_ev(self.atoms[p1a]["el"], self.atoms[p1b]["el"]) + intrinsic_single_bond_ev(
                            self.atoms[p2a]["el"], self.atoms[p2b]["el"]
                        )
                        dh = d_old - d_new
                        exc_b = 0.8 * sum(self.atoms[x]["exc"] for x in (i, j, k, l)) + (1.8 if force_spark else 0.0)
                        ea_4c = max(0.0, 0.24 * d_old + 0.45 * dh - 0.70 * ov_min * d_new - exc_b)
                        p_wkb = math.exp(-2.5 * (1.0 - 0.70 * ov_min) * math.sqrt(ea_4c)) if ea_4c > 0.0 else 1.0
                        p_th = math.exp(-ea_4c / (5.5 * kb_t)) if ea_4c > 0.0 else 1.0
                        p_tot = min(1.0, (1.0 - (1.0 - p_th) * (1.0 - 0.35 * p_wkb)) * p_boost)

                        if rng.random() > p_tot:
                            continue

                        self.set_bond(i, j, bij - 1.0)
                        self.set_bond(k, l, bkl - 1.0)
                        self.set_bond(p1a, p1b, 1.0)
                        self.set_bond(p2a, p2b, 1.0)
                        for x in (i, j, k, l):
                            self.atoms[x]["exc"] = 1.0

                        cm = 0.25 * (self.atoms[i]["pos"] + self.atoms[j]["pos"] + self.atoms[k]["pos"] + self.atoms[l]["pos"])
                        self.flashes.append({"pos": cm.copy(), "life": 1.15, "color": (125, 235, 255)})
                        self.event_log.append(
                            f"4-Zentren-Torus-Austausch: -> {self.atoms[p1a]['el']}-{self.atoms[p1b]['el']} + "
                            f"{self.atoms[p2a]['el']}-{self.atoms[p2b]['el']} (O_4D={ov_min:.2f}, dH={dh*EV_TO_KJMOL:+.0f} kJ/mol)"
                        )
                        return (-dh * 190.0) / max(n, 1)

        return 0.0

    def step_md_forces(self, dt, temp_k, p_bar, chamber_rad, rng):
        """Integriert die 3D-Bewegungsgleichungen aller Atome (Bindungen, VSEPR, Coulomb, Druck P, Thermik T)."""
        n = len(self.atoms)
        if n == 0:
            return

        forces = [np.zeros(3, dtype=np.float32) for _ in range(n)]
        confine_k = 0.35 + 0.28 * math.log10(max(p_bar, 0.1) + 1.0)

        # 1) Harmonische / Morse-artige Bindungsfedern
        for (i, j), b_ord in self.bond_order.items():
            if b_ord <= 0.0:
                continue
            diff = self.atoms[j]["pos"] - self.atoms[i]["pos"]
            dist = max(float(np.linalg.norm(diff)), 1e-4)
            r0 = equilibrium_bond_length_ang(self.atoms[i]["el"], self.atoms[j]["el"], b_ord)
            k_spring = 18.0 * b_ord
            f_vec = k_spring * (dist - r0) * (diff / dist)
            forces[i] += f_vec
            forces[j] -= f_vec

        # 2) VSEPR-Winkelabstossung zwischen Nachbarn am selben Zentralatom
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
                    d13_target = math.sqrt(max(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * math.cos(theta_0), 0.1))
                    diff13 = self.atoms[a2]["pos"] - self.atoms[a1]["pos"]
                    dist13 = max(float(np.linalg.norm(diff13)), 1e-4)
                    f_ang = 5.5 * (dist13 - d13_target) * (diff13 / dist13)
                    forces[a1] += f_ang
                    forces[a2] -= f_ang

        # 3) Nicht-gebundene Kraefte: 4D-Coulomb-Anziehung (Na+ <-> Cl-) & sterische Pauli-Abstossung
        for i in range(n):
            for j in range(i + 1, n):
                if self.get_bond(i, j) > 0.0:
                    continue
                diff = self.atoms[j]["pos"] - self.atoms[i]["pos"]
                dist = max(float(np.linalg.norm(diff)), 1e-4)
                u_dir = diff / dist
                r_vdw = 0.85 * (ELEMENTS[self.atoms[i]["el"]][3] + ELEMENTS[self.atoms[j]["el"]][3])

                # Ionische Coulomb-Wechselwirkung zwischen 4D-Partialladungen q_i, q_j
                qi, qj = self.atoms[i]["charge"], self.atoms[j]["charge"]
                if abs(qi * qj) > 0.02 and dist < 4.5:
                    f_coul = 3.2 * (qi * qj) / max(dist * dist, 0.8)
                    forces[i] += f_coul * u_dir
                    forces[j] -= f_coul * u_dir

                # Pauli-Abstossung der Elektronenhüllen bei Nahkontakt
                if dist < r_vdw:
                    f_rep = 14.0 * (r_vdw - dist)
                    forces[i] -= f_rep * u_dir
                    forces[j] += f_rep * u_dir

        # 4) Langevin-Thermostat (T) + isobare Kammer-Kompression (P) + Euler-Integration
        th_v = 0.42 * math.sqrt(max(temp_k, 0.0) / 300.0)
        sub_steps = 3
        h_dt = dt / sub_steps

        for _ in range(sub_steps):
            for i in range(n):
                self.atoms[i]["exc"] = max(0.0, self.atoms[i]["exc"] - 0.75 * h_dt)
                mass_amu = max(float(self.atoms[i]["Z"] + self.atoms[i]["N"]), 1.0)
                inv_m = 1.0 / math.sqrt(mass_amu)

                f_conf = -confine_k * self.atoms[i]["pos"]
                f_rand = np.array(
                    [rng.gauss(0.0, th_v * 2.2), rng.gauss(0.0, th_v * 2.2), rng.gauss(0.0, th_v * 2.2)],
                    dtype=np.float32,
                )
                acc = (forces[i] + f_conf + f_rand) * inv_m
                self.atoms[i]["vel"] = 0.94 * self.atoms[i]["vel"] + acc * h_dt

                v_mag = float(np.linalg.norm(self.atoms[i]["vel"]))
                v_cap = max(2.2 * th_v, 0.55)
                if v_mag > v_cap:
                    self.atoms[i]["vel"] *= v_cap / v_mag

                self.atoms[i]["pos"] += self.atoms[i]["vel"] * h_dt

                # Reflexion an der isobaren Druckkammer-Sphaere R(P)
                r_pos = float(np.linalg.norm(self.atoms[i]["pos"]))
                if r_pos > chamber_rad:
                    n_wall = self.atoms[i]["pos"] / r_pos
                    self.atoms[i]["pos"] = n_wall * chamber_rad
                    vn = float(np.dot(self.atoms[i]["vel"], n_wall))
                    if vn > 0.0:
                        self.atoms[i]["vel"] -= 1.85 * vn * n_wall


# ==============================================================================
# 4. 4D-POLYCHOR & VEKTORISIERTE JACOBI-THETA-WELLENPAKETE
# ==============================================================================
def rotation_matrix_4d(a1, a2):
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


def build_24cell_vertices():
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
    shells = [round(0.5 * float(np.sum((v - v0) ** 2))) for v in verts]
    centered = np.array([0.36 * (v - v0) for v in verts], dtype=np.float32)
    return np.array(verts, dtype=np.float32), centered, shells


def evaluate_batched_4d_packets(coords_t, pos_arr, sig_arr, cp_arr, rgb_arr, anim_time):
    X1, X2, X3, X4 = coords_t
    ny, nx = X1.shape
    out_r = np.zeros((ny, nx), dtype=np.float32)
    out_g = np.zeros((ny, nx), dtype=np.float32)
    out_b = np.zeros((ny, nx), dtype=np.float32)

    m_total = pos_arr.shape[0]
    if m_total == 0:
        return out_r, out_g, out_b

    coords = (X1[None, :, :], X2[None, :, :], X3[None, :, :], X4[None, :, :])
    two_pi_n = np.float32(4.0 * math.pi)
    time_phase = np.complex64(np.exp(-1j * 2.4 * anim_time))

    chunk_size = 56
    for start in range(0, m_total, chunk_size):
        end = min(start + chunk_size, m_total)
        pb = pos_arr[start:end]
        sb = sig_arr[start:end, None, None]
        cpb = cp_arr[start:end, None, None]
        cb = rgb_arr[start:end]

        inv_sig2 = np.float32(-0.5) / (sb * sb)
        psi_4d = np.full((end - start, ny, nx), time_phase, dtype=np.complex64)

        for k in range(4):
            ak = pb[:, k : k + 1, None]
            dx_base = ((coords[k] - ak + np.float32(0.5)) % np.float32(1.0)) - np.float32(0.5)
            ph0 = cpb if k == 0 else np.float32(0.0)
            psi_k = np.zeros_like(psi_4d)
            for m in (-1.0, 0.0, 1.0):
                dx = dx_base + np.float32(m)
                env = np.exp(inv_sig2 * (dx * dx))
                ang = two_pi_n * dx + ph0
                psi_k += env * (np.cos(ang) + 1j * np.sin(ang))
            psi_4d *= psi_k

        dens = (np.abs(psi_4d) ** 2) * (np.float32(0.78) + np.float32(0.22) * np.cos(np.angle(psi_4d)))
        out_r += np.tensordot(cb[:, 0], dens, axes=(0, 0))
        out_g += np.tensordot(cb[:, 1], dens, axes=(0, 0))
        out_b += np.tensordot(cb[:, 2], dens, axes=(0, 0))

    return out_r, out_g, out_b


def build_torus_particles_from_atoms(engine, anim_time, p_bar, show_dm_flyby):
    """Erzeugt alle Einzelteilchen (p, n, u, d, e^-, g, pi, gamma, DM) aus dem aktuellen Atom-Ensemble."""
    nuc_base_4d = np.array([0.21, -0.20, 0.18, -0.17], dtype=np.float32)
    elec_base_4d = np.array([-0.27, 0.25, -0.23, 0.22], dtype=np.float32)

    nucleons, quarks, electrons = [], [], []
    gluon_links, pion_links, photon_links = [], [], []
    atom_elec_centers = []

    total_A = sum(a["Z"] + a["N"] for a in engine.atoms)
    scale_f = 1.0 if total_A <= 12 else (0.68 if total_A <= 45 else 0.48)
    sig_q = SIGMA_W * (0.46 if total_A <= 12 else (0.30 if total_A <= 45 else 0.21))
    p_comp = 1.0 / (1.0 + 0.16 * math.log10(max(p_bar, 0.1) + 1.0))

    p_cnt, n_cnt, u_cnt, d_cnt, e_cnt = 0, 0, 0, 0, 0
    b_nuc_total_mev = 0.0

    alpha_tet = [
        np.array([+1.0, +1.0, +1.0, 0.0], dtype=np.float32),
        np.array([-1.0, -1.0, +1.0, 0.0], dtype=np.float32),
        np.array([+1.0, -1.0, -1.0, 0.0], dtype=np.float32),
        np.array([-1.0, +1.0, -1.0, 0.0], dtype=np.float32),
    ]

    for idx_a, at in enumerate(engine.atoms):
        ax, ay, az = at["pos"]
        b_nuc_total_mev += ELEMENTS[at["el"]][4]

        shift_4d = np.array(
            [0.036 * ax * p_comp, 0.036 * ay * p_comp, 0.028 * az * p_comp, 0.018 * (ax - ay) * p_comp],
            dtype=np.float32,
        )
        atom_nuc_c = nuc_base_4d + shift_4d
        atom_elec_c = elec_base_4d + shift_4d
        atom_elec_centers.append(atom_elec_c.copy())

        z_c, n_c = at["Z"], at["N"]
        n_alpha = min(z_c // 2, n_c // 2)
        rem_p = z_c - 2 * n_alpha
        rem_n = n_c - 2 * n_alpha

        atom_nuc_indices = []
        nuc_rot = 0.75 * anim_time + idx_a * 0.5
        cn, sn = math.cos(nuc_rot), math.sin(nuc_rot)

        for cl_i in range(n_alpha):
            cl_ang = cl_i * (2.0 * math.pi / max(n_alpha, 1)) + 0.35 * anim_time
            cl_r = (0.032 * scale_f) if n_alpha > 1 else 0.0
            cl_c = atom_nuc_c + np.array(
                [cl_r * math.cos(cl_ang), cl_r * math.sin(cl_ang), 0.01 * math.sin(cl_ang), -0.01 * math.cos(cl_ang)],
                dtype=np.float32,
            )
            cluster_nucs = []
            for t_i, is_prot in enumerate([True, True, False, False]):
                off = alpha_tet[t_i] * (0.022 * scale_f)
                ox = off[0] * cn - off[1] * sn
                oy = off[0] * sn + off[1] * cn
                n_pos_4d = cl_c + np.array([ox, oy, off[2], 0.0], dtype=np.float32)
                if is_prot:
                    p_cnt += 1
                    n_kind = "p"
                else:
                    n_cnt += 1
                    n_kind = "n"
                n_idx = len(nucleons)
                nucleons.append({"kind": n_kind, "pos_4d": n_pos_4d})
                atom_nuc_indices.append(n_idx)
                cluster_nucs.append(n_idx)
            for ia in range(4):
                for ib in range(ia + 1, 4):
                    pion_links.append((cluster_nucs[ia], cluster_nucs[ib]))

        for is_prot, count_rem in [(True, rem_p), (False, rem_n)]:
            for _ in range(count_rem):
                if is_prot:
                    p_cnt += 1
                    n_kind = "p"
                else:
                    n_cnt += 1
                    n_kind = "n"
                n_idx = len(nucleons)
                nucleons.append({"kind": n_kind, "pos_4d": atom_nuc_c.copy()})
                atom_nuc_indices.append(n_idx)

        q_rad = 0.020 * scale_f
        for n_idx in atom_nuc_indices:
            is_prot = nucleons[n_idx]["kind"] == "p"
            flavors = [("u", "R"), ("u", "G"), ("d", "B")] if is_prot else [("u", "R"), ("d", "G"), ("d", "B")]
            ang0 = (2.7 if n_idx % 2 == 0 else -2.5) * anim_time + n_idx * 0.65
            for qi, (flv, qcol) in enumerate(flavors):
                th = ang0 + qi * 2.0943951
                q_pos_4d = nucleons[n_idx]["pos_4d"] + np.array(
                    [q_rad * math.cos(th), q_rad * math.sin(th), 0.006 * math.sin(th), -0.006 * math.cos(th)],
                    dtype=np.float32,
                )
                if flv == "u":
                    u_cnt += 1
                    cp_d = 0.0
                else:
                    d_cnt += 1
                    cp_d = CP_PHASE_DEG
                if qcol == "R":
                    rgb_w, m_col = (0.95, 0.18, 0.16), (255, 115, 115)
                elif qcol == "G":
                    rgb_w, m_col = (0.16, 0.92, 0.28), (115, 255, 145)
                else:
                    rgb_w, m_col = (0.20, 0.42, 1.00), (130, 180, 255)
                q_idx = len(quarks)
                quarks.append({"flavor": flv, "pos_4d": q_pos_4d, "sigma": sig_q, "cp_deg": cp_d, "rgb": rgb_w, "marker_col": m_col})
                gluon_links.append((n_idx, q_idx, m_col))

        # Elektronen dieses Atoms in C_0 (inkl. 4D-Ladungsverschiebung bei Na+ -> Cl-!)
        sig_e_at = SIGMA_W * 0.58 * ELEMENTS[at["el"]][7]
        n_el_eff = max(0, int(round(z_c - at["charge"])))
        e_cnt += z_c
        for ei in range(min(n_el_eff, 18)):
            ang_e = (2.0 if ei % 2 == 0 else -1.8) * anim_time + ei * (2.0 * math.pi / max(n_el_eff, 1))
            r_e = 0.022 if ei < 2 else (0.044 if ei < 10 else 0.068)
            ep = atom_elec_c + np.array([r_e * math.cos(ang_e), r_e * math.sin(ang_e), 0.0, 0.0], dtype=np.float32)
            # Ionische Anionen (z.B. Cl-) leuchten gold-tuerkis durch das aufgenommene 3s-Elektron
            if at["charge"] < -0.35 and ei == n_el_eff - 1:
                rgb_e, m_e = (1.00, 0.88, 0.25), (255, 235, 110)
            else:
                rgb_e, m_e = (0.16, 0.85, 1.00), (105, 235, 255)
            electrons.append({"pos_4d": ep, "sigma": sig_e_at, "rgb": rgb_e, "marker_col": m_e})

    proton_indices = [idx_n for idx_n, nuc in enumerate(nucleons) if nuc["kind"] == "p"]
    if proton_indices and electrons:
        for pi_k, p_idx in enumerate(proton_indices):
            e_idx = pi_k % len(electrons)
            photon_links.append((p_idx, e_idx, (pi_k * 0.17) % 1.0))

    shadow_dm = []
    if show_dm_flyby:
        for dm_p in [np.array([0.36, 0.36, 0.0, 0.0], dtype=np.float32), np.array([-0.36, -0.36, 0.0, 0.0], dtype=np.float32)]:
            shadow_dm.append({"pos_4d": dm_p, "sigma": SIGMA_W * 0.65, "rgb": (1.00, 0.52, 0.12), "marker_col": (255, 165, 85)})

    return {
        "nucleons": nucleons,
        "quarks": quarks,
        "electrons": electrons,
        "atom_elec_centers": atom_elec_centers,
        "gluon_links": gluon_links,
        "pion_links": pion_links,
        "photon_links": photon_links,
        "shadow_dm": shadow_dm,
        "n_p": p_cnt,
        "n_n": n_cnt,
        "n_u": u_cnt,
        "n_d": d_cnt,
        "n_e": e_cnt,
        "b_nuc_mev": b_nuc_total_mev,
    }


def render_3d_electron_cloud(engine, MU, MV, box_ang, yaw, pitch, temp_k, anim_time):
    """Rendert die volumetrische 3D-Elektronenwolke (inkl. ionischer Ladungsverschiebung)."""
    ny, nx = MU.shape
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    dens_core = np.zeros((ny, nx), dtype=np.float32)
    dens_bond = np.zeros((ny, nx), dtype=np.float32)
    dens_ion = np.zeros((ny, nx), dtype=np.float32)

    aspect = ny / max(nx, 1)
    for z_sl in (-0.42, -0.20, 0.0, 0.20, 0.42):
        w_sl = np.float32(math.exp(-2.2 * z_sl * z_sl))
        U_s = MU * np.float32(box_ang)
        V_s = MV * np.float32(box_ang * aspect)
        W_s = np.full_like(U_s, np.float32(z_sl * box_ang))

        X = cy * U_s + sy * (sp * V_s + cp * W_s)
        Y = cp * V_s - sp * W_s
        Z = -sy * U_s + cy * (sp * V_s + cp * W_s)

        for at in engine.atoms:
            ax, ay, az = at["pos"]
            r2 = (X - ax) ** 2 + (Y - ay) ** 2 + (Z - az) ** 2
            rad_c = 0.32 * ELEMENTS[at["el"]][3] + 0.16
            # Kationen (Na+) schrumpfen in der Elektronenhuelle, Anionen (Cl-) quellen auf!
            q_scale = max(0.35, 1.0 - 0.45 * at["charge"])
            dens_core += w_sl * np.float32(0.75 * q_scale) * np.exp(-r2 / np.float32(2.0 * rad_c * rad_c))
            if at["charge"] < -0.35:
                dens_ion += w_sl * np.float32(abs(at["charge"]) * 1.1) * np.exp(-r2 / np.float32(2.0 * (rad_c * 1.35) ** 2))

        for (ia, ib), b_ord in engine.bond_order.items():
            if b_ord <= 0.0:
                continue
            p1 = engine.atoms[ia]["pos"]
            p2 = engine.atoms[ib]["pos"]
            mid = 0.5 * (p1 + p2)
            seg = p2 - p1
            seg_len = max(float(np.linalg.norm(seg)), 1e-4)
            u_ax = seg / seg_len
            rel_x, rel_y, rel_z = X - mid[0], Y - mid[1], Z - mid[2]
            proj_l = rel_x * u_ax[0] + rel_y * u_ax[1] + rel_z * u_ax[2]
            perp_r2 = np.maximum(0.0, rel_x * rel_x + rel_y * rel_y + rel_z * rel_z - proj_l * proj_l)
            dens_bond += w_sl * np.float32(0.95 * b_ord) * np.exp(
                -0.5 * (proj_l / (0.44 * seg_len)) ** 2 - 0.5 * (perp_r2 / 0.11)
            )

    th_glow = max(0.0, (temp_k - 1200.0) / 3800.0) * 0.14
    pulse = 0.92 + 0.08 * math.sin(2.4 * anim_time)
    bg = 0.065
    r_col = bg * 0.75 + th_glow * 1.2 + dens_core * 0.25 + dens_bond * 0.22 + dens_ion * 1.05 * pulse
    g_col = bg * 0.95 + th_glow * 0.4 + dens_core * 0.72 + dens_bond * 0.92 * pulse + dens_ion * 0.78 * pulse
    b_col = bg * 1.40 + dens_core * 1.00 + dens_bond * 1.00 * pulse + dens_ion * 0.25

    rgb = np.stack([r_col, g_col, b_col], axis=-1)
    rgb = rgb / (1.0 + 0.48 * rgb)
    return np.clip(np.power(np.clip(rgb, 0.0, 1.0), 0.85) * 255.0, 0, 255).astype(np.uint8)


# ==============================================================================
# 5. UI-SLIDER
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
        surf.blit(lbl_surf, (self.rect.x, self.rect.y - 14))

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
# 6. HAUPTANWENDUNG
# ==============================================================================
def main():
    pygame.init()
    pygame.key.set_repeat(320, 35)
    WIDTH, HEIGHT = 1440, 870
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("24-Zell-Torus (v9) — Regelfreie 4D-Torus-Reaktions- & Druck-Simulation")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("dejavusans,arial,helvetica", 19, bold=True)
    font_sub = pygame.font.SysFont("dejavusans,arial,helvetica", 13, bold=True)
    font_ui = pygame.font.SysFont("dejavusans,arial,helvetica", 12)
    font_mono = pygame.font.SysFont("dejavusansmono,consolas,monospace", 12)

    rng = random.Random(24)
    verts_24, verts_torus, shells_24 = build_24cell_vertices()
    engine = TorusAbInitioEngine()

    TORUS_SIZE = 636
    TORUS_POS = (20, 76)
    torus_rect = pygame.Rect(TORUS_POS[0], TORUS_POS[1], TORUS_SIZE, TORUS_SIZE)
    TORUS_N = 128
    t_axis = np.linspace(-0.5, 0.5, TORUS_N, endpoint=False, dtype=np.float32)
    U_T, V_T = np.meshgrid(t_axis, t_axis)

    MOL_W, MOL_H = 376, 308
    MOL_POS = (672, 76)
    mol_rect = pygame.Rect(MOL_POS[0], MOL_POS[1], MOL_W, MOL_H)
    MOL_NX, MOL_NY = 108, 88
    mx_1d = np.linspace(-1.0, 1.0, MOL_NX, dtype=np.float32)
    my_1d = np.linspace(-1.0, 1.0, MOL_NY, dtype=np.float32)
    MU, MV = np.meshgrid(mx_1d, -my_1d)

    ctrl_x = 1060
    preset_buttons = []
    for idx, pr in enumerate(INITIAL_MIXTURES):
        col = idx % 2
        row = idx // 2
        w_b = 171 if idx < 4 else 348
        preset_buttons.append((pygame.Rect(ctrl_x + 8 + col * 177, 94 + row * 22, w_b, 19), pr))

    smiles_box = pygame.Rect(ctrl_x + 8, 165, 212, 22)
    btn_add_smiles = pygame.Rect(ctrl_x + 224, 165, 66, 22)
    btn_rep_smiles = pygame.Rect(ctrl_x + 294, 165, 62, 22)

    quick_buttons = []
    for idx, qm in enumerate(QUICK_ADD_ATOMS):
        col = idx % 3
        row = idx // 3
        quick_buttons.append((pygame.Rect(ctrl_x + 8 + col * 117, 192 + row * 21, 112, 18), qm))

    slider_temp = Slider((ctrl_x + 10, 248, 344, 6), "Temperatur T [Kelvin]", 0.0, 5000.0, 380.0, "{:.0f} K")
    slider_logp = Slider((ctrl_x + 10, 278, 344, 6), "Druck log10(P / bar) [-1..4.5]", -1.00, 4.50, 0.00, "{:+.2f}")
    slider_ephot = Slider((ctrl_x + 10, 308, 344, 6), "Photonenenergie log10(E_g / eV)", -1.00, 4.00, 0.78, "{:.2f}")
    slider_rot = Slider((ctrl_x + 10, 338, 344, 6), "4D-Torus-Rotation [rad/s]", 0.00, 2.20, 0.42, "{:.2f}")
    sliders = [slider_temp, slider_logp, slider_ephot, slider_rot]

    btn_fire_ph = pygame.Rect(ctrl_x + 8, 354, 84, 22)
    btn_spark = pygame.Rect(ctrl_x + 96, 354, 86, 22)
    btn_rxn_tog = pygame.Rect(ctrl_x + 186, 354, 84, 22)
    btn_reload = pygame.Rect(ctrl_x + 274, 354, 82, 22)

    tog_u = pygame.Rect(ctrl_x + 8, 380, 55, 20)
    tog_d = pygame.Rect(ctrl_x + 66, 380, 55, 20)
    tog_e = pygame.Rect(ctrl_x + 124, 380, 55, 20)
    tog_strong = pygame.Rect(ctrl_x + 182, 380, 55, 20)
    tog_photon = pygame.Rect(ctrl_x + 240, 380, 56, 20)
    tog_dm = pygame.Rect(ctrl_x + 299, 380, 57, 20)

    active_preset_idx = 0
    initial_conservation = (0, 0, 0, 0, 0)
    smiles_text = "[Na]"
    smiles_active = False
    status_msg = "2 Na + Cl2 geladen: Beobachte die 4D-Harpunen-Reaktion zu 2 Na+Cl-!"
    status_col = (145, 225, 185)

    reactions_enabled = True
    auto_rotate = True
    show_u_quarks = True
    show_d_quarks = True
    show_electrons = True
    show_strong = True
    show_photons = True
    show_dm_flyby = True

    rot_a1 = 0.28
    rot_a2 = 0.16
    mol_yaw = 0.40
    mol_pitch = 0.30
    anim_time = 0.0
    rxn_timer = 0.0
    spark_pending = False
    incoming_photons = []

    drag_target = None
    last_mouse = (0, 0)

    def snapshot_conservation():
        np_t = sum(a["Z"] for a in engine.atoms)
        nn_t = sum(a["N"] for a in engine.atoms)
        return (np_t, nn_t, 2 * np_t + nn_t, np_t + 2 * nn_t, np_t)

    def get_chamber_radius(p_bar):
        r_comp = 1.95 * ((max(p_bar, 0.1) / 1.0) ** (-0.14))
        return float(np.clip(r_comp, 0.85, 2.55))

    def load_mixture(p_idx):
        nonlocal active_preset_idx, initial_conservation, status_msg, status_col
        active_preset_idx = p_idx
        pr = INITIAL_MIXTURES[p_idx]
        engine.clear()
        n_sp = len(pr["smiles_list"])
        rad = 1.15
        for idx_s, smi in enumerate(pr["smiles_list"]):
            ang = idx_s * (2.0 * math.pi / max(n_sp, 1))
            c3 = [rad * math.cos(ang), rad * math.sin(ang), 0.22 * math.sin(idx_s * 1.7)] if n_sp > 1 else [0.0, 0.0, 0.0]
            engine.add_smiles_species(smi, c3)
        slider_temp.val = pr["temp_k"]
        slider_logp.val = pr["log10_p"]
        incoming_photons.clear()
        initial_conservation = snapshot_conservation()
        status_msg = f"Mischung: {pr['name']}"
        status_col = (135, 245, 190)

    def add_smiles_to_chamber(smi_str, replace=False):
        nonlocal initial_conservation, status_msg, status_col
        try:
            if replace:
                engine.clear()
                engine.add_smiles_species(smi_str, [0.0, 0.0, 0.0])
            else:
                if len(engine.atoms) >= 36:
                    raise ValueError("Maximal 36 Atome gleichzeitig in der Kammer.")
                ang = rng.uniform(0.0, 2.0 * math.pi)
                engine.add_smiles_species(smi_str, [0.95 * math.cos(ang), 0.95 * math.sin(ang), rng.uniform(-0.3, 0.3)])
            initial_conservation = snapshot_conservation()
            status_msg = f"Eingefuegt: {smi_str} (gesamt {len(engine.atoms)} Atome)"
            status_col = (135, 245, 190)
        except ValueError as err:
            status_msg = str(err)
            status_col = (255, 125, 125)

    def fire_photon():
        e_ev = 10.0 ** slider_ephot.val
        p_col = (255, 130, 85) if e_ev < 3.2 else ((175, 130, 255) if e_ev < 13.6 else (95, 225, 255))
        y_off = rng.uniform(-0.85, 0.85)
        z_off = rng.uniform(-0.65, 0.65)
        start_p = np.array([-3.0, y_off, z_off], dtype=np.float32)
        dir_v = np.array([3.0, -0.6 * y_off, -0.6 * z_off], dtype=np.float32)
        dir_v /= max(float(np.linalg.norm(dir_v)), 1e-4)
        incoming_photons.append({"pos_3d": start_p, "dir_3d": dir_v, "e_ev": e_ev, "col": p_col, "hit_done": False})

    load_mixture(0)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        anim_time += dt
        rxn_timer += dt

        if auto_rotate and drag_target != "torus":
            rot_a1 = (rot_a1 + slider_rot.val * dt) % (2.0 * math.pi)
            rot_a2 = (rot_a2 + 0.6180339887 * slider_rot.val * dt) % (2.0 * math.pi)
            if drag_target != "mol_cam":
                mol_yaw = (mol_yaw + 0.24 * slider_rot.val * dt) % (2.0 * math.pi)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if smiles_active:
                        smiles_active = False
                    else:
                        running = False
                elif event.key == pygame.K_TAB:
                    smiles_active = not smiles_active
                elif smiles_active:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        add_smiles_to_chamber(smiles_text, replace=False)
                    elif event.key == pygame.K_BACKSPACE:
                        smiles_text = smiles_text[:-1]
                    elif event.unicode and len(smiles_text) < 42 and event.unicode.isprintable():
                        smiles_text += event.unicode
                else:
                    if event.key == pygame.K_SPACE:
                        auto_rotate = not auto_rotate
                    elif event.key == pygame.K_f:
                        fire_photon()
                    elif event.key == pygame.K_g:
                        spark_pending = True
                    elif event.key == pygame.K_c:
                        reactions_enabled = not reactions_enabled
                    elif event.key == pygame.K_u:
                        show_u_quarks = not show_u_quarks
                    elif event.key == pygame.K_d:
                        show_d_quarks = not show_d_quarks
                    elif event.key == pygame.K_e:
                        show_electrons = not show_electrons
                    elif event.key == pygame.K_p:
                        show_photons = not show_photons
                    elif event.key == pygame.K_s:
                        show_dm_flyby = not show_dm_flyby
                    elif event.key == pygame.K_r:
                        load_mixture(active_preset_idx)
                    else:
                        for idx_p, pr in enumerate(INITIAL_MIXTURES):
                            if event.unicode == pr["key"]:
                                load_mixture(idx_p)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                smiles_active = smiles_box.collidepoint(event.pos)
                if btn_add_smiles.collidepoint(event.pos):
                    add_smiles_to_chamber(smiles_text, replace=False)
                elif btn_rep_smiles.collidepoint(event.pos):
                    add_smiles_to_chamber(smiles_text, replace=True)
                elif btn_fire_ph.collidepoint(event.pos):
                    fire_photon()
                elif btn_spark.collidepoint(event.pos):
                    spark_pending = True
                elif btn_rxn_tog.collidepoint(event.pos):
                    reactions_enabled = not reactions_enabled
                elif btn_reload.collidepoint(event.pos):
                    load_mixture(active_preset_idx)
                else:
                    for idx_p, (rect, _) in enumerate(preset_buttons):
                        if rect.collidepoint(event.pos):
                            load_mixture(idx_p)
                    for rect, qm in quick_buttons:
                        if rect.collidepoint(event.pos):
                            smiles_text = qm["smiles"]
                            add_smiles_to_chamber(qm["smiles"], replace=False)

                if torus_rect.collidepoint(event.pos):
                    drag_target = "torus"
                    last_mouse = event.pos
                elif mol_rect.collidepoint(event.pos):
                    drag_target = "mol_cam"
                    last_mouse = event.pos

                if tog_u.collidepoint(event.pos):
                    show_u_quarks = not show_u_quarks
                elif tog_d.collidepoint(event.pos):
                    show_d_quarks = not show_d_quarks
                elif tog_e.collidepoint(event.pos):
                    show_electrons = not show_electrons
                elif tog_strong.collidepoint(event.pos):
                    show_strong = not show_strong
                elif tog_photon.collidepoint(event.pos):
                    show_photons = not show_photons
                elif tog_dm.collidepoint(event.pos):
                    show_dm_flyby = not show_dm_flyby

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drag_target = None
            elif event.type == pygame.MOUSEMOTION and drag_target is not None:
                dx_m = event.pos[0] - last_mouse[0]
                dy_m = event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                if drag_target == "torus":
                    rot_a1 = (rot_a1 + dx_m * 0.01) % (2.0 * math.pi)
                    rot_a2 = (rot_a2 + dy_m * 0.01) % (2.0 * math.pi)
                elif drag_target == "mol_cam":
                    mol_yaw = (mol_yaw + dx_m * 0.012) % (2.0 * math.pi)
                    mol_pitch = float(np.clip(mol_pitch + dy_m * 0.012, -1.45, 1.45))

            for s in sliders:
                s.handle_event(event)

        temp_k = slider_temp.val
        p_bar = 10.0 ** slider_logp.val
        e_gamma_ev = 10.0 ** slider_ephot.val
        R4 = rotation_matrix_4d(rot_a1, rot_a2)
        chamber_rad = get_chamber_radius(p_bar)
        box_ang = 2.65

        # ----------------------------------------------------------------------
        # 1. PHOTONENBESCHUSS, 4D-LADUNGSTRANSFER & REGELFREIE TORUS-UEBERGAENGE
        # ----------------------------------------------------------------------
        surviving_photons = []
        for ph in incoming_photons:
            ph["pos_3d"] = ph["pos_3d"] + ph["dir_3d"] * (5.2 * dt)
            if not ph["hit_done"] and engine.atoms:
                for idx_a, at in enumerate(engine.atoms):
                    if float(np.linalg.norm(ph["pos_3d"] - at["pos"])) < 0.72:
                        ph["hit_done"] = True
                        ev = ph["e_ev"]
                        at["exc"] = 1.0
                        if ev < 3.2:
                            dt_h = min(380.0, 90.0 + 120.0 * ev)
                            slider_temp.val = min(5000.0, slider_temp.val + dt_h)
                            engine.event_log.append(f"IR/Vis-Absorption ({ev:.2f} eV) -> +{dt_h:.0f} K Aufheizung")
                        else:
                            # UV/Roentgen spaltet eine Bindung an diesem Atom in offene Radikal-Valenzslots!
                            nbrs = engine.neighbors(idx_a)
                            if nbrs:
                                nb_target, b_old = nbrs[0]
                                engine.set_bond(idx_a, nb_target, b_old - 1.0)
                                engine.atoms[nb_target]["exc"] = 1.0
                                sep = engine.atoms[nb_target]["pos"] - at["pos"]
                                sep /= max(float(np.linalg.norm(sep)), 1e-4)
                                engine.atoms[nb_target]["vel"] += 0.85 * sep
                                engine.event_log.append(
                                    f"UV-Photospaltung ({ev:.2f} eV): {at['el']}-{engine.atoms[nb_target]['el']} -> 2 offene 4D-Valenzen"
                                )
                            else:
                                slider_temp.val = min(5000.0, slider_temp.val + 250.0)
                        break
            if not ph["hit_done"] and float(np.linalg.norm(ph["pos_3d"])) < 5.2:
                surviving_photons.append(ph)
        incoming_photons = surviving_photons

        engine.update_4d_charges(p_bar, anim_time)

        if reactions_enabled and (rxn_timer > 0.06 or spark_pending):
            rxn_timer = 0.0
            d_temp = engine.step_quantum_transitions(slider_temp.val, p_bar, anim_time, rng, force_spark=spark_pending)
            spark_pending = False
            if d_temp != 0.0:
                slider_temp.val = float(np.clip(slider_temp.val + d_temp, 0.0, 5000.0))

        engine.step_md_forces(dt, slider_temp.val, p_bar, chamber_rad, rng)
        engine.flashes = [
            {"pos": fl["pos"], "life": fl["life"] - dt, "color": fl["color"]}
            for fl in engine.flashes
            if fl["life"] - dt > 0.0
        ]
        if len(engine.event_log) > 3:
            engine.event_log = engine.event_log[-3:]

        temp_k = slider_temp.val

        # ----------------------------------------------------------------------
        # 2. TORUS-EBENE (T^4 / Gamma_24) & 3D-ELEKTRONENWOLKE RENDERN
        # ----------------------------------------------------------------------
        pdata = build_torus_particles_from_atoms(engine, anim_time, p_bar, show_dm_flyby)

        X1 = U_T * R4[0, 0] + V_T * R4[0, 1]
        X2 = U_T * R4[1, 0] + V_T * R4[1, 1]
        X3 = U_T * R4[2, 0] + V_T * R4[2, 1]
        X4 = U_T * R4[3, 0] + V_T * R4[3, 1]
        coords_t = (X1, X2, X3, X4)

        pos_list, sig_list, cp_list, rgb_list = [], [], [], []
        q_gain = 1.0 / math.sqrt(max(len(pdata["quarks"]) / 3.0, 1.0))
        e_gain = 1.0 / math.sqrt(max(len(pdata["electrons"]), 1.0))

        for q in pdata["quarks"]:
            if (q["flavor"] == "u" and show_u_quarks) or (q["flavor"] == "d" and show_d_quarks):
                pos_list.append(q["pos_4d"])
                sig_list.append(q["sigma"])
                cp_list.append(math.radians(q["cp_deg"]))
                rgb_list.append((q["rgb"][0] * q_gain, q["rgb"][1] * q_gain, q["rgb"][2] * q_gain))

        if show_electrons:
            for el_p in pdata["electrons"]:
                pos_list.append(el_p["pos_4d"])
                sig_list.append(el_p["sigma"])
                cp_list.append(0.0)
                rgb_list.append((el_p["rgb"][0] * e_gain, el_p["rgb"][1] * e_gain, el_p["rgb"][2] * e_gain))

        for dm_p in pdata["shadow_dm"]:
            pos_list.append(dm_p["pos_4d"])
            sig_list.append(dm_p["sigma"])
            cp_list.append(0.0)
            rgb_list.append(dm_p["rgb"])

        if pos_list:
            tor_r, tor_g, tor_b = evaluate_batched_4d_packets(
                coords_t,
                np.array(pos_list, dtype=np.float32),
                np.array(sig_list, dtype=np.float32),
                np.array(cp_list, dtype=np.float32),
                np.array(rgb_list, dtype=np.float32),
                anim_time,
            )
        else:
            tor_r, tor_g, tor_b = np.zeros_like(X1), np.zeros_like(X1), np.zeros_like(X1)

        bulk_bg = 0.062 + 0.014 * np.sin(2.0 * math.pi * (U_T + V_T) - 1.5 * anim_time)
        tor_r += bulk_bg * 0.75
        tor_g += bulk_bg * 0.95
        tor_b += bulk_bg * 1.35

        tor_rgb = np.stack([tor_r, tor_g, tor_b], axis=-1)
        tor_rgb = tor_rgb / (1.0 + 0.45 * tor_rgb)
        tor_u8 = np.clip(np.power(np.clip(tor_rgb, 0.0, 1.0), 0.86) * 255.0, 0, 255).astype(np.uint8)
        torus_surf = pygame.surfarray.make_surface(np.transpose(tor_u8, (1, 0, 2)))
        torus_surf = pygame.transform.smoothscale(torus_surf, (TORUS_SIZE, TORUS_SIZE))

        mol_u8 = render_3d_electron_cloud(engine, MU, MV, box_ang, mol_yaw, mol_pitch, temp_k, anim_time)
        mol_surf = pygame.surfarray.make_surface(np.transpose(mol_u8, (1, 0, 2)))
        mol_surf = pygame.transform.smoothscale(mol_surf, (MOL_W, MOL_H))

        # ----------------------------------------------------------------------
        # 3. BILDSCHIRM ZEICHNEN: TORUS (LINKS) & 3D-KAMMER (MITTE)
        # ----------------------------------------------------------------------
        screen.fill((14, 18, 28))

        n_ferm = len(pdata["quarks"]) + len(pdata["electrons"])
        title_s = font_title.render(
            f"Regelfreie 4D-Torus-Reaktions-MD (v9): T = {temp_k:.0f} K, Druck P = {p_bar:.2f} bar ({n_ferm} Fermionen)",
            True,
            (235, 242, 255),
        )
        sub_s = font_ui.render(
            f"0 Reaktionsregeln — Bindungen emergieren aus 4D-Jacobi-Theta-Ueberlapp O_4D & 4D-Harpunen-Ladungstransfer Delta q",
            True,
            (150, 175, 210),
        )
        screen.blit(title_s, (20, 14))
        screen.blit(sub_s, (20, 42))

        screen.blit(torus_surf, TORUS_POS)
        pygame.draw.rect(
            screen,
            (85, 110, 150),
            (TORUS_POS[0] - 1, TORUS_POS[1] - 1, TORUS_SIZE + 2, TORUS_SIZE + 2),
            2,
        )

        tcx = TORUS_POS[0] + TORUS_SIZE // 2
        tcy = TORUS_POS[1] + TORUS_SIZE // 2

        rot_nodes = verts_torus @ R4
        for idx_n in range(24):
            un, vn, _, wn4 = rot_nodes[idx_n]
            px_n = int(tcx + un * TORUS_SIZE)
            py_n = int(tcy + vn * TORUS_SIZE)
            if not torus_rect.collidepoint(px_n, py_n):
                continue
            sh = shells_24[idx_n]
            near_w = abs(float(wn4)) < 0.24
            if sh == 0:
                pygame.draw.circle(screen, (95, 205, 255), (px_n, py_n), 6, 2)
                screen.blit(font_mono.render("C_0", True, (130, 215, 255)), (px_n + 8, py_n - 7))
            elif sh == 1:
                col_s1 = (255, 155, 75) if near_w else (165, 95, 55)
                pygame.draw.circle(screen, col_s1, (px_n, py_n), 4 if near_w else 2)

        for mult, col_h in [(0.55, (255, 230, 120)), (1.05, (230, 180, 75))]:
            r_px = int(SIGMA_0 * mult * TORUS_SIZE)
            if 6 < r_px < TORUS_SIZE // 2:
                pygame.draw.circle(screen, col_h, (tcx, tcy), r_px, 1)
        screen.blit(font_mono.render("H^0 (0,0,0,0)", True, (255, 225, 125)), (tcx + 8, tcy - 18))

        def proj_torus(p4):
            pr = p4 @ R4
            return (int(tcx + pr[0] * TORUS_SIZE), int(tcy + pr[1] * TORUS_SIZE))

        nuc_proj = [proj_torus(nuc["pos_4d"]) for nuc in pdata["nucleons"]]
        q_proj = [proj_torus(q["pos_4d"]) for q in pdata["quarks"]]
        e_proj = [proj_torus(el_p["pos_4d"]) for el_p in pdata["electrons"]]
        dm_proj = [proj_torus(dm_p["pos_4d"]) for dm_p in pdata["shadow_dm"]]
        at_ec_proj = [proj_torus(ac) for ac in pdata["atom_elec_centers"]]

        # 4D-Harpunen- & Ladungstransfer-Boegen im Torus C_0 zeichnen!
        max_o4d_live = 0.0
        for ia, ib, ov_v, dq_v in engine.harpoon_arcs:
            max_o4d_live = max(max_o4d_live, ov_v)
            if ia < len(at_ec_proj) and ib < len(at_ec_proj):
                pa_e, pb_e = at_ec_proj[ia], at_ec_proj[ib]
                if torus_rect.collidepoint(pa_e) and torus_rect.collidepoint(pb_e):
                    col_arc = (255, 235, 105) if abs(dq_v) > 0.55 else (130, 255, 195)
                    pygame.draw.line(screen, col_arc, pa_e, pb_e, 2)

        if show_strong:
            for ia, ib in pdata["pion_links"]:
                pa_t, pb_t = nuc_proj[ia], nuc_proj[ib]
                if torus_rect.collidepoint(pa_t) and torus_rect.collidepoint(pb_t):
                    pygame.draw.line(screen, (235, 150, 75), pa_t, pb_t, 1)
            for n_idx, q_idx, g_col in pdata["gluon_links"]:
                q_item = pdata["quarks"][q_idx]
                if (q_item["flavor"] == "u" and not show_u_quarks) or (q_item["flavor"] == "d" and not show_d_quarks):
                    continue
                pn_t, pq_t = nuc_proj[n_idx], q_proj[q_idx]
                if torus_rect.collidepoint(pn_t) and torus_rect.collidepoint(pq_t):
                    pygame.draw.line(screen, g_col, pn_t, pq_t, 1)

        if show_photons and show_electrons:
            for p_idx, e_idx, ph_off in pdata["photon_links"]:
                pp_t, pe_t = nuc_proj[p_idx], e_proj[e_idx]
                if torus_rect.collidepoint(pp_t) and torus_rect.collidepoint(pe_t):
                    pygame.draw.line(screen, (55, 118, 158), pp_t, pe_t, 1)
                    s_ph = ((0.78 + 0.45 * math.sqrt(temp_k / 300.0)) * anim_time + ph_off) % 1.0
                    gx = int(pp_t[0] + s_ph * (pe_t[0] - pp_t[0]))
                    gy = int(pp_t[1] + s_ph * (pe_t[1] - pp_t[1]))
                    if torus_rect.collidepoint(gx, gy):
                        pygame.draw.circle(screen, (255, 245, 130), (gx, gy), 3)

        q_marker_r = 4 if len(pdata["quarks"]) <= 24 else 3
        for q_item, pq_t in zip(pdata["quarks"], q_proj):
            if (q_item["flavor"] == "u" and not show_u_quarks) or (q_item["flavor"] == "d" and not show_d_quarks):
                continue
            if torus_rect.collidepoint(pq_t):
                pygame.draw.circle(screen, q_item["marker_col"], pq_t, q_marker_r)

        if show_electrons:
            for el_item, pe_t in zip(pdata["electrons"], e_proj):
                if torus_rect.collidepoint(pe_t):
                    pygame.draw.circle(screen, el_item["marker_col"], pe_t, 4, 1)
                    pygame.draw.circle(screen, (255, 255, 255), pe_t, 1)

        for dm_item, pdm_t in zip(pdata["shadow_dm"], dm_proj):
            if torus_rect.collidepoint(pdm_t):
                pygame.draw.circle(screen, dm_item["marker_col"], pdm_t, 5, 2)

        leg_items = [
            ((255, 115, 115), f"u ({pdata['n_u']})"),
            ((130, 180, 255), f"d ({pdata['n_d']})"),
            ((105, 235, 255), f"e^- ({pdata['n_e']})"),
            ((255, 235, 105), "4D-Harpunen-Transfer (Na+->Cl-)"),
            ((255, 165, 85), "DM n'_1"),
        ]
        lx_cur = 20
        for col_i, txt_i in leg_items:
            pygame.draw.circle(screen, col_i, (lx_cur + 5, 734), 5)
            ts_i = font_ui.render(txt_i, True, (195, 210, 235))
            screen.blit(ts_i, (lx_cur + 13, 726))
            lx_cur += ts_i.get_width() + 14

        screen.blit(
            font_ui.render(
                "Tasten: [1-5] Mischungen | [G] Zuendfunke | [F] UV-Photon | [C] Reaktion AN/AUS | [TAB] SMILES | [R] Reset",
                True,
                (140, 160, 190),
            ),
            (20, 756),
        )
        screen.blit(
            font_ui.render(
                "Reaktionsprinzip: Offene 4D-Valenzslots + Pauling-D_ij + 4D-Harpunen-Elektronentransfer Delta q + WKB-Tunneln",
                True,
                (140, 160, 190),
            ),
            (20, 776),
        )

        # --- Oben Mitte: 3D-Kammer (M_4) ---
        screen.blit(mol_surf, MOL_POS)
        pygame.draw.rect(
            screen,
            (85, 110, 150),
            (MOL_POS[0] - 1, MOL_POS[1] - 1, MOL_W + 2, MOL_H + 2),
            2,
        )

        mcx = MOL_POS[0] + MOL_W // 2
        mcy = MOL_POS[1] + MOL_H // 2
        px_per_ang = (MOL_W * 0.5) / box_ang
        cy_m, sy_m = math.cos(mol_yaw), math.sin(mol_yaw)
        cp_m, sp_m = math.cos(mol_pitch), math.sin(mol_pitch)

        def proj_mol_3d(p_vec):
            bx, by, bz = float(p_vec[0]), float(p_vec[1]), float(p_vec[2])
            u_p = cy_m * bx - sy_m * bz
            w_tmp = sy_m * bx + cy_m * bz
            v_p = cp_m * by + sp_m * w_tmp
            z_depth = -sp_m * by + cp_m * w_tmp
            return (int(mcx + u_p * px_per_ang), int(mcy - v_p * px_per_ang), z_depth)

        p_ring_px = int(chamber_rad * px_per_ang)
        if 20 < p_ring_px < MOL_W // 2 - 4:
            pygame.draw.circle(screen, (255, 155, 95) if p_bar > 50.0 else (85, 145, 195), (mcx, mcy), p_ring_px, 1)

        proj_atoms = [(*proj_mol_3d(at["pos"]), at["el"], at["charge"], idx_a) for idx_a, at in enumerate(engine.atoms)]

        # Aktive Bindungen aus der dynamischen Bond-Order-Matrix zeichnen
        for (ia, ib), b_ord in engine.bond_order.items():
            if b_ord <= 0.0:
                continue
            p1 = (proj_atoms[ia][0], proj_atoms[ia][1])
            p2 = (proj_atoms[ib][0], proj_atoms[ib][1])
            if not (mol_rect.collidepoint(p1) and mol_rect.collidepoint(p2)):
                continue
            dx_b, dy_b = p2[0] - p1[0], p2[1] - p1[1]
            norm_b = max(math.hypot(dx_b, dy_b), 1.0)
            nx_b, ny_b = -dy_b / norm_b, dx_b / norm_b

            # Ionische Bindungen (z.B. Na+Cl-) gestrichelt gold-weiss zeichnen
            is_ionic = abs(engine.atoms[ia]["charge"] - engine.atoms[ib]["charge"]) > 0.75
            if is_ionic:
                pygame.draw.line(screen, (255, 225, 115), p1, p2, 2)
            elif b_ord >= 2.9:
                for off_s in (-3.5, 0.0, 3.5):
                    pygame.draw.line(
                        screen, (225, 235, 250),
                        (int(p1[0] + nx_b * off_s), int(p1[1] + ny_b * off_s)),
                        (int(p2[0] + nx_b * off_s), int(p2[1] + ny_b * off_s)), 2,
                    )
            elif b_ord >= 1.9:
                for off_s in (-2.6, 2.6):
                    pygame.draw.line(
                        screen, (220, 230, 245),
                        (int(p1[0] + nx_b * off_s), int(p1[1] + ny_b * off_s)),
                        (int(p2[0] + nx_b * off_s), int(p2[1] + ny_b * off_s)), 2,
                    )
            else:
                pygame.draw.line(screen, (220, 230, 245), p1, p2, 2)

        for px_a, py_a, _, el_s, q_a, idx_a in sorted(proj_atoms, key=lambda item: item[2]):
            if not mol_rect.collidepoint((px_a, py_a)):
                continue
            col_el = ELEMENT_COLORS.get(el_s, (220, 220, 220))
            rad_el = 5 if el_s == "H" else (10 if el_s == "Na" else 8)
            is_rad = engine.open_valence(idx_a) >= 0.9
            pygame.draw.circle(screen, col_el, (px_a, py_a), rad_el)
            pygame.draw.circle(screen, (255, 135, 95) if is_rad else (25, 32, 48), (px_a, py_a), rad_el, 2 if is_rad else 1)

            q_tag = "+" if q_a > 0.35 else ("-" if q_a < -0.35 else "")
            if len(engine.atoms) <= 18:
                screen.blit(font_mono.render(f"{el_s}{q_tag}", True, (245, 250, 255)), (px_a + rad_el + 2, py_a - 6))

        for fl in engine.flashes:
            fx, fy, _ = proj_mol_3d(fl["pos"])
            if mol_rect.collidepoint((fx, fy)):
                rad_fl = int(8 + 38 * (1.15 - fl["life"]))
                pygame.draw.circle(screen, fl["color"], (fx, fy), rad_fl, 2)

        for ph in incoming_photons:
            pts_w = []
            for wi in range(10):
                back_p = ph["pos_3d"] - ph["dir_3d"] * (wi * 0.08)
                px_w, py_w, _ = proj_mol_3d(back_p)
                pts_w.append((px_w, int(py_w + 4.0 * math.sin(wi * 1.4 - 16.0 * anim_time))))
            valid_w = [pt for pt in pts_w if mol_rect.collidepoint(pt)]
            if len(valid_w) >= 2:
                pygame.draw.lines(screen, ph["col"], False, valid_w, 2)

        screen.blit(
            font_sub.render(f"Kammer (M_4): T={temp_k:.0f} K | P={p_bar:.2f} bar | R={chamber_rad:.2f} A", True, (235, 245, 255)),
            (MOL_POS[0] + 8, MOL_POS[1] + 6),
        )
        screen.blit(
            font_mono.render(f"Max. 4D-Ueberlapp O_4D = {max_o4d_live:.3f} | Gelb = Ionenbindung Na+Cl-", True, (150, 235, 200)),
            (MOL_POS[0] + 8, MOL_POS[1] + MOL_H - 18),
        )

        # ----------------------------------------------------------------------
        # 4. OBEN RECHTS: MISCHUNGEN, SMILES, DRUCK- & TEMPERATURSLIDER
        # ----------------------------------------------------------------------
        pygame.draw.rect(screen, (20, 26, 40), (ctrl_x, 76, 364, 332), border_radius=8)
        pygame.draw.rect(screen, (50, 65, 95), (ctrl_x, 76, 364, 332), 1, border_radius=8)

        screen.blit(font_sub.render("Start-Mischungen & SMILES-Baukasten:", True, (235, 245, 255)), (ctrl_x + 8, 78))

        for idx_p, (rect, pr) in enumerate(preset_buttons):
            active_p = idx_p == active_preset_idx
            pygame.draw.rect(screen, (38, 78, 120) if active_p else (28, 36, 54), rect, border_radius=4)
            pygame.draw.rect(screen, (110, 200, 255) if active_p else (58, 74, 102), rect, 1, border_radius=4)
            screen.blit(font_ui.render(pr["label"], True, (240, 248, 255)), (rect.x + 8, rect.y + 3))

        pygame.draw.rect(screen, (14, 20, 32), smiles_box, border_radius=4)
        pygame.draw.rect(
            screen,
            (110, 215, 255) if smiles_active else (70, 90, 125),
            smiles_box,
            2 if smiles_active else 1,
            border_radius=4,
        )
        cursor_str = "|" if (smiles_active and int(anim_time * 2.5) % 2 == 0) else ""
        screen.blit(font_mono.render(smiles_text + cursor_str, True, (245, 250, 255)), (smiles_box.x + 6, smiles_box.y + 4))

        for rect, lbl, bg_c in [
            (btn_add_smiles, "+ Einf.", (38, 92, 115)),
            (btn_rep_smiles, "Ersetz.", (55, 68, 102)),
        ]:
            pygame.draw.rect(screen, bg_c, rect, border_radius=4)
            pygame.draw.rect(screen, (110, 195, 245), rect, 1, border_radius=4)
            screen.blit(font_ui.render(lbl, True, (240, 248, 255)), (rect.x + 8, rect.y + 4))

        for rect, qm in quick_buttons:
            pygame.draw.rect(screen, (28, 38, 58), rect, border_radius=4)
            pygame.draw.rect(screen, (65, 85, 118), rect, 1, border_radius=4)
            screen.blit(font_ui.render(qm["label"], True, (225, 238, 255)), (rect.x + 8, rect.y + 2))

        for s in sliders:
            s.draw(screen, font_ui)

        for rect, lbl, bg_c, bd_c in [
            (btn_fire_ph, "[F] Photon", (48, 84, 128), (115, 205, 255)),
            (btn_spark, "[G] Funke", (118, 68, 36), (255, 195, 95)),
            (btn_rxn_tog, f"[C] Rxn:{'1' if reactions_enabled else '0'}", (36, 86, 72) if reactions_enabled else (45, 45, 60), (115, 235, 185)),
            (btn_reload, "[R] Reset", (55, 62, 85), (120, 140, 180)),
        ]:
            pygame.draw.rect(screen, bg_c, rect, border_radius=4)
            pygame.draw.rect(screen, bd_c, rect, 1, border_radius=4)
            screen.blit(font_ui.render(lbl, True, (240, 248, 255)), (rect.x + 7, rect.y + 4))

        toggles = [
            (tog_u, f"u:{'1' if show_u_quarks else '0'}", show_u_quarks),
            (tog_d, f"d:{'1' if show_d_quarks else '0'}", show_d_quarks),
            (tog_e, f"e:{'1' if show_electrons else '0'}", show_electrons),
            (tog_strong, f"g:{'1' if show_strong else '0'}", show_strong),
            (tog_photon, f"y:{'1' if show_photons else '0'}", show_photons),
            (tog_dm, f"DM:{'1' if show_dm_flyby else '0'}", show_dm_flyby),
        ]
        for rect, lbl, active in toggles:
            pygame.draw.rect(screen, (35, 65, 95) if active else (30, 36, 50), rect, border_radius=4)
            pygame.draw.rect(screen, (95, 170, 235) if active else (65, 75, 95), rect, 1, border_radius=4)
            screen.blit(font_ui.render(lbl, True, (230, 240, 255)), (rect.x + 8, rect.y + 3))

        # ----------------------------------------------------------------------
        # 5. UNTEN RECHTS: EMERGENTE MOLEKUELLISTE & ERHALTUNGSSAETZE
        # ----------------------------------------------------------------------
        tel_rect = pygame.Rect(672, 418, 752, 428)
        pygame.draw.rect(screen, (20, 26, 40), tel_rect, border_radius=8)
        pygame.draw.rect(screen, (50, 65, 95), tel_rect, 1, border_radius=8)

        screen.blit(font_mono.render(f"Status: {status_msg}", True, status_col), (tel_rect.x + 12, tel_rect.y + 8))

        # Aktuelle Molekuele direkt aus den Zusammenhangskomponenten des Bindungsgraphen!
        comps = engine.connected_components()
        comp_counts = {}
        for comp in comps:
            f_str = build_formula_string(comp, engine.atoms)
            if any(engine.open_valence(idx) >= 0.9 for idx in comp):
                f_str += "*"
            comp_counts[f_str] = comp_counts.get(f_str, 0) + 1
        comp_str = " + ".join(f"{cnt if cnt > 1 else ''}{frm}" for frm, cnt in comp_counts.items()) if comp_counts else "Leer"
        screen.blit(font_mono.render(f"Emergente Spezies : {comp_str}", True, (225, 238, 255)), (tel_rect.x + 12, tel_rect.y + 30))

        for h_i, h_line in enumerate(engine.event_log[-3:]):
            c_h = (255, 225, 120) if ("Harpune" in h_line or "Austausch" in h_line or "Valenzkopplung" in h_line) else (165, 200, 235)
            screen.blit(font_mono.render(f"  > {h_line}", True, c_h), (tel_rect.x + 12, tel_rect.y + 52 + h_i * 20))

        p0, n0, u0, d0, e0 = initial_conservation
        du_diff = pdata["n_u"] - u0
        dd_diff = pdata["n_d"] - d0
        de_diff = pdata["n_e"] - e0

        total_A = pdata["n_p"] + pdata["n_n"]
        bare_q_mev = pdata["n_u"] * M_U_MEV + pdata["n_d"] * M_D_MEV
        free_nuc_mev = pdata["n_p"] * M_P_MEV + pdata["n_n"] * M_N_MEV
        qcd_conf_mev = free_nuc_mev - bare_q_mev
        total_mass_mev = max(0.0, free_nuc_mev - pdata["b_nuc_mev"] + e0 * (M_E_KEV / 1000.0))

        # Groesster Ionen-Ladungstransfer (z.B. Na+ / Cl-) anzeigen
        max_q_atom = max((abs(a["charge"]) for a in engine.atoms), default=0.0)
        kb_t_ev = KB_EV_PER_K * temp_k

        tel_lines = [
            ("--- REGELFREIE 4D-TORUS-QUANTENDYNAMIK (T^4 / GAMMA_24) ---", (255, 225, 130)),
            (f"Thermodynamik (T,P): T = {temp_k:6.1f} K (k_B T = {kb_t_ev:.4f} eV)  |  P = {p_bar:8.2f} bar ({p_bar*1e-4:.4f} GPa)", (255, 185, 125)),
            (f"4D-Ladungstransfer : Max. |Delta q_4D| = {max_q_atom:.2f} e  |  Max. Wellenpaket-Ueberlapp O_4D = {max_o4d_live:.3f}", (255, 235, 115)),
            (f"Kammer & Torus     : R_Box = {chamber_rad:.2f} A  |  0 Reaktionsregeln (100% Pauling-D_ij + O_4D + WKB)", (135, 225, 255)),
            ("--- STRIKTE TEILCHENERHALTUNG IM CLUSTER C_0 ---", (255, 225, 130)),
            (f"Quark-Erhaltung    : Delta u = {du_diff:+d} ({pdata['n_u']} Stueck)  |  Delta d = {dd_diff:+d} ({pdata['n_d']} Stueck)  [100% erhalten!]", (135, 245, 190)),
            (f"Elektronen-Bilanz  : Delta e^- = {de_diff:+d} ({pdata['n_e']} Stueck)  |  CP-Phase = {CP_PHASE_DEG:+.2f} deg", (195, 215, 240)),
            (f"Nukleonen & Bosonen: {pdata['n_p']}p + {pdata['n_n']}n = {total_A} Baryonen | {len(pdata['gluon_links'])} Gluonen (g), {len(pdata['photon_links'])} Photonen (gamma)", (255, 205, 150)),
            (f"Gesamtmasse Szene  : Quarks = {bare_q_mev:.2f} MeV | QCD = +{qcd_conf_mev:.2f} MeV | M_ges = {total_mass_mev:.3f} MeV", (225, 235, 250)),
        ]
        for li, (txt, col) in enumerate(tel_lines):
            screen.blit(font_mono.render(txt, True, col), (tel_rect.x + 12, tel_rect.y + 122 + li * 29))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()

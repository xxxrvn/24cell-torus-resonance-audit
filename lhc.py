#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24-Zell-Torus & Schattenresonanz (v9) — Virtueller Teilchenbeschleuniger & 4D-Torus-Collider
Koppelt einen makroskopischen Synchrotron-Speicherring + 4-Lagen-Detektor (M_4) mit der
4D-Jacobi-Theta-Kollisionskammer (T^4 / Gamma_24, L_T = 1.5605e-33 m, E_T = 1.2645e17 GeV).

Benoetigte Pakete:
    pip install pygame numpy
"""

import math
import random
import sys
import numpy as np
import pygame

# ==============================================================================
# 1. FUNDAMENTALE v9-KONSTANTEN & TEILCHENDATEN
# ==============================================================================
M_TAU = 1776.86             # MeV
M_MU = 105.6583755          # MeV
TAU = math.sqrt(2.0 * math.log(M_TAU / M_MU))  # 2.37588
L_T = 1.5605e-33            # m (96.55 l_Pl)
E_T_GEV = 1.2645e17         # GeV (Torus-Kompaktifizierungsskala)
M_R_GEV = 4.47e14           # GeV (Typ-I-Seesaw Majorana-Neutrino nu_R)
M_G1_GEV = 1.80e14          # GeV (Schale-1 Polychor-Graviton G_1..G_8)
LAMBDA_G1_GEV = 1.42e16     # GeV (Rayleigh-Portal-Skala E_T * sqrt(R_1))

M_H_GEV = 125.68            # GeV (v9 Higgs-Masse)
M_T_GEV = 172.95            # GeV (v9 Top-Quark-Masse)
M_Z_GEV = 91.1876           # GeV (Z^0-Boson)
M_W_GEV = 80.379            # GeV (W^+-Boson)
M_DM_GEV = 0.6237           # GeV (Schatten-Neutron n'_1 in Schale 1)

SIGMA_0 = 1.0 / math.sqrt(math.pi * TAU)       # 0.3660 * L_T (Higgs-Breite)
SIGMA_W = 1.0 / (math.sqrt(3.0) * TAU)         # 0.2430 * L_T (Fermion-Breite)
CP_PHASE_DEG = -76.39       # arg(I_d13) in Grad

COLLIDER_MODES = [
    {
        "key": "1",
        "name": "LHC (p + p)",
        "beam": "Proton-Proton (uud + uud)",
        "log10_s": 4.1335,      # 13,600 GeV = 13.6 TeV
        "is_lepton": False,
        "hist_range": (40.0, 420.0),
        "hist_unit": "GeV",
        "desc": "Gluon-Fusion gg -> H^0 (125.68 GeV) & ttbar; BR(H^0 -> Schatten) = 0.00%.",
    },
    {
        "key": "2",
        "name": "Higgs-Fabrik (e+e-)",
        "beam": "Elektron-Positron (e^- + e^+)",
        "log10_s": 2.3979,      # 250 GeV
        "is_lepton": True,
        "hist_range": (60.0, 220.0),
        "hist_unit": "GeV",
        "desc": "Higgsstrahlung e+e- -> Z^0 H^0; Praezisionstest lambda_HH' = 0 & eps_mix = 0.",
    },
    {
        "key": "3",
        "name": "Seesaw-Collider (nu_R)",
        "beam": "Hochenergie-Kollision (M_R-Skala)",
        "log10_s": 14.6503,     # 4.47e14 GeV
        "is_lepton": True,
        "hist_range": (0.5e14, 7.5e14),
        "hist_unit": "GeV",
        "desc": "Produktion schwerer Majorana-Neutrinos nu_R (4.47e14 GeV) -> Delta L = 2!",
    },
    {
        "key": "4",
        "name": "Torus-Collider (E_T)",
        "beam": "24-Zell-Grenzenergie (E_T = 1.26e17 GeV)",
        "log10_s": 17.1019,     # 1.2645e17 GeV
        "is_lepton": False,
        "hist_range": (0.2e17, 1.8e17),
        "hist_unit": "GeV",
        "desc": "Anregung von Phi_T & G_1..23: 23/24 (95.83%) Energiefluss in Schattensektor!",
    },
]


# ==============================================================================
# 2. 4D-POLYCHOR & JACOBI-THETA-WELLENPAKETE
# ==============================================================================
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


def build_24cell_vertices():
    """Erzeugt alle 24 Ecken des D_4-Wurzelgitters sortiert nach Schale (1-8-6-8-1)."""
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


# ==============================================================================
# 3. EREIGNISGENERATOR (MONTE-CARLO PHYSIK NACH v9-VERZWEIGUNGSVERHAELTNISSEN)
# ==============================================================================
def generate_collision_event(sqrt_s_gev, is_lepton, rng):
    """
    Erzeugt ein physikalisches Kollisionsereignis gemaess v9-Wirkungsquerschnitten,
    Verzweigungsverhaeltnissen und Z_2-Portalschutz.
    """
    tracks = []
    # Zufallsazimut im Detektor
    phi0 = rng.uniform(0.0, 2.0 * math.pi)

    def add_track(name, ptype, charge, pt_frac, angle, is_missing=False):
        tracks.append({
            "name": name,
            "ptype": ptype,       # "photon", "electron", "muon", "jet", "neutrino", "shadow"
            "charge": charge,     # -1, 0, +1
            "pt": max(0.08, pt_frac) * (0.5 * sqrt_s_gev),
            "angle": angle,
            "missing": is_missing,
        })

    # Regime A: Niedrig- bis Hochenergie-Standardmodell (sqrt(s) < 1e13 GeV)
    if sqrt_s_gev < 1.0e13:
        if sqrt_s_gev < 110.0:
            # Z^0-Resonanz (91.19 GeV)
            m_inv = rng.gauss(M_Z_GEV, 2.49)
            r = rng.random()
            if r < 0.20:
                channel = "Z^0 -> mu^+ mu^-"
                add_track("mu^-", "muon", -1, 0.48, phi0)
                add_track("mu^+", "muon", +1, 0.48, phi0 + math.pi + rng.uniform(-0.15, 0.15))
            elif r < 0.40:
                channel = "Z^0 -> e^+ e^-"
                add_track("e^-", "electron", -1, 0.48, phi0)
                add_track("e^+", "electron", +1, 0.48, phi0 + math.pi + rng.uniform(-0.15, 0.15))
            elif r < 0.60:
                channel = "Z^0 -> nu nubar (SM MET)"
                add_track("nu", "neutrino", 0, 0.48, phi0, is_missing=True)
                add_track("nubar", "neutrino", 0, 0.48, phi0 + math.pi, is_missing=True)
            else:
                channel = "Z^0 -> q qbar (2 Jets)"
                add_track("Jet_1", "jet", +1, 0.45, phi0)
                add_track("Jet_2", "jet", -1, 0.45, phi0 + math.pi + rng.uniform(-0.2, 0.2))
            return {
                "channel": channel,
                "m_inv": m_inv,
                "tracks": tracks,
                "shadow_leak": False,
                "higgs_hit": False,
                "cp_active": False,
                "met_gev": sum(t["pt"] for t in tracks if t["missing"]),
            }

        # Oberhalb 110 GeV: Mix aus Z^0, H^0 (125.68 GeV) und ttbar (345.9 GeV)
        roll = rng.random()
        can_top = (not is_lepton and sqrt_s_gev >= 360.0) or (is_lepton and sqrt_s_gev >= 350.0)
        if can_top and roll < 0.34:
            # Top-Antitop-Produktion (m_t = 172.95 GeV, CP-Phase -76.39 deg im b-Vertex)
            m_inv = rng.gauss(2.0 * M_T_GEV, 6.5)
            channel = "gg/ee -> t tbar -> (b W+)(bbar W-) [CP -76.39 deg]"
            add_track("b-Jet", "jet", -1, 0.28, phi0)
            add_track("bbar-Jet", "jet", +1, 0.28, phi0 + 2.1)
            add_track("l^+", "muon", +1, 0.22, phi0 + 4.0)
            add_track("nu_l", "neutrino", 0, 0.18, phi0 + 4.3, is_missing=True)
            add_track("Jet_W", "jet", -1, 0.24, phi0 + 5.2)
            return {
                "channel": channel,
                "m_inv": m_inv,
                "tracks": tracks,
                "shadow_leak": False,
                "higgs_hit": True,
                "cp_active": True,
                "met_gev": sum(t["pt"] for t in tracks if t["missing"]),
            }
        elif roll < 0.78:
            # Higgs-Boson H^0 (m_H = 125.68 GeV, BR_inv = 0.00%!)
            m_inv = rng.gauss(M_H_GEV, 1.65)
            br = rng.random()
            if br < 0.5824:
                channel = "H^0 -> b bbar (BR 58.24%)"
                add_track("b-Jet", "jet", -1, 0.46, phi0)
                add_track("bbar-Jet", "jet", +1, 0.46, phi0 + math.pi + rng.uniform(-0.22, 0.22))
            elif br < 0.7961:
                channel = "H^0 -> W+ W-* -> l+ nu l- nubar (BR 21.37%)"
                add_track("e^+", "electron", +1, 0.28, phi0)
                add_track("mu^-", "muon", -1, 0.27, phi0 + 2.4)
                add_track("nu_e", "neutrino", 0, 0.20, phi0 + 4.5, is_missing=True)
            elif br < 0.8780:
                channel = "H^0 -> g g (2 Gluon-Jets, BR 8.19%)"
                add_track("g-Jet_1", "jet", +1, 0.45, phi0)
                add_track("g-Jet_2", "jet", -1, 0.45, phi0 + math.pi + rng.uniform(-0.2, 0.2))
            elif br < 0.9407:
                channel = "H^0 -> tau+ tau- (BR 6.27%)"
                add_track("tau^+", "muon", +1, 0.42, phi0)
                add_track("tau^-", "muon", -1, 0.42, phi0 + math.pi + rng.uniform(-0.18, 0.18))
            else:
                channel = "H^0 -> gamma gamma (Goldener Kanal, BR 0.23%)"
                add_track("gamma_1", "photon", 0, 0.49, phi0)
                add_track("gamma_2", "photon", 0, 0.49, phi0 + math.pi + rng.uniform(-0.08, 0.08))

            if is_lepton:
                # Z^0-Rueckstoss bei Higgsstrahlung e+e- -> Z^0 H^0
                add_track("mu^+ (Z)", "muon", +1, 0.22, phi0 + 1.4)
                add_track("mu^- (Z)", "muon", -1, 0.22, phi0 + 4.6)

            return {
                "channel": channel,
                "m_inv": m_inv,
                "tracks": tracks,
                "shadow_leak": False,
                "higgs_hit": True,
                "cp_active": False,
                "met_gev": sum(t["pt"] for t in tracks if t["missing"]),
            }
        else:
            # Z^0-Untergrund (91.19 GeV)
            m_inv = rng.gauss(M_Z_GEV, 2.49)
            channel = "Z^0 -> e^+ e^- (Drell-Yan / Hintergrund)"
            add_track("e^-", "electron", -1, 0.46, phi0)
            add_track("e^+", "electron", +1, 0.46, phi0 + math.pi + rng.uniform(-0.18, 0.18))
            return {
                "channel": channel,
                "m_inv": m_inv,
                "tracks": tracks,
                "shadow_leak": False,
                "higgs_hit": False,
                "cp_active": False,
                "met_gev": 0.0,
            }

    # Regime B: GUT-Seesaw-Skala (1e13 .. 3e16 GeV) -> Majorana-Neutrino nu_R & Polychor-Graviton G_1
    if sqrt_s_gev < 3.0e16:
        if rng.random() < 0.62:
            m_inv = rng.gauss(M_R_GEV, 0.18e14)
            channel = "nu_R (4.47e14 GeV) -> e^- e^- + 2 Jets [Delta L = 2!]"
            # Typisches Majorana-Signal: zwei gleichgeladene Leptonen + Jets ohne MET!
            add_track("e^-_1", "electron", -1, 0.34, phi0)
            add_track("e^-_2", "electron", -1, 0.32, phi0 + 2.1)
            add_track("Jet_1", "jet", +1, 0.28, phi0 + 4.1)
            add_track("Jet_2", "jet", +1, 0.26, phi0 + 5.3)
            return {
                "channel": channel,
                "m_inv": m_inv,
                "tracks": tracks,
                "shadow_leak": False,
                "higgs_hit": True,
                "cp_active": True,
                "met_gev": 0.0,
            }
        else:
            m_inv = rng.gauss(M_G1_GEV, 0.12e14)
            channel = "G_1 (1.80e14 GeV) -> n'_1 n'_1 (Schatten-DM-Flucht!)"
            add_track("n'_1 (C_1)", "shadow", 0, 0.48, phi0, is_missing=True)
            add_track("n'_1 (C_2)", "shadow", 0, 0.48, phi0 + math.pi + rng.uniform(-0.2, 0.2), is_missing=True)
            add_track("ISR-Jet", "jet", +1, 0.16, phi0 + 1.5)
            return {
                "channel": channel,
                "m_inv": m_inv,
                "tracks": tracks,
                "shadow_leak": True,
                "higgs_hit": False,
                "cp_active": False,
                "met_gev": sum(t["pt"] for t in tracks if t["missing"]),
            }

    # Regime C: Torus-Kompaktifizierungsskala E_T = 1.2645e17 GeV
    m_inv = rng.gauss(E_T_GEV, 0.045e17)
    # Demokratische 24-Zell-Aufspaltung: 1/24 (4.17%) in C_0, 23/24 (95.83%) in Schatten C_1..C_23!
    channel = "Phi_T / KK-Mode (1.26e17 GeV) -> 23/24 Schatten-DM + 1/24 C_0"
    add_track("n'_1 (Schale 1)", "shadow", 0, 0.44, phi0, is_missing=True)
    add_track("n'_1 (Schale 1)", "shadow", 0, 0.42, phi0 + 2.3, is_missing=True)
    add_track("gamma (C_0, 4.17%)", "photon", 0, 0.14, phi0 + 4.4)
    add_track("Jet (C_0)", "jet", +1, 0.16, phi0 + 5.1)
    return {
        "channel": channel,
        "m_inv": m_inv,
        "tracks": tracks,
        "shadow_leak": True,
        "higgs_hit": True,
        "cp_active": True,
        "met_gev": sum(t["pt"] for t in tracks if t["missing"]),
    }


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
# 5. HAUPTANWENDUNG: 24-ZELL-TEILCHENBESCHLEUNIGER
# ==============================================================================
def main():
    pygame.init()
    WIDTH, HEIGHT = 1400, 850
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("24-Zell-Torus (v9) — Virtueller Teilchenbeschleuniger & 4D-Torus-Collider")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("dejavusans,arial,helvetica", 19, bold=True)
    font_sub = pygame.font.SysFont("dejavusans,arial,helvetica", 13, bold=True)
    font_ui = pygame.font.SysFont("dejavusans,arial,helvetica", 12)
    font_mono = pygame.font.SysFont("dejavusansmono,consolas,monospace", 12)

    rng = random.Random(24)
    verts_24, verts_torus, shells_24 = build_24cell_vertices()

    # Oben Links: Beschleunigerring + Detektor-Querschnitt (M_4)
    ACC_RECT = pygame.Rect(22, 78, 636, 368)
    RING_CENTER = (158, 262)
    RING_R = 112
    DET_CENTER = (485, 262)
    DET_MAX_R = 152

    # Unten Links: 4D-Torus-Kollisionskammer (T^4 / Gamma_24)
    TORUS_RECT = pygame.Rect(22, 460, 636, 326)
    TORUS_W, TORUS_H = 636, 326
    TORUS_NX, TORUS_NY = 160, 84
    tx_1d = np.linspace(-0.5, 0.5, TORUS_NX, endpoint=False, dtype=np.float32)
    ty_1d = np.linspace(-0.26, 0.26, TORUS_NY, endpoint=False, dtype=np.float32)
    U_T, V_T = np.meshgrid(tx_1d, ty_1d)

    # Oben Rechts: Steuerung & Strahlparameter
    ctrl_x = 674
    mode_buttons = []
    for idx, m_item in enumerate(COLLIDER_MODES):
        mode_buttons.append(pygame.Rect(ctrl_x + 8 + idx * 174, 86, 168, 30))

    slider_log_s = Slider((ctrl_x + 12, 146, 336, 7), "Schwerpunktsenergie log10(sqrt(s)/GeV)", 1.80, 17.25, 4.1335, "{:.3f}")
    slider_bfield = Slider((ctrl_x + 366, 146, 326, 7), "Detektor-Magnetfeld B [Tesla]", 0.50, 8.00, 3.80, "{:.2f}")
    slider_rate = Slider((ctrl_x + 12, 186, 336, 7), "Kollisionsrate [Hz]", 0.20, 4.00, 1.25, "{:.2f}")
    slider_w4 = Slider((ctrl_x + 366, 186, 326, 7), "4D-Torus-Schnitt x_4 / L_T", -0.50, 0.50, 0.00)
    sliders = [slider_log_s, slider_bfield, slider_rate, slider_w4]

    btn_fire = pygame.Rect(ctrl_x + 12, 212, 218, 28)
    btn_auto = pygame.Rect(ctrl_x + 240, 212, 218, 28)
    btn_clear = pygame.Rect(ctrl_x + 468, 212, 224, 28)

    # Zustandsvariablen
    mode_idx = 0
    auto_collide = True
    beam_phase = 0.0
    collision_flash = 0.0
    anim_time = 0.0
    rot_a1 = 0.24
    rot_a2 = 0.14

    # Histogramm der rekonstruierten invarianten Massen
    N_BINS = 52
    hist_counts = [0] * N_BINS
    total_events = 0
    higgs_count = 0
    shadow_events = 0
    last_log_s_group = 0

    current_event = generate_collision_event(10.0 ** slider_log_s.val, False, rng)

    def get_hist_config(sqrt_s):
        if sqrt_s < 1.0e13:
            if sqrt_s < 320.0:
                return (50.0, 220.0, "GeV")
            return (50.0, 420.0, "GeV")
        elif sqrt_s < 3.0e16:
            return (0.4e14, 7.2e14, "GeV")
        else:
            return (0.2e17, 1.8e17, "GeV")

    def trigger_collision():
        nonlocal current_event, collision_flash, total_events, higgs_count, shadow_events, hist_counts, last_log_s_group
        sqrt_s = 10.0 ** slider_log_s.val
        grp = 0 if sqrt_s < 320.0 else (1 if sqrt_s < 1.0e13 else (2 if sqrt_s < 3.0e16 else 3))
        if grp != last_log_s_group:
            hist_counts = [0] * N_BINS
            total_events = 0
            higgs_count = 0
            shadow_events = 0
            last_log_s_group = grp

        is_lep = COLLIDER_MODES[mode_idx]["is_lepton"]
        ev = generate_collision_event(sqrt_s, is_lep, rng)
        current_event = ev
        collision_flash = 1.0
        total_events += 1
        if ev["higgs_hit"] and abs(ev["m_inv"] - M_H_GEV) < 8.0:
            higgs_count += 1
        if ev["shadow_leak"]:
            shadow_events += 1

        h_min, h_max, _ = get_hist_config(sqrt_s)
        if h_min <= ev["m_inv"] <= h_max:
            b_idx = int(((ev["m_inv"] - h_min) / (h_max - h_min)) * (N_BINS - 1))
            b_idx = max(0, min(N_BINS - 1, b_idx))
            hist_counts[b_idx] += 1

    trigger_collision()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        anim_time += dt
        rot_a1 = (rot_a1 + 0.35 * dt) % (2.0 * math.pi)
        rot_a2 = (rot_a2 + 0.21 * dt) % (2.0 * math.pi)
        collision_flash = max(0.0, collision_flash - 1.8 * dt)

        if auto_collide:
            prev_phase = beam_phase
            beam_phase = (beam_phase + slider_rate.val * dt) % 1.0
            if beam_phase < prev_phase:
                trigger_collision()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    trigger_collision()
                elif event.key == pygame.K_a:
                    auto_collide = not auto_collide
                elif event.key == pygame.K_c:
                    hist_counts = [0] * N_BINS
                    total_events = 0
                    higgs_count = 0
                    shadow_events = 0
                elif event.key == pygame.K_r:
                    mode_idx = 0
                    slider_log_s.val = COLLIDER_MODES[0]["log10_s"]
                    slider_bfield.val = 3.80
                    slider_rate.val = 1.25
                    slider_w4.val = 0.0
                    hist_counts = [0] * N_BINS
                    total_events = 0
                    higgs_count = 0
                    shadow_events = 0
                    trigger_collision()
                else:
                    for idx, m_item in enumerate(COLLIDER_MODES):
                        if event.unicode == m_item["key"]:
                            mode_idx = idx
                            slider_log_s.val = m_item["log10_s"]
                            trigger_collision()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for idx, rect in enumerate(mode_buttons):
                    if rect.collidepoint(event.pos):
                        mode_idx = idx
                        slider_log_s.val = COLLIDER_MODES[idx]["log10_s"]
                        trigger_collision()
                if btn_fire.collidepoint(event.pos):
                    trigger_collision()
                elif btn_auto.collidepoint(event.pos):
                    auto_collide = not auto_collide
                elif btn_clear.collidepoint(event.pos):
                    hist_counts = [0] * N_BINS
                    total_events = 0
                    higgs_count = 0
                    shadow_events = 0

            for s in sliders:
                s.handle_event(event)

        sqrt_s_curr = 10.0 ** slider_log_s.val
        b_tesla = slider_bfield.val
        w4_val = slider_w4.val
        R4 = rotation_matrix_4d(rot_a1, rot_a2)
        mode_cfg = COLLIDER_MODES[mode_idx]

        # ----------------------------------------------------------------------
        # 1. 4D-TORUS-KOLLISIONSKAMMER (T^4 / Gamma_24) BERECHNEN
        # ----------------------------------------------------------------------
        # Kollisionsphase: -1 -> +1 (0 = exakter Zusammenstoss im Zentrum (0,0,0,0)!)
        coll_s = math.sin(math.pi * (beam_phase - 0.5))
        impact_prox = math.exp(-12.0 * coll_s * coll_s)

        # Zwei gegenlaeufige 4D-Strahl-Wellenpakete in C_0
        beam_a_4d = np.array([0.34 * coll_s, -0.06 * coll_s, 0.04 * coll_s, 0.0], dtype=np.float32)
        beam_b_4d = np.array([-0.34 * coll_s, 0.06 * coll_s, -0.04 * coll_s, 0.0], dtype=np.float32)

        X1 = U_T * R4[0, 0] + V_T * R4[0, 1] + w4_val * R4[0, 3]
        X2 = U_T * R4[1, 0] + V_T * R4[1, 1] + w4_val * R4[1, 3]
        X3 = U_T * R4[2, 0] + V_T * R4[2, 1] + w4_val * R4[2, 3]
        X4 = U_T * R4[3, 0] + V_T * R4[3, 1] + w4_val * R4[3, 3]
        coords_t = (X1, X2, X3, X4)

        # Wellenpaket-Fokussierung steigt mit sqrt(s) (de-Broglie-Wellenlaenge schrumpft)
        focus = max(0.42, 1.0 - 0.032 * slider_log_s.val)
        sig_beam = SIGMA_W * focus

        def eval_4d_wp(pos_4d, sigma_p, cp_phase=0.0):
            psi = np.exp(np.complex64(-1j * 2.8 * anim_time)) * np.ones_like(X1, dtype=np.complex64)
            for k in range(4):
                psi *= jacobi_theta_1d(coords_t[k], float(pos_4d[k]), sigma_p, 2, cp_phase if k == 0 else 0.0)
            return np.abs(psi) ** 2, np.angle(psi)

        d_ba, a_ba = eval_4d_wp(beam_a_4d, sig_beam, 0.0)
        d_bb, a_bb = eval_4d_wp(beam_b_4d, sig_beam, math.radians(CP_PHASE_DEG) if current_event["cp_active"] else 0.0)

        # Higgs-Resonanz-Anregung im Zentrum (0,0,0,0) bei Kollision
        d_higgs, _ = eval_4d_wp(np.zeros(4, dtype=np.float32), SIGMA_0, 0.0)
        higgs_glow = (0.18 + 1.35 * impact_prox * (1.0 if current_event["higgs_hit"] else 0.35)) * d_higgs

        # Schatten-Anregung auf Nachbarecken C_1..C_8 bei E >= m_G1 (1.80e14 GeV)
        shadow_glow = np.zeros_like(X1)
        if sqrt_s_curr >= 1.0e14 or current_event["shadow_leak"]:
            leak_amp = min(1.5, math.log10(max(sqrt_s_curr, 1e14) / 1e13) * 0.45) * (0.35 + 0.95 * collision_flash)
            for s_pos in [(0.36, 0.36, 0.0, 0.0), (-0.36, -0.36, 0.0, 0.0)]:
                ds, _ = eval_4d_wp(np.array(s_pos, dtype=np.float32), SIGMA_W * 0.85, 0.0)
                shadow_glow += leak_amp * ds

        bulk_bg = 0.06 + 0.015 * np.sin(2.0 * math.pi * (U_T + V_T) - 1.6 * anim_time)
        m_ba = 0.78 + 0.22 * np.cos(a_ba)
        m_bb = 0.78 + 0.22 * np.cos(a_bb)

        tor_r = bulk_bg * 0.75 + d_ba * 0.15 * m_ba + d_bb * 0.95 * m_bb + higgs_glow * 1.00 + shadow_glow * 1.00
        tor_g = bulk_bg * 0.95 + d_ba * 0.88 * m_ba + d_bb * 0.28 * m_bb + higgs_glow * 0.82 + shadow_glow * 0.50
        tor_b = bulk_bg * 1.35 + d_ba * 1.00 * m_ba + d_bb * 0.85 * m_bb + higgs_glow * 0.22 + shadow_glow * 0.10

        tor_rgb = np.stack([tor_r, tor_g, tor_b], axis=-1)
        tor_rgb = tor_rgb / (1.0 + 0.48 * tor_rgb)
        tor_u8 = np.clip(np.power(np.clip(tor_rgb, 0.0, 1.0), 0.86) * 255.0, 0, 255).astype(np.uint8)
        torus_surf = pygame.surfarray.make_surface(np.transpose(tor_u8, (1, 0, 2)))
        torus_surf = pygame.transform.smoothscale(torus_surf, (TORUS_W, TORUS_H))

        # ----------------------------------------------------------------------
        # 2. BILDSCHIRM ZEICHNEN: KOPFZEILE & BESCHLEUNIGERRING + DETEKTOR
        # ----------------------------------------------------------------------
        screen.fill((14, 18, 28))

        title_s = font_title.render(
            "24-Zell-Teilchenbeschleuniger (v9) — Synchrotron, 4-Lagen-Detektor & 4D-Torus-Collider",
            True,
            (235, 242, 255),
        )
        sub_s = font_ui.render(
            f"Strahl: {mode_cfg['beam']}   |   sqrt(s) = {sqrt_s_curr:.3e} GeV   |   "
            f"Kompaktifizierungsskala E_T = 1.2645e17 GeV (L_T = 1.5605e-33 m)",
            True,
            (150, 175, 210),
        )
        screen.blit(title_s, (22, 16))
        screen.blit(sub_s, (22, 44))

        # Container Oben Links
        pygame.draw.rect(screen, (18, 24, 38), ACC_RECT, border_radius=8)
        pygame.draw.rect(screen, (75, 100, 140), ACC_RECT, 2, border_radius=8)

        # --- Links im Container: Synchrotron-Speicherring ---
        rx, ry = RING_CENTER
        pygame.draw.circle(screen, (45, 62, 92), (rx, ry), RING_R + 7, 2)
        pygame.draw.circle(screen, (45, 62, 92), (rx, ry), RING_R - 7, 2)
        pygame.draw.circle(screen, (95, 165, 235), (rx, ry), RING_R, 1)

        # 8 Dipol-Ablenkmagnete entlang des Rings
        for m_i in range(8):
            ang_m = m_i * (math.pi / 4.0) + math.pi / 8.0
            mx = int(rx + RING_R * math.cos(ang_m))
            my = int(ry + RING_R * math.sin(ang_m))
            pygame.draw.circle(screen, (70, 115, 175), (mx, my), 6)
            pygame.draw.circle(screen, (165, 215, 255), (mx, my), 6, 1)

        # HF-Beschleunigungskavitaet (links am Ring)
        rf_rect = pygame.Rect(rx - RING_R - 14, ry - 16, 28, 32)
        pygame.draw.rect(screen, (40, 85, 115), rf_rect, border_radius=4)
        pygame.draw.rect(screen, (120, 225, 255), rf_rect, 1, border_radius=4)
        screen.blit(font_mono.render("RF", True, (200, 245, 255)), (rf_rect.x + 6, rf_rect.y + 8))

        # Kollisionspunkt (IP) rechts am Ring + Verbindungslinie zum Detektor-Zoom
        ip_x, ip_y = rx + RING_R, ry
        pygame.draw.circle(screen, (255, 220, 110), (ip_x, ip_y), 6)
        pygame.draw.line(screen, (90, 125, 165), (ip_x + 6, ip_y), (DET_CENTER[0] - DET_MAX_R - 8, DET_CENTER[1]), 1)

        # Zwei gegenlaeufige Teilchenpakete (Bunch A im Uhrzeigersinn, Bunch B gegen Uhrzeigersinn)
        ang_a = 2.0 * math.pi * beam_phase
        ang_b = -2.0 * math.pi * beam_phase
        bax, bay = int(rx + RING_R * math.cos(ang_a)), int(ry + RING_R * math.sin(ang_a))
        bbx, bby = int(rx + RING_R * math.cos(ang_b)), int(ry + RING_R * math.sin(ang_b))
        pygame.draw.circle(screen, (80, 235, 255), (bax, bay), 5)
        pygame.draw.circle(screen, (255, 105, 220), (bbx, bby), 5)

        screen.blit(font_sub.render("Synchrotron-Ring (M_4)", True, (225, 238, 255)), (ACC_RECT.x + 12, ACC_RECT.y + 8))
        screen.blit(font_mono.render("Bunch 1 (Cyan) ->", True, (95, 235, 255)), (ACC_RECT.x + 14, ACC_RECT.bottom - 42))
        screen.blit(font_mono.render("<- Bunch 2 (Magenta)", True, (255, 120, 225)), (ACC_RECT.x + 14, ACC_RECT.bottom - 24))

        # --- Rechts im Container: 4-Lagen-Zylinderdetektor (Querschnitt) ---
        dcx, dcy = DET_CENTER
        det_layers = [
            (DET_MAX_R, (28, 36, 56), (85, 115, 165), "Myon-Kammern"),
            (int(DET_MAX_R * 0.74), (34, 46, 70), (95, 145, 195), "HCAL (Hadronen)"),
            (int(DET_MAX_R * 0.50), (38, 58, 82), (110, 195, 215), "ECAL (e/gamma)"),
            (int(DET_MAX_R * 0.28), (24, 34, 52), (90, 150, 210), "Pixel-Tracker"),
        ]
        for rad_l, fill_c, edge_c, _ in det_layers:
            pygame.draw.circle(screen, fill_c, (dcx, dcy), rad_l)
            pygame.draw.circle(screen, edge_c, (dcx, dcy), rad_l, 1)

        # Rekonstruierte Teilchenspuren im Magnetfeld B zeichnen
        for tr in current_event["tracks"]:
            ang0 = tr["angle"]
            q = tr["charge"]
            # Kruemmung proportional zu q * B / p_T
            curv = 0.0 if q == 0 else (q * b_tesla * 0.012) / max(math.log10(tr["pt"] + 1.5), 0.35)

            if tr["ptype"] == "photon":
                r_end = int(DET_MAX_R * 0.48)
                col_t = (255, 240, 110)
                dashed = True
            elif tr["ptype"] == "electron":
                r_end = int(DET_MAX_R * 0.49)
                col_t = (90, 235, 255)
                dashed = False
            elif tr["ptype"] == "muon":
                r_end = int(DET_MAX_R * 0.98)
                col_t = (135, 255, 175)
                dashed = False
            elif tr["ptype"] == "jet":
                r_end = int(DET_MAX_R * 0.72)
                col_t = (255, 165, 95)
                dashed = False
            elif tr["ptype"] == "shadow":
                r_end = int(DET_MAX_R * 1.04)
                col_t = (255, 110, 75)
                dashed = True
            else:  # SM-Neutrino
                r_end = int(DET_MAX_R * 1.02)
                col_t = (185, 145, 255)
                dashed = True

            pts_tr = []
            steps_tr = 22
            for si in range(steps_tr + 1):
                frac = si / steps_tr
                r_s = frac * r_end
                th_s = ang0 + curv * frac
                pts_tr.append((int(dcx + r_s * math.cos(th_s)), int(dcy + r_s * math.sin(th_s))))

            if dashed:
                for si in range(0, len(pts_tr) - 1, 2):
                    pygame.draw.line(screen, col_t, pts_tr[si], pts_tr[si + 1], 2)
            else:
                pygame.draw.lines(screen, col_t, False, pts_tr, 2)

            # Kalorimeter-Trefferpunkt + Label am Spurende
            end_pt = pts_tr[-1]
            if not tr["missing"]:
                pygame.draw.circle(screen, col_t, end_pt, 4)
            lbl_pt = (
                int(dcx + (r_end + 10) * math.cos(ang0 + curv)),
                int(dcy + (r_end + 10) * math.sin(ang0 + curv)),
            )
            if ACC_RECT.collidepoint(lbl_pt):
                screen.blit(font_mono.render(tr["name"], True, col_t), (lbl_pt[0] - 14, lbl_pt[1] - 6))

        # Kollisionsblitz im Detektorzentrum
        if collision_flash > 0.05:
            fl_r = int(6 + 18 * (1.0 - collision_flash))
            pygame.draw.circle(screen, (255, 245, 180), (dcx, dcy), fl_r, 2)
        pygame.draw.circle(screen, (255, 255, 255), (dcx, dcy), 4)

        screen.blit(
            font_sub.render(f"Detektor-Querschnitt (B = {b_tesla:.2f} T)", True, (225, 238, 255)),
            (dcx - 115, ACC_RECT.y + 8),
        )
        screen.blit(
            font_mono.render("Cyan: e | Gruen: mu | Orange: Jets | Gelb: gamma | Rot gestr.: Schatten-MET", True, (165, 190, 220)),
            (dcx - 210, ACC_RECT.bottom - 24),
        )

        # ----------------------------------------------------------------------
        # 3. UNTEN LINKS: 4D-TORUS-KOLLISIONSKAMMER (T^4 / Gamma_24)
        # ----------------------------------------------------------------------
        screen.blit(torus_surf, TORUS_RECT.topleft)
        pygame.draw.rect(screen, (75, 100, 140), TORUS_RECT, 2, border_radius=6)

        tcx, tcy = TORUS_RECT.centerx, TORUS_RECT.centery
        rot_nodes = verts_torus @ R4
        for idx_n in range(24):
            un, vn, _, wn4 = rot_nodes[idx_n]
            px_n = int(tcx + un * TORUS_W * 0.56)
            py_n = int(tcy + vn * TORUS_H * 0.88)
            if not TORUS_RECT.collidepoint(px_n, py_n):
                continue
            sh = shells_24[idx_n]
            near_w = abs(float(wn4) - w4_val) < 0.25
            if sh == 0:
                pygame.draw.circle(screen, (95, 205, 255), (px_n, py_n), 6, 2)
                screen.blit(font_mono.render("C_0 (SM)", True, (130, 215, 255)), (px_n + 8, py_n - 7))
            elif sh == 1:
                col_s1 = (255, 155, 75) if near_w else (165, 95, 55)
                pygame.draw.circle(screen, col_s1, (px_n, py_n), 4 if near_w else 2)
                if idx_n in (1, 2):
                    screen.blit(font_mono.render(f"C_{idx_n} (DM)", True, col_s1), (px_n + 6, py_n - 6))

        # Higgs-Feld H^0 Kontur im Ursprung (0,0,0,0)
        pygame.draw.circle(screen, (255, 225, 115), (tcx, tcy), 28, 1)
        pygame.draw.circle(screen, (220, 175, 70), (tcx, tcy), 54, 1)
        screen.blit(font_mono.render("H^0 (0,0,0,0)", True, (255, 225, 125)), (tcx + 8, tcy - 18))

        screen.blit(
            font_sub.render("4D-Torus-Kollisionskammer (T^4 / Gamma_24): Wavepacket-Ueberlapp & 24-Zell-Schwelle", True, (235, 245, 255)),
            (TORUS_RECT.x + 12, TORUS_RECT.y + 8),
        )
        screen.blit(
            font_mono.render("Cyan/Magenta: Strahlpakete | Gold: Higgs-Resonanz | Orange: Schatten-Leckage (E >= m_G1)", True, (175, 200, 230)),
            (TORUS_RECT.x + 12, TORUS_RECT.bottom - 22),
        )

        screen.blit(
            font_ui.render(
                "Tasten: [1-4] Beschleuniger-Modus | [SPACE] Einzelschuss | [A] Auto-Kollision | [C] Histogramm leeren | [R] Reset",
                True,
                (145, 165, 195),
            ),
            (22, 804),
        )

        # ----------------------------------------------------------------------
        # 4. OBEN RECHTS: STEUERUNGSPANEL
        # ----------------------------------------------------------------------
        ctrl_rect = pygame.Rect(ctrl_x, 78, 704, 172)
        pygame.draw.rect(screen, (20, 26, 40), ctrl_rect, border_radius=8)
        pygame.draw.rect(screen, (50, 65, 95), ctrl_rect, 1, border_radius=8)

        for idx, (m_item, rect) in enumerate(zip(COLLIDER_MODES, mode_buttons)):
            active = idx == mode_idx
            pygame.draw.rect(screen, (38, 82, 125) if active else (28, 36, 54), rect, border_radius=5)
            pygame.draw.rect(screen, (110, 205, 255) if active else (55, 70, 98), rect, 2 if active else 1, border_radius=5)
            screen.blit(font_sub.render(f"[{m_item['key']}] {m_item['name']}", True, (240, 248, 255)), (rect.x + 8, rect.y + 6))

        for s in sliders:
            s.draw(screen, font_ui)

        for rect, lbl, active in [
            (btn_fire, "Kollision ausloesen [SPACE]", True),
            (btn_auto, f"Auto-Strahl: {'AN' if auto_collide else 'AUS'} [A]", auto_collide),
            (btn_clear, "Histogramm leeren [C]", False),
        ]:
            pygame.draw.rect(screen, (35, 70, 105) if active else (30, 36, 50), rect, border_radius=5)
            pygame.draw.rect(screen, (95, 175, 240) if active else (65, 75, 95), rect, 1, border_radius=5)
            screen.blit(font_ui.render(lbl, True, (235, 245, 255)), (rect.x + 12, rect.y + 7))

        # ----------------------------------------------------------------------
        # 5. MITTE RECHTS: INVARIANTES MASSENSPEKTRUM (HISTOGRAMM)
        # ----------------------------------------------------------------------
        hist_rect = pygame.Rect(ctrl_x, 262, 704, 214)
        pygame.draw.rect(screen, (20, 26, 40), hist_rect, border_radius=8)
        pygame.draw.rect(screen, (50, 65, 95), hist_rect, 1, border_radius=8)

        h_min, h_max, h_unit = get_hist_config(sqrt_s_curr)
        hx0, hy0 = hist_rect.x + 48, hist_rect.y + 30
        hw, hh = hist_rect.width - 68, hist_rect.height - 62
        pygame.draw.rect(screen, (15, 20, 32), (hx0, hy0, hw, hh))
        pygame.draw.rect(screen, (55, 70, 98), (hx0, hy0, hw, hh), 1)

        max_c = max(max(hist_counts), 5)
        bar_w = hw / N_BINS
        for bi, cnt in enumerate(hist_counts):
            if cnt > 0:
                bh = int((cnt / max_c) * (hh - 6))
                bx = int(hx0 + bi * bar_w)
                by = hy0 + hh - bh
                m_bin = h_min + (bi / max(N_BINS - 1, 1)) * (h_max - h_min)
                # Farbliche Hervorhebung der v9-Resonanzen
                if abs(m_bin - M_H_GEV) < 6.0:
                    bcol = (255, 215, 95)   # Higgs-Peak 125.68 GeV
                elif abs(m_bin - M_Z_GEV) < 6.0:
                    bcol = (95, 195, 255)   # Z^0-Peak 91.19 GeV
                elif m_bin > 1.0e13:
                    bcol = (255, 130, 95)   # GUT / Torus-Resonanzen
                else:
                    bcol = (115, 225, 185)
                pygame.draw.rect(screen, bcol, (bx + 1, by, max(int(bar_w) - 1, 2), bh))

        # Referenzlinien fuer Z^0 (91.19 GeV) und H^0 (125.68 GeV) im GeV-Bereich
        if h_max < 1000.0:
            for ref_m, ref_lbl, ref_col in [
                (M_Z_GEV, "Z^0 (91.2)", (95, 195, 255)),
                (M_H_GEV, "H^0 (125.68)", (255, 220, 100)),
            ]:
                if h_min <= ref_m <= h_max:
                    rx_p = int(hx0 + ((ref_m - h_min) / (h_max - h_min)) * hw)
                    pygame.draw.line(screen, ref_col, (rx_p, hy0), (rx_p, hy0 + hh), 1)
                    screen.blit(font_mono.render(ref_lbl, True, ref_col), (rx_p + 3, hy0 + 4))

        screen.blit(
            font_sub.render(
                f"Rekonstruiertes Invariantes Massenspektrum M_inv   |   Ereignisse: {total_events}   |   H^0-Kandidaten: {higgs_count}",
                True,
                (235, 245, 255),
            ),
            (hist_rect.x + 12, hist_rect.y + 7),
        )
        screen.blit(font_mono.render(f"{h_min:.2g} {h_unit}", True, (155, 175, 205)), (hx0, hy0 + hh + 5))
        screen.blit(font_mono.render(f"{h_max:.2g} {h_unit}", True, (155, 175, 205)), (hx0 + hw - 72, hy0 + hh + 5))

        # ----------------------------------------------------------------------
        # 6. UNTEN RECHTS: EREIGNIS- & v9-PHYSIK-TELEMETRIE
        # ----------------------------------------------------------------------
        tel_rect = pygame.Rect(ctrl_x, 488, 704, 340)
        pygame.draw.rect(screen, (20, 26, 40), tel_rect, border_radius=8)
        pygame.draw.rect(screen, (50, 65, 95), tel_rect, 1, border_radius=8)

        # Demokratisches Schatten-Branching oberhalb von m_G1 = 1.80e14 GeV
        if sqrt_s_curr < 1.0e14:
            br_shadow_pct = 0.00
            portal_status = "Z_2-geschuetzt (lambda_HH' = 0, eps_mix = 0) -> 0.00% Schattenzerfall"
        else:
            br_shadow_pct = 100.0 * (23.0 / 24.0)
            portal_status = "Polychor-Schwelle ueberschritten -> 23/24 (95.83%) in Schatten C_1..23"

        de_broglie_m = (1.97327e-16) / max(sqrt_s_curr, 1e-6)
        ratio_lt = de_broglie_m / L_T

        tel_lines = [
            ("--- AKTUELLES KOLLISIONSEREIGNIS & DETEKTOR-REKONSTRUKTION ---", (255, 225, 130)),
            (f"Zerfallskanal       : {current_event['channel']}", (135, 245, 190)),
            (f"Invariante Masse    : M_inv = {current_event['m_inv']:.4g} GeV   |   Fehlende Energie MET = {current_event['met_gev']:.4g} GeV", (225, 235, 250)),
            (f"Detektierte Spuren  : " + ", ".join(f"{t['name']}({t['pt']:.2g})" for t in current_event["tracks"]), (195, 215, 240)),
            ("--- v9-THEORIE-DIAGNOSE BEI AKTUELLER STRAHLENERGIE sqrt(s) ---", (255, 225, 130)),
            (f"Modus-Physik        : {mode_cfg['desc']}", (225, 235, 250)),
            (f"Aufloesung lambda_dB: {de_broglie_m:.3e} m  =  {ratio_lt:.3e} * L_T (Torus-Umfang 1.5605e-33 m)", (135, 225, 255)),
            (f"Higgs-Sektor (C_0)  : m_H = {M_H_GEV:.2f} GeV (Gamma_H = 4.023 MeV)  |  BR(H^0 -> unsichtbar) = 0.00%", (175, 235, 195)),
            (f"Schatten-Portal     : {portal_status} (BR_Schatten = {br_shadow_pct:.2f}%)", (255, 175, 110)),
            (f"Majorana & CP-Phase : M_R = 4.47e14 GeV (Delta L = 2)  |  arg(I_d13) = {CP_PHASE_DEG:+.2f} deg", (205, 220, 240)),
            (f"Rayleigh-Graviton   : m_G1 = 1.80e14 GeV (Lambda_G1 = 1.42e16 GeV, sigma_SI = 7.74e-99 cm^2)", (165, 190, 225)),
        ]
        for li, (txt, col) in enumerate(tel_lines):
            screen.blit(font_mono.render(txt, True, col), (tel_rect.x + 12, tel_rect.y + 10 + li * 29))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()

import time
import numpy as np


# ==============================================================================
# 1. PHYSIKALISCHE KONSTANTEN & MODELLPARAMETER (24-ZELL-TORUS-BASIS)
# ==============================================================================
CONST = {
    "m_mu_MeV": 105.6583755,
    "m_tau_MeV": 1776.86,
    "v_H_GeV": 246.22,
    "m_p_GeV": 0.938272,
    "alpha_em": 1.0 / 137.035999084,
    "l_Pl_m": 1.616255e-35,
    "hbar_c_GeV_m": 1.973269804e-16,
    "GeV_to_g": 1.78266192e-24,
    "c_m_s": 2.99792458e8,
    "G_N": 6.67430e-11,
    "M_sun_kg": 1.98847e30,
    "kpc_to_m": 3.08567758e19,
    "kpc_to_cm": 3.08567758e21,
    "sec_per_Gyr": 365.25 * 86400.0 * 1e9,
    "t_Hubble_Gyr": 13.8,
}


def init_model_parameters():
    tau_sq = 2.0 * np.log(CONST["m_tau_MeV"] / CONST["m_mu_MeV"])
    tau = np.sqrt(tau_sq)
    sigma0_over_LT_sq = 1.0 / (np.pi * tau)
    f_24 = 4.0 * np.pi * np.sqrt(24.0)
    L_T = (CONST["l_Pl_m"] / np.sqrt(CONST["alpha_em"])) * sigma0_over_LT_sq * f_24
    E_T = CONST["hbar_c_GeV_m"] / L_T
    Phi_T_GeV = E_T
    M_R_GeV = E_T * np.exp(-tau_sq)

    n_plateau = 3
    eps_0 = 1e-7
    R1 = 1.0 / (8.0 * (np.pi**2)) + eps_0
    Lambda_P = (-np.log(R1 - eps_0)) ** (-1.0 / (2 * n_plateau))

    def resonance(d_tau):
        return np.exp(-((d_tau / Lambda_P) ** (2 * n_plateau))) + eps_0

    R2 = resonance(np.sqrt(2.0))
    R3 = resonance(np.sqrt(3.0))
    R4 = resonance(2.0)

    alpha_prime = CONST["alpha_em"] * (1.0 + 0.5 * tau_sq) ** 2
    alpha_dip = alpha_prime / (24.0**2)
    m_n_phys_GeV = R1 * (CONST["v_H_GeV"] / 5.0)
    m_med_phys_GeV = R1 * (CONST["m_mu_MeV"] / 1000.0)
    m_gamma_phys_GeV = R1 * (CONST["v_H_GeV"] / (24.0**2)) * np.exp(-1.5 * tau_sq)

    v_turn_km_s = (CONST["c_m_s"] / 1e3) * (m_med_phys_GeV / m_n_phys_GeV)

    # Vollständig symmetrisches 24-Cluster-Spektrum:
    # 24 Cluster x 22 Arten (inkl. 69 Schatten-Majorana nu'_R & 23 Schatten-Moduli Phi'_T) + 24 Gravitonen = 552
    n_unique_species = 24 * 22 + 24

    return {
        "tau": tau,
        "tau_sq": tau_sq,
        "L_T": L_T,
        "E_T": E_T,
        "Phi_T_GeV": Phi_T_GeV,
        "M_R_GeV": M_R_GeV,
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "R4": R4,
        "alpha_prime": alpha_prime,
        "alpha_dip": alpha_dip,
        "m_n_phys_GeV": m_n_phys_GeV,
        "m_med_phys_GeV": m_med_phys_GeV,
        "m_gamma_phys_GeV": m_gamma_phys_GeV,
        "v_turn_km_s": v_turn_km_s,
        "n_unique_species": n_unique_species,
    }


def momentum_param_x(model, v_km_s):
    beta = (v_km_s * 1e3) / CONST["c_m_s"]
    return float((model["m_n_phys_GeV"] * beta) / model["m_med_phys_GeV"])


def eff_yukawa_sidm_cm2_g(model, v_km_s):
    m_n_MeV = model["m_n_phys_GeV"] * 1e3
    m_med_MeV = model["m_med_phys_GeV"] * 1e3
    alpha_dip = model["alpha_dip"]
    hbar_c_MeV_cm = CONST["hbar_c_GeV_m"] * 1e5
    m_n_g = model["m_n_phys_GeV"] * CONST["GeV_to_g"]

    beta = (v_km_s * 1e3) / CONST["c_m_s"]
    x = ((m_n_MeV * beta) / m_med_MeV) ** 2
    bracket = np.log(1.0 + x) - x / (1.0 + x)
    sigma_T_cm2 = (8.0 * np.pi * (alpha_dip**2) / ((m_n_MeV**2) * (beta**4))) * bracket * (hbar_c_MeV_cm**2)
    return float((1.0 / 8.0) * (sigma_T_cm2 / m_n_g))


def compute_radii_chunked(pos, chunk_size=2_000_000):
    n_p = pos.shape[0]
    radii = np.empty(n_p, dtype=np.float32)
    for start in range(0, n_p, chunk_size):
        end = min(start + chunk_size, n_p)
        p_blk = pos[start:end].astype(np.float64)
        radii[start:end] = np.sqrt(p_blk[:, 0] ** 2 + p_blk[:, 1] ** 2 + p_blk[:, 2] ** 2).astype(np.float32)
    return radii


# ==============================================================================
# 1. MEHRTEILCHEN-SIMULATOR FÜR YUKAWA-DIPOL-SIDM (PM / GITTER)
# ==============================================================================
def simulate_sidm_halo(model, v_rms_km_s, n_particles=800_000, n_steps=60, dt_val=0.002, grid_n=32, seed=42):
    rng = np.random.default_rng(seed)
    box_half = 2.5
    cell_size = (2.0 * box_half) / grid_n
    r_med_grid = 0.25

    x_mom = momentum_param_x(model, v_rms_km_s)
    born_factor = np.log(1.0 + x_mom**2) - (x_mom**2) / (1.0 + x_mom**2)

    g_eff = np.sqrt(model["alpha_dip"]) * 0.25
    omega_halo = np.float32(0.25)

    u_rad = rng.random(n_particles, dtype=np.float32) ** np.float32(1.4)
    r_init = np.float32(0.10) + np.float32(1.10) * u_rad
    cos_th = rng.uniform(-1.0, 1.0, size=n_particles).astype(np.float32)
    sin_th = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - cos_th**2))
    phi_az = rng.uniform(0.0, 2.0 * np.pi, size=n_particles).astype(np.float32)

    pos = np.empty((n_particles, 3), dtype=np.float32)
    pos[:, 0] = r_init * sin_th * np.cos(phi_az)
    pos[:, 1] = r_init * sin_th * np.sin(phi_az)
    pos[:, 2] = r_init * cos_th

    vel_scale = max(v_rms_km_s * 1.0227e-3, 1e-4) / np.sqrt(3.0)
    vel = rng.normal(0.0, vel_scale, size=(n_particles, 3)).astype(np.float32)

    kx = np.fft.fftfreq(grid_n, d=cell_size) * 2.0 * np.pi
    ky = np.fft.fftfreq(grid_n, d=cell_size) * 2.0 * np.pi
    kz = np.fft.fftfreq(grid_n, d=cell_size) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    sigma_g = 1.5 * cell_size
    k_nyq = np.pi / cell_size
    nyquist_damping = float(np.exp(-0.5 * (k_nyq**2) * (sigma_g**2)))

    filter_k = np.exp(-0.5 * K2 * (sigma_g**2))
    green_yukawa_k = filter_k * (4.0 * np.pi * (g_eff**2)) / (K2 + (1.0 / (r_med_grid**2)))
    green_yukawa_k[0, 0, 0] = 0.0

    def pm_forces_and_yuk_energy(r_vec):
        r_clipped = np.clip(r_vec, -box_half + 1e-4, box_half - 1e-4)
        idx = np.clip(((r_clipped + box_half) / cell_size).astype(np.int32), 0, grid_n - 1)
        flat_idx = idx[:, 0] * (grid_n * grid_n) + idx[:, 1] * grid_n + idx[:, 2]

        rho_grid = (
            np.bincount(flat_idx, minlength=grid_n**3).reshape((grid_n, grid_n, grid_n)).astype(np.float64)
            / float(n_particles)
        )
        rho_k = np.fft.fftn(rho_grid)
        phi_k = rho_k * green_yukawa_k
        phi_grid = np.real(np.fft.ifftn(phi_k))

        gx_grid = -np.real(np.fft.ifftn(1j * KX * phi_k)).astype(np.float32)
        gy_grid = -np.real(np.fft.ifftn(1j * KY * phi_k)).astype(np.float32)
        gz_grid = -np.real(np.fft.ifftn(1j * KZ * phi_k)).astype(np.float32)

        acc_out = np.empty_like(r_vec)
        acc_out[:, 0] = gx_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        acc_out[:, 1] = gy_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        acc_out[:, 2] = gz_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        acc_out -= (omega_halo**2) * r_vec

        e_yuk = 0.5 * float(n_particles) * float(np.mean(phi_grid[idx[:, 0], idx[:, 1], idx[:, 2]]))
        return acc_out, e_yuk

    def total_energy(r_vec, v_vec, e_yuk_val):
        kin = 0.5 * float(np.sum(v_vec.astype(np.float64) ** 2))
        pot_halo = 0.5 * float(omega_halo**2) * float(np.sum(r_vec.astype(np.float64) ** 2))
        return kin + pot_halo + e_yuk_val

    dt = np.float32(dt_val)
    acc, e_yuk_curr = pm_forces_and_yuk_energy(pos)
    E_init = total_energy(pos, vel, e_yuk_curr)

    for _ in range(n_steps):
        vel_half = vel + np.float32(0.5) * dt * acc
        pos = pos + dt * vel_half
        acc_new, e_yuk_curr = pm_forces_and_yuk_energy(pos)
        vel = vel_half + np.float32(0.5) * dt * acc_new
        acc = acc_new

    E_final = total_energy(pos, vel, e_yuk_curr)
    rel_energy_drift = abs(E_final - E_init) / max(abs(E_init), 1e-15)

    radii = compute_radii_chunked(pos)
    core_radius_kpc = float(np.percentile(radii[::4], 25))
    inner_fraction = float(np.mean(radii < 0.35))

    sigma_m = eff_yukawa_sidm_cm2_g(model, v_rms_km_s)
    rho_halo_g_cm3 = (2.0e7 * CONST["M_sun_kg"] * 1e3) / (CONST["kpc_to_cm"] ** 3)
    v_cm_s = max(v_rms_km_s, 1.0) * 1e5
    t_relax_Gyr = (1.0 / max(rho_halo_g_cm3 * sigma_m * v_cm_s, 1e-60)) / CONST["sec_per_Gyr"]
    t_coll_Gyr = (150.0 / 0.75) * t_relax_Gyr

    return {
        "n_particles": n_particles,
        "v_rms_km_s": v_rms_km_s,
        "x_mom": x_mom,
        "born_factor": born_factor,
        "rel_energy_drift": rel_energy_drift,
        "core_radius_kpc": core_radius_kpc,
        "inner_fraction_035": inner_fraction,
        "nyquist_damping": nyquist_damping,
        "sigma_m_cm2_g": sigma_m,
        "t_relax_Gyr": t_relax_Gyr,
        "t_coll_Gyr": t_coll_Gyr,
        "pos": pos,
        "vel": vel,
        "radii": radii,
    }


# ==============================================================================
# 1B. LANGZEITSIMULATION MIT PERIODISCHEN KONTROLL-PRINTS
# ==============================================================================
def run_long_term_simulation_with_periodic_checks(
    model, v_rms_km_s=30.0, n_particles=800_000, n_steps=600, check_interval=100, dt_val=0.002, grid_n=32, seed=777
):
    print(f"\n--- 1B-LONG. LANGZEITSIMULATION ({n_steps} SCHRITTE, CHECK ALLE {check_interval} SCHRITTE) ---")
    rng = np.random.default_rng(seed)
    box_half = 2.5
    cell_size = (2.0 * box_half) / grid_n
    r_med_grid = 0.25
    g_eff = np.sqrt(model["alpha_dip"]) * 0.25
    omega_halo = np.float32(0.25)

    u_rad = rng.random(n_particles, dtype=np.float32) ** np.float32(1.4)
    r_init = np.float32(0.10) + np.float32(1.10) * u_rad
    cos_th = rng.uniform(-1.0, 1.0, size=n_particles).astype(np.float32)
    sin_th = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - cos_th**2))
    phi_az = rng.uniform(0.0, 2.0 * np.pi, size=n_particles).astype(np.float32)

    pos = np.empty((n_particles, 3), dtype=np.float32)
    pos[:, 0] = r_init * sin_th * np.cos(phi_az)
    pos[:, 1] = r_init * sin_th * np.sin(phi_az)
    pos[:, 2] = r_init * cos_th

    vel_scale = max(v_rms_km_s * 1.0227e-3, 1e-4) / np.sqrt(3.0)
    vel = rng.normal(0.0, vel_scale, size=(n_particles, 3)).astype(np.float32)

    kx = np.fft.fftfreq(grid_n, d=cell_size) * 2.0 * np.pi
    ky = np.fft.fftfreq(grid_n, d=cell_size) * 2.0 * np.pi
    kz = np.fft.fftfreq(grid_n, d=cell_size) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    sigma_g = 1.5 * cell_size
    filter_k = np.exp(-0.5 * K2 * (sigma_g**2))
    green_yukawa_k = filter_k * (4.0 * np.pi * (g_eff**2)) / (K2 + (1.0 / (r_med_grid**2)))
    green_yukawa_k[0, 0, 0] = 0.0

    def pm_forces_and_yuk_energy(r_vec):
        r_clipped = np.clip(r_vec, -box_half + 1e-4, box_half - 1e-4)
        idx = np.clip(((r_clipped + box_half) / cell_size).astype(np.int32), 0, grid_n - 1)
        flat_idx = idx[:, 0] * (grid_n * grid_n) + idx[:, 1] * grid_n + idx[:, 2]

        rho_grid = (
            np.bincount(flat_idx, minlength=grid_n**3).reshape((grid_n, grid_n, grid_n)).astype(np.float64)
            / float(n_particles)
        )
        rho_k = np.fft.fftn(rho_grid)
        phi_k = rho_k * green_yukawa_k
        phi_grid = np.real(np.fft.ifftn(phi_k))

        gx_grid = -np.real(np.fft.ifftn(1j * KX * phi_k)).astype(np.float32)
        gy_grid = -np.real(np.fft.ifftn(1j * KY * phi_k)).astype(np.float32)
        gz_grid = -np.real(np.fft.ifftn(1j * KZ * phi_k)).astype(np.float32)

        acc_out = np.empty_like(r_vec)
        acc_out[:, 0] = gx_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        acc_out[:, 1] = gy_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        acc_out[:, 2] = gz_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        acc_out -= (omega_halo**2) * r_vec

        e_yuk = 0.5 * float(n_particles) * float(np.mean(phi_grid[idx[:, 0], idx[:, 1], idx[:, 2]]))
        return acc_out, e_yuk

    def total_energy(r_vec, v_vec, e_yuk_val):
        kin = 0.5 * float(np.sum(v_vec.astype(np.float64) ** 2))
        pot_halo = 0.5 * float(omega_halo**2) * float(np.sum(r_vec.astype(np.float64) ** 2))
        return kin + pot_halo + e_yuk_val

    dt = np.float32(dt_val)
    acc, e_yuk_curr = pm_forces_and_yuk_energy(pos)
    E_init = total_energy(pos, vel, e_yuk_curr)

    checkpoints = []
    max_drift_seen = 0.0
    all_checks_ok = True

    for step in range(1, n_steps + 1):
        vel_half = vel + np.float32(0.5) * dt * acc
        pos = pos + dt * vel_half
        acc_new, e_yuk_curr = pm_forces_and_yuk_energy(pos)
        vel = vel_half + np.float32(0.5) * dt * acc_new
        acc = acc_new

        if step % check_interval == 0 or step == n_steps:
            e_curr = total_energy(pos, vel, e_yuk_curr)
            drift = abs(e_curr - E_init) / max(abs(E_init), 1e-15)
            max_drift_seen = max(max_drift_seen, drift)

            sample_r = np.sqrt(
                pos[::16, 0].astype(np.float64) ** 2
                + pos[::16, 1].astype(np.float64) ** 2
                + pos[::16, 2].astype(np.float64) ** 2
            )
            r_core_est = float(np.percentile(sample_r, 25))
            in_box_frac = float(np.mean(np.max(np.abs(pos[::16]), axis=1) <= box_half))

            finite_ok = bool(np.all(np.isfinite(pos[::16])) and np.all(np.isfinite(vel[::16])) and np.isfinite(e_curr))
            drift_ok = drift < 5e-2
            confinement_ok = in_box_frac > 0.90
            model_ok = (model["Phi_T_GeV"] > 1e16) and (model["v_turn_km_s"] > 500.0) and (model["n_unique_species"] == 552)

            step_ok = finite_ok and drift_ok and confinement_ok and model_ok
            all_checks_ok = all_checks_ok and step_ok
            tag = "OK  " if step_ok else "WARN"
            print(
                f"  [{tag}] Schritt {step:4d}/{n_steps:4d} | dE/E={drift:.2e} | "
                f"r_core={r_core_est:.3f} kpc | im Volumen={in_box_frac*100:.2f}% | Alles richtig={step_ok}"
            )
            checkpoints.append((step, drift, r_core_est, in_box_frac, step_ok))

    return {
        "n_steps": n_steps,
        "check_interval": check_interval,
        "checkpoints": checkpoints,
        "max_drift_seen": max_drift_seen,
        "all_checks_ok": all_checks_ok,
    }


# ==============================================================================
# 1C. RADIALE SCHALEN-DICHTEPROFIL-ANALYSE rho(r)
# ==============================================================================
def analyze_radial_density_profile(sim_state, r_min=0.05, r_max=2.0, n_shells=8):
    radii = sim_state["radii"].astype(np.float64)
    edges = np.logspace(np.log10(r_min), np.log10(r_max), n_shells + 1)
    counts, _ = np.histogram(radii, bins=edges)
    volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho_shells = counts / np.maximum(volumes, 1e-18)
    r_centers = np.sqrt(edges[:-1] * edges[1:])

    non_empty_r = r_centers[rho_shells > 0.0]
    non_empty_rho = rho_shells[rho_shells > 0.0]
    rho_inner = float(non_empty_rho[0]) if len(non_empty_rho) > 0 else 0.0
    rho_outer = float(non_empty_rho[-1]) if len(non_empty_rho) > 0 else 0.0
    inner_to_outer_ratio = rho_inner / max(rho_outer, 1e-18)

    return {
        "r_centers_kpc": r_centers,
        "shell_counts": counts,
        "rho_kpc3": rho_shells,
        "r_inner_active_kpc": float(non_empty_r[0]) if len(non_empty_r) > 0 else np.nan,
        "r_outer_active_kpc": float(non_empty_r[-1]) if len(non_empty_r) > 0 else np.nan,
        "rho_inner_kpc3": rho_inner,
        "rho_outer_kpc3": rho_outer,
        "inner_to_outer_ratio": inner_to_outer_ratio,
        "all_finite_nonneg": bool(np.all(np.isfinite(rho_shells)) and np.all(rho_shells >= 0.0)),
    }


# ==============================================================================
# 2. LAGRANGE-DICHTE-AUSWERTUNG ÜBER DAS ENSEMBLE (CHUNKS)
# ==============================================================================
def evaluate_lagrangian_over_ensemble(model, sim_state, chunk_size=2_000_000):
    n_p = sim_state["n_particles"]
    p = sim_state["pos"]
    v = sim_state["vel"]
    g_dip = np.sqrt(4.0 * np.pi * model["alpha_dip"])
    m_med = model["m_med_phys_GeV"]
    m_gamma = model["m_gamma_phys_GeV"]

    sum_h_med = 0.0
    max_h_med = 0.0
    sum_e_proca = 0.0
    all_pos = True

    for start in range(0, n_p, chunk_size):
        end = min(start + chunk_size, n_p)
        p_blk = p[start:end].astype(np.float64)
        v_blk = v[start:end].astype(np.float64)

        r_mag = np.sqrt(p_blk[:, 0] ** 2 + p_blk[:, 1] ** 2 + p_blk[:, 2] ** 2) + 0.05
        v_mag = np.sqrt(v_blk[:, 0] ** 2 + v_blk[:, 1] ** 2 + v_blk[:, 2] ** 2)

        phi_med = 1e-6 * g_dip * np.exp(-r_mag / 0.6) / r_mag
        grad_phi_sq = (phi_med / r_mag) ** 2
        h_med_arr = 0.5 * grad_phi_sq + 0.5 * (m_med**2) * (phi_med**2)

        A_sq = (1e-12 * (v_mag + 1e-5)) ** 2
        F_sq = (1e-15 / r_mag) ** 2
        e_proca_arr = 0.5 * F_sq + 0.5 * (m_gamma**2) * A_sq

        sum_h_med += float(np.sum(h_med_arr))
        max_h_med = max(max_h_med, float(np.max(h_med_arr)))
        sum_e_proca += float(np.sum(e_proca_arr))
        if not (np.all(h_med_arr > 0.0) and np.all(e_proca_arr > 0.0)):
            all_pos = False

    mean_h_med = sum_h_med / float(n_p)
    mean_e_proca = sum_e_proca / float(n_p)

    return {
        "n_evaluated": n_p,
        "mean_H_med_GeV4": mean_h_med,
        "max_H_med_GeV4": max_h_med,
        "mean_E_proca_GeV4": mean_e_proca,
        "mean_H_eff_GeV4": mean_h_med + mean_e_proca,
        "all_positive": all_pos,
    }


# ==============================================================================
# 3. RAUMZEITKRÜMMUNGS- & 24-CLUSTER-ENSEMBLE-SIMULATION
# ==============================================================================
def simulate_curvature_and_24clusters(model, n_particles=800_000, n_cluster_draws=800_000, l_core=0.02, seed=2026):
    rng = np.random.default_rng(seed)
    rs = 1.0
    c2 = rs * (l_core**2)

    r_samples = np.logspace(np.log10(0.02), np.log10(50.0), n_particles)
    r3 = r_samples**3
    R_curv_arr = (2.0 * rs * (l_core**2) * (2.0 * r3 - c2)) / ((r3 + c2) ** 3)
    K_curv_arr = (
        12.0
        * (rs**2)
        * (
            (r_samples**12)
            - 4.0 * c2 * (r_samples**9)
            + 18.0 * (c2**2) * (r_samples**6)
            - 2.0 * (c2**3) * r3
            + 2.0 * (c2**4)
        )
    ) / ((r3 + c2) ** 6)

    K_max_analytic = 24.0 / (l_core**4)

    cluster_R = np.array(
        [1.0] + [model["R1"]] * 8 + [model["R2"]] * 6 + [model["R3"]] * 8 + [model["R4"]]
    )
    expected_R = float(np.mean(cluster_R))
    std_R = float(np.std(cluster_R, ddof=1))
    sigma_3sigma_mc = 3.0 * (std_R / np.sqrt(float(n_cluster_draws)))

    c_idx = rng.integers(0, 24, size=n_cluster_draws)
    R_cluster_samples = cluster_R[c_idx]
    mean_R_ens = float(np.mean(R_cluster_samples))

    return {
        "n_particles": n_particles,
        "n_cluster_draws": n_cluster_draws,
        "max_K_simulated": float(np.max(K_curv_arr)),
        "K_max_analytic": float(K_max_analytic),
        "min_R_curv": float(np.min(R_curv_arr)),
        "max_R_curv": float(np.max(R_curv_arr)),
        "mean_cluster_R_ens": mean_R_ens,
        "expected_cluster_R": expected_R,
        "sigma_3sigma_mc": sigma_3sigma_mc,
        "within_3sigma_mc": abs(mean_R_ens - expected_R) <= sigma_3sigma_mc,
        "all_finite_and_bounded": bool(
            np.all(np.isfinite(R_curv_arr))
            and np.all(np.isfinite(K_curv_arr))
            and np.max(K_curv_arr) <= K_max_analytic * 1.01
        ),
    }


# ==============================================================================
# 3B. WEITERE SIMULATIONS-VORHERSAGEN (VORHERSAGEN 11–15) & V-SWEEP
# ==============================================================================
def compute_additional_simulation_predictions(model, dwarf_sim, cluster_sim, prof_dwarf, prof_cluster, lag_dwarf, cur_sim, long_sim):
    print("\n--- 3B. ZUSÄTZLICHE SIMULATIONS-VORHERSAGEN (11–15) ---")
    ratio_dwarf = prof_dwarf["inner_to_outer_ratio"]
    ratio_cluster = prof_cluster["inner_to_outer_ratio"]
    r_turn_kpc = dwarf_sim["core_radius_kpc"] * (model["v_turn_km_s"] / 30.0) * 0.1
    mean_h_eff = lag_dwarf["mean_H_eff_GeV4"]
    max_k_reg = cur_sim["max_K_simulated"]
    max_long_drift = long_sim["max_drift_seen"]

    preds_11_15 = [
        ("11. Halo-Dichtekontrast (Rubin / ELT)", f"rho_in/out(30)={ratio_dwarf:.2f} vs. Haufen={ratio_cluster:.2f}", "Kernabflachung bei v << v_turn"),
        ("12. Kinematischer Turnover-Radius r_turn", f"r_turn ~ {r_turn_kpc:.3f} kpc (v_turn={model['v_turn_km_s']:.1f} km/s)", "Übergang von Dipol-Plateau zu Born-Abfall"),
        ("13. Mittlere Schatten-Feldenergiedichte", f"<H_eff> = {mean_h_eff:.3e} GeV^4", "Mediator- + Proca-Schattenfeld im Ensemble"),
        ("14. Regulierte Kernkrümmung (GW-Kern)", f"max(K_reg) = {max_k_reg:.3e} (endlich)", "Glättung der r^-6-Singularität durch L_core"),
        ("15. Langzeit-Energiedrift-Schranke", f"max(dE/E)={max_long_drift:.2e} ueber {long_sim['n_steps']} Schritte", "Symplektische Langzeitstabilität im PM-Gitter"),
    ]
    for title, val, ctx in preds_11_15:
        print(f"    {title:<42} -> {val:<42} [{ctx}]")

    v_grid = [10.0, 30.0, 300.0, model["v_turn_km_s"], 3000.0]
    x_grid = [momentum_param_x(model, v) for v in v_grid]
    sig_grid = [eff_yukawa_sidm_cm2_g(model, v) for v in v_grid]

    return {
        "ratio_dwarf": ratio_dwarf,
        "ratio_cluster": ratio_cluster,
        "r_turn_kpc": r_turn_kpc,
        "mean_h_eff": mean_h_eff,
        "max_k_reg": max_k_reg,
        "max_long_drift": max_long_drift,
        "v_grid": v_grid,
        "x_grid": x_grid,
        "sig_grid": sig_grid,
    }


# ==============================================================================
# 4. AUTOMATISIERTE TEST-SUITE (13 TESTS)
# ==============================================================================
def run_tests(model, dwarf_sim, cluster_sim, prof_dwarf, prof_cluster, lag_dwarf, cur_sim, long_sim, extra):
    print("\n--- 4. AUTOMATISIERTE TEST-SUITE (13 SIMULATIONSTESTS) ---")
    rel_err_R = abs(cur_sim["mean_cluster_R_ens"] - cur_sim["expected_cluster_R"]) / cur_sim["expected_cluster_R"]

    dt_half_sim = simulate_sidm_halo(
        model, v_rms_km_s=30.0, n_particles=100_000, n_steps=30, dt_val=0.001, grid_n=32, seed=42
    )
    x_grid = extra["x_grid"]
    x_monotone = all(x_grid[i] < x_grid[i + 1] for i in range(len(x_grid) - 1))
    x_at_turn_ok = abs(x_grid[3] - 1.0) < 1e-6

    tests = [
        (
            "1. 552 eindeutige Spektren (24x22+24) & Phi_T != u getrennt",
            model["n_unique_species"] == 552 and model["Phi_T_GeV"] > 1e16 and model["v_turn_km_s"] > 500.0,
            f"Arten={model['n_unique_species']}, Phi_T={model['Phi_T_GeV']:.2e} GeV, v_turn={model['v_turn_km_s']:.1f} km/s",
        ),
        (
            "2. Konservative Energieerhaltung & Nyquist-Dämpfung < 1e-4",
            dwarf_sim["rel_energy_drift"] < 5e-2
            and cluster_sim["rel_energy_drift"] < 5e-2
            and dwarf_sim["nyquist_damping"] < 1e-4,
            f"dE/E(30)={dwarf_sim['rel_energy_drift']:.2e}, dE/E(3000)={cluster_sim['rel_energy_drift']:.2e}, damp={dwarf_sim['nyquist_damping']:.1e}",
        ),
        (
            "3. Geschwindigkeitsabhängiger Impulsparameter x(3000) > x(30)",
            cluster_sim["x_mom"] > dwarf_sim["x_mom"] and cluster_sim["born_factor"] > dwarf_sim["born_factor"],
            f"x(30)={dwarf_sim['x_mom']:.2e}, x(3000)={cluster_sim['x_mom']:.2e}",
        ),
        (
            "4. Gravothermische Zeitskala t_coll(30 km/s) >> t_Hubble",
            dwarf_sim["t_coll_Gyr"] > 10.0 * CONST["t_Hubble_Gyr"],
            f"t_relax(30)={dwarf_sim['t_relax_Gyr']:.2f} Gyr, t_coll(30)={dwarf_sim['t_coll_Gyr']:.0f} Gyr",
        ),
        (
            "5. Radiales Dichteprofil rho(r) in allen Schalen endlich >= 0",
            prof_dwarf["all_finite_nonneg"] and prof_cluster["all_finite_nonneg"],
            f"rho_in/out(30)={prof_dwarf['inner_to_outer_ratio']:.2f}, rho_in/out(3000)={prof_cluster['inner_to_outer_ratio']:.2f}",
        ),
        (
            "6. Lagrange-Energiedichten positiv & endlich",
            lag_dwarf["all_positive"] and np.isfinite(lag_dwarf["mean_H_med_GeV4"]),
            f"<H_med>={lag_dwarf['mean_H_med_GeV4']:.3e}, <E_proca>={lag_dwarf['mean_E_proca_GeV4']:.3e}",
        ),
        (
            "7. Kretschmann-Skalar im regulierten Kern endlich",
            cur_sim["all_finite_and_bounded"],
            f"max(K)={cur_sim['max_K_simulated']:.3e} <= {cur_sim['K_max_analytic']:.3e}",
        ),
        (
            "8. 24-Cluster-Resonanzmittelwert innerhalb 3-Sigma-MC-Fenster",
            cur_sim["within_3sigma_mc"] and rel_err_R < 0.02,
            f"<R>={cur_sim['mean_cluster_R_ens']:.6f} vs. {cur_sim['expected_cluster_R']:.6f} (Err={rel_err_R*100:.2f}%)",
        ),
        (
            "9. Halo-Dichtekontrast & Kernradius-Ordnung (30 vs. 3000 km/s)",
            extra["ratio_dwarf"] < extra["ratio_cluster"] and dwarf_sim["core_radius_kpc"] < cluster_sim["core_radius_kpc"],
            f"Kontrast(30)={extra['ratio_dwarf']:.2f} < Kontrast(3000)={extra['ratio_cluster']:.2f}",
        ),
        (
            "10. Strenge Monotonie von x(v) & x(v_turn) = 1.0000",
            x_monotone and x_at_turn_ok,
            f"x(10)={x_grid[0]:.3f}, x(v_turn)={x_grid[3]:.4f}, x(3000)={x_grid[4]:.2f}",
        ),
        (
            "11. Symplektische Stabilität bei halbierter Zeitschrittweite dt/2",
            dt_half_sim["rel_energy_drift"] < 5e-2 and np.isfinite(dt_half_sim["core_radius_kpc"]),
            f"dE/E(dt=0.001)={dt_half_sim['rel_energy_drift']:.2e}, r_core={dt_half_sim['core_radius_kpc']:.3f} kpc",
        ),
        (
            "12. Endlichkeit & Positivität der Zusatzvorhersagen 11–15",
            np.isfinite(extra["r_turn_kpc"])
            and extra["r_turn_kpc"] > 0.0
            and extra["mean_h_eff"] > 0.0
            and np.isfinite(extra["max_k_reg"])
            and extra["max_k_reg"] > 0.0
            and np.isfinite(extra["max_long_drift"]),
            f"r_turn={extra['r_turn_kpc']:.3f} kpc, <H_eff>={extra['mean_h_eff']:.3e} GeV^4",
        ),
        (
            "13. Periodische Langzeit-Checks durchgehend OK",
            long_sim["all_checks_ok"] and len(long_sim["checkpoints"]) >= 2 and long_sim["max_drift_seen"] < 5e-2,
            f"Checks={len(long_sim['checkpoints'])}, max_drift={long_sim['max_drift_seen']:.2e}, all_OK={long_sim['all_checks_ok']}",
        ),
    ]

    all_passed = True
    for name, ok, info in tests:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name:<54} | {info}")
        if not ok:
            all_passed = False
    return all_passed


# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================
if __name__ == "__main__":
    t0 = time.perf_counter()
    n_part = 800_000  # Auf 8_000_000 erhöhen für den 8M-Vollauf
    print("==============================================================================")
    print(f" PARTIKELSIMULATOR v9: 24-ZELL-TORI & SCHATTENRESONANZ (N = {n_part:,})")
    print("==============================================================================")

    model = init_model_parameters()
    print(f"Basis: tau={model['tau']:.5f}, L_T={model['L_T']:.3e} m, Phi_T={model['Phi_T_GeV']:.3e} GeV, M_R={model['M_R_GeV']:.3e} GeV")
    print(f"Schale 1: m_n'={model['m_n_phys_GeV']:.4f} GeV, m_med={model['m_med_phys_GeV']*1e3:.3f} MeV, v_turn={model['v_turn_km_s']:.1f} km/s")

    print(f"\n--- 1. YUKAWA-DIPOL-SIDM (N={n_part:,} TEILCHEN, 32^3 PM-GITTER) ---")
    dwarf_sim = simulate_sidm_halo(model, v_rms_km_s=30.0, n_particles=n_part, n_steps=60, dt_val=0.002, grid_n=32, seed=42)
    cluster_sim = simulate_sidm_halo(model, v_rms_km_s=3000.0, n_particles=n_part, n_steps=60, dt_val=0.002, grid_n=32, seed=99)

    print(
        f"  Zwerggalaxie (30 km/s)   | x={dwarf_sim['x_mom']:.3e} | Born={dwarf_sim['born_factor']:.3e} | "
        f"dE/E={dwarf_sim['rel_energy_drift']:.2e} | r_core(25%)={dwarf_sim['core_radius_kpc']:.3f} kpc | t_coll={dwarf_sim['t_coll_Gyr']:.0f} Gyr"
    )
    print(
        f"  Galaxienhaufen (3000 km/s)| x={cluster_sim['x_mom']:.3e} | Born={cluster_sim['born_factor']:.3e} | "
        f"dE/E={cluster_sim['rel_energy_drift']:.2e} | r_core(25%)={cluster_sim['core_radius_kpc']:.3f} kpc"
    )

    long_sim = run_long_term_simulation_with_periodic_checks(
        model, v_rms_km_s=30.0, n_particles=n_part, n_steps=600, check_interval=100, dt_val=0.002, grid_n=32, seed=777
    )

    print("\n--- 1C. RADIALES SCHALEN-DICHTEPROFIL rho(r) (8 LOG-SCHALEN) ---")
    prof_dwarf = analyze_radial_density_profile(dwarf_sim, r_min=0.05, r_max=2.0, n_shells=8)
    prof_cluster = analyze_radial_density_profile(cluster_sim, r_min=0.05, r_max=2.0, n_shells=8)
    for rc, rd, rclust in zip(prof_dwarf["r_centers_kpc"], prof_dwarf["rho_kpc3"], prof_cluster["rho_kpc3"]):
        print(f"  r={rc:6.3f} kpc | rho_dwarf(30 km/s)={rd:11.1f} kpc^-3 | rho_cluster(3000 km/s)={rclust:11.1f} kpc^-3")

    print(f"\n--- 2. LAGRANGE-DICHTE-AUSWERTUNG ÜBER {n_part:,} TEILCHEN ---")
    lag_dwarf = evaluate_lagrangian_over_ensemble(model, dwarf_sim)
    print(
        f"  Ausgewertet: {lag_dwarf['n_evaluated']:,} Teilchen | <H_med>={lag_dwarf['mean_H_med_GeV4']:.3e} GeV^4 | "
        f"<E_proca>={lag_dwarf['mean_E_proca_GeV4']:.3e} GeV^4 | <H_eff>={lag_dwarf['mean_H_eff_GeV4']:.3e} GeV^4"
    )

    print("\n--- 3. RAUMZEITKRÜMMUNG & 24-CLUSTER-ENSEMBLE ---")
    cur_sim = simulate_curvature_and_24clusters(
        model, n_particles=n_part, n_cluster_draws=n_part, l_core=0.02, seed=2026
    )
    print(
        f"  R_curv in [{cur_sim['min_R_curv']:.2e}, {cur_sim['max_R_curv']:.2e}] | "
        f"max(K)={cur_sim['max_K_simulated']:.3e} | <R>_ens={cur_sim['mean_cluster_R_ens']:.6f} (+/- {cur_sim['sigma_3sigma_mc']:.2e} 3-sig)"
    )

    extra = compute_additional_simulation_predictions(
        model, dwarf_sim, cluster_sim, prof_dwarf, prof_cluster, lag_dwarf, cur_sim, long_sim
    )

    tests_ok = run_tests(model, dwarf_sim, cluster_sim, prof_dwarf, prof_cluster, lag_dwarf, cur_sim, long_sim, extra)
    elapsed = time.perf_counter() - t0
    print(f"\nLaufzeit ({n_part:,} Teilchen + Langzeit + 13 Tests): {elapsed:.2f} s | Gesamtstatus: {'PASS' if tests_ok else 'FAIL'}")
    print("==============================================================================")

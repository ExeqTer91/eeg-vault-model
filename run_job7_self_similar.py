#!/usr/bin/env python3
"""
JOB 7: SELF-SIMILAR ANABOLIC CASCADE
Tests whether anabolic flux patterns repeat across timescales
with e-1 as the scaling ratio.

Parts A-D:
A) Multi-scale flux measurement (e-ladder vs phi-ladder vs 2-ladder)
B) Radial peripheral shell structure
C) Power spectrum of anabolic flux
D) Cross-scale correlation and information flow direction

Key observable: order parameter r(t) and its fluctuations at multiple timescales.
"Flux" = rate of change of local synchronization = dr/dt (signed).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import welch
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
E_MINUS_1 = np.e - 1
INV_E = 1 / np.e

OUTPUT_DIR = 'outputs/publication_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})


def run_kuramoto_trajectory(K, N_osc, noise_sigma, dt=0.001, T_total=40.0, fs_out=1000):
    np.random.seed(hash((K, N_osc, int(noise_sigma * 1000), 77)) % 2**31)
    f_min, f_max = 2.0, 45.0
    freqs = f_min * (f_max / f_min) ** (np.arange(N_osc) / max(N_osc - 1, 1))
    omega = 2 * np.pi * freqs

    n_steps = int(T_total / dt)
    theta = np.random.uniform(0, 2 * np.pi, N_osc)

    out_dt = 1.0 / fs_out
    n_out = int(T_total * fs_out)
    r_out = np.zeros(n_out)
    psi_out = np.zeros(n_out)
    phase_snapshots = np.zeros((n_out, N_osc))
    out_idx = 0
    next_out_t = 0.0

    for step in range(n_steps):
        t = step * dt
        z = np.mean(np.exp(1j * theta))
        r_abs = np.abs(z)
        r_angle = np.angle(z)
        coupling = K * r_abs * np.sin(r_angle - theta)
        noise_term = noise_sigma * np.sqrt(dt) * np.random.randn(N_osc)
        dtheta = (omega + coupling) * dt + noise_term
        theta += dtheta

        if out_idx < n_out and t >= next_out_t:
            r_out[out_idx] = r_abs
            psi_out[out_idx] = r_angle
            phase_snapshots[out_idx] = theta % (2 * np.pi)
            out_idx += 1
            next_out_t += out_dt

    actual = min(out_idx, n_out)
    return {
        'r': r_out[:actual],
        'psi': psi_out[:actual],
        'phases': phase_snapshots[:actual],
        'omega': omega,
        'freqs': freqs,
        'N': N_osc,
        'K': K,
        'fs': fs_out,
        'T': T_total,
    }


def compute_windowed_r(r_ts, window_samples):
    if window_samples < 1 or len(r_ts) < window_samples:
        return r_ts.copy()
    return uniform_filter1d(r_ts, size=window_samples)


def compute_flux_at_scale(r_ts, window_samples):
    r_smooth = compute_windowed_r(r_ts, window_samples)
    dr = np.diff(r_smooth)
    return dr, r_smooth


def symmetric_kl(p, q, epsilon=1e-10):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return (kl_pq + kl_qp) / 2


print("=" * 60)
print("JOB 7: SELF-SIMILAR ANABOLIC CASCADE")
print("=" * 60)

N_CONFIGS = 25
N_OSC = 100
T_TOTAL = 40.0
FS_OUT = 1000
DT = 0.001

config_params = []
for i in range(N_CONFIGS):
    K = 1.5 + i * 0.25
    noise = 0.3 + (i % 5) * 0.1
    config_params.append((K, N_OSC, noise))

print(f"\nRunning {N_CONFIGS} Kuramoto configs for {T_TOTAL}s each...")
print(f"  N_osc = {N_OSC}, dt = {DT}, fs_out = {FS_OUT} Hz")

simulations = []
for ci, (K, N, noise) in enumerate(config_params):
    if (ci + 1) % 5 == 0 or ci == 0:
        print(f"  Config {ci+1}/{N_CONFIGS}: K={K:.2f}, N={N}, noise={noise:.2f}")
    sim = run_kuramoto_trajectory(K, N, noise, dt=DT, T_total=T_TOTAL, fs_out=FS_OUT)
    simulations.append(sim)
    if ci == 0:
        print(f"    r(t) range: [{sim['r'].min():.4f}, {sim['r'].max():.4f}], "
              f"mean = {sim['r'].mean():.4f}")

print(f"\n  All {N_CONFIGS} simulations complete.")

r_means = [sim['r'][len(sim['r'])//2:].mean() for sim in simulations]
r_stds = [sim['r'][len(sim['r'])//2:].std() for sim in simulations]
print(f"  Order parameter (2nd half): mean = {np.mean(r_means):.4f} ± {np.mean(r_stds):.4f}")

print("\n" + "=" * 60)
print("PART A: MULTI-SCALE FLUX MEASUREMENT")
print("=" * 60)

base_window_sec = 0.01

e_windows = [base_window_sec * np.e**k for k in range(7)]
phi_windows = [base_window_sec * PHI**k for k in range(7)]
two_windows = [base_window_sec * 2**k for k in range(7)]

e_windows = [w for w in e_windows if w * FS_OUT < T_TOTAL * FS_OUT / 4]
phi_windows = [w for w in phi_windows if w * FS_OUT < T_TOTAL * FS_OUT / 4]
two_windows = [w for w in two_windows if w * FS_OUT < T_TOTAL * FS_OUT / 4]

print(f"\n  e-ladder windows (s): {[f'{w:.4f}' for w in e_windows]}")
print(f"  φ-ladder windows (s): {[f'{w:.4f}' for w in phi_windows]}")
print(f"  2-ladder windows (s): {[f'{w:.4f}' for w in two_windows]}")

n_bins = 60
bin_range = (-4, 4)

ladders = {'e': e_windows, 'phi': phi_windows, 'two': two_windows}
ladder_results = {}

for ladder_name, windows in ladders.items():
    print(f"\n  --- {ladder_name}-ladder ---")
    scale_hists = {}
    scale_stats = {}

    for wi, ws in enumerate(windows):
        ws_samples = max(2, int(ws * FS_OUT))
        all_flux_normalized = []

        for sim in simulations:
            r_ts = sim['r']
            anabolic, r_smooth = compute_flux_at_scale(r_ts, ws_samples)
            if len(anabolic) < 20:
                continue
            mean_a = np.mean(anabolic)
            std_a = np.std(anabolic)
            if std_a > 1e-15:
                flux_norm = (anabolic - mean_a) / std_a
                all_flux_normalized.extend(flux_norm.tolist())

        if len(all_flux_normalized) > 100:
            arr = np.array(all_flux_normalized)
            hist, edges = np.histogram(arr, bins=n_bins, range=bin_range, density=True)
            scale_hists[wi] = hist.tolist()
            scale_stats[wi] = {
                'n_samples': len(arr),
                'kurtosis': float(stats.kurtosis(arr)),
                'skewness': float(stats.skew(arr)),
            }
            print(f"    Scale {wi} ({ws:.4f}s, {ws_samples} samp): "
                  f"n={len(arr)}, kurt={stats.kurtosis(arr):.3f}, skew={stats.skew(arr):.3f}")
        else:
            scale_hists[wi] = None
            print(f"    Scale {wi} ({ws:.4f}s): insufficient data")

    per_scale_kl = []
    for wi in range(len(windows) - 1):
        h1 = scale_hists.get(wi)
        h2 = scale_hists.get(wi + 1)
        if h1 is not None and h2 is not None:
            skl = symmetric_kl(h1, h2)
            per_scale_kl.append(skl)
            print(f"    SKL {wi}→{wi+1}: {skl:.6f}")
        else:
            per_scale_kl.append(float('nan'))

    valid_kls = [k for k in per_scale_kl if np.isfinite(k)]
    mean_kl = float(np.mean(valid_kls)) if valid_kls else float('inf')
    ladder_results[ladder_name] = {
        'per_scale_kl': per_scale_kl,
        'mean_kl': mean_kl,
        'scale_hists': scale_hists,
        'scale_stats': scale_stats,
        'windows': [float(w) for w in windows],
    }
    print(f"    Mean SKL: {mean_kl:.6f}")

valid_ladders = {k: v['mean_kl'] for k, v in ladder_results.items() if np.isfinite(v['mean_kl'])}
if valid_ladders:
    best_ladder = min(valid_ladders, key=valid_ladders.get)
else:
    best_ladder = 'none'

print(f"\n  COLLAPSE QUALITY (lowest SKL = most self-similar):")
for ln, lv in sorted(valid_ladders.items(), key=lambda x: x[1]):
    marker = ' ← lowest' if ln == best_ladder else ''
    label = {'e': 'e', 'phi': 'φ', 'two': '2'}[ln]
    print(f"    {label}-ladder: {lv:.6f}{marker}")

print(f"\n  --- Bootstrap CI for ladder SKL differences (paired, config-level) ---")

n_configs_all = len(simulations)
paired_kls = {ln: np.full(n_configs_all, np.nan) for ln in ladders}

for ci_idx, sim in enumerate(simulations):
    for ln, windows in ladders.items():
        config_kls = []
        for wi in range(len(windows) - 1):
            ws1 = max(2, int(windows[wi] * FS_OUT))
            ws2 = max(2, int(windows[wi+1] * FS_OUT))
            dr1, _ = compute_flux_at_scale(sim['r'], ws1)
            dr2, _ = compute_flux_at_scale(sim['r'], ws2)
            if len(dr1) < 20 or len(dr2) < 20:
                continue
            std1 = np.std(dr1)
            std2 = np.std(dr2)
            if std1 < 1e-15 or std2 < 1e-15:
                continue
            n1 = (dr1 - np.mean(dr1)) / std1
            n2 = (dr2 - np.mean(dr2)) / std2
            h1, _ = np.histogram(n1, bins=n_bins, range=bin_range, density=True)
            h2, _ = np.histogram(n2, bins=n_bins, range=bin_range, density=True)
            config_kls.append(symmetric_kl(h1, h2))
        if config_kls:
            paired_kls[ln][ci_idx] = np.mean(config_kls)

valid_mask = np.all([np.isfinite(paired_kls[ln]) for ln in ladders], axis=0)
n_valid = int(np.sum(valid_mask))
print(f"    Configs with valid data for all ladders: {n_valid}/{n_configs_all}")

paired_e = paired_kls['e'][valid_mask]
paired_phi = paired_kls['phi'][valid_mask]
paired_two = paired_kls['two'][valid_mask]

n_boot = 2000
boot_diff_e_phi = np.zeros(n_boot)
boot_diff_phi_two = np.zeros(n_boot)
boot_diff_e_two = np.zeros(n_boot)

for b in range(n_boot):
    idx = np.random.choice(n_valid, size=n_valid, replace=True)
    boot_diff_e_phi[b] = np.mean(paired_e[idx]) - np.mean(paired_phi[idx])
    boot_diff_phi_two[b] = np.mean(paired_phi[idx]) - np.mean(paired_two[idx])
    boot_diff_e_two[b] = np.mean(paired_e[idx]) - np.mean(paired_two[idx])

for ln in ladders:
    vals = paired_kls[ln][valid_mask]
    boot_m = [np.mean(vals[np.random.choice(n_valid, n_valid, replace=True)]) for _ in range(n_boot)]
    label = {'e': 'e', 'phi': 'φ', 'two': '2'}[ln]
    print(f"    {label}-ladder: {np.mean(vals):.6f} [95% CI: {np.percentile(boot_m, 2.5):.6f}, {np.percentile(boot_m, 97.5):.6f}]")

ci_e_phi = (np.percentile(boot_diff_e_phi, 2.5), np.percentile(boot_diff_e_phi, 97.5))
ci_phi_two = (np.percentile(boot_diff_phi_two, 2.5), np.percentile(boot_diff_phi_two, 97.5))
ci_e_two = (np.percentile(boot_diff_e_two, 2.5), np.percentile(boot_diff_e_two, 97.5))

print(f"\n    Paired differences (positive = first ladder has HIGHER SKL = WORSE collapse):")
print(f"    e - φ: {np.mean(boot_diff_e_phi):.6f} [CI: {ci_e_phi[0]:.6f}, {ci_e_phi[1]:.6f}]")
print(f"    φ - 2: {np.mean(boot_diff_phi_two):.6f} [CI: {ci_phi_two[0]:.6f}, {ci_phi_two[1]:.6f}]")
print(f"    e - 2: {np.mean(boot_diff_e_two):.6f} [CI: {ci_e_two[0]:.6f}, {ci_e_two[1]:.6f}]")

e_vs_phi_sig = not (ci_e_phi[0] <= 0 <= ci_e_phi[1])
phi_vs_two_sig = not (ci_phi_two[0] <= 0 <= ci_phi_two[1])
e_vs_two_sig = not (ci_e_two[0] <= 0 <= ci_e_two[1])

print(f"\n    e vs φ: {'SIGNIFICANT' if e_vs_phi_sig else 'NOT significant'} (CI {'excludes' if e_vs_phi_sig else 'includes'} 0)")
print(f"    φ vs 2: {'SIGNIFICANT' if phi_vs_two_sig else 'NOT significant'} (CI {'excludes' if phi_vs_two_sig else 'includes'} 0)")
print(f"    e vs 2: {'SIGNIFICANT' if e_vs_two_sig else 'NOT significant'} (CI {'excludes' if e_vs_two_sig else 'includes'} 0)")

phi_best_significant = e_vs_phi_sig and phi_vs_two_sig and (np.mean(boot_diff_e_phi) > 0) and (np.mean(boot_diff_phi_two) < 0)

if phi_best_significant:
    print(f"\n  CONCLUSION: φ-ladder produces SIGNIFICANTLY better collapse than both e and 2 ladders.")
    best_ladder_note = 'phi_significant'
    ladder_distinguishable = True
elif e_vs_phi_sig or phi_vs_two_sig:
    print(f"\n  CONCLUSION: Partial significance — some ladders differ but φ is not uniquely best.")
    best_ladder_note = 'partial_significance'
    ladder_distinguishable = True
else:
    print(f"\n  CONCLUSION: All three ladders produce statistically EQUIVALENT collapse.")
    print(f"  The self-similar scaling constant cannot be determined from these data.")
    best_ladder_note = 'indistinguishable'
    ladder_distinguishable = False

pairwise_test = {
    'e_vs_phi_ci': [float(ci_e_phi[0]), float(ci_e_phi[1])],
    'phi_vs_two_ci': [float(ci_phi_two[0]), float(ci_phi_two[1])],
    'e_vs_two_ci': [float(ci_e_two[0]), float(ci_e_two[1])],
    'e_vs_phi_significant': bool(e_vs_phi_sig),
    'phi_vs_two_significant': bool(phi_vs_two_sig),
    'phi_best_significant': bool(phi_best_significant),
    'n_paired_configs': n_valid,
}

scale_collapse = {
    'e_ladder_mean_KL': ladder_results['e']['mean_kl'],
    'phi_ladder_mean_KL': ladder_results['phi']['mean_kl'],
    'two_ladder_mean_KL': ladder_results['two']['mean_kl'],
    'lowest_skl_ladder': best_ladder,
    'ladders_distinguishable': ladder_distinguishable,
    'bootstrap_note': best_ladder_note,
    'per_scale_KL': {k: v['per_scale_kl'] for k, v in ladder_results.items()},
    'bootstrap_ci': pairwise_test,
}

print("\n" + "=" * 60)
print("PART B: RADIAL PERIPHERAL STRUCTURE")
print("=" * 60)

n_shells = 5
all_shell_fluxes = []

for si, sim in enumerate(simulations):
    phases = sim['phases']
    N_t, N_osc_local = phases.shape
    omega = sim['omega']
    fs = sim['fs']

    late_start = N_t // 2
    late_phases = phases[late_start:]

    mean_phase_complex = np.mean(np.exp(1j * late_phases), axis=0)
    mean_field = np.mean(mean_phase_complex)
    mean_field_angle = np.angle(mean_field)

    angular_dist = np.abs(np.angle(mean_phase_complex / np.abs(mean_phase_complex) *
                                    np.exp(-1j * mean_field_angle)))

    sorted_idx = np.argsort(angular_dist)
    shell_size = N_osc_local // n_shells

    shell_data = []
    for sh in range(n_shells):
        start = sh * shell_size
        end = start + shell_size if sh < n_shells - 1 else N_osc_local
        osc_idx = sorted_idx[start:end]

        shell_phases = late_phases[:, osc_idx]
        z_shell = np.mean(np.exp(1j * shell_phases), axis=1)
        r_shell = np.abs(z_shell)

        dr = np.diff(r_shell)
        anabolic_flux = np.std(dr)
        r_mean = float(np.mean(r_shell))
        r_std = float(np.std(r_shell))
        r_fluct = r_std / (r_mean + 1e-10)

        shell_data.append({
            'shell': sh,
            'n_osc': len(osc_idx),
            'r_mean': r_mean,
            'r_std': r_std,
            'r_fluctuation': r_fluct,
            'anabolic_flux': float(anabolic_flux),
            'angular_distance': float(np.mean(angular_dist[osc_idx])),
        })

    all_shell_fluxes.append(shell_data)

mean_shell_flux = np.zeros(n_shells)
mean_shell_r = np.zeros(n_shells)
mean_shell_fluct = np.zeros(n_shells)
std_shell_flux = np.zeros(n_shells)

for sh in range(n_shells):
    fluxes = [res[sh]['anabolic_flux'] for res in all_shell_fluxes]
    rs = [res[sh]['r_mean'] for res in all_shell_fluxes]
    flucts = [res[sh]['r_fluctuation'] for res in all_shell_fluxes]
    mean_shell_flux[sh] = np.mean(fluxes)
    std_shell_flux[sh] = np.std(fluxes)
    mean_shell_r[sh] = np.mean(rs)
    mean_shell_fluct[sh] = np.mean(flucts)

print(f"\n  Shell structure (averaged over {len(simulations)} configs):")
for sh in range(n_shells):
    print(f"    Shell {sh}: r = {mean_shell_r[sh]:.4f}, "
          f"flux = {mean_shell_flux[sh]:.6f}, "
          f"fluctuation = {mean_shell_fluct[sh]:.4f}")

adjacent_ratios_flux = []
adjacent_ratios_fluct = []
for sh in range(n_shells - 1):
    if mean_shell_flux[sh] > 1e-12:
        ratio_f = mean_shell_flux[sh+1] / mean_shell_flux[sh]
        adjacent_ratios_flux.append(ratio_f)
    if mean_shell_fluct[sh] > 1e-12:
        ratio_fl = mean_shell_fluct[sh+1] / mean_shell_fluct[sh]
        adjacent_ratios_fluct.append(ratio_fl)

print(f"\n  Adjacent shell flux ratios: {[f'{r:.4f}' for r in adjacent_ratios_flux]}")
print(f"  Adjacent shell fluctuation ratios: {[f'{r:.4f}' for r in adjacent_ratios_fluct]}")

if adjacent_ratios_flux:
    mean_ratio = float(np.mean(adjacent_ratios_flux))
    diffs = {
        'e_inv': abs(mean_ratio - INV_E),
        'phi_inv': abs(mean_ratio - PHI_INV),
        'half': abs(mean_ratio - 0.5),
        'one': abs(mean_ratio - 1.0),
    }
    nearest = min(diffs, key=diffs.get)
    print(f"\n  Mean flux ratio: {mean_ratio:.4f}")
    print(f"    e⁻¹ = {INV_E:.4f} (diff: {diffs['e_inv']:.4f})")
    print(f"    φ⁻¹ = {PHI_INV:.4f} (diff: {diffs['phi_inv']:.4f})")
    print(f"    0.5 (diff: {diffs['half']:.4f})")
    print(f"    1.0 (diff: {diffs['one']:.4f})")
    print(f"  Nearest: {nearest}")
else:
    mean_ratio = 0
    nearest = 'undetermined'

n_coherent = sum(1 for sh in range(n_shells) if mean_shell_r[sh] > 0.1)

radial_shells = {
    'n_shells': n_shells,
    'n_coherent_shells': n_coherent,
    'flux_ratio_adjacent_shells': float(mean_ratio) if adjacent_ratios_flux else 0,
    'nearest_constant': nearest,
    'per_shell': [
        {'shell': sh, 'r_mean': float(mean_shell_r[sh]),
         'anabolic_flux': float(mean_shell_flux[sh]),
         'fluctuation': float(mean_shell_fluct[sh])}
        for sh in range(n_shells)
    ],
    'adjacent_flux_ratios': [float(r) for r in adjacent_ratios_flux],
    'adjacent_fluct_ratios': [float(r) for r in adjacent_ratios_fluct],
}

print("\n" + "=" * 60)
print("PART C: POWER SPECTRUM OF ANABOLIC FLUX")
print("=" * 60)

window_flux_sec = 0.05
window_flux_samp = max(2, int(window_flux_sec * FS_OUT))

all_psd_freqs = None
all_psds = []
all_betas = []

for sim in simulations:
    r_ts = sim['r']
    anabolic, r_smooth = compute_flux_at_scale(r_ts, window_flux_samp)
    if len(anabolic) < 100:
        continue

    flux_fs = sim['fs']
    nperseg = min(2048, len(anabolic) // 4)
    if nperseg < 64:
        continue
    f_psd, psd_val = welch(anabolic, fs=flux_fs, nperseg=nperseg)

    valid = (f_psd > 0.1) & (f_psd < flux_fs / 4) & (psd_val > 0)
    if np.sum(valid) < 10:
        continue

    log_f = np.log10(f_psd[valid])
    log_p = np.log10(psd_val[valid])
    slope, intercept, r_val, p_val, _ = stats.linregress(log_f, log_p)
    beta = -slope
    r2 = r_val**2

    all_betas.append(beta)
    all_psds.append(psd_val)
    if all_psd_freqs is None:
        all_psd_freqs = f_psd

all_betas = np.array(all_betas)

if len(all_betas) > 0:
    print(f"\n  Power spectrum analysis ({len(all_betas)} configs):")
    print(f"    β = {np.mean(all_betas):.4f} ± {np.std(all_betas):.4f}")
    print(f"    Median β = {np.median(all_betas):.4f}")
    print(f"\n  Comparisons:")
    print(f"    β ≈ 1 (1/f, critical): diff = {abs(np.mean(all_betas) - 1):.4f}")
    print(f"    β ≈ 2 (Brownian): diff = {abs(np.mean(all_betas) - 2):.4f}")
    print(f"    β ≈ 1/φ = {PHI_INV:.4f}: diff = {abs(np.mean(all_betas) - PHI_INV):.4f}")
    print(f"    β ≈ e⁻¹ = {INV_E:.4f}: diff = {abs(np.mean(all_betas) - INV_E):.4f}")

    min_len_psd = min(len(p) for p in all_psds)
    psd_matrix = np.array([p[:min_len_psd] for p in all_psds])
    mean_psd = np.mean(psd_matrix, axis=0)
    psd_freqs = all_psd_freqs[:min_len_psd] if all_psd_freqs is not None else np.arange(min_len_psd)
else:
    print("\n  No valid power spectra computed")
    mean_psd = np.array([0])
    psd_freqs = np.array([0])

spectral_analysis = {
    'mean_beta': float(np.mean(all_betas)) if len(all_betas) > 0 else 0,
    'std_beta': float(np.std(all_betas)) if len(all_betas) > 0 else 0,
    'median_beta': float(np.median(all_betas)) if len(all_betas) > 0 else 0,
    'n_configs': len(all_betas),
}

print("\n" + "=" * 60)
print("PART D: CROSS-SCALE CORRELATION")
print("=" * 60)

fast_window_sec = 0.01
slow_windows_test = {
    'e': fast_window_sec * np.e,
    'phi': fast_window_sec * PHI,
    'two': fast_window_sec * 2,
}

cross_scale_results = {}

for scale_name, slow_sec in slow_windows_test.items():
    fast_samp = max(2, int(fast_window_sec * FS_OUT))
    slow_samp = max(2, int(slow_sec * FS_OUT))

    all_cc = []
    all_peak_lags = []

    for sim in simulations:
        r_ts = sim['r']
        anab_fast, _ = compute_flux_at_scale(r_ts, fast_samp)
        anab_slow, _ = compute_flux_at_scale(r_ts, slow_samp)

        min_n = min(len(anab_fast), len(anab_slow))
        if min_n < 100:
            continue
        anab_fast = anab_fast[:min_n]
        anab_slow = anab_slow[:min_n]

        std_f = np.std(anab_fast)
        std_s = np.std(anab_slow)
        if std_f < 1e-15 or std_s < 1e-15:
            continue

        ff = (anab_fast - np.mean(anab_fast)) / std_f
        fs_n = (anab_slow - np.mean(anab_slow)) / std_s

        max_lag = min(min_n // 8, 100)
        cc_vals = np.zeros(2 * max_lag + 1)
        lag_axis_local = np.arange(-max_lag, max_lag + 1)

        for li, lag in enumerate(lag_axis_local):
            if lag >= 0:
                n_overlap = min_n - lag
                if n_overlap > 10:
                    cc_vals[li] = np.mean(ff[:n_overlap] * fs_n[lag:lag + n_overlap])
            else:
                n_overlap = min_n + lag
                if n_overlap > 10:
                    cc_vals[li] = np.mean(ff[-lag:-lag + n_overlap] * fs_n[:n_overlap])

        all_cc.append(cc_vals)
        peak_idx = np.argmax(np.abs(cc_vals))
        all_peak_lags.append(lag_axis_local[peak_idx])

    if all_cc:
        min_cc_len = min(len(c) for c in all_cc)
        cc_matrix = np.array([c[:min_cc_len] for c in all_cc])
        mean_cc = np.mean(cc_matrix, axis=0)
        lag_axis = np.arange(-(min_cc_len // 2), min_cc_len // 2 + 1)[:min_cc_len]

        peak_idx = np.argmax(np.abs(mean_cc))
        peak_lag = lag_axis[peak_idx]
        max_cc_val = float(mean_cc[peak_idx])

        if peak_lag < -2:
            direction = 'top_down'
        elif peak_lag > 2:
            direction = 'bottom_up'
        else:
            direction = 'symmetric'

        print(f"\n  {scale_name}-scale (fast={fast_window_sec}s, slow={slow_sec:.4f}s):")
        print(f"    Peak lag: {peak_lag} samples ({direction})")
        print(f"    Max cross-corr: {max_cc_val:.6f}")
        print(f"    Individual peak lags: {np.mean(all_peak_lags):.1f} ± {np.std(all_peak_lags):.1f}")

        cross_scale_results[scale_name] = {
            'fast_window': fast_window_sec,
            'slow_window': float(slow_sec),
            'peak_lag': int(peak_lag),
            'direction': direction,
            'max_cross_correlation': float(max_cc_val),
            'mean_cc': mean_cc.tolist(),
            'lag_axis': lag_axis.tolist(),
            'n_configs': len(all_cc),
        }
    else:
        cross_scale_results[scale_name] = {
            'direction': 'insufficient_data', 'max_cross_correlation': 0
        }

best_cc_scale = max(cross_scale_results, key=lambda k: abs(cross_scale_results[k].get('max_cross_correlation', 0)))
print(f"\n  Strongest cross-scale coupling: {best_cc_scale}-scale")

print("\n" + "=" * 60)
print("GENERATING FIGURE 17")
print("=" * 60)

fig = plt.figure(figsize=(14, 14))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
bin_centers = np.linspace(-4, 4, n_bins)

ladder_colors = {'e': 'steelblue', 'phi': 'goldenrod', 'two': 'firebrick'}
ladder_labels = {'e': 'e', 'phi': 'φ', 'two': '2'}

best_hists = ladder_results[best_ladder]['scale_hists']
best_windows = ladder_results[best_ladder]['windows']
n_scales_plot = sum(1 for v in best_hists.values() if v is not None)
scale_colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(n_scales_plot, 1)))
ci = 0
for wi, h in sorted(best_hists.items(), key=lambda x: int(x[0])):
    if h is not None:
        ax1.plot(bin_centers, h, color=scale_colors[ci], lw=1.5,
                 label=f'{best_windows[int(wi)]:.4f}s', alpha=0.85)
        ci += 1

ax1.set_xlabel('Normalized Anabolic Flux (σ units)')
ax1.set_ylabel('Density')
collapse_title = f'{ladder_labels[best_ladder]}-Ladder' if ladder_distinguishable else 'All Ladders Equivalent'
ax1.set_title(f'A. Flux Distribution Collapse ({collapse_title})')
ax1.legend(fontsize=6, title='Window', title_fontsize=7, ncol=2)

kl_text = 'Mean SKL:\n'
for ln in ['e', 'phi', 'two']:
    label = ladder_labels[ln]
    val = ladder_results[ln]['mean_kl']
    kl_text += f'  {label}: {val:.5f}\n'
if ladder_distinguishable:
    kl_text += f'Best: {ladder_labels[best_ladder]}\n'
else:
    kl_text += 'All equivalent\n(bootstrap CIs overlap)'
ax1.text(0.98, 0.95, kl_text.strip(), transform=ax1.transAxes, fontsize=7,
         va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
         family='monospace')

ax2 = fig.add_subplot(gs[0, 1])
shell_x = np.arange(n_shells)
ax2.bar(shell_x, mean_shell_flux, yerr=std_shell_flux, color='steelblue', alpha=0.7,
        edgecolor='black', lw=0.5, capsize=4)

if n_shells > 2 and mean_shell_flux[0] > 0:
    try:
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)
        popt_shell, _ = curve_fit(exp_decay, shell_x, mean_shell_flux,
                                   p0=[mean_shell_flux[0], 0.5], maxfev=5000)
        x_fine = np.linspace(0, n_shells - 1, 100)
        ax2.plot(x_fine, exp_decay(x_fine, *popt_shell), 'r-', lw=2,
                 label=f'Exp fit (rate={popt_shell[1]:.3f})')

        e_decay = mean_shell_flux[0] * INV_E ** shell_x
        phi_decay = mean_shell_flux[0] * PHI_INV ** shell_x
        ax2.plot(shell_x, e_decay, 's--', color='orange', ms=5, lw=1.5, alpha=0.7,
                 label=f'e⁻¹ decay')
        ax2.plot(shell_x, phi_decay, 'd--', color='goldenrod', ms=5, lw=1.5, alpha=0.7,
                 label=f'φ⁻¹ decay')
    except Exception:
        pass

ax2.set_xlabel('Shell Number (0 = core)')
ax2.set_ylabel('Mean Anabolic Flux')
ax2.set_title('B. Radial Shell Flux Decay')
ax2.legend(fontsize=7)
if adjacent_ratios_flux:
    ax2.text(0.98, 0.95,
             f'Mean ratio: {mean_ratio:.4f}\nNearest: {nearest}\n'
             f'Coherent shells: {n_coherent}/{n_shells}',
             transform=ax2.transAxes, fontsize=7, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax3 = fig.add_subplot(gs[1, 0])
if len(psd_freqs) > 1 and len(mean_psd) > 1:
    valid_psd = (psd_freqs > 0) & (mean_psd > 0)
    if np.sum(valid_psd) > 5:
        ax3.loglog(psd_freqs[valid_psd], mean_psd[valid_psd], 'steelblue', lw=2, label='Mean flux PSD')

        log_f = np.log10(psd_freqs[valid_psd])
        log_p = np.log10(mean_psd[valid_psd])
        fit_mask = psd_freqs[valid_psd] > 0.1
        if np.sum(fit_mask) > 5:
            slope, intercept, _, _, _ = stats.linregress(log_f[fit_mask], log_p[fit_mask])
            beta_fit = -slope
            f_line = psd_freqs[valid_psd][fit_mask]
            p_line = 10**(intercept + slope * np.log10(f_line))
            ax3.loglog(f_line, p_line, 'r--', lw=1.5,
                       label=f'β = {beta_fit:.3f}')

ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Power Spectral Density')
ax3.set_title('C. Power Spectrum of Anabolic Flux')
ax3.legend(fontsize=7)

if len(all_betas) > 0:
    beta_nearest = '1/f' if abs(np.mean(all_betas) - 1) < abs(np.mean(all_betas) - 2) else 'Brownian'
    ax3.text(0.98, 0.05,
             f'β = {np.mean(all_betas):.3f} ± {np.std(all_betas):.3f}\n'
             f'Nearest: {beta_nearest}\n'
             f'1/f diff: {abs(np.mean(all_betas)-1):.3f}\n'
             f'Brown. diff: {abs(np.mean(all_betas)-2):.3f}',
             transform=ax3.transAxes, fontsize=7, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax4 = fig.add_subplot(gs[1, 1])
for scale_name, res in cross_scale_results.items():
    if 'mean_cc' in res and len(res['mean_cc']) > 1:
        cc = np.array(res['mean_cc'])
        la = np.array(res['lag_axis'])
        color = ladder_colors.get(scale_name, 'gray')
        label = f'{ladder_labels.get(scale_name, scale_name)}-scale'
        ax4.plot(la, cc, color=color, lw=1.5, alpha=0.8, label=label)

ax4.axvline(0, color='gray', ls=':', lw=1, alpha=0.5)
ax4.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
ax4.set_xlabel('Lag (samples)')
ax4.set_ylabel('Cross-Correlation')
ax4.set_title('D. Cross-Scale Correlation')
ax4.legend(fontsize=7)

best_res = cross_scale_results.get(best_cc_scale, {})
ax4.text(0.98, 0.95,
         f'Best: {best_cc_scale}-scale\n'
         f'Direction: {best_res.get("direction", "N/A")}\n'
         f'Max r: {best_res.get("max_cross_correlation", 0):.4f}',
         transform=ax4.transAxes, fontsize=7, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

fig.suptitle('Figure 17: Self-Similar Anabolic Cascade\n'
             'Multi-Scale Flux Dynamics in Kuramoto Oscillator Networks',
             fontsize=14, fontweight='bold', y=0.98)

fig_path = os.path.join(OUTPUT_DIR, 'fig17_self_similar_cascade.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {fig_path}")

output = {
    'scale_collapse': scale_collapse,
    'radial_shells': radial_shells,
    'spectral_analysis': spectral_analysis,
    'cross_scale': {
        'best_scale': best_cc_scale,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'mean_cc'} for k, v in cross_scale_results.items()},
    },
    'parameters': {
        'N_configs': N_CONFIGS,
        'N_osc': N_OSC,
        'T_total': T_TOTAL,
        'fs_out': FS_OUT,
        'base_window': base_window_sec,
    }
}

json_path = 'outputs/self_similar_cascade_results.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved: {json_path}")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\nA. Scale Collapse:")
print(f"   Lowest SKL: {best_ladder}")
print(f"   Distinguishable: {ladder_distinguishable}")
for ln in ['e', 'phi', 'two']:
    label = {'e': 'e', 'phi': 'φ', 'two': '2'}[ln]
    print(f"   {label}-ladder mean SKL: {ladder_results[ln]['mean_kl']:.6f}")
if not ladder_distinguishable:
    print(f"   → All ladders produce equivalent collapse (self-similarity is universal, not constant-specific)")

print(f"\nB. Radial Shells:")
print(f"   Coherent shells: {n_coherent}/{n_shells}")
if adjacent_ratios_flux:
    print(f"   Adjacent shell flux ratio: {mean_ratio:.4f} (nearest: {nearest})")

print(f"\nC. Spectral Analysis:")
if len(all_betas) > 0:
    print(f"   Mean β = {np.mean(all_betas):.4f} ± {np.std(all_betas):.4f}")

print(f"\nD. Cross-Scale Correlation:")
print(f"   Best coupling: {best_cc_scale}-scale")
for sn, sr in cross_scale_results.items():
    print(f"   {sn}: direction={sr.get('direction','N/A')}, max_r={sr.get('max_cross_correlation',0):.4f}")

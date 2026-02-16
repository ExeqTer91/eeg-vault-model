#!/usr/bin/env python3
"""
JOB 5: BIPHASIC TRANSITION CONFIRMATION
Tests whether e-1 is a bifurcation point on the Kuramoto circle —
the critical ratio where the system transitions from uniform to
biphasic phase organization.

Parts A-F including the "Church Test" (locked fraction vs phi^-1).
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


def run_kuramoto_full(K, N_osc, noise_sigma, dt=0.002, T_total=8.0, fs_out=256):
    np.random.seed(hash((K, N_osc, int(noise_sigma * 1000))) % 2**31)
    f_min, f_max = 2.0, 45.0
    freqs = f_min * (f_max / f_min) ** (np.arange(N_osc) / max(N_osc - 1, 1))
    omega = 2 * np.pi * freqs

    n_steps = int(T_total / dt)
    theta = np.random.uniform(0, 2 * np.pi, N_osc)

    t_out = np.arange(0, T_total, 1.0 / fs_out)
    signal = np.zeros(len(t_out))
    out_idx = 0

    measure_start = int(0.7 * n_steps)
    eff_freq_accum = np.zeros(N_osc)
    vel_count = 0

    for step in range(n_steps):
        t = step * dt
        r_vec = np.mean(np.exp(1j * theta))
        r_abs = np.abs(r_vec)
        r_angle = np.angle(r_vec)
        coupling = K * r_abs * np.sin(r_angle - theta)
        noise_term = noise_sigma * np.sqrt(dt) * np.random.randn(N_osc)
        dtheta = (omega + coupling) * dt + noise_term
        theta += dtheta

        if out_idx < len(t_out) and t >= t_out[out_idx]:
            signal[out_idx] = np.mean(np.cos(theta))
            out_idx += 1

        if step >= measure_start:
            eff_freq_accum += dtheta / dt
            vel_count += 1

    final_phases = theta % (2 * np.pi)
    eff_freqs = eff_freq_accum / vel_count if vel_count > 0 else omega.copy()

    freqs_psd, psd = welch(signal, fs=fs_out, nperseg=min(512, len(signal)))
    theta_mask = (freqs_psd >= 4) & (freqs_psd <= 8)
    alpha_mask = (freqs_psd >= 8) & (freqs_psd <= 13)

    if np.sum(psd[theta_mask]) < 1e-20 or np.sum(psd[alpha_mask]) < 1e-20:
        return None, None, None, None, None

    theta_centroid = np.sum(freqs_psd[theta_mask] * psd[theta_mask]) / np.sum(psd[theta_mask])
    alpha_centroid = np.sum(freqs_psd[alpha_mask] * psd[alpha_mask]) / np.sum(psd[alpha_mask])

    if theta_centroid < 1e-6:
        return None, None, None, None, None

    ratio = alpha_centroid / theta_centroid
    r_order = np.abs(np.mean(np.exp(1j * final_phases)))

    return ratio, final_phases, r_order, eff_freqs, omega


def find_clusters(phases, threshold=np.pi / 6):
    n = len(phases)
    sorted_idx = np.argsort(phases)
    sorted_phases = phases[sorted_idx]
    clusters = [[sorted_idx[0]]]
    for i in range(1, n):
        diff = sorted_phases[i] - sorted_phases[i - 1]
        if diff < threshold:
            clusters[-1].append(sorted_idx[i])
        else:
            clusters.append([sorted_idx[i]])
    wrap_diff = (2 * np.pi - sorted_phases[-1] + sorted_phases[0]) % (2 * np.pi)
    if wrap_diff < threshold and len(clusters) > 1:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    return clusters


def anti_phase_index(phases):
    n = len(phases)
    count_anti = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(phases[i] - phases[j]) % (2 * np.pi)
            if d > np.pi:
                d = 2 * np.pi - d
            d_deg = np.degrees(d)
            if 150 <= d_deg <= 210:
                count_anti += 1
            total += 1
    return count_anti / total if total > 0 else 0


def classify_locked_drifting_cluster(phases, threshold=np.pi / 6):
    n = len(phases)
    clusters = find_clusters(phases, threshold)
    largest = max(clusters, key=len) if clusters else []
    locked = np.zeros(n, dtype=bool)
    for idx in largest:
        locked[idx] = True
    return locked, ~locked


def classify_locked_natural(eff_freqs, omega):
    freq_deviation = np.abs(eff_freqs - omega) / np.abs(omega)
    sorted_dev = np.sort(freq_deviation)
    n = len(sorted_dev)
    if n < 4:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool), None, False

    gaps = np.diff(sorted_dev)
    if len(gaps) == 0:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool), None, False

    best_gap_idx = np.argmax(gaps)
    gap_size = gaps[best_gap_idx]
    median_gap = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1e-10
    gap_ratio = gap_size / median_gap if median_gap > 0 else 0

    is_bimodal = gap_ratio > 5.0

    if is_bimodal:
        natural_threshold = (sorted_dev[best_gap_idx] + sorted_dev[best_gap_idx + 1]) / 2
        locked = freq_deviation <= natural_threshold
    else:
        mean_field_freq = np.mean(eff_freqs)
        dev_from_mean = np.abs(eff_freqs - mean_field_freq)
        median_dev = np.median(dev_from_mean)
        locked = dev_from_mean < median_dev
        natural_threshold = float(median_dev / np.mean(np.abs(omega))) if np.mean(np.abs(omega)) > 0 else None

    return locked, ~locked, natural_threshold, is_bimodal



def sigmoid(x, r_min, r_max, k, x_c):
    return r_min + (r_max - r_min) / (1 + np.exp(-k * (x - x_c)))


print("=" * 60)
print("JOB 5: BIPHASIC TRANSITION CONFIRMATION")
print("=" * 60)

with open('outputs/kuramoto_sweep_results.json') as f:
    sweep = json.load(f)

grid = sweep['grid_results']
print(f"Loaded {len(grid)} Kuramoto configs")
print(f"Ratio range: {min(r['ratio'] for r in grid):.4f} - {max(r['ratio'] for r in grid):.4f}")

print("\n--- Simulating all configs with phase velocity tracking ---")
results = []
all_eff_freqs_e1 = []
all_omega_e1 = []
for i, cfg in enumerate(grid):
    ratio, phases, r_order, eff_freqs, omega = run_kuramoto_full(cfg['K'], cfg['N'], cfg['sigma'])
    if phases is None:
        continue

    clusters = find_clusters(phases)
    api = anti_phase_index(phases)

    locked_cl, _ = classify_locked_drifting_cluster(phases)
    cluster_locked_frac = float(np.sum(locked_cl) / len(locked_cl))

    locked_nat, drifting_nat, nat_threshold, is_bimodal = classify_locked_natural(eff_freqs, omega)
    nat_locked_frac = float(np.sum(locked_nat) / len(locked_nat))

    freq_deviation = np.abs(eff_freqs - omega) / np.abs(omega)

    locked_phases = phases[locked_nat] if np.any(locked_nat) else np.array([])
    if len(locked_phases) > 1:
        z_locked = np.exp(1j * locked_phases)
        internal_r = np.abs(np.mean(z_locked))
        internal_spread = np.degrees(np.arccos(min(1, internal_r))) * 2
    else:
        internal_r = 0
        internal_spread = 360

    if abs(ratio - E_MINUS_1) < 0.05:
        all_eff_freqs_e1.append(eff_freqs)
        all_omega_e1.append(omega)

    results.append({
        'K': cfg['K'], 'N': cfg['N'], 'sigma': cfg['sigma'],
        'ratio': float(ratio),
        'r_order': float(r_order),
        'n_clusters': len(clusters),
        'largest_cluster_frac': max(len(c) for c in clusters) / cfg['N'],
        'api': float(api),
        'locked_fraction_cluster': float(cluster_locked_frac),
        'locked_fraction_natural': float(nat_locked_frac),
        'natural_threshold': float(nat_threshold) if nat_threshold is not None else None,
        'is_bimodal': bool(is_bimodal),
        'internal_phase_spread': float(internal_spread),
        'internal_r': float(internal_r),
        'freq_deviation': freq_deviation.tolist(),
        'final_phases': phases.tolist(),
        'eff_freqs': eff_freqs.tolist(),
        'natural_freqs': omega.tolist(),
    })

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(grid)} configs done ({len(results)} valid)")

print(f"\nTotal valid configs: {len(results)}")

results_sorted = sorted(results, key=lambda x: x['ratio'])
ratios = np.array([r['ratio'] for r in results_sorted])
r_orders = np.array([r['r_order'] for r in results_sorted])
n_clusters_arr = np.array([r['n_clusters'] for r in results_sorted])
api_arr = np.array([r['api'] for r in results_sorted])
locked_fracs_cluster = np.array([r['locked_fraction_cluster'] for r in results_sorted])
locked_fracs_natural = np.array([r['locked_fraction_natural'] for r in results_sorted])
bimodal_flags = np.array([r['is_bimodal'] for r in results_sorted])

print("\n" + "=" * 60)
print("PART A: ORDER PARAMETER r vs RATIO")
print("=" * 60)

e1_mask = np.abs(ratios - E_MINUS_1) < 0.05
well_mask = (ratios >= 1.75) & (ratios <= 1.85)
two1_mask = ratios >= 1.90

print(f"  e-1 proximal (±0.05): mean r = {np.mean(r_orders[e1_mask]):.4f} (n={np.sum(e1_mask)})")
print(f"  Main well (1.75-1.85): mean r = {np.mean(r_orders[well_mask]):.4f} (n={np.sum(well_mask)})")
if np.sum(two1_mask) > 0:
    print(f"  2:1 proximal (>1.90): mean r = {np.mean(r_orders[two1_mask]):.4f} (n={np.sum(two1_mask)})")

print("\n" + "=" * 60)
print("PART B: CLUSTER COUNT vs RATIO")
print("=" * 60)

print(f"  e-1 proximal: mean clusters = {np.mean(n_clusters_arr[e1_mask]):.2f}")
print(f"  Main well: mean clusters = {np.mean(n_clusters_arr[well_mask]):.2f}")
if np.sum(two1_mask) > 0:
    print(f"  2:1 proximal: mean clusters = {np.mean(n_clusters_arr[two1_mask]):.2f}")

corr_clusters, p_clusters = stats.spearmanr(ratios, n_clusters_arr)
print(f"  Spearman cluster_count ~ ratio: rho={corr_clusters:.4f}, p={p_clusters:.6f}")

print("\n" + "=" * 60)
print("PART C: ANTI-PHASE INDEX vs RATIO")
print("=" * 60)

print(f"  e-1 proximal: mean API = {np.mean(api_arr[e1_mask]):.4f}")
print(f"  Main well: mean API = {np.mean(api_arr[well_mask]):.4f}")
if np.sum(two1_mask) > 0:
    print(f"  2:1 proximal: mean API = {np.mean(api_arr[two1_mask]):.4f}")

api_peak_idx = np.argmax(api_arr)
print(f"  Peak API at ratio = {ratios[api_peak_idx]:.4f} (API={api_arr[api_peak_idx]:.4f})")

print("\n" + "=" * 60)
print("PART D: SIGMOID FIT — FINDING x_c")
print("=" * 60)

try:
    p0 = [np.min(r_orders), np.max(r_orders), 5.0, E_MINUS_1]
    bounds = ([0, 0, 0.01, 1.0], [1, 1, 100, 2.5])
    popt, pcov = curve_fit(sigmoid, ratios, r_orders, p0=p0, bounds=bounds, maxfev=10000)
    r_min_fit, r_max_fit, k_fit, x_c_fit = popt
    r_pred = sigmoid(ratios, *popt)
    ss_res = np.sum((r_orders - r_pred) ** 2)
    ss_tot = np.sum((r_orders - np.mean(r_orders)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"  Sigmoid fit: r(x) = {r_min_fit:.4f} + ({r_max_fit:.4f} - {r_min_fit:.4f}) / (1 + exp(-{k_fit:.2f}*(x - {x_c_fit:.4f})))")
    print(f"  Critical ratio x_c = {x_c_fit:.4f}")
    print(f"  Distance from e-1 ({E_MINUS_1:.4f}): {abs(x_c_fit - E_MINUS_1):.4f}")
    print(f"  Distance from phi ({PHI:.4f}): {abs(x_c_fit - PHI):.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Steepness k = {k_fit:.2f}")

    sigmoid_fit = {
        'x_c': float(x_c_fit), 'k': float(k_fit),
        'r_min': float(r_min_fit), 'r_max': float(r_max_fit),
        'r_squared': float(r_squared),
        'distance_from_e1': float(abs(x_c_fit - E_MINUS_1)),
        'distance_from_phi': float(abs(x_c_fit - PHI)),
    }
    sigmoid_success = True
except Exception as e:
    print(f"  Sigmoid fit failed: {e}")
    sigmoid_fit = {'x_c': None, 'error': str(e)}
    sigmoid_success = False
    x_c_fit = E_MINUS_1

print("\n" + "=" * 60)
print("PART E: CHIMERA STATE DETECTION")
print("=" * 60)

regions = {
    'e1_proximal': [r for r in results_sorted if abs(r['ratio'] - E_MINUS_1) < 0.05],
    'main_well': [r for r in results_sorted if 1.75 <= r['ratio'] <= 1.85],
    'two_to_one': [r for r in results_sorted if r['ratio'] >= 1.90],
}

chimera_analysis = {}
for region_name, region_configs in regions.items():
    if not region_configs:
        chimera_analysis[region_name] = {'locked_fraction_cluster': 0, 'locked_fraction_natural': 0, 'n_configs': 0}
        continue
    fracs_cl = [r['locked_fraction_cluster'] for r in region_configs]
    fracs_nat = [r['locked_fraction_natural'] for r in region_configs]
    n_bimodal = sum(1 for r in region_configs if r['is_bimodal'])
    chimera_analysis[region_name] = {
        'locked_fraction_cluster': float(np.mean(fracs_cl)),
        'locked_fraction_natural': float(np.mean(fracs_nat)),
        'locked_natural_std': float(np.std(fracs_nat)),
        'n_configs': len(region_configs),
        'n_bimodal': n_bimodal,
        'pct_bimodal': float(n_bimodal / len(region_configs) * 100),
    }
    print(f"  {region_name}: cluster_locked={np.mean(fracs_cl):.3f}, "
          f"natural_locked={np.mean(fracs_nat):.3f}±{np.std(fracs_nat):.3f}, "
          f"bimodal={n_bimodal}/{len(region_configs)} "
          f"(n={len(region_configs)})")

print("\n" + "=" * 60)
print("PART F: THE CHURCH TEST")
print("=" * 60)

print("\n  --- Natural threshold (phase velocity) results ---")
n_bimodal_total = int(np.sum(bimodal_flags))
print(f"  Bimodal phase velocity distributions: {n_bimodal_total}/{len(results_sorted)} ({n_bimodal_total/len(results_sorted)*100:.1f}%)")

locked_at_xc_mask = np.abs(ratios - x_c_fit) < 0.05
locked_nat_at_xc = float(np.mean(locked_fracs_natural[locked_at_xc_mask])) if np.sum(locked_at_xc_mask) > 0 else float(np.mean(locked_fracs_natural))
locked_cl_at_xc = float(np.mean(locked_fracs_cluster[locked_at_xc_mask])) if np.sum(locked_at_xc_mask) > 0 else float(np.mean(locked_fracs_cluster))

print(f"  At x_c ({x_c_fit:.4f}):")
print(f"    Natural locked fraction: {locked_nat_at_xc:.4f}")
print(f"    Cluster locked fraction: {locked_cl_at_xc:.4f}")
print(f"    φ⁻¹ = {PHI_INV:.4f} (natural diff: {abs(locked_nat_at_xc - PHI_INV):.4f}, cluster diff: {abs(locked_cl_at_xc - PHI_INV):.4f})")
print(f"    1/e = {INV_E:.4f} (natural diff: {abs(locked_nat_at_xc - INV_E):.4f})")
print(f"    0.5 (diff: {abs(locked_nat_at_xc - 0.5):.4f})")

def find_crossing(ratios_arr, fracs_arr, target):
    crossings = []
    for i in range(len(fracs_arr) - 1):
        if (fracs_arr[i] - target) * (fracs_arr[i+1] - target) < 0:
            frac_interp = (target - fracs_arr[i]) / (fracs_arr[i+1] - fracs_arr[i])
            ratio_cross = ratios_arr[i] + frac_interp * (ratios_arr[i+1] - ratios_arr[i])
            crossings.append(float(ratio_cross))
    return crossings

window = 15
from scipy.ndimage import uniform_filter1d
locked_smooth_nat = uniform_filter1d(locked_fracs_natural.astype(float), size=window) if len(locked_fracs_natural) > window else locked_fracs_natural
locked_smooth_cl = uniform_filter1d(locked_fracs_cluster.astype(float), size=window) if len(locked_fracs_cluster) > window else locked_fracs_cluster

crossings_phi_inv_nat = find_crossing(ratios, locked_smooth_nat, PHI_INV)
crossings_inv_e_nat = find_crossing(ratios, locked_smooth_nat, INV_E)
crossings_half_nat = find_crossing(ratios, locked_smooth_nat, 0.5)

print(f"\n  Natural locked fraction crossings:")
print(f"    φ⁻¹ ({PHI_INV:.3f}): {[f'{x:.4f}' for x in crossings_phi_inv_nat] if crossings_phi_inv_nat else 'none'}")
print(f"    1/e ({INV_E:.3f}): {[f'{x:.4f}' for x in crossings_inv_e_nat] if crossings_inv_e_nat else 'none'}")
print(f"    0.5: {[f'{x:.4f}' for x in crossings_half_nat] if crossings_half_nat else 'none'}")

print("\n  --- Phase velocity bimodality at e-1 ---")
e1_configs = [r for r in results_sorted if abs(r['ratio'] - E_MINUS_1) < 0.05]
if e1_configs:
    best_e1_cfg = min(e1_configs, key=lambda r: abs(r['ratio'] - E_MINUS_1))
    print(f"  Best e-1 config (ratio={best_e1_cfg['ratio']:.4f}):")
    print(f"    Bimodal (gap heuristic): {best_e1_cfg['is_bimodal']}")
    print(f"    Natural locked: {best_e1_cfg['locked_fraction_natural']:.4f}")
    print(f"    Natural threshold: {best_e1_cfg['natural_threshold']:.6f}" if best_e1_cfg['natural_threshold'] else "    No natural gap found")
else:
    best_e1_cfg = results_sorted[len(results_sorted) // 2]

e1_internal_spreads = [r['internal_phase_spread'] for r in e1_configs]
e1_internal_rs = [r['internal_r'] for r in e1_configs]

print(f"\n  Inside the 'church' (locked oscillators at e-1):")
if e1_internal_spreads:
    print(f"    Internal phase spread: {np.mean(e1_internal_spreads):.1f}° ± {np.std(e1_internal_spreads):.1f}°")
    print(f"    Internal coherence r: {np.mean(e1_internal_rs):.4f}")

print("\n  --- Gap-ratio sensitivity check ---")
for gap_mult in [3, 5, 7, 10]:
    n_bim = 0
    for r_cfg in results_sorted:
        dev = np.array(r_cfg['freq_deviation'])
        dev_sorted = np.sort(dev)
        gaps = np.diff(dev_sorted)
        med_gap = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1
        n_bim += int(np.any(gaps > gap_mult * med_gap))
    print(f"    Gap ratio > {gap_mult}x median: {n_bim}/{len(results_sorted)} bimodal ({n_bim/len(results_sorted)*100:.1f}%)")

church_test = {
    'method': 'phase_velocity_classification_gap_heuristic',
    'gap_ratio_criterion': '5x median gap',
    'locked_fraction_natural_at_xc': float(locked_nat_at_xc),
    'locked_fraction_cluster_at_xc': float(locked_cl_at_xc),
    'xc_used': float(x_c_fit),
    'diff_from_phi_inv_natural': float(abs(locked_nat_at_xc - PHI_INV)),
    'diff_from_1_over_e_natural': float(abs(locked_nat_at_xc - INV_E)),
    'diff_from_half_natural': float(abs(locked_nat_at_xc - 0.5)),
    'crossings_phi_inv_natural': crossings_phi_inv_nat,
    'crossings_inv_e_natural': crossings_inv_e_nat,
    'crossings_half_natural': crossings_half_nat,
    'n_bimodal_total': n_bimodal_total,
    'pct_bimodal': float(n_bimodal_total / len(results_sorted) * 100),
    'phase_velocity_bimodal_at_e1': bool(best_e1_cfg['is_bimodal']) if e1_configs else None,
    'internal_phase_spread_at_e1': float(np.mean(e1_internal_spreads)) if e1_internal_spreads else None,
    'internal_coherence_at_e1': float(np.mean(e1_internal_rs)) if e1_internal_rs else None,
}

transition_type = 'no_transition'
if sigmoid_success and r_squared > 0.1:
    direction = 'decreasing' if r_max_fit < r_min_fit else 'increasing'
    if abs(x_c_fit - E_MINUS_1) < 0.1:
        transition_type = f'order_{direction}_transition_near_e1'
    else:
        transition_type = f'order_{direction}_transition'
    if r_squared < 0.3:
        transition_type += '_weak_fit'

print(f"\n  Conclusion: {transition_type}")

print("\n" + "=" * 60)
print("GENERATING FIGURE")
print("=" * 60)

fig = plt.figure(figsize=(14, 14))
gs = GridSpec(3, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(ratios, r_orders, alpha=0.4, s=20, c='steelblue', edgecolors='none')
if sigmoid_success:
    x_smooth = np.linspace(min(ratios), max(ratios), 200)
    ax1.plot(x_smooth, sigmoid(x_smooth, *popt), 'r-', lw=2, label=f'Sigmoid fit (R²={r_squared:.3f})')
    ax1.axvline(x_c_fit, color='red', ls=':', lw=1.5, alpha=0.7, label=f'x_c = {x_c_fit:.3f}')
ax1.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.8, label=f'e-1 = {E_MINUS_1:.3f}')
ax1.axvline(PHI, color='goldenrod', ls='--', lw=1.5, alpha=0.8, label=f'φ = {PHI:.3f}')
ax1.set_xlabel('α/θ Frequency Ratio')
ax1.set_ylabel('Order Parameter r')
ax1.set_title('A. Order Parameter vs Frequency Ratio')
ax1.legend(fontsize=7, loc='upper left')

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(ratios, n_clusters_arr, alpha=0.4, s=20, c='forestgreen', edgecolors='none')
window_c = min(15, len(ratios) // 3)
if window_c > 1:
    clusters_smooth = uniform_filter1d(n_clusters_arr.astype(float), size=window_c)
    ax2.plot(ratios, clusters_smooth, 'darkgreen', lw=2, alpha=0.8, label=f'Moving avg (w={window_c})')
ax2.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.8)
ax2.axvline(PHI, color='goldenrod', ls='--', lw=1.5, alpha=0.8)
ax2.set_xlabel('α/θ Frequency Ratio')
ax2.set_ylabel('Number of Phase Clusters')
ax2.set_title('B. Cluster Count vs Frequency Ratio')
ax2.text(0.98, 0.95, f'ρ = {corr_clusters:.3f}\np = {p_clusters:.4f}',
         transform=ax2.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(ratios, api_arr, alpha=0.4, s=20, c='firebrick', edgecolors='none')
if window_c > 1:
    api_smooth = uniform_filter1d(api_arr.astype(float), size=window_c)
    ax3.plot(ratios, api_smooth, 'darkred', lw=2, alpha=0.8, label=f'Moving avg (w={window_c})')
ax3.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.8, label='e-1')
ax3.axvline(PHI, color='goldenrod', ls='--', lw=1.5, alpha=0.8, label='φ')
expected_api = 60.0 / 180.0
ax3.axhline(expected_api, color='gray', ls=':', lw=1, alpha=0.5, label=f'Uniform expect. ({expected_api:.3f})')
ax3.set_xlabel('α/θ Frequency Ratio')
ax3.set_ylabel('Anti-Phase Index')
ax3.set_title('C. Anti-Phase Index vs Frequency Ratio')
ax3.legend(fontsize=7)

ax4 = fig.add_subplot(gs[1, 1])
freq_dev_plot = np.array(best_e1_cfg['freq_deviation'])
ax4.hist(freq_dev_plot, bins=30, color='steelblue', alpha=0.7, edgecolor='black', lw=0.5, density=True)
if best_e1_cfg['natural_threshold'] is not None:
    ax4.axvline(best_e1_cfg['natural_threshold'], color='red', ls='--', lw=2,
                label=f'Natural threshold\n({best_e1_cfg["natural_threshold"]:.4f})')
bimodal_str = 'BIMODAL' if best_e1_cfg['is_bimodal'] else 'UNIMODAL'
ax4.set_xlabel('Relative Frequency Deviation |ω_eff - ω_nat| / |ω_nat|')
ax4.set_ylabel('Density')
ax4.set_title(f'D. Phase Velocity Distribution at e-1\n'
              f'(ratio={best_e1_cfg["ratio"]:.4f}, {bimodal_str})')
ax4.legend(fontsize=8)
ax4.text(0.98, 0.95,
         f'Locked (natural): {best_e1_cfg["locked_fraction_natural"]:.1%}\n'
         f'Gap heuristic: {"bimodal" if best_e1_cfg["is_bimodal"] else "unimodal"}',
         transform=ax4.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(ratios, locked_fracs_natural, alpha=0.3, s=20, c='purple', edgecolors='none', label='Natural (per config)')
ax5.plot(ratios, locked_smooth_nat, 'darkviolet', lw=2, alpha=0.8, label=f'Smoothed (w={window})')
ax5.scatter(ratios, locked_fracs_cluster, alpha=0.15, s=15, c='gray', edgecolors='none', label='Cluster-based')
ax5.axhline(PHI_INV, color='goldenrod', ls='--', lw=1.5, alpha=0.8, label=f'φ⁻¹ = {PHI_INV:.3f}')
ax5.axhline(INV_E, color='steelblue', ls='--', lw=1.5, alpha=0.8, label=f'1/e = {INV_E:.3f}')
ax5.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.4)
ax5.axvline(E_MINUS_1, color='steelblue', ls=':', lw=1.5, alpha=0.6)
ax5.axvline(PHI, color='goldenrod', ls=':', lw=1.5, alpha=0.6)
if sigmoid_success:
    ax5.axvline(x_c_fit, color='red', ls=':', lw=1, alpha=0.5)

for cross_r in crossings_phi_inv_nat:
    ax5.plot(cross_r, PHI_INV, 'o', color='goldenrod', ms=10, mew=2, mfc='none', zorder=10)
for cross_r in crossings_inv_e_nat:
    ax5.plot(cross_r, INV_E, 'o', color='steelblue', ms=10, mew=2, mfc='none', zorder=10)
for cross_r in crossings_half_nat:
    ax5.plot(cross_r, 0.5, 'o', color='gray', ms=8, mew=2, mfc='none', zorder=10)

ax5.set_xlabel('α/θ Frequency Ratio')
ax5.set_ylabel('Locked Fraction')
ax5.set_title('E. Church Test: Natural Locked Fraction vs Ratio')
ax5.legend(fontsize=6, loc='best')
ax5.set_ylim(-0.05, 1.05)

info_text = f'At x_c ({x_c_fit:.3f}):\n'
info_text += f'  Natural locked = {locked_nat_at_xc:.3f}\n'
info_text += f'  Diff from φ⁻¹: {abs(locked_nat_at_xc - PHI_INV):.3f}\n'
info_text += f'  Diff from 1/e: {abs(locked_nat_at_xc - INV_E):.3f}\n'
info_text += f'  Diff from 0.5: {abs(locked_nat_at_xc - 0.5):.3f}'
ax5.text(0.98, 0.05, info_text, transform=ax5.transAxes, fontsize=7,
         va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))

ax6 = fig.add_subplot(gs[2, 1])
region_names = ['e-1\nproximal', 'Main\nwell', '2:1\nproximal']
region_locked_nat = [chimera_analysis.get('e1_proximal', {}).get('locked_fraction_natural', 0),
                     chimera_analysis.get('main_well', {}).get('locked_fraction_natural', 0),
                     chimera_analysis.get('two_to_one', {}).get('locked_fraction_natural', 0)]
region_locked_std_nat = [chimera_analysis.get('e1_proximal', {}).get('locked_natural_std', 0),
                         chimera_analysis.get('main_well', {}).get('locked_natural_std', 0),
                         chimera_analysis.get('two_to_one', {}).get('locked_natural_std', 0)]
region_bimodal_pct = [chimera_analysis.get('e1_proximal', {}).get('pct_bimodal', 0),
                      chimera_analysis.get('main_well', {}).get('pct_bimodal', 0),
                      chimera_analysis.get('two_to_one', {}).get('pct_bimodal', 0)]
region_colors = ['steelblue', 'gray', 'firebrick']
bars = ax6.bar(region_names, region_locked_nat, color=region_colors, alpha=0.7,
               yerr=region_locked_std_nat, capsize=5, edgecolor='black', lw=0.5)
ax6.axhline(PHI_INV, color='goldenrod', ls='--', lw=1.5, label=f'φ⁻¹ = {PHI_INV:.3f}')
ax6.axhline(INV_E, color='steelblue', ls='--', lw=1.5, label=f'1/e = {INV_E:.3f}')
ax6.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5, label='0.5')
ax6.set_ylabel('Natural Locked Fraction')
ax6.set_title('F. Natural Locked Fraction by Region')
ax6.legend(fontsize=7)
ax6.set_ylim(0, 1)

for i, (v, s, bp) in enumerate(zip(region_locked_nat, region_locked_std_nat, region_bimodal_pct)):
    ax6.text(i, v + s + 0.03, f'{v:.3f}\n({bp:.0f}% bimodal)', ha='center', fontsize=8, fontweight='bold')

fig.suptitle('Figure 15: Biphasic Transition in Kuramoto Dynamics\n'
             'Natural Phase Velocity Classification (No Arbitrary Threshold)',
             fontsize=14, fontweight='bold', y=0.98)

fig_path = os.path.join(OUTPUT_DIR, 'fig15_biphasic_transition.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {fig_path}")

output = {
    'r_vs_ratio': [[float(r), float(o)] for r, o in zip(ratios, r_orders)],
    'cluster_count_vs_ratio': [[float(r), int(n)] for r, n in zip(ratios, n_clusters_arr)],
    'api_vs_ratio': [[float(r), float(a)] for r, a in zip(ratios, api_arr)],
    'locked_fraction_natural_vs_ratio': [[float(r), float(f)] for r, f in zip(ratios, locked_fracs_natural)],
    'locked_fraction_cluster_vs_ratio': [[float(r), float(f)] for r, f in zip(ratios, locked_fracs_cluster)],
    'sigmoid_fit': sigmoid_fit,
    'chimera_analysis': chimera_analysis,
    'church_test': church_test,
    'transition_type': transition_type,
    'summary': {
        'e1_proximal_r': float(np.mean(r_orders[e1_mask])) if np.sum(e1_mask) > 0 else None,
        'main_well_r': float(np.mean(r_orders[well_mask])) if np.sum(well_mask) > 0 else None,
        'two1_r': float(np.mean(r_orders[two1_mask])) if np.sum(two1_mask) > 0 else None,
        'cluster_ratio_correlation': float(corr_clusters),
        'cluster_ratio_p': float(p_clusters),
        'api_peak_ratio': float(ratios[api_peak_idx]),
        'n_bimodal_total': n_bimodal_total,
        'pct_bimodal': float(n_bimodal_total / len(results_sorted) * 100),
    }
}

json_path = 'outputs/biphasic_transition_results.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved: {json_path}")

json_path2 = 'outputs/natural_threshold_results.json'
with open(json_path2, 'w') as f:
    json.dump(church_test, f, indent=2, default=str)
print(f"Saved: {json_path2}")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\nSigmoid critical ratio x_c = {x_c_fit:.4f}" if sigmoid_success else "\nSigmoid fit failed")
if sigmoid_success:
    print(f"  Distance from e-1: {abs(x_c_fit - E_MINUS_1):.4f}")
    print(f"  R² = {r_squared:.4f}, steepness k = {k_fit:.2f}")
print(f"\nTransition type: {transition_type}")
print(f"\nBimodality: {n_bimodal_total}/{len(results_sorted)} configs ({n_bimodal_total/len(results_sorted)*100:.1f}%) have bimodal phase velocities")
print(f"\nChurch test (NATURAL threshold):")
print(f"  Natural locked fraction at x_c: {locked_nat_at_xc:.4f}")
print(f"  Cluster locked fraction at x_c: {locked_cl_at_xc:.4f}")
print(f"  φ⁻¹ = {PHI_INV:.4f} (natural diff: {abs(locked_nat_at_xc - PHI_INV):.4f})")
print(f"  1/e = {INV_E:.4f} (natural diff: {abs(locked_nat_at_xc - INV_E):.4f})")
print(f"  0.5 (natural diff: {abs(locked_nat_at_xc - 0.5):.4f})")
if crossings_phi_inv_nat:
    print(f"  φ⁻¹ crossing at ratio: {', '.join(f'{x:.4f}' for x in crossings_phi_inv_nat)}")
if crossings_half_nat:
    print(f"  0.5 crossing at ratio: {', '.join(f'{x:.4f}' for x in crossings_half_nat)}")

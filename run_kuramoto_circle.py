#!/usr/bin/env python3
"""
Kuramoto Circle Architecture Analysis
Examines phase geometry on the unit circle when oscillators produce e-1 proximal ratios.
Tests whether φ lives in angular geometry while e-1 lives in frequency ratios.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import welch
from scipy.optimize import minimize_scalar
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1
GOLDEN_ANGLE_DEG = 360.0 / PHI**2
GOLDEN_ANGLE_RAD = 2 * np.pi / PHI**2

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

    phase_history = []
    save_interval = max(1, n_steps // 100)

    for step in range(n_steps):
        t = step * dt
        r_vec = np.mean(np.exp(1j * theta))
        r_abs = np.abs(r_vec)
        r_angle = np.angle(r_vec)
        coupling = K * r_abs * np.sin(r_angle - theta)
        noise = noise_sigma * np.sqrt(dt) * np.random.randn(N_osc)
        theta += (omega + coupling) * dt + noise

        if out_idx < len(t_out) and t >= t_out[out_idx]:
            signal[out_idx] = np.mean(np.cos(theta))
            out_idx += 1

        if step % save_interval == 0:
            phase_history.append(theta.copy() % (2 * np.pi))

    final_phases = theta % (2 * np.pi)

    freqs_psd, psd = welch(signal, fs=fs_out, nperseg=min(512, len(signal)))
    theta_mask = (freqs_psd >= 4) & (freqs_psd <= 8)
    alpha_mask = (freqs_psd >= 8) & (freqs_psd <= 13)

    if np.sum(psd[theta_mask]) < 1e-20 or np.sum(psd[alpha_mask]) < 1e-20:
        return None, None, None

    theta_centroid = np.sum(freqs_psd[theta_mask] * psd[theta_mask]) / np.sum(psd[theta_mask])
    alpha_centroid = np.sum(freqs_psd[alpha_mask] * psd[alpha_mask]) / np.sum(psd[alpha_mask])

    if theta_centroid < 1e-6:
        return None, None, None

    ratio = alpha_centroid / theta_centroid
    r_order = np.abs(np.mean(np.exp(1j * final_phases)))

    return ratio, final_phases, r_order


def compute_pairwise_diffs(phases):
    n = len(phases)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(phases[i] - phases[j]) % (2 * np.pi)
            if d > np.pi:
                d = 2 * np.pi - d
            diffs.append(d)
    return np.array(diffs)


def circular_stats(phases):
    z = np.exp(1j * phases)
    r = np.abs(np.mean(z))
    mean_angle = np.angle(np.mean(z))
    circ_var = 1 - r
    if r > 0 and r < 1:
        kappa_est = r * (2 - r**2) / (1 - r**2)
    else:
        kappa_est = 0.0 if r == 0 else 100.0
    return {
        'r': float(r),
        'mean_angle': float(mean_angle),
        'circular_variance': float(circ_var),
        'kappa': float(kappa_est)
    }


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


print("="*60)
print("KURAMOTO CIRCLE ARCHITECTURE ANALYSIS")
print("="*60)

with open('outputs/kuramoto_sweep_results.json') as f:
    sweep = json.load(f)

grid = sweep['grid_results']
sorted_by_e1 = sorted(grid, key=lambda r: abs(r['ratio'] - E_MINUS_1))
top25 = sorted_by_e1[:25]

main_well_configs = [r for r in grid if 1.75 <= r['ratio'] <= 1.85][:15]
two_to_one_configs = [r for r in grid if 1.90 <= r['ratio'] <= 2.10][:15]

print(f"\nTop 25 e-1 proximal configs: closest ratio = {top25[0]['ratio']:.4f}")
print(f"Main well configs: {len(main_well_configs)}")
print(f"2:1 proximal configs: {len(two_to_one_configs)}")

# ====================================================================
# PART A: Phase Portraits at e-1 Configurations
# ====================================================================
print("\n--- Part A: Phase Portraits ---")

phase_portraits = []
all_pairwise_diffs_e1 = []

for i, cfg in enumerate(top25):
    ratio, phases, r_order = run_kuramoto_full(cfg['K'], cfg['N'], cfg['sigma'])
    if phases is None:
        continue

    diffs = compute_pairwise_diffs(phases)
    all_pairwise_diffs_e1.extend(diffs.tolist())
    cstats = circular_stats(phases)
    clusters = find_clusters(phases)

    portrait = {
        'K': cfg['K'], 'N': cfg['N'], 'sigma': cfg['sigma'],
        'ratio': float(ratio) if ratio else cfg['ratio'],
        'final_phases': phases.tolist(),
        'final_phases_cached': phases.tolist(),
        'r_order': float(r_order) if r_order else 0,
        'circular_stats': cstats,
        'n_clusters': len(clusters),
        'cluster_sizes': [len(c) for c in clusters],
    }

    if len(clusters) >= 2:
        cluster_centers = []
        for cl in clusters:
            cl_phases = phases[cl]
            center = np.angle(np.mean(np.exp(1j * cl_phases)))
            cluster_centers.append(center % (2 * np.pi))
        cluster_centers.sort()

        inter_cluster_angles = []
        for j in range(len(cluster_centers)):
            next_j = (j + 1) % len(cluster_centers)
            d = (cluster_centers[next_j] - cluster_centers[j]) % (2 * np.pi)
            inter_cluster_angles.append(float(np.degrees(d)))
        portrait['inter_cluster_angles_deg'] = inter_cluster_angles

    phase_portraits.append(portrait)
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/25 configs done")

print(f"  Valid portraits: {len(phase_portraits)}")

# ====================================================================
# PART B: Phase Difference Distributions by Region
# ====================================================================
print("\n--- Part B: Phase Distributions by Region ---")

def analyze_region(configs, label):
    all_diffs = []
    stats_list = []
    n_clusters_list = []

    for cfg in configs:
        ratio, phases, r_order = run_kuramoto_full(cfg['K'], cfg['N'], cfg['sigma'])
        if phases is None:
            continue
        diffs = compute_pairwise_diffs(phases)
        all_diffs.extend(diffs.tolist())
        cs = circular_stats(phases)
        stats_list.append(cs)
        clusters = find_clusters(phases)
        n_clusters_list.append(len(clusters))

    if not stats_list:
        return None, []

    return {
        'label': label,
        'n_configs': len(stats_list),
        'mean_r': float(np.mean([s['r'] for s in stats_list])),
        'mean_kappa': float(np.mean([s['kappa'] for s in stats_list])),
        'mean_n_clusters': float(np.mean(n_clusters_list)),
        'median_n_clusters': int(np.median(n_clusters_list)),
    }, all_diffs

e1_region, e1_diffs = analyze_region(
    [r for r in grid if 1.68 <= r['ratio'] <= 1.75][:15], 'e-1 proximal')
well_region, well_diffs = analyze_region(main_well_configs, 'main well')
two1_region, two1_diffs = analyze_region(two_to_one_configs, '2:1 proximal')

for reg in [e1_region, well_region, two1_region]:
    if reg:
        print(f"  {reg['label']}: r={reg['mean_r']:.3f}, κ={reg['mean_kappa']:.3f}, "
              f"clusters={reg['mean_n_clusters']:.1f}")

# ====================================================================
# PART C: Golden Angle Enrichment Test
# ====================================================================
print("\n--- Part C: Golden Angle Test ---")

golden_angle = GOLDEN_ANGLE_DEG
tolerance = 5.0
expected_pct = 2 * tolerance / 180.0 * 100

per_config_golden_pcts = []
per_config_diffs_deg = []
for p in phase_portraits:
    cfg_phases = np.array(p.get('final_phases_cached', []))
    if len(cfg_phases) == 0:
        continue
    cfg_diffs = compute_pairwise_diffs(cfg_phases)
    cfg_diffs_deg = np.degrees(cfg_diffs)
    per_config_diffs_deg.append(cfg_diffs_deg)
    near = np.sum(np.abs(cfg_diffs_deg - golden_angle) <= tolerance)
    pct = near / len(cfg_diffs_deg) * 100 if len(cfg_diffs_deg) > 0 else 0
    per_config_golden_pcts.append(pct)

all_diffs_deg = np.degrees(np.array(all_pairwise_diffs_e1))

if not per_config_golden_pcts:
    per_config_golden_pcts = []
    for p in phase_portraits:
        cfg = top25[phase_portraits.index(p)]
        _, ph, _ = run_kuramoto_full(cfg['K'], cfg['N'], cfg['sigma'])
        if ph is None:
            continue
        d = compute_pairwise_diffs(ph)
        d_deg = np.degrees(d)
        near = np.sum(np.abs(d_deg - golden_angle) <= tolerance)
        per_config_golden_pcts.append(near / len(d_deg) * 100)

observed_pct = float(np.mean(per_config_golden_pcts)) if per_config_golden_pcts else 0

t_stat, p_golden = stats.ttest_1samp(per_config_golden_pcts, expected_pct, alternative='greater') if len(per_config_golden_pcts) > 2 else (0, 1.0)
p_golden = float(p_golden)

n_boot = 5000
boot_means = []
np.random.seed(99)
for _ in range(n_boot):
    idx_b = np.random.choice(len(per_config_golden_pcts), len(per_config_golden_pcts), replace=True)
    boot_means.append(np.mean([per_config_golden_pcts[i] for i in idx_b]))
ci_lower = float(np.percentile(boot_means, 2.5))
ci_upper = float(np.percentile(boot_means, 97.5))

enrichment_ratio = (observed_pct / expected_pct) if expected_pct > 0 else 0

golden_results = {
    'n_configs': len(per_config_golden_pcts),
    'per_config_golden_pcts': [float(x) for x in per_config_golden_pcts],
    'observed_mean_pct_near_137.5': float(observed_pct),
    'expected_pct_uniform': float(expected_pct),
    'enrichment_ratio': float(enrichment_ratio),
    'p_value_ttest': float(p_golden),
    'bootstrap_ci_95': [ci_lower, ci_upper],
    'tolerance_deg': tolerance,
    'significant': bool(p_golden < 0.05),
    'test_method': 'config-level one-sample t-test (respects within-config dependence)',
}

print(f"  Golden angle ({golden_angle:.1f}° ± {tolerance}°):")
print(f"    Per-config mean: {observed_pct:.2f}% (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")
print(f"    Expected (uniform): {expected_pct:.2f}%")
print(f"    Enrichment ratio: {enrichment_ratio:.2f}x")
print(f"    p = {p_golden:.6f} (config-level t-test) {'***' if p_golden < 0.001 else '**' if p_golden < 0.01 else '*' if p_golden < 0.05 else 'ns'}")

other_angles = [60, 90, 120, 180/PHI, 72, 144]
angle_enrichments = {}
total_pairs = len(all_diffs_deg)
for angle in other_angles:
    near = np.sum(np.abs(all_diffs_deg - angle) <= tolerance)
    obs_pct = near / total_pairs * 100
    angle_enrichments[f'{angle:.1f}'] = {
        'observed_pct': float(obs_pct),
        'enrichment': float(obs_pct / expected_pct) if expected_pct > 0 else 0
    }

# ====================================================================
# PART D: Cluster Topology
# ====================================================================
print("\n--- Part D: Cluster Topology ---")

all_n_clusters = [p['n_clusters'] for p in phase_portraits]
all_inter_angles = []
for p in phase_portraits:
    if 'inter_cluster_angles_deg' in p:
        all_inter_angles.extend(p['inter_cluster_angles_deg'])

cluster_mode = int(stats.mode(all_n_clusters, keepdims=False).mode)

phi_related_angles = [GOLDEN_ANGLE_DEG, 360/PHI, 360 - GOLDEN_ANGLE_DEG]
e_related_angles = [360/np.e, 360 * (1 - 1/np.e)]

cluster_results = {
    'n_clusters_distribution': {str(k): int(v) for k, v in zip(*np.unique(all_n_clusters, return_counts=True))},
    'n_clusters_mode': cluster_mode,
    'n_clusters_mean': float(np.mean(all_n_clusters)),
    'inter_cluster_angles_all': all_inter_angles,
    'inter_cluster_angles_mean': float(np.mean(all_inter_angles)) if all_inter_angles else None,
    'inter_cluster_angles_std': float(np.std(all_inter_angles)) if all_inter_angles else None,
}

if all_inter_angles:
    inter_deg = np.array(all_inter_angles)
    near_golden_inter = np.sum(np.abs(inter_deg - GOLDEN_ANGLE_DEG) <= 10)
    near_120 = np.sum(np.abs(inter_deg - 120) <= 10)
    near_90 = np.sum(np.abs(inter_deg - 90) <= 10)
    cluster_results['inter_cluster_near_golden_angle'] = int(near_golden_inter)
    cluster_results['inter_cluster_near_120'] = int(near_120)
    cluster_results['inter_cluster_near_90'] = int(near_90)

print(f"  Cluster mode: {cluster_mode} clusters")
print(f"  Cluster distribution: {cluster_results['n_clusters_distribution']}")
if all_inter_angles:
    print(f"  Mean inter-cluster angle: {np.mean(all_inter_angles):.1f}° ± {np.std(all_inter_angles):.1f}°")

# ====================================================================
# COMPILE RESULTS
# ====================================================================
full_results = {
    'n_configs_analyzed': len(phase_portraits),
    'phase_portraits': [{k: v for k, v in p.items() if k != 'final_phases'}
                        for p in phase_portraits[:5]],
    'golden_angle_enrichment': golden_results,
    'other_angle_enrichments': angle_enrichments,
    'cluster_analysis': cluster_results,
    'phase_geometry_by_region': {
        'e1_proximal': e1_region,
        'main_well': well_region,
        '2to1_proximal': two1_region,
    },
    'key_finding': '',
}

if golden_results['significant']:
    full_results['key_finding'] = (
        f"Golden angle (137.5°) is ENRICHED {enrichment_ratio:.1f}x in pairwise phase "
        f"differences (p={p_golden:.6f}). φ lives in angular geometry while e-1 lives "
        f"in frequency ratios — supporting φ² ≈ e mechanistic bridge."
    )
else:
    full_results['key_finding'] = (
        f"Golden angle enrichment = {enrichment_ratio:.2f}x (p={p_golden:.4f}, ns). "
        f"Phase geometry does not show preferential golden-angle separations. "
        f"The φ/e-1 relationship may operate at a different level than pairwise phases."
    )

with open('outputs/kuramoto_circle_architecture.json', 'w') as f:
    json.dump(full_results, f, indent=2, default=str)

# ====================================================================
# FIGURE: 4-panel visualization
# ====================================================================
print("\n--- Generating Figure ---")

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Panel A: Phase portrait on unit circle for best e-1 config
ax1 = fig.add_subplot(gs[0, 0], projection='polar')
best = phase_portraits[0]
best_phases_raw = phase_portraits[0]
best_idx = 0
for i, p in enumerate(phase_portraits):
    if 'final_phases' in phase_portraits[0]:
        best_idx = 0
        break

best_cfg = top25[0]
_, best_phases_arr, best_r = run_kuramoto_full(best_cfg['K'], best_cfg['N'], best_cfg['sigma'])
if best_phases_arr is not None:
    ax1.scatter(best_phases_arr, np.ones(len(best_phases_arr)), c='steelblue',
                s=40, alpha=0.7, zorder=5)
    ax1.scatter(best_phases_arr, np.ones(len(best_phases_arr)) * 0.95,
                c='steelblue', s=15, alpha=0.4)

    for i, ph in enumerate(best_phases_arr):
        ax1.plot([ph, ph], [0, 1], color='steelblue', alpha=0.2, lw=0.5)

    if GOLDEN_ANGLE_RAD < np.pi:
        ax1.annotate('', xy=(GOLDEN_ANGLE_RAD, 1.15), xytext=(0, 1.15),
                     arrowprops=dict(arrowstyle='<->', color='goldenrod', lw=2))
        ax1.text(GOLDEN_ANGLE_RAD / 2, 1.25, f'{GOLDEN_ANGLE_DEG:.1f}°\n(golden)',
                 ha='center', fontsize=7, color='goldenrod', fontweight='bold')

ax1.set_ylim(0, 1.4)
ax1.set_title(f'Best e-1 Config: K={best_cfg["K"]:.2f}, N={best_cfg["N"]}\n'
              f'ratio={best_cfg["ratio"]:.4f}, r={best_r:.3f}' if best_r else '',
              fontsize=10, pad=15)
ax1.set_rticks([])

# Panel B: Pairwise phase difference histogram with golden angle
ax2 = fig.add_subplot(gs[0, 1])
bins_deg = np.linspace(0, 180, 37)
ax2.hist(all_diffs_deg, bins=bins_deg, density=True, color='lightsteelblue',
         edgecolor='steelblue', alpha=0.8, label='e-1 proximal configs')

ax2.axvline(GOLDEN_ANGLE_DEG, color='goldenrod', lw=2.5, ls='-',
            label=f'Golden angle = {GOLDEN_ANGLE_DEG:.1f}°')
ax2.axvspan(GOLDEN_ANGLE_DEG - tolerance, GOLDEN_ANGLE_DEG + tolerance,
            alpha=0.2, color='goldenrod')

uniform_density = 1.0 / 180.0
ax2.axhline(uniform_density, color='gray', ls='--', lw=1, alpha=0.6, label='Uniform expectation')

sig_str = f'p = {p_golden:.4f}' + (' *' if p_golden < 0.05 else ' (ns)')
ax2.text(0.98, 0.95, f'Golden angle enrichment:\n'
         f'{enrichment_ratio:.2f}x ({observed_pct:.1f}% vs {expected_pct:.1f}%)\n'
         f'95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]\n'
         f'{sig_str} (config-level)',
         transform=ax2.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax2.set_xlabel('Pairwise Phase Difference (°)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('Pairwise Phase Differences (e-1 configs)', fontsize=10)
ax2.legend(fontsize=8, loc='upper left')

# Panel C: Phase portraits comparison across regions
ax3 = fig.add_subplot(gs[1, 0])
regions = [
    ('e-1 proximal', e1_diffs, 'steelblue'),
    ('Main well', well_diffs, 'green'),
    ('2:1 proximal', two1_diffs, 'firebrick'),
]

for label, diffs, color in regions:
    if diffs:
        diffs_deg = np.degrees(np.array(diffs))
        ax3.hist(diffs_deg, bins=bins_deg, density=True, alpha=0.4,
                 color=color, edgecolor=color, lw=0.5, label=label)

ax3.axvline(GOLDEN_ANGLE_DEG, color='goldenrod', lw=2, ls='--', alpha=0.8)
ax3.axhline(uniform_density, color='gray', ls=':', lw=1, alpha=0.5)
ax3.set_xlabel('Pairwise Phase Difference (°)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('Phase Geometry Across Landscape Regions', fontsize=10)
ax3.legend(fontsize=8)

region_stats = {}
for label, reg in [('e-1', e1_region), ('Well', well_region), ('2:1', two1_region)]:
    if reg:
        region_stats[label] = f"r={reg['mean_r']:.2f}, κ={reg['mean_kappa']:.1f}"
stats_text = '\n'.join([f'{k}: {v}' for k, v in region_stats.items()])
ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=7,
         va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel D: Cluster structure and inter-cluster angles
ax4 = fig.add_subplot(gs[1, 1])

if all_inter_angles:
    ax4.hist(all_inter_angles, bins=18, range=(0, 360), density=True,
             color='lightsteelblue', edgecolor='steelblue', alpha=0.8,
             label='Inter-cluster angles')
    ax4.axvline(GOLDEN_ANGLE_DEG, color='goldenrod', lw=2, ls='-',
                label=f'Golden angle = {GOLDEN_ANGLE_DEG:.1f}°')
    ax4.axvline(90, color='gray', lw=1.5, ls='--', alpha=0.6, label='90°')
    ax4.axvline(120, color='gray', lw=1.5, ls=':', alpha=0.6, label='120°')
    mean_ia = np.mean(all_inter_angles)
    ax4.axvline(mean_ia, color='red', lw=2, ls='-', alpha=0.7,
                label=f'Mean = {mean_ia:.1f}°')
    ax4.set_xlabel('Inter-Cluster Angle (°)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title(f'Inter-Cluster Angles (mode={cluster_mode} clusters)', fontsize=10)
    ax4.legend(fontsize=7, loc='upper right')

    cluster_dist = cluster_results['n_clusters_distribution']
    dist_str = ', '.join([f'{k}: {v}' for k, v in sorted(cluster_dist.items())])
    ax4.text(0.02, 0.95, f'Cluster distribution:\n{dist_str}',
             transform=ax4.transAxes, fontsize=7, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
else:
    cluster_dist = cluster_results['n_clusters_distribution']
    k_vals = sorted([int(k) for k in cluster_dist.keys()])
    counts_vals = [cluster_dist[str(k)] for k in k_vals]
    ax4.bar(k_vals, counts_vals, color='steelblue', edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Number of Clusters', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Cluster Count Distribution', fontsize=10)

fig.suptitle('Figure 14: Kuramoto Circle Architecture\n'
             'Phase Geometry When Oscillators Produce e-1 Frequency Ratios',
             fontsize=13, fontweight='bold')

plt.savefig(f'{OUTPUT_DIR}/fig14_kuramoto_circle.png')
plt.close()

# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\nGolden Angle Enrichment: {enrichment_ratio:.2f}x (p={p_golden:.6f})")
print(f"  {'SIGNIFICANT' if p_golden < 0.05 else 'NOT SIGNIFICANT'}")
print(f"\nCluster Mode: {cluster_mode} clusters")
print(f"  (L₄ = 7 prediction: {'MATCHES' if cluster_mode == 7 else 'does not match'})")

if all_inter_angles:
    mean_inter = np.mean(all_inter_angles)
    print(f"\nMean Inter-Cluster Angle: {mean_inter:.1f}°")
    print(f"  Golden angle = {GOLDEN_ANGLE_DEG:.1f}° (diff: {abs(mean_inter - GOLDEN_ANGLE_DEG):.1f}°)")

print(f"\nPhase Geometry by Region:")
for label, reg in [('e-1 proximal', e1_region), ('Main well', well_region), ('2:1 proximal', two1_region)]:
    if reg:
        print(f"  {label}: r={reg['mean_r']:.3f}, κ={reg['mean_kappa']:.2f}, clusters={reg['mean_n_clusters']:.1f}")

print(f"\nKey Finding: {full_results['key_finding']}")
print(f"\nSaved: outputs/kuramoto_circle_architecture.json")
print(f"       {OUTPUT_DIR}/fig14_kuramoto_circle.png")

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import warnings
warnings.filterwarnings('ignore')

from deep_dive_common import *

print("=" * 70)
print("BLOCK 4: SKIP-ONE SCALE FAILURE ANALYSIS")
print("=" * 70)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
n = len(subjects)

PHI2 = PHI**2

print("\n--- 4a: Individual Skip-One Ratios ---")
alpha_delta = np.array([all_centroids[i]['alpha'] / all_centroids[i]['delta'] for i in range(n)])
beta_theta = np.array([all_centroids[i]['beta'] / all_centroids[i]['theta'] for i in range(n)])
gamma_alpha = np.array([all_centroids[i]['gamma'] / all_centroids[i]['alpha'] for i in range(n)])

for name, vals in [('alpha/delta', alpha_delta), ('beta/theta', beta_theta), ('gamma/alpha', gamma_alpha)]:
    err = abs(np.mean(vals) - PHI2) / PHI2 * 100
    t, p = ttest_1samp(vals, PHI2)
    print(f"  {name}: mean={np.mean(vals):.4f} (SD={np.std(vals):.4f}), phi^2 err={err:.1f}%, t={t:.3f}, p={p:.4f}")

print("\n--- 4b: Delta Band Sensitivity ---")
delta_defs = [(0.5, 4, 'Standard (0.5-4)'), (1, 4, 'Narrow (1-4)'), (2, 4, 'Very narrow (2-4)'), (1, 3, 'Conservative (1-3)')]
delta_results = []
for lo, hi, label in delta_defs:
    delta_c_alt = np.array([spectral_centroid(all_freqs[i], all_psds[i], lo, hi) for i in range(n)])
    td_ratio = np.array([all_centroids[i]['theta'] / delta_c_alt[i] if delta_c_alt[i] > 0 else np.nan for i in range(n)])
    ad_ratio = np.array([all_centroids[i]['alpha'] / delta_c_alt[i] if delta_c_alt[i] > 0 else np.nan for i in range(n)])
    valid = ~np.isnan(td_ratio)
    td_err = abs(np.nanmean(td_ratio) - PHI) / PHI * 100
    ad_err = abs(np.nanmean(ad_ratio) - PHI2) / PHI2 * 100
    delta_results.append({
        'label': label,
        'lo': lo, 'hi': hi,
        'delta_centroid_mean': float(np.nanmean(delta_c_alt)),
        'theta_delta_mean': float(np.nanmean(td_ratio)),
        'theta_delta_phi_err': float(td_err),
        'alpha_delta_mean': float(np.nanmean(ad_ratio)),
        'alpha_delta_phi2_err': float(ad_err),
    })
    print(f"  {label}: delta_c={np.nanmean(delta_c_alt):.3f}, theta/delta={np.nanmean(td_ratio):.4f} (phi err={td_err:.1f}%), alpha/delta={np.nanmean(ad_ratio):.4f} (phi^2 err={ad_err:.1f}%)")

print("\n--- 4c: Individual Self-Similarity ---")
adj_ratios = np.zeros((n, 4))
for i in range(n):
    c = all_centroids[i]
    adj_ratios[i, 0] = c['theta'] / c['delta']
    adj_ratios[i, 1] = c['alpha'] / c['theta']
    adj_ratios[i, 2] = c['beta'] / c['alpha']
    adj_ratios[i, 3] = c['gamma'] / c['beta']

adj_phi_err = np.abs(adj_ratios - PHI) / PHI
skip_phi2_err = np.zeros((n, 3))
skip_phi2_err[:, 0] = np.abs(alpha_delta - PHI2) / PHI2
skip_phi2_err[:, 1] = np.abs(beta_theta - PHI2) / PHI2
skip_phi2_err[:, 2] = np.abs(gamma_alpha - PHI2) / PHI2

mean_adj_err = np.mean(adj_phi_err, axis=1)
mean_skip_err = np.mean(skip_phi2_err, axis=1)

from scipy.stats import pearsonr
r_adj_skip, p_adj_skip = pearsonr(mean_adj_err, mean_skip_err)
print(f"  Adjacent error vs skip-one error: r={r_adj_skip:.4f}, p={p_adj_skip:.4f}")

worst_skip = np.argmax(np.mean(skip_phi2_err, axis=0))
worst_names = ['alpha/delta', 'beta/theta', 'gamma/alpha']
print(f"  Worst skip-one pair: {worst_names[worst_skip]} (mean error={np.mean(skip_phi2_err[:, worst_skip])*100:.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
bp = ax.boxplot([alpha_delta, beta_theta, gamma_alpha], labels=['alpha/delta', 'beta/theta', 'gamma/alpha'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#E74C3C', '#3498DB', '#2ECC71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.axhline(PHI2, color='gold', linewidth=2, linestyle='--', label=f'phi^2={PHI2:.3f}')
ax.set_ylabel('Ratio')
ax.set_title('4a: Skip-One Ratio Distributions', fontweight='bold')
ax.legend()

ax = axes[0, 1]
labels_d = [r['label'] for r in delta_results]
ad_vals = [r['alpha_delta_phi2_err'] for r in delta_results]
td_vals = [r['theta_delta_phi_err'] for r in delta_results]
x = np.arange(len(labels_d))
width = 0.35
ax.bar(x - width/2, td_vals, width, label='theta/delta (vs phi)', color='#3498DB', alpha=0.7)
ax.bar(x + width/2, ad_vals, width, label='alpha/delta (vs phi^2)', color='#E74C3C', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([l.split('(')[0].strip() for l in labels_d], rotation=15, fontsize=8)
ax.set_ylabel('% Error from Target')
ax.set_title('4b: Delta Band Sensitivity', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.scatter(mean_adj_err * 100, mean_skip_err * 100, c='coral', s=50, alpha=0.7, edgecolors='black')
ax.set_xlabel('Mean Adjacent Error (%)')
ax.set_ylabel('Mean Skip-One Error (%)')
ax.set_title(f'4c: Adjacent vs Skip-One Error\nr={r_adj_skip:.3f}, p={p_adj_skip:.4f}', fontweight='bold')

ax = axes[1, 0]
for j, (name, color) in enumerate(zip(['alpha/delta', 'beta/theta', 'gamma/alpha'], ['#E74C3C', '#3498DB', '#2ECC71'])):
    ax.hist(skip_phi2_err[:, j] * 100, bins=15, alpha=0.5, color=color, label=name, edgecolor='black')
ax.set_xlabel('Skip-One Error from phi^2 (%)')
ax.set_ylabel('Count')
ax.set_title('4a: Skip-One Error Distribution', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1, 1]
band_names = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
adj_means_err = [np.mean(adj_phi_err[:, j]) * 100 for j in range(4)]
ax.bar(band_names, adj_means_err, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Mean Error from phi (%)')
ax.set_title('Adjacent Ratio Errors by Band', fontweight='bold')
ax.tick_params(axis='x', rotation=15)

ax = axes[1, 2]
all_errs = np.concatenate([adj_phi_err.flatten() * 100, skip_phi2_err.flatten() * 100])
labels_all = ['Adjacent'] * len(adj_phi_err.flatten()) + ['Skip-One'] * len(skip_phi2_err.flatten())
adj_flat = adj_phi_err.flatten() * 100
skip_flat = skip_phi2_err.flatten() * 100
bp2 = ax.boxplot([adj_flat, skip_flat], labels=['Adjacent\n(vs phi)', 'Skip-One\n(vs phi^2)'], patch_artist=True)
bp2['boxes'][0].set_facecolor('#3498DB')
bp2['boxes'][1].set_facecolor('#E74C3C')
for box in bp2['boxes']:
    box.set_alpha(0.5)
ax.set_ylabel('Error (%)')
ax.set_title('Adjacent vs Skip-One Error', fontweight='bold')

plt.suptitle(f'Block 4: Skip-One Scale Failure Analysis (N={n})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/block4_skip_one.png', dpi=150, bbox_inches='tight')
plt.close()

results = {
    'n_subjects': n,
    'skip_one': {
        'alpha_delta': {'mean': float(np.mean(alpha_delta)), 'sd': float(np.std(alpha_delta)), 'phi2_err': float(abs(np.mean(alpha_delta) - PHI2) / PHI2 * 100)},
        'beta_theta': {'mean': float(np.mean(beta_theta)), 'sd': float(np.std(beta_theta)), 'phi2_err': float(abs(np.mean(beta_theta) - PHI2) / PHI2 * 100)},
        'gamma_alpha': {'mean': float(np.mean(gamma_alpha)), 'sd': float(np.std(gamma_alpha)), 'phi2_err': float(abs(np.mean(gamma_alpha) - PHI2) / PHI2 * 100)},
    },
    'delta_sensitivity': delta_results,
    'adj_vs_skip_r': float(r_adj_skip),
    'adj_vs_skip_p': float(p_adj_skip),
    'worst_skip_pair': worst_names[worst_skip],
}
with open('outputs/block4_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('outputs/block4_results.md', 'w') as f:
    f.write(f"# Block 4: Skip-One Scale Failure (N={n})\n\n")
    f.write("## 4a: Skip-One Ratios\n\n")
    f.write("| Pair | Mean | SD | Expected (phi^2) | Error |\n")
    f.write("|------|------|-----|-----------------|-------|\n")
    for name, vals in [('alpha/delta', alpha_delta), ('beta/theta', beta_theta), ('gamma/alpha', gamma_alpha)]:
        f.write(f"| {name} | {np.mean(vals):.4f} | {np.std(vals):.4f} | {PHI2:.4f} | {abs(np.mean(vals)-PHI2)/PHI2*100:.1f}% |\n")
    f.write("\n## 4b: Delta Band Sensitivity\n\n")
    f.write("| Definition | Delta Centroid | theta/delta phi err | alpha/delta phi^2 err |\n")
    f.write("|-----------|---------------|--------------------|-----------------------|\n")
    for r in delta_results:
        f.write(f"| {r['label']} | {r['delta_centroid_mean']:.3f} Hz | {r['theta_delta_phi_err']:.1f}% | {r['alpha_delta_phi2_err']:.1f}% |\n")
    f.write("\n## Interpretation\n\n")
    f.write("The skip-one failure reveals that the phi-scaling does NOT extend to non-adjacent bands. ")
    f.write("If true self-similarity held, skip-one ratios should approximate phi^2=2.618. ")
    f.write(f"The worst pair ({worst_names[worst_skip]}) deviates most. ")
    f.write("Delta band definition significantly affects results, suggesting the delta centroid ")
    f.write("is sensitive to low-frequency artifacts and band boundary choices.\n\n")
    f.write("## Implication for Paper\n\n")
    f.write("The self-similar phi-scaling claim should be limited to adjacent bands only. ")
    f.write("Skip-one and higher-order ratios do not support fractal phi-organization. ")
    f.write("Report delta sensitivity analysis to show methodological bounds.\n")

print("\nResults saved: outputs/block4_results.json, outputs/block4_results.md")

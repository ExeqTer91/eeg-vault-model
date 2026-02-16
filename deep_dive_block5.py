import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp
import warnings
warnings.filterwarnings('ignore')

from deep_dive_common import *

print("=" * 70)
print("BLOCK 5: INDIVIDUAL META-RATIOS & SELF-SIMILARITY")
print("=" * 70)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
n = len(subjects)

adj_ratios = np.zeros((n, 4))
meta_ratios = np.zeros(n)
skip_one = np.zeros((n, 3))
full_span = np.zeros(n)
attractor_equiv = np.zeros(n)

for i in range(n):
    c = all_centroids[i]
    adj_ratios[i, 0] = c['theta'] / c['delta']
    adj_ratios[i, 1] = c['alpha'] / c['theta']
    adj_ratios[i, 2] = c['beta'] / c['alpha']
    adj_ratios[i, 3] = c['gamma'] / c['beta']

    if adj_ratios[i, 0] > 0:
        meta_ratios[i] = adj_ratios[i, 1] / adj_ratios[i, 0]
    skip_one[i, 0] = c['alpha'] / c['delta']
    skip_one[i, 1] = c['beta'] / c['theta']
    skip_one[i, 2] = c['gamma'] / c['alpha']
    full_span[i] = c['gamma'] / c['delta']

    if adj_ratios[i, 3] > 0:
        attractor_equiv[i] = adj_ratios[i, 1] / adj_ratios[i, 3]

print("--- 5a: Individual Meta-Ratios ---")
t_meta, p_meta = ttest_1samp(meta_ratios, PHI)
within_10 = np.sum(np.abs(meta_ratios - PHI) / PHI < 0.10)
within_20 = np.sum(np.abs(meta_ratios - PHI) / PHI < 0.20)
print(f"  Mean meta-ratio: {np.mean(meta_ratios):.4f} (SD={np.std(meta_ratios):.4f})")
print(f"  t-test vs phi: t={t_meta:.4f}, p={p_meta:.6f}")
print(f"  Within 10% of phi: {within_10}/{n} ({within_10/n*100:.1f}%)")
print(f"  Within 20% of phi: {within_20}/{n} ({within_20/n*100:.1f}%)")

print("\n--- 5b: Full Self-Similarity Profile ---")
adj_means = np.mean(adj_ratios, axis=1)
adj_errors = np.abs(adj_means - PHI) / PHI * 100

phi_6 = PHI**6
full_span_err = np.abs(full_span - phi_6) / phi_6 * 100

self_sim_scores = np.zeros(n)
for i in range(n):
    adj_err = np.mean(np.abs(adj_ratios[i] - PHI) / PHI)
    meta_err = abs(meta_ratios[i] - PHI) / PHI
    self_sim_scores[i] = 1.0 / (1.0 + adj_err + meta_err)

best_5 = np.argsort(self_sim_scores)[-5:][::-1]
worst_5 = np.argsort(self_sim_scores)[:5]
print(f"  Most self-similar subjects: {[subjects[i]['name'] for i in best_5]}")
print(f"    Scores: {[f'{self_sim_scores[i]:.3f}' for i in best_5]}")
print(f"  Least self-similar: {[subjects[i]['name'] for i in worst_5]}")
print(f"    Scores: {[f'{self_sim_scores[i]:.3f}' for i in worst_5]}")

r_ss_pci, p_ss_pci = pearsonr(self_sim_scores, all_pcis)
print(f"  Self-similarity vs PCI: r={r_ss_pci:.4f}, p={p_ss_pci:.4f}")

print("\n--- 5c: Attractor Equivalence ---")
t_eq, p_eq = ttest_1samp(attractor_equiv, 1.0)
within_10_eq = np.sum(np.abs(attractor_equiv - 1.0) < 0.10)
print(f"  Mean (alpha/theta) / (gamma/beta): {np.mean(attractor_equiv):.4f} (SD={np.std(attractor_equiv):.4f})")
print(f"  t-test vs 1.0: t={t_eq:.4f}, p={p_eq:.6f}")
print(f"  Within 10% of 1.0: {within_10_eq}/{n} ({within_10_eq/n*100:.1f}%)")

print("\n--- Adjacent ratio details ---")
band_names = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
for j, name in enumerate(band_names):
    vals = adj_ratios[:, j]
    err = abs(np.mean(vals) - PHI) / PHI * 100
    print(f"  {name}: mean={np.mean(vals):.4f}, SD={np.std(vals):.4f}, phi err={err:.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.hist(meta_ratios, bins=15, color='#9B59B6', alpha=0.7, edgecolor='black')
ax.axvline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi={PHI:.3f}')
ax.axvline(np.mean(meta_ratios), color='blue', linewidth=2, linestyle='-', label=f'mean={np.mean(meta_ratios):.3f}')
ax.set_xlabel('Meta-Ratio')
ax.set_ylabel('Count')
ax.set_title(f'5a: Individual Meta-Ratios\nt={t_meta:.2f}, p={p_meta:.4f}', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[0, 1]
bp = ax.boxplot([adj_ratios[:, j] for j in range(4)], labels=band_names, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label='phi')
ax.set_ylabel('Ratio')
ax.set_title('5b: Adjacent Ratios by Band', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.hist(attractor_equiv, bins=15, color='#1ABC9C', alpha=0.7, edgecolor='black')
ax.axvline(1.0, color='red', linewidth=2, linestyle='--', label='Expected=1.0')
ax.axvline(np.mean(attractor_equiv), color='blue', linewidth=2, linestyle='-', label=f'mean={np.mean(attractor_equiv):.3f}')
ax.set_xlabel('(alpha/theta) / (gamma/beta)')
ax.set_ylabel('Count')
ax.set_title(f'5c: Attractor Equivalence\nt={t_eq:.2f}, p={p_eq:.4f}', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.scatter(self_sim_scores, all_pcis, c='coral', s=50, alpha=0.7, edgecolors='black')
ax.set_xlabel('Self-Similarity Score')
ax.set_ylabel('PCI')
ax.set_title(f'5b: Self-Similarity vs PCI\nr={r_ss_pci:.3f}, p={p_ss_pci:.4f}', fontweight='bold')

ax = axes[1, 1]
band_means = [np.mean(adj_ratios[:, j]) for j in range(4)]
band_errs = [np.std(adj_ratios[:, j]) / np.sqrt(n) for j in range(4)]
ax.bar(band_names, band_means, yerr=band_errs, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'], alpha=0.7, edgecolor='black', capsize=5)
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label='phi')
ax.set_ylabel('Mean Ratio')
ax.set_title('Adjacent Ratio Means (+/- SE)', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1, 2]
skip_names = ['alpha/delta', 'beta/theta', 'gamma/alpha']
skip_means = [np.mean(skip_one[:, j]) for j in range(3)]
expected_phi2 = PHI**2
ax.bar(skip_names, skip_means, color=['#E74C3C', '#3498DB', '#2ECC71'], alpha=0.7, edgecolor='black')
ax.axhline(expected_phi2, color='gold', linewidth=2, linestyle='--', label=f'phi^2={expected_phi2:.3f}')
ax.set_ylabel('Mean Ratio')
ax.set_title(f'Skip-One Ratios (expected phi^2)', fontweight='bold')
ax.legend(fontsize=8)

plt.suptitle(f'Block 5: Individual Meta-Ratios & Self-Similarity (N={n})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/block5_meta_ratios.png', dpi=150, bbox_inches='tight')
plt.close()

results = {
    'n_subjects': n,
    'meta_ratio_mean': float(np.mean(meta_ratios)),
    'meta_ratio_sd': float(np.std(meta_ratios)),
    'meta_ratio_t': float(t_meta),
    'meta_ratio_p': float(p_meta),
    'within_10pct_phi': int(within_10),
    'within_20pct_phi': int(within_20),
    'adj_ratio_means': {name: float(np.mean(adj_ratios[:, j])) for j, name in enumerate(band_names)},
    'adj_ratio_phi_errors': {name: float(abs(np.mean(adj_ratios[:, j]) - PHI) / PHI * 100) for j, name in enumerate(band_names)},
    'attractor_equiv_mean': float(np.mean(attractor_equiv)),
    'attractor_equiv_t': float(t_eq),
    'attractor_equiv_p': float(p_eq),
    'self_sim_vs_pci_r': float(r_ss_pci),
    'self_sim_vs_pci_p': float(p_ss_pci),
    'skip_one_means': {name: float(np.mean(skip_one[:, j])) for j, name in enumerate(skip_names)},
    'skip_one_phi2_errors': {name: float(abs(np.mean(skip_one[:, j]) - expected_phi2) / expected_phi2 * 100) for j, name in enumerate(skip_names)},
}
with open('outputs/block5_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('outputs/block5_results.md', 'w') as f:
    f.write(f"# Block 5: Individual Meta-Ratios & Self-Similarity (N={n})\n\n")
    f.write("## 5a: Meta-Ratios\n\n")
    f.write(f"- Mean: {np.mean(meta_ratios):.4f} (SD={np.std(meta_ratios):.4f})\n")
    f.write(f"- t-test vs phi: t={t_meta:.4f}, p={p_meta:.6f}\n")
    f.write(f"- Within 10% of phi: {within_10}/{n} ({within_10/n*100:.1f}%)\n")
    f.write(f"- Within 20% of phi: {within_20}/{n} ({within_20/n*100:.1f}%)\n\n")
    f.write("## 5b: Adjacent Ratios\n\n")
    f.write("| Band Pair | Mean | Phi Error |\n")
    f.write("|-----------|------|----------|\n")
    for j, name in enumerate(band_names):
        m = np.mean(adj_ratios[:, j])
        e = abs(m - PHI) / PHI * 100
        f.write(f"| {name} | {m:.4f} | {e:.1f}% |\n")
    f.write("\n## 5c: Attractor Equivalence\n\n")
    f.write(f"- Mean (alpha/theta)/(gamma/beta): {np.mean(attractor_equiv):.4f}\n")
    f.write(f"- t-test vs 1.0: t={t_eq:.4f}, p={p_eq:.6f}\n\n")
    f.write("## Skip-One Ratios\n\n")
    f.write("| Pair | Mean | Expected (phi^2) | Error |\n")
    f.write("|------|------|-----------------|-------|\n")
    for j, name in enumerate(skip_names):
        m = np.mean(skip_one[:, j])
        e = abs(m - expected_phi2) / expected_phi2 * 100
        f.write(f"| {name} | {m:.4f} | {expected_phi2:.4f} | {e:.1f}% |\n")
    f.write("\n## Interpretation\n\n")
    f.write("The meta-ratio tests whether the ratio-of-ratios itself converges to phi, ")
    f.write("which would indicate true self-similarity across frequency scales. ")
    if p_meta < 0.05:
        f.write(f"The meta-ratio ({np.mean(meta_ratios):.4f}) is significantly different from phi, ")
        f.write("meaning exact self-similarity does NOT hold at individual level.\n")
    else:
        f.write(f"The meta-ratio ({np.mean(meta_ratios):.4f}) is not significantly different from phi, ")
        f.write("consistent with (but not proving) self-similar organization.\n")
    f.write("\n## Implication for Paper\n\n")
    f.write("Report per-band ratio profiles showing which adjacent pairs are closest to phi. ")
    f.write("The meta-ratio and attractor equivalence tests directly probe self-similarity claims.\n")

print("\nResults saved: outputs/block5_results.json, outputs/block5_results.md")

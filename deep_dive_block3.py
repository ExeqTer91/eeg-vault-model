import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

from deep_dive_common import *

print("=" * 70)
print("BLOCK 3: ATTRACTOR LANDSCAPE")
print("=" * 70)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
n = len(subjects)

print("\n--- 3a: Cluster Analysis ---")
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

X = all_ratios.reshape(-1, 1)

sil_scores = {}
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores[k] = sil
    centers = sorted(km.cluster_centers_.flatten())
    print(f"  K-means k={k}: silhouette={sil:.4f}, centers={[f'{c:.4f}' for c in centers]}")

best_k_km = max(sil_scores, key=sil_scores.get)
print(f"  Best k (K-means): {best_k_km} (silhouette={sil_scores[best_k_km]:.4f})")

gmm_bics = {}
gmm_models = {}
for k in [1, 2, 3, 4]:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(X)
    bic = gmm.bic(X)
    gmm_bics[k] = bic
    gmm_models[k] = gmm
    if k > 1:
        means = sorted(gmm.means_.flatten())
        weights = gmm.weights_
        print(f"  GMM k={k}: BIC={bic:.2f}, means={[f'{m:.4f}' for m in means]}, weights={[f'{w:.3f}' for w in sorted(weights, reverse=True)]}")
    else:
        print(f"  GMM k={k}: BIC={bic:.2f}, mean={gmm.means_[0,0]:.4f}")

best_k_gmm = min(gmm_bics, key=gmm_bics.get)
print(f"  Best k (GMM by BIC): {best_k_gmm}")

print("\n--- 3b: Bimodality Test (Hartigan's Dip) ---")
try:
    from diptest import diptest
    dip_stat, dip_p = diptest(all_ratios)
    print(f"  Hartigan's dip test: D={dip_stat:.4f}, p={dip_p:.4f}")
    bimodal = dip_p < 0.05
    print(f"  Bimodal: {'YES' if bimodal else 'NO'}")
except ImportError:
    sorted_data = np.sort(all_ratios)
    n_d = len(sorted_data)
    max_dip = 0
    for i in range(n_d):
        ecdf = (i + 1) / n_d
        uniform_cdf = (sorted_data[i] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
        dip = abs(ecdf - uniform_cdf)
        if dip > max_dip:
            max_dip = dip
    dip_stat = max_dip
    dip_p = -1
    print(f"  Manual dip statistic: D={dip_stat:.4f} (diptest not installed, no p-value)")
    bimodal = False

print("\n--- 3c: Individual Classification ---")
attractors = {'phi': PHI, 'e-1': E_MINUS_1, '2:1': HARMONIC, 'pi/2': np.pi / 2}
classification = []
for i in range(n):
    r = all_ratios[i]
    closest = min(attractors.items(), key=lambda x: abs(r - x[1]))
    classification.append(closest[0])

for name in attractors:
    count = classification.count(name)
    print(f"  Closest to {name} ({attractors[name]:.4f}): {count}/{n} ({count/n*100:.1f}%)")

print("\n--- 3d: Attractor Stability (within-subject variance) ---")
N_WINDOWS = 8
within_vars = np.zeros(n)
window_ratios_all = []

for i, subj in enumerate(subjects):
    data = subj['data']
    fs = subj['fs']
    n_samp = data.shape[1]
    win_size = n_samp // N_WINDOWS
    ratios_w = []
    for w in range(N_WINDOWS):
        seg = data[:, w * win_size:(w + 1) * win_size]
        nperseg = min(int(4 * fs), seg.shape[1])
        if nperseg < int(fs):
            continue
        avg_psd = None
        for ch in range(seg.shape[0]):
            freqs, psd = welch(seg[ch], fs=fs, nperseg=nperseg)
            if avg_psd is None:
                avg_psd = psd.copy()
            else:
                avg_psd += psd
        avg_psd /= seg.shape[0]
        tc = spectral_centroid(freqs, avg_psd, 4, 8)
        ac = spectral_centroid(freqs, avg_psd, 8, 13)
        ratios_w.append(ac / tc)
    within_vars[i] = np.var(ratios_w) if len(ratios_w) > 1 else 0
    window_ratios_all.append(ratios_w)

between_var = np.var(all_ratios)
mean_within_var = np.mean(within_vars)
print(f"  Between-subject variance: {between_var:.6f}")
print(f"  Mean within-subject variance: {mean_within_var:.6f}")
print(f"  Ratio (between/within): {between_var / max(mean_within_var, 1e-10):.2f}")

switchers = 0
for i in range(n):
    wr = window_ratios_all[i]
    if len(wr) < 2:
        continue
    pcis_w = [compute_pci(r) for r in wr]
    signs = [1 if p > 0 else -1 for p in pcis_w]
    if len(set(signs)) > 1:
        switchers += 1
print(f"  Subjects switching between phi/harmonic states: {switchers}/{n} ({switchers/n*100:.1f}%)")

print("\n--- 3e: Predictors of Attractor Membership ---")
phi_org = all_pcis > 0
total_power = np.zeros(n)
for i in range(n):
    total_power[i] = np.sum(all_psds[i])

from scipy.stats import mannwhitneyu as mwu
if np.sum(phi_org) > 1 and np.sum(~phi_org) > 1:
    u_pow, p_pow = mwu(total_power[phi_org], total_power[~phi_org])
    print(f"  Total power: phi-org={np.mean(total_power[phi_org]):.2e}, harm={np.mean(total_power[~phi_org]):.2e}, p={p_pow:.4f}")

    alpha_pow_ratio = np.array([all_centroids[i]['alpha'] for i in range(n)])
    u_ap, p_ap = mwu(alpha_pow_ratio[phi_org], alpha_pow_ratio[~phi_org])
    print(f"  Alpha centroid: phi-org={np.mean(alpha_pow_ratio[phi_org]):.2f}, harm={np.mean(alpha_pow_ratio[~phi_org]):.2f}, p={p_ap:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.hist(all_ratios, bins=15, color='#E74C3C', alpha=0.7, edgecolor='black')
for name, val in attractors.items():
    ax.axvline(val, linewidth=1.5, linestyle='--', label=f'{name}={val:.3f}')
ax.set_xlabel('Alpha/Theta Ratio')
ax.set_ylabel('Count')
ax.set_title('3a: Ratio Distribution with Attractors', fontweight='bold')
ax.legend(fontsize=7)

ax = axes[0, 1]
ks = list(sil_scores.keys())
ax.bar(ks, [sil_scores[k] for k in ks], color='#3498DB', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Score')
ax.set_title(f'3a: K-Means Silhouette\nBest k={best_k_km}', fontweight='bold')

ax = axes[0, 2]
ks_g = list(gmm_bics.keys())
ax.bar(ks_g, [gmm_bics[k] for k in ks_g], color='#2ECC71', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Components')
ax.set_ylabel('BIC')
ax.set_title(f'3a: GMM BIC\nBest k={best_k_gmm}', fontweight='bold')

ax = axes[1, 0]
counts = [classification.count(name) for name in attractors]
ax.bar(list(attractors.keys()), counts, color=['gold', 'purple', 'blue', 'green'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Count')
ax.set_title('3c: Nearest Attractor Classification', fontweight='bold')

ax = axes[1, 1]
ax.scatter(range(n), within_vars, c='coral', s=50, alpha=0.7, edgecolors='black')
ax.axhline(between_var, color='blue', linewidth=2, linestyle='--', label=f'Between-subj var={between_var:.5f}')
ax.set_xlabel('Subject')
ax.set_ylabel('Within-Subject Variance')
ax.set_title(f'3d: Attractor Stability\n{switchers}/{n} switch states', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1, 2]
best_gmm = gmm_models[best_k_gmm]
x_plot = np.linspace(all_ratios.min() - 0.2, all_ratios.max() + 0.2, 300).reshape(-1, 1)
log_dens = best_gmm.score_samples(x_plot)
ax.hist(all_ratios, bins=15, density=True, alpha=0.5, color='gray', edgecolor='black')
ax.plot(x_plot, np.exp(log_dens), 'r-', linewidth=2, label=f'GMM k={best_k_gmm}')
ax.set_xlabel('Alpha/Theta Ratio')
ax.set_ylabel('Density')
ax.set_title(f'3a: Best GMM Fit (k={best_k_gmm})', fontweight='bold')
ax.legend()

plt.suptitle(f'Block 3: Attractor Landscape (N={n})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/block3_attractor_landscape.png', dpi=150, bbox_inches='tight')
plt.close()

results = {
    'n_subjects': n,
    'kmeans_silhouettes': {str(k): float(v) for k, v in sil_scores.items()},
    'best_k_kmeans': best_k_km,
    'gmm_bics': {str(k): float(v) for k, v in gmm_bics.items()},
    'best_k_gmm': best_k_gmm,
    'dip_stat': float(dip_stat),
    'dip_p': float(dip_p),
    'bimodal': bool(bimodal),
    'attractor_counts': {name: classification.count(name) for name in attractors},
    'between_var': float(between_var),
    'mean_within_var': float(mean_within_var),
    'var_ratio': float(between_var / max(mean_within_var, 1e-10)),
    'switchers': switchers,
    'switchers_pct': float(switchers / n * 100),
}
with open('outputs/block3_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('outputs/block3_results.md', 'w') as f:
    f.write(f"# Block 3: Attractor Landscape (N={n})\n\n")
    f.write("## 3a: Cluster Analysis\n\n")
    f.write("### K-Means\n")
    for k, s in sil_scores.items():
        f.write(f"- k={k}: silhouette={s:.4f}\n")
    f.write(f"- Best k: {best_k_km}\n\n")
    f.write("### Gaussian Mixture Model\n")
    for k, b in gmm_bics.items():
        f.write(f"- k={k}: BIC={b:.2f}\n")
    f.write(f"- Best k: {best_k_gmm}\n\n")
    f.write("## 3b: Bimodality\n\n")
    f.write(f"- Dip statistic: {dip_stat:.4f}")
    if dip_p >= 0:
        f.write(f", p={dip_p:.4f}")
    f.write(f"\n- Distribution is {'bimodal' if bimodal else 'unimodal'}\n\n")
    f.write("## 3c: Nearest Attractor\n\n")
    for name in attractors:
        c = classification.count(name)
        f.write(f"- {name} ({attractors[name]:.4f}): {c}/{n} ({c/n*100:.1f}%)\n")
    f.write("\n## 3d: Stability\n\n")
    f.write(f"- Between/within variance ratio: {between_var/max(mean_within_var,1e-10):.2f}\n")
    f.write(f"- Subjects switching states: {switchers}/{n} ({switchers/n*100:.1f}%)\n\n")
    f.write("## Interpretation\n\n")
    if best_k_gmm == 1:
        f.write("The best GMM has 1 component — the population is unimodal, not a mixture of discrete attractors. ")
        f.write("The distribution is a single cluster, and the 'attractor' interpretation is not supported.\n")
    else:
        f.write(f"The best GMM has {best_k_gmm} components, suggesting a mixture of sub-populations. ")
    f.write("\n## Implication for Paper\n\n")
    f.write("If the distribution is unimodal, the paper cannot claim discrete phi/harmonic attractors. ")
    f.write("Instead, the ratio space is a continuous distribution. ")
    f.write("The attractor metaphor should be replaced with a description of the population distribution shape.\n")

print("\nResults saved: outputs/block3_results.json, outputs/block3_results.md")

import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
from scipy.stats import pearsonr, ttest_1samp, ttest_rel, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from deep_dive_common import (
    PHI, load_and_compute, compute_subject_psd, spectral_centroid, compute_pci
)
from fooof import FOOOF

os.makedirs('outputs', exist_ok=True)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
N = len(subjects)
print(f"N={N} subjects loaded\n")

PHI_SQ = PHI ** 2
INV_PHI = 1.0 / PHI
SQRT_PHI = np.sqrt(PHI)
E_VAL = np.e
E_MINUS_1 = np.e - 1
PI_HALF = np.pi / 2

CONSTANTS = {
    '1/phi^2': PHI**-2, '1/phi': INV_PHI, 'sqrt(phi)': SQRT_PHI,
    'phi': PHI, 'phi^2': PHI_SQ, 'phi^3': PHI**3,
    'e-1': E_MINUS_1, 'e': E_VAL, 'pi/2': PI_HALF, '2': 2.0, '3': 3.0
}

results = {}
all_tests = []


def nearest_constant(val):
    best_name, best_err = None, float('inf')
    for name, c in CONSTANTS.items():
        if c == 0:
            continue
        err = abs(val - c) / abs(c) * 100
        if err < best_err:
            best_err = err
            best_name = name
    return best_name, best_err


def record_test(label, value, category):
    name, err = nearest_constant(value)
    all_tests.append({
        'label': label, 'value': float(value),
        'nearest': name, 'error_pct': float(err),
        'category': category
    })
    return name, err


def fit_fooof_all():
    fooof_results = []
    for i in range(N):
        freqs = all_freqs[i]
        psd = all_psds[i]
        mask = (freqs >= 1) & (freqs <= 50)
        fm = FOOOF(peak_width_limits=[1, 12], max_n_peaks=8, min_peak_height=0.05, verbose=False)
        try:
            fm.fit(freqs[mask], psd[mask])
            fooof_results.append({
                'peaks': fm.peak_params_,
                'aperiodic': fm.aperiodic_params_,
                'name': subjects[i]['name']
            })
        except:
            fooof_results.append({'peaks': np.array([]).reshape(0,3), 'aperiodic': None, 'name': subjects[i]['name']})
    return fooof_results

print("Fitting FOOOF...")
fooof_all = fit_fooof_all()
print("FOOOF done.\n")

band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
band_ranges = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

all_band_powers = np.zeros((N, 5))
for i in range(N):
    freqs = all_freqs[i]
    psd = all_psds[i]
    for j, (lo, hi) in enumerate(band_ranges):
        mask = (freqs >= lo) & (freqs <= hi)
        all_band_powers[i, j] = np.sum(psd[mask]) * (freqs[1] - freqs[0])

all_adj_ratios = np.zeros((N, 4))
adj_ratio_names = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
for i in range(N):
    c = all_centroids[i]
    all_adj_ratios[i, 0] = c['theta'] / c['delta']
    all_adj_ratios[i, 1] = c['alpha'] / c['theta']
    all_adj_ratios[i, 2] = c['beta'] / c['alpha']
    all_adj_ratios[i, 3] = c['gamma'] / c['beta']

all_meta_ratios = all_adj_ratios[:, 0] / all_adj_ratios[:, 1]


print("=" * 70)
print("BLOCK 10A: RATIO-OF-EVERYTHING MATRIX")
print("=" * 70)

quantities = {}
for i in range(N):
    c = all_centroids[i]
    subj_q = {}
    for j, bn in enumerate(band_names):
        subj_q[f'centroid_{bn}'] = c[bn]
        subj_q[f'power_{bn}'] = all_band_powers[i, j]
    subj_q['pci'] = all_pcis[i]
    subj_q['alpha_theta_ratio'] = all_ratios[i]
    subj_q['meta_ratio'] = all_meta_ratios[i]
    subj_q['total_power'] = np.sum(all_band_powers[i])
    fr = fooof_all[i]
    if fr['aperiodic'] is not None:
        subj_q['aperiodic_slope'] = fr['aperiodic'][-1]
        subj_q['aperiodic_offset'] = fr['aperiodic'][0]
    quantities[i] = subj_q

q_names = sorted(quantities[0].keys())
n_q = len(q_names)
n_pairs = n_q * (n_q - 1) // 2

print(f"  Quantities per subject: {n_q}")
print(f"  Unique ratio pairs: {n_pairs}")

ratio_results = []
for j in range(n_q):
    for k in range(j+1, n_q):
        vals = []
        for i in range(N):
            if q_names[j] in quantities[i] and q_names[k] in quantities[i]:
                v1 = quantities[i][q_names[j]]
                v2 = quantities[i][q_names[k]]
                if v2 != 0 and np.isfinite(v1) and np.isfinite(v2):
                    vals.append(v1 / v2)
        if len(vals) >= 5:
            mean_r = np.mean(vals)
            if np.isfinite(mean_r) and 0.1 < abs(mean_r) < 100:
                name, err = record_test(f'{q_names[j]}/{q_names[k]}', mean_r, '10A')
                ratio_results.append({
                    'pair': f'{q_names[j]}/{q_names[k]}',
                    'mean': float(mean_r),
                    'nearest': name,
                    'error_pct': float(err),
                    'n': len(vals)
                })
                inv_mean = 1.0 / mean_r if mean_r != 0 else float('inf')
                if np.isfinite(inv_mean) and 0.1 < abs(inv_mean) < 100:
                    record_test(f'{q_names[k]}/{q_names[j]}', inv_mean, '10A')

ratio_results.sort(key=lambda x: x['error_pct'])
print(f"  Total ratios computed: {len(ratio_results)}")
print(f"\n  TOP 10 closest to any mathematical constant:")
for r in ratio_results[:10]:
    print(f"    {r['pair']}: {r['mean']:.4f} -> {r['nearest']} ({r['error_pct']:.2f}% error)")

results['10A'] = {
    'n_quantities': n_q,
    'n_ratios': len(ratio_results),
    'top_20': ratio_results[:20]
}
print()


print("=" * 70)
print("BLOCK 10B: POWER x FREQUENCY CROSS-PRODUCTS (Spectral Energy)")
print("=" * 70)

spectral_energies = np.zeros((N, 5))
for i in range(N):
    c = all_centroids[i]
    for j, bn in enumerate(band_names):
        spectral_energies[i, j] = c[bn] * all_band_powers[i, j]

energy_ratios = np.zeros((N, 4))
energy_ratio_names = ['E_theta/E_delta', 'E_alpha/E_theta', 'E_beta/E_alpha', 'E_gamma/E_beta']
for j in range(4):
    energy_ratios[:, j] = spectral_energies[:, j+1] / np.maximum(spectral_energies[:, j], 1e-30)

print("  Adjacent spectral energy ratios:")
for j in range(4):
    mean_er = np.mean(energy_ratios[:, j])
    name, err = record_test(energy_ratio_names[j], mean_er, '10B')
    print(f"    {energy_ratio_names[j]}: {mean_er:.4f} -> {name} ({err:.1f}% error)")

e_total = np.sum(spectral_energies, axis=1)
frac_energies = spectral_energies / e_total[:, None]
mean_frac = np.mean(frac_energies, axis=0)
print(f"\n  Fractional spectral energies:")
for j, bn in enumerate(band_names):
    print(f"    {bn}: {mean_frac[j]:.4f}")

phi_geometric = np.array([PHI**(-4), PHI**(-3), PHI**(-2), PHI**(-1), 1.0])
phi_geometric /= phi_geometric.sum()
frac_err = np.mean(np.abs(mean_frac - phi_geometric) / phi_geometric) * 100
print(f"  Error from phi-geometric series: {frac_err:.1f}%")
record_test('frac_energy_phi_geometric_err', frac_err, '10B')

freq_ratio_errs = []
energy_ratio_errs = []
for j in range(4):
    freq_r = np.mean(all_adj_ratios[:, j])
    energy_r = np.mean(energy_ratios[:, j])
    freq_ratio_errs.append(abs(freq_r - PHI) / PHI * 100)
    energy_ratio_errs.append(abs(energy_r - PHI) / PHI * 100)

print(f"\n  Frequency ratio mean phi-error: {np.mean(freq_ratio_errs):.1f}%")
print(f"  Energy ratio mean phi-error: {np.mean(energy_ratio_errs):.1f}%")
energy_tighter = np.sum(np.array(energy_ratio_errs) < np.array(freq_ratio_errs))
print(f"  Energy ratios tighter for {energy_tighter}/4 pairs")

results['10B'] = {
    'energy_ratios': {energy_ratio_names[j]: float(np.mean(energy_ratios[:, j])) for j in range(4)},
    'frac_energies': {band_names[j]: float(mean_frac[j]) for j in range(5)},
    'frac_phi_geometric_error_pct': float(frac_err),
    'freq_ratio_mean_err': float(np.mean(freq_ratio_errs)),
    'energy_ratio_mean_err': float(np.mean(energy_ratio_errs)),
    'energy_tighter_count': int(energy_tighter)
}
print()


print("=" * 70)
print("BLOCK 10C: AMPLITUDE-FREQUENCY COUPLING (1/f Residuals)")
print("=" * 70)

residual_ratios_all = []
phi_residual_indices = []

for i in range(N):
    c = all_centroids[i]
    centroids = np.array([c[bn] for bn in band_names])
    powers = all_band_powers[i]

    log_f = np.log10(centroids)
    log_p = np.log10(powers + 1e-30)

    coeffs = np.polyfit(log_f, log_p, 1)
    fitted = np.polyval(coeffs, log_f)
    residuals = log_p - fitted

    for j in range(4):
        if residuals[j] != 0:
            rr = residuals[j+1] / residuals[j]
            if np.isfinite(rr) and 0.01 < abs(rr) < 100:
                residual_ratios_all.append(rr)

    phi_pattern = np.array([(-1)**k * PHI**(-k) for k in range(5)])
    phi_pattern /= np.linalg.norm(phi_pattern)
    res_norm = residuals / (np.linalg.norm(residuals) + 1e-30)
    phi_res_idx = abs(np.dot(res_norm, phi_pattern))
    phi_residual_indices.append(phi_res_idx)

phi_residual_indices = np.array(phi_residual_indices)

if len(residual_ratios_all) > 0:
    residual_ratios_all = np.array(residual_ratios_all)
    valid = np.isfinite(residual_ratios_all) & (np.abs(residual_ratios_all) < 50)
    residual_ratios_all = residual_ratios_all[valid]
    if len(residual_ratios_all) > 0:
        mean_rr = np.mean(residual_ratios_all)
        name, err = record_test('residual_ratio_mean', mean_rr, '10C')
        print(f"  Mean residual ratio: {mean_rr:.4f} -> {name} ({err:.1f}%)")

r_pri, p_pri = pearsonr(phi_residual_indices, all_pcis)
print(f"  Phi-residual index vs PCI: r={r_pri:.3f}, p={p_pri:.4f}")
record_test('phi_residual_vs_pci_r', abs(r_pri), '10C')

results['10C'] = {
    'mean_residual_ratio': float(mean_rr) if len(residual_ratios_all) > 0 else None,
    'phi_residual_vs_pci_r': float(r_pri),
    'phi_residual_vs_pci_p': float(p_pri),
    'mean_phi_residual_index': float(np.mean(phi_residual_indices))
}
print()


print("=" * 70)
print("BLOCK 10D: CROSS-BAND ASYMMETRY RATIOS")
print("=" * 70)

phi_pair = np.column_stack([all_adj_ratios[:, 1], all_adj_ratios[:, 3]])
harm_pair = np.column_stack([all_adj_ratios[:, 0], all_adj_ratios[:, 2]])
phi_pair_mean = np.mean(phi_pair, axis=1)
harm_pair_mean = np.mean(harm_pair, axis=1)
asymmetry = phi_pair_mean / harm_pair_mean

mean_asym = np.mean(asymmetry)
name_a, err_a = record_test('asymmetry_phi_vs_harm', mean_asym, '10D')
print(f"  Mean asymmetry (phi-pair/harm-pair): {mean_asym:.4f} -> {name_a} ({err_a:.1f}%)")

diag1 = all_adj_ratios[:, 1] / all_adj_ratios[:, 2]
diag2 = all_adj_ratios[:, 3] / all_adj_ratios[:, 0]
mean_d1 = np.mean(diag1)
mean_d2 = np.mean(diag2)
record_test('diagonal_alpha_theta_over_beta_alpha', mean_d1, '10D')
record_test('diagonal_gamma_beta_over_theta_delta', mean_d2, '10D')
print(f"  Diagonal 1 (a/t)/(b/a): {mean_d1:.4f}")
print(f"  Diagonal 2 (g/b)/(t/d): {mean_d2:.4f}")

double_diag = (all_adj_ratios[:, 1] * all_adj_ratios[:, 3]) / (all_adj_ratios[:, 0] * all_adj_ratios[:, 2])
mean_dd = np.mean(double_diag)
name_dd, err_dd = record_test('double_diagonal', mean_dd, '10D')
print(f"  Double diagonal (phi-product / harm-product): {mean_dd:.4f} -> {name_dd} ({err_dd:.1f}%)")

results['10D'] = {
    'asymmetry': float(mean_asym), 'asymmetry_nearest': name_a, 'asymmetry_err': float(err_a),
    'diagonal_1': float(mean_d1), 'diagonal_2': float(mean_d2),
    'double_diagonal': float(mean_dd), 'double_diagonal_nearest': name_dd
}
print()


print("=" * 70)
print("BLOCK 10E: SUBJECT x BAND INTERACTION (PCA/SVD)")
print("=" * 70)

pca = PCA(n_components=4)
pca_scores = pca.fit_transform(all_adj_ratios)
explained = pca.explained_variance_ratio_
loadings = pca.components_

print(f"  PCA explained variance: {', '.join(f'{v:.3f}' for v in explained)}")
print(f"  Cumulative: {', '.join(f'{v:.3f}' for v in np.cumsum(explained))}")

for j in range(min(3, len(explained))):
    print(f"  PC{j+1} loadings: {', '.join(f'{band_names[k+1]}/{band_names[k]}={loadings[j,k]:.3f}' for k in range(4))}")

sv_ratios = []
svs = np.sqrt(pca.explained_variance_)
for j in range(len(svs)-1):
    if svs[j+1] > 0:
        sv_r = svs[j] / svs[j+1]
        name_sv, err_sv = record_test(f'sv{j+1}/sv{j+2}', sv_r, '10E')
        sv_ratios.append({'ratio': f'sigma{j+1}/sigma{j+2}', 'value': float(sv_r), 'nearest': name_sv, 'error': float(err_sv)})
        print(f"  Singular value ratio sigma{j+1}/sigma{j+2}: {sv_r:.4f} -> {name_sv} ({err_sv:.1f}%)")

r_pc1_pci, p_pc1_pci = pearsonr(pca_scores[:, 0], all_pcis)
print(f"  PC1 vs PCI: r={r_pc1_pci:.3f}, p={p_pc1_pci:.4f}")

results['10E'] = {
    'explained_variance': [float(v) for v in explained],
    'sv_ratios': sv_ratios,
    'pc1_vs_pci_r': float(r_pc1_pci), 'pc1_vs_pci_p': float(p_pc1_pci),
    'loadings': {f'PC{j+1}': {adj_ratio_names[k]: float(loadings[j,k]) for k in range(4)} for j in range(min(3, len(explained)))}
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(range(4), explained, color='steelblue')
axes[0].set_xticks(range(4))
axes[0].set_xticklabels([f'PC{j+1}' for j in range(4)])
axes[0].set_ylabel('Explained Variance')
axes[0].set_title('10E: PCA of Adjacent Ratio Matrix')
axes[1].scatter(pca_scores[:, 0], pca_scores[:, 1], c=all_pcis, cmap='coolwarm', s=50)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('10E: Subjects in PC1-PC2 space (color=PCI)')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='PCI')
plt.tight_layout()
plt.savefig('outputs/fig_10E_pca.png', dpi=150)
plt.close()
print()


print("=" * 70)
print("BLOCK 10F: TEMPORAL x SPECTRAL DIAGONAL")
print("=" * 70)

win_sec = 4.0
overlap = 0.5
step_sec = win_sec * (1 - overlap)

autocorr_lags_phi = []
pci_fluct_freqs = []
band_pci_corrs = {bn: [] for bn in band_names}

for i in range(N):
    data = subjects[i]['data']
    fs = subjects[i]['fs']
    n_samp = data.shape[1]
    win_samp = int(win_sec * fs)
    step_samp = int(step_sec * fs)

    if n_samp < win_samp * 2:
        continue

    max_ch = min(data.shape[0], 10)
    pci_ts = []
    band_power_ts = {bn: [] for bn in band_names}
    starts = range(0, n_samp - win_samp, step_samp)

    for s in starts:
        seg = data[:max_ch, s:s+win_samp]
        nperseg = min(int(2 * fs), win_samp)
        avg_psd = None
        for ch in range(seg.shape[0]):
            f, p = welch(seg[ch], fs=fs, nperseg=nperseg)
            if avg_psd is None:
                avg_psd = p.copy()
            else:
                avg_psd += p
        avg_psd /= seg.shape[0]

        tc = spectral_centroid(f, avg_psd, 4, 8)
        ac = spectral_centroid(f, avg_psd, 8, 13)
        ratio = ac / tc if tc > 0 else 1.0
        pci_ts.append(compute_pci(ratio))

        for j, (lo, hi) in enumerate(band_ranges):
            mask = (f >= lo) & (f <= hi)
            band_power_ts[band_names[j]].append(np.sum(avg_psd[mask]) * (f[1] - f[0]))

    pci_ts = np.array(pci_ts)

    if len(pci_ts) >= 10:
        pci_centered = pci_ts - np.mean(pci_ts)
        acf = np.correlate(pci_centered, pci_centered, mode='full')
        acf = acf[len(acf)//2:]
        acf /= acf[0] + 1e-30

        for lag in range(1, len(acf)):
            if acf[lag] <= INV_PHI:
                autocorr_lags_phi.append(lag * step_sec)
                break

        if len(pci_ts) >= 16:
            fft_pci = np.abs(np.fft.rfft(pci_centered))
            fft_freqs = np.fft.rfftfreq(len(pci_centered), d=step_sec)
            if len(fft_pci) > 1:
                peak_idx = np.argmax(fft_pci[1:]) + 1
                pci_fluct_freqs.append(fft_freqs[peak_idx])

        for bn in band_names:
            bp = np.array(band_power_ts[bn])
            if len(bp) == len(pci_ts):
                r_bp, _ = pearsonr(pci_ts, bp)
                band_pci_corrs[bn].append(r_bp)

if autocorr_lags_phi:
    mean_lag = np.mean(autocorr_lags_phi)
    print(f"  Mean lag to 1/phi autocorrelation: {mean_lag:.2f} sec ({len(autocorr_lags_phi)} subjects)")
else:
    mean_lag = float('nan')
    print(f"  No subjects reached 1/phi autocorrelation threshold")

if pci_fluct_freqs:
    mean_fluct = np.mean(pci_fluct_freqs)
    print(f"  Mean PCI fluctuation frequency: {mean_fluct:.4f} Hz")
    e_minus_phi_sq = np.e - PHI_SQ
    print(f"  e - phi^2 = {e_minus_phi_sq:.4f} Hz")
    err_fluct = abs(mean_fluct - e_minus_phi_sq) / abs(e_minus_phi_sq) * 100
    print(f"  Error: {err_fluct:.1f}%")
    record_test('pci_fluct_freq', mean_fluct, '10F')

print(f"  Band-PCI correlations (mean across subjects):")
for bn in band_names:
    corrs = band_pci_corrs[bn]
    if corrs:
        mean_c = np.mean(corrs)
        print(f"    {bn}: r={mean_c:.3f} (N={len(corrs)})")

results['10F'] = {
    'mean_lag_to_inv_phi': float(mean_lag) if np.isfinite(mean_lag) else None,
    'n_lag_subjects': len(autocorr_lags_phi),
    'mean_pci_fluct_freq': float(np.mean(pci_fluct_freqs)) if pci_fluct_freqs else None,
    'band_pci_corrs': {bn: float(np.mean(band_pci_corrs[bn])) if band_pci_corrs[bn] else None for bn in band_names}
}
print()


print("=" * 70)
print("BLOCK 10G: INTER-SUBJECT RATIO DISTANCES")
print("=" * 70)

dists = pdist(all_adj_ratios, metric='euclidean')
dist_matrix = squareform(dists)
mean_dist = np.mean(dists)
median_dist = np.median(dists)

name_d, err_d = record_test('mean_inter_subject_distance', mean_dist, '10G')
print(f"  Mean inter-subject distance: {mean_dist:.4f} -> {name_d} ({err_d:.1f}%)")
print(f"  Median: {median_dist:.4f}")

from scipy.stats import normaltest
stat_norm, p_norm = normaltest(dists)
print(f"  Distance normality: stat={stat_norm:.2f}, p={p_norm:.4f}")

km2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(all_adj_ratios)
labels = km2.labels_
c0 = all_adj_ratios[labels == 0].mean(axis=0)
c1 = all_adj_ratios[labels == 1].mean(axis=0)
centroid_dist = np.linalg.norm(c0 - c1)

within_0 = np.mean([np.linalg.norm(all_adj_ratios[j] - c0) for j in range(N) if labels[j] == 0])
within_1 = np.mean([np.linalg.norm(all_adj_ratios[j] - c1) for j in range(N) if labels[j] == 1])
mean_within = (within_0 + within_1) / 2

sep_ratio = centroid_dist / mean_within if mean_within > 0 else float('inf')
name_sep, err_sep = record_test('cluster_separation_ratio', sep_ratio, '10G')
print(f"  Cluster separation / within-spread: {sep_ratio:.4f} -> {name_sep} ({err_sep:.1f}%)")
print(f"  Centroid distance: {centroid_dist:.4f}, mean within: {mean_within:.4f}")

from scipy.sparse.csgraph import minimum_spanning_tree
mst = minimum_spanning_tree(squareform(dists))
mst_edges = mst.toarray()
edge_lengths = mst_edges[mst_edges > 0]
mean_edge = np.mean(edge_lengths) if len(edge_lengths) > 0 else 0
name_mst, err_mst = record_test('mst_mean_edge', mean_edge, '10G')
print(f"  MST mean edge: {mean_edge:.4f} -> {name_mst} ({err_mst:.1f}%)")

results['10G'] = {
    'mean_distance': float(mean_dist), 'median_distance': float(median_dist),
    'separation_ratio': float(sep_ratio), 'nearest_sep': name_sep, 'sep_error': float(err_sep),
    'mst_mean_edge': float(mean_edge)
}
print()


print("=" * 70)
print("BLOCK 10H: THE PRODUCT CHAIN")
print("=" * 70)

partial_products = {}
pp_names = [
    ('alpha/delta', [0, 1], 2),
    ('beta/theta', [1, 2], 2),
    ('gamma/alpha', [2, 3], 2),
    ('beta/delta', [0, 1, 2], 3),
    ('gamma/theta', [1, 2, 3], 3),
    ('gamma/delta', [0, 1, 2, 3], 4)
]

for label, indices, expected_n in pp_names:
    prod = np.prod(all_adj_ratios[:, indices], axis=1)
    mean_prod = np.mean(prod)
    expected_phi_n = PHI ** expected_n
    err = abs(mean_prod - expected_phi_n) / expected_phi_n * 100
    name_pp, err_pp = record_test(f'chain_{label}', mean_prod, '10H')
    print(f"  {label} (n={expected_n}): mean={mean_prod:.4f}, phi^{expected_n}={expected_phi_n:.4f}, error={err:.1f}%")
    partial_products[label] = {'mean': float(mean_prod), 'expected_phi_n': float(expected_phi_n), 'error_pct': float(err)}

adj_errors = all_adj_ratios - PHI
error_corrs = []
for j in range(3):
    r_ec, p_ec = pearsonr(adj_errors[:, j], adj_errors[:, j+1])
    error_corrs.append({'pair': f'{adj_ratio_names[j]} vs {adj_ratio_names[j+1]}', 'r': float(r_ec), 'p': float(p_ec)})
    print(f"  Error correlation {adj_ratio_names[j]} vs {adj_ratio_names[j+1]}: r={r_ec:.3f}, p={p_ec:.4f}")
    record_test(f'error_corr_{j}_{j+1}', abs(r_ec), '10H')

n_negative = sum(1 for ec in error_corrs if ec['r'] < 0)
print(f"  Anti-correlated error pairs: {n_negative}/3")
print(f"  {'HOMEOSTATIC mechanism suggested' if n_negative >= 2 else 'No homeostasis detected'}")

results['10H'] = {
    'partial_products': partial_products,
    'error_correlations': error_corrs,
    'n_anticorrelated': n_negative
}
print()


print("=" * 70)
print("BLOCK 10I: THE GOLDEN CONSTANT HUNT (Systematic)")
print("=" * 70)

print("  --- Ratios of statistics ---")
mean_at = np.mean(all_ratios)
std_at = np.std(all_ratios)
if std_at > 0:
    snr = mean_at / std_at
    record_test('alpha_theta_SNR', snr, '10I')
    print(f"  Alpha/theta SNR (mean/std): {snr:.4f}")

mean_pci_val = np.mean(all_pcis)
std_pci_val = np.std(all_pcis)
if std_pci_val > 0:
    pci_snr = abs(mean_pci_val) / std_pci_val
    record_test('pci_SNR', pci_snr, '10I')
    print(f"  PCI SNR: {pci_snr:.4f}")

print("\n  --- Geometric means ---")
gm_all = np.exp(np.mean(np.log(all_adj_ratios), axis=1))
mean_gm_all = np.mean(gm_all)
record_test('geom_mean_all_adj', mean_gm_all, '10I')
print(f"  GM(all 4 adjacent): {mean_gm_all:.4f}")

gm_phi = np.exp(np.mean(np.log(all_adj_ratios[:, [1, 3]]), axis=1))
gm_harm = np.exp(np.mean(np.log(all_adj_ratios[:, [0, 2]]), axis=1))
mean_gm_phi = np.mean(gm_phi)
mean_gm_harm = np.mean(gm_harm)
gm_ratio = mean_gm_phi / mean_gm_harm if mean_gm_harm > 0 else float('inf')
record_test('GM_phi_pair', mean_gm_phi, '10I')
record_test('GM_harm_pair', mean_gm_harm, '10I')
record_test('GM_phi/GM_harm', gm_ratio, '10I')
print(f"  GM(phi-pair): {mean_gm_phi:.4f}")
print(f"  GM(harm-pair): {mean_gm_harm:.4f}")
print(f"  GM ratio: {gm_ratio:.4f}")

print("\n  --- Entropic measures ---")
for i in range(N):
    psd = all_psds[i]
    freqs = all_freqs[i]
    mask = (freqs >= 1) & (freqs <= 45)
    p_norm = psd[mask] / np.sum(psd[mask])
    p_norm = p_norm[p_norm > 0]
    se = -np.sum(p_norm * np.log(p_norm))
    quantities[i]['spectral_entropy'] = se

se_vals = [quantities[i].get('spectral_entropy', np.nan) for i in range(N)]
se_vals = np.array([v for v in se_vals if np.isfinite(v)])
if len(se_vals) > 0:
    mean_se = np.mean(se_vals)
    log_phi = np.log(PHI)
    se_ratio = mean_se / log_phi if log_phi > 0 else float('inf')
    record_test('spectral_entropy', mean_se, '10I')
    record_test('entropy/log_phi', se_ratio, '10I')
    print(f"  Mean spectral entropy: {mean_se:.4f}")
    print(f"  Entropy / log(phi): {se_ratio:.4f}")

print("\n  --- Dimensionless combinations ---")
dev_prod = (all_adj_ratios[:, 1] - 1) * (all_adj_ratios[:, 0] - 1)
mean_dev_prod = np.mean(dev_prod)
record_test('(a/t-1)*(t/d-1)', mean_dev_prod, '10I')
print(f"  (alpha/theta - 1) * (theta/delta - 1): {mean_dev_prod:.4f}")

log_ratio = np.log(all_adj_ratios[:, 1]) / np.log(np.maximum(all_adj_ratios[:, 0], 1.01))
mean_log_ratio = np.mean(log_ratio)
record_test('log(a/t)/log(t/d)', mean_log_ratio, '10I')
print(f"  log(alpha/theta) / log(theta/delta): {mean_log_ratio:.4f}")

results['10I'] = {
    'alpha_theta_snr': float(snr) if std_at > 0 else None,
    'pci_snr': float(pci_snr) if std_pci_val > 0 else None,
    'gm_all_adj': float(mean_gm_all),
    'gm_phi_pair': float(mean_gm_phi),
    'gm_harm_pair': float(mean_gm_harm),
    'gm_ratio': float(gm_ratio),
    'mean_spectral_entropy': float(mean_se) if len(se_vals) > 0 else None,
}
print()


print("=" * 70)
print("MULTIPLE COMPARISON CORRECTION")
print("=" * 70)

n_total_tests = len(all_tests)
print(f"  Total comparisons made: {n_total_tests}")

within_3 = [t for t in all_tests if t['error_pct'] < 3.0]
within_5 = [t for t in all_tests if t['error_pct'] < 5.0]
within_3.sort(key=lambda x: x['error_pct'])
within_5.sort(key=lambda x: x['error_pct'])

print(f"  Within 3% of any constant: {len(within_3)}")
print(f"  Within 5% of any constant: {len(within_5)}")

n_constants = len(CONSTANTS)
expected_within_3_pct = 2 * 0.03 * n_constants
expected_within_5_pct = 2 * 0.05 * n_constants

for t in within_3:
    value = t['value']
    target_val = CONSTANTS.get(t['nearest'], PHI)
    if target_val == 0:
        continue
    n_subj_key = None
    for i in range(N):
        break
    uncorrected_p = None

    print(f"  HIT: {t['label']} = {t['value']:.4f} -> {t['nearest']} ({t['error_pct']:.2f}%)")

print(f"\n  Bonferroni threshold: 0.05 / {n_total_tests} = {0.05/n_total_tests:.6f}")
print(f"  With {n_total_tests} comparisons against {n_constants} constants,")
print(f"  chance of at least one <3% match: ~{1-(1-0.06)**n_total_tests:.1f}")
print(f"  (Most 'hits' are expected by chance alone)")

results['multiple_comparisons'] = {
    'total_tests': n_total_tests,
    'n_constants': n_constants,
    'within_3pct': len(within_3),
    'within_5pct': len(within_5),
    'bonferroni_threshold': float(0.05 / n_total_tests),
    'top_hits': within_3[:20]
}


print("\n\n" + "=" * 70)
print("BLOCK 10 COMPLETE — HIT LIST")
print("=" * 70)

print(f"\n  TOP 20 tightest matches (across all {n_total_tests} tests):")
all_tests.sort(key=lambda x: x['error_pct'])
for j, t in enumerate(all_tests[:20]):
    marker = " ***" if t['error_pct'] < 1.0 else ""
    print(f"  {j+1:2d}. {t['label']}: {t['value']:.4f} -> {t['nearest']} ({t['error_pct']:.2f}%){marker}")


with open('outputs/block10_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nResults saved: outputs/block10_results.json")
print(f"Figure: outputs/fig_10E_pca.png")

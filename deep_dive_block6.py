import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from deep_dive_common import *

print("=" * 70)
print("BLOCK 6: CROSS-FREQUENCY COUPLING")
print("=" * 70)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
n = len(subjects)

print("\n--- 6a: Phase-Amplitude Coupling (Theta-Gamma) ---")

def compute_pac_mi(data, fs, phase_band=(4, 8), amp_band=(30, 45), n_bins=18):
    nyq = fs / 2.0
    b_phase, a_phase = butter(4, [phase_band[0]/nyq, phase_band[1]/nyq], btype='band')
    b_amp, a_amp = butter(4, [amp_band[0]/nyq, amp_band[1]/nyq], btype='band')

    n_ch = min(data.shape[0], 10)
    mi_values = []
    for ch in range(n_ch):
        sig = data[ch]
        phase_sig = filtfilt(b_phase, a_phase, sig)
        amp_sig = filtfilt(b_amp, a_amp, sig)
        phase_analytic = np.angle(hilbert(phase_sig))
        amp_envelope = np.abs(hilbert(amp_sig))

        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp = np.zeros(n_bins)
        for b in range(n_bins):
            idx = (phase_analytic >= bin_edges[b]) & (phase_analytic < bin_edges[b + 1])
            if np.sum(idx) > 0:
                mean_amp[b] = np.mean(amp_envelope[idx])
        mean_amp_norm = mean_amp / (np.sum(mean_amp) + 1e-20)
        uniform = np.ones(n_bins) / n_bins
        kl = np.sum(mean_amp_norm * np.log((mean_amp_norm + 1e-20) / uniform))
        mi = kl / np.log(n_bins)
        mi_values.append(mi)
    return np.mean(mi_values)

pac_mi = np.zeros(n)
for i, subj in enumerate(subjects):
    data = subj['data']
    fs = subj['fs']
    max_samp = min(data.shape[1], int(30 * fs))
    pac_mi[i] = compute_pac_mi(data[:, :max_samp], fs)

r_pac_pci, p_pac_pci = pearsonr(pac_mi, all_pcis)
r_pac_ratio, p_pac_ratio = pearsonr(pac_mi, all_ratios)
print(f"  Mean PAC MI: {np.mean(pac_mi):.6f} (SD={np.std(pac_mi):.6f})")
print(f"  PAC vs PCI: r={r_pac_pci:.4f}, p={p_pac_pci:.4f}")
print(f"  PAC vs ratio: r={r_pac_ratio:.4f}, p={p_pac_ratio:.4f}")

print("\n--- 6b: Theta-Gamma Ratio (vs phi^4) ---")
phi4 = PHI**4
tg_ratios = np.array([all_centroids[i]['gamma'] / all_centroids[i]['theta'] for i in range(n)])
tg_err = abs(np.mean(tg_ratios) - phi4) / phi4 * 100
from scipy.stats import ttest_1samp
t_tg, p_tg = ttest_1samp(tg_ratios, phi4)
print(f"  Mean gamma/theta: {np.mean(tg_ratios):.4f} (SD={np.std(tg_ratios):.4f})")
print(f"  Expected phi^4: {phi4:.4f}")
print(f"  Error: {tg_err:.1f}%")
print(f"  t-test vs phi^4: t={t_tg:.4f}, p={p_tg:.6f}")

print("\n--- 6c: Full Pairwise Ratio Matrix ---")
band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
nb = len(band_names)
ratio_matrix = np.zeros((nb, nb))
for i_b in range(nb):
    for j_b in range(nb):
        if i_b == j_b:
            ratio_matrix[i_b, j_b] = 1.0
            continue
        vals = np.array([all_centroids[s][band_names[j_b]] / all_centroids[s][band_names[i_b]] for s in range(n)])
        ratio_matrix[i_b, j_b] = np.mean(vals)

print("  Pairwise ratio matrix (row/col):")
print(f"  {'':>8}", end='')
for name in band_names:
    print(f"{name:>8}", end='')
print()
for i_b in range(nb):
    print(f"  {band_names[i_b]:>8}", end='')
    for j_b in range(nb):
        print(f"{ratio_matrix[i_b, j_b]:>8.3f}", end='')
    print()

phi_n_targets = {1: PHI, 2: PHI**2, 3: PHI**3, 4: PHI**4}
phi_match = np.zeros((nb, nb))
for i_b in range(nb):
    for j_b in range(nb):
        if i_b == j_b:
            continue
        r = ratio_matrix[i_b, j_b]
        best_n = min(phi_n_targets.items(), key=lambda x: abs(r - x[1]))
        phi_match[i_b, j_b] = abs(r - best_n[1]) / best_n[1] * 100

print("\n  Best phi^n match errors (%):")
print(f"  {'':>8}", end='')
for name in band_names:
    print(f"{name:>8}", end='')
print()
for i_b in range(nb):
    print(f"  {band_names[i_b]:>8}", end='')
    for j_b in range(nb):
        if i_b == j_b:
            print(f"{'---':>8}", end='')
        else:
            print(f"{phi_match[i_b, j_b]:>7.1f}%", end='')
    print()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.scatter(pac_mi, all_pcis, c='steelblue', s=50, alpha=0.7, edgecolors='black')
ax.set_xlabel('PAC Modulation Index')
ax.set_ylabel('PCI')
ax.set_title(f'6a: PAC vs PCI\nr={r_pac_pci:.3f}, p={p_pac_pci:.4f}', fontweight='bold')

ax = axes[0, 1]
ax.hist(tg_ratios, bins=15, color='#F39C12', alpha=0.7, edgecolor='black')
ax.axvline(phi4, color='gold', linewidth=2, linestyle='--', label=f'phi^4={phi4:.3f}')
ax.axvline(np.mean(tg_ratios), color='blue', linewidth=2, linestyle='-', label=f'mean={np.mean(tg_ratios):.3f}')
ax.set_xlabel('Gamma/Theta Ratio')
ax.set_ylabel('Count')
ax.set_title(f'6b: Gamma/Theta vs phi^4\nerror={tg_err:.1f}%', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[0, 2]
im = ax.imshow(ratio_matrix, cmap='RdYlBu_r', aspect='auto')
ax.set_xticks(range(nb))
ax.set_xticklabels(band_names, fontsize=9)
ax.set_yticks(range(nb))
ax.set_yticklabels(band_names, fontsize=9)
plt.colorbar(im, ax=ax, label='Mean Ratio')
ax.set_title('6c: Full Pairwise Ratio Matrix', fontweight='bold')
for i_b in range(nb):
    for j_b in range(nb):
        ax.text(j_b, i_b, f'{ratio_matrix[i_b, j_b]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if ratio_matrix[i_b, j_b] > 5 else 'black')

ax = axes[1, 0]
ax.scatter(pac_mi, all_ratios, c='coral', s=50, alpha=0.7, edgecolors='black')
ax.set_xlabel('PAC Modulation Index')
ax.set_ylabel('Alpha/Theta Ratio')
ax.axhline(PHI, color='gold', linewidth=1.5, linestyle='--')
ax.set_title(f'6a: PAC vs Ratio\nr={r_pac_ratio:.3f}, p={p_pac_ratio:.4f}', fontweight='bold')

ax = axes[1, 1]
im2 = ax.imshow(phi_match, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(nb))
ax.set_xticklabels(band_names, fontsize=9)
ax.set_yticks(range(nb))
ax.set_yticklabels(band_names, fontsize=9)
plt.colorbar(im2, ax=ax, label='Error from best phi^n (%)')
ax.set_title('6c: Phi^n Match Errors', fontweight='bold')
for i_b in range(nb):
    for j_b in range(nb):
        if i_b == j_b:
            ax.text(j_b, i_b, '---', ha='center', va='center', fontsize=7)
        else:
            ax.text(j_b, i_b, f'{phi_match[i_b, j_b]:.0f}%', ha='center', va='center', fontsize=7)

ax = axes[1, 2]
ax.hist(pac_mi, bins=15, color='#9B59B6', alpha=0.7, edgecolor='black')
ax.set_xlabel('PAC Modulation Index')
ax.set_ylabel('Count')
ax.set_title(f'PAC MI Distribution (mean={np.mean(pac_mi):.5f})', fontweight='bold')

plt.suptitle(f'Block 6: Cross-Frequency Coupling (N={n})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/block6_cross_freq.png', dpi=150, bbox_inches='tight')
plt.close()

results = {
    'n_subjects': n,
    'pac_mi_mean': float(np.mean(pac_mi)),
    'pac_mi_sd': float(np.std(pac_mi)),
    'pac_vs_pci_r': float(r_pac_pci),
    'pac_vs_pci_p': float(p_pac_pci),
    'pac_vs_ratio_r': float(r_pac_ratio),
    'pac_vs_ratio_p': float(p_pac_ratio),
    'tg_ratio_mean': float(np.mean(tg_ratios)),
    'tg_ratio_sd': float(np.std(tg_ratios)),
    'tg_phi4': float(phi4),
    'tg_err_pct': float(tg_err),
    'tg_t': float(t_tg),
    'tg_p': float(p_tg),
    'ratio_matrix': ratio_matrix.tolist(),
}
with open('outputs/block6_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('outputs/block6_results.md', 'w') as f:
    f.write(f"# Block 6: Cross-Frequency Coupling (N={n})\n\n")
    f.write("## 6a: Phase-Amplitude Coupling\n\n")
    f.write(f"- Mean theta-gamma PAC MI: {np.mean(pac_mi):.6f}\n")
    f.write(f"- PAC vs PCI: r={r_pac_pci:.4f}, p={p_pac_pci:.4f}\n")
    f.write(f"- PAC vs ratio: r={r_pac_ratio:.4f}, p={p_pac_ratio:.4f}\n")
    if abs(r_pac_pci) > 0.3 and p_pac_pci < 0.05:
        f.write("- **Significant PAC-PCI correlation**: phi-organized subjects show different CFC\n")
    else:
        f.write("- No significant PAC-PCI correlation\n")
    f.write("\n## 6b: Gamma/Theta Ratio\n\n")
    f.write(f"- Mean: {np.mean(tg_ratios):.4f}\n")
    f.write(f"- Expected phi^4: {phi4:.4f}\n")
    f.write(f"- Error: {tg_err:.1f}%\n")
    f.write(f"- t-test: t={t_tg:.4f}, p={p_tg:.6f}\n\n")
    f.write("## 6c: Pairwise Ratio Matrix\n\n")
    f.write("See outputs/block6_cross_freq.png for heatmaps.\n\n")
    f.write("## Interpretation\n\n")
    if abs(r_pac_pci) > 0.3 and p_pac_pci < 0.05:
        f.write("PAC correlates with phi-organization, providing functional validation. ")
        f.write("Subjects closer to phi show different cross-frequency coupling patterns.\n")
    else:
        f.write("PAC does not significantly correlate with phi-organization. ")
        f.write("The spectral ratio structure is independent of cross-frequency coupling dynamics.\n")
    f.write("\n## Implication for Paper\n\n")
    f.write("Cross-frequency coupling is theoretically linked to phi-organization via nested oscillations. ")
    f.write("The PAC-PCI correlation (or lack thereof) directly tests this claim.\n")

print("\nResults saved: outputs/block6_results.json, outputs/block6_results.md")

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

from deep_dive_common import *

print("=" * 70)
print("BLOCK 1: PER-DATASET SURROGATE ANALYSIS")
print("=" * 70)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
n = len(subjects)

alpha_s_idx = [i for i, s in enumerate(subjects) if s['name'].startswith('alpha_s') and not s['name'].startswith('alpha_subj')]
alpha_subj_idx = [i for i, s in enumerate(subjects) if s['name'].startswith('alpha_subj')]
print(f"Dataset A (alpha_s): N={len(alpha_s_idx)}")
print(f"Dataset B (alpha_subj): N={len(alpha_subj_idx)}")

N_SURROGATES = 200
np.random.seed(42)

cropped_data_cache = {}
for idx, subj in enumerate(subjects):
    data = subj['data']
    fs = subj['fs']
    max_ch = min(data.shape[0], 10)
    max_samp = min(data.shape[1], int(15 * fs))
    cropped_data_cache[idx] = (data[:max_ch, :max_samp], fs)

def run_surrogate_test(indices, label, n_surr=N_SURROGATES):
    n_sub = len(indices)
    obs_ratios = all_ratios[indices]
    obs_pcis = all_pcis[indices]
    obs_mean_phi_err = np.mean(np.abs(obs_ratios - PHI))
    obs_pci_rate = np.mean(obs_pcis > 0) * 100

    surr_mean_errs = np.zeros(n_surr)
    surr_pci_rates = np.zeros(n_surr)

    for s in range(n_surr):
        if (s + 1) % 50 == 0:
            print(f"  [{label}] Surrogate {s+1}/{n_surr}")
        surr_ratios = np.zeros(n_sub)
        surr_pcis_arr = np.zeros(n_sub)
        for j, idx in enumerate(indices):
            cropped, fs = cropped_data_cache[idx]
            surr_data = phase_randomize_shared(cropped)
            nperseg = min(int(4 * fs), surr_data.shape[1])
            avg_psd = None
            for ch in range(surr_data.shape[0]):
                freqs, psd = welch(surr_data[ch], fs=fs, nperseg=nperseg)
                if avg_psd is None:
                    avg_psd = psd.copy()
                else:
                    avg_psd += psd
            avg_psd /= surr_data.shape[0]
            tc = spectral_centroid(freqs, avg_psd, 4, 8)
            ac = spectral_centroid(freqs, avg_psd, 8, 13)
            r = ac / tc
            surr_ratios[j] = r
            surr_pcis_arr[j] = compute_pci(r)
        surr_mean_errs[s] = np.mean(np.abs(surr_ratios - PHI))
        surr_pci_rates[s] = np.mean(surr_pcis_arr > 0) * 100

    z = (obs_mean_phi_err - np.mean(surr_mean_errs)) / max(np.std(surr_mean_errs), 1e-10)
    p_phi = np.mean(surr_mean_errs <= obs_mean_phi_err)
    p_pci = np.mean(surr_pci_rates >= obs_pci_rate)

    return {
        'label': label,
        'n': n_sub,
        'obs_mean_phi_err': float(obs_mean_phi_err),
        'surr_mean_phi_err': float(np.mean(surr_mean_errs)),
        'surr_sd_phi_err': float(np.std(surr_mean_errs)),
        'z_score': float(z),
        'p_phi': float(p_phi),
        'obs_pci_rate': float(obs_pci_rate),
        'surr_pci_rate_mean': float(np.mean(surr_pci_rates)),
        'p_pci': float(p_pci),
        'obs_mean_ratio': float(np.mean(obs_ratios)),
        'surr_mean_errs': surr_mean_errs.tolist(),
        'surr_pci_rates': surr_pci_rates.tolist(),
    }

print("\n--- Running surrogates for Dataset A (alpha_s) ---")
res_a = run_surrogate_test(alpha_s_idx, 'alpha_s')
print(f"  Z={res_a['z_score']:.3f}, p(phi)={res_a['p_phi']:.4f}, p(PCI)={res_a['p_pci']:.4f}")

print("\n--- Running surrogates for Dataset B (alpha_subj) ---")
res_b = run_surrogate_test(alpha_subj_idx, 'alpha_subj')
print(f"  Z={res_b['z_score']:.3f}, p(phi)={res_b['p_phi']:.4f}, p(PCI)={res_b['p_pci']:.4f}")

print("\n--- Running surrogates for Combined (all subjects) ---")
res_all = run_surrogate_test(list(range(n)), 'combined')
print(f"  Z={res_all['z_score']:.3f}, p(phi)={res_all['p_phi']:.4f}, p(PCI)={res_all['p_pci']:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col, (res, label) in enumerate([(res_a, 'Dataset A (alpha_s)'), (res_b, 'Dataset B (alpha_subj)'), (res_all, 'Combined')]):
    ax = axes[0, col]
    ax.hist(res['surr_mean_errs'], bins=40, color='#3498DB', alpha=0.7, edgecolor='black')
    ax.axvline(res['obs_mean_phi_err'], color='red', linewidth=2.5, label=f'Observed')
    ax.set_xlabel('Mean |Ratio - Phi|')
    ax.set_ylabel('Count')
    ax.set_title(f'{label}\nZ={res["z_score"]:.2f}, p={res["p_phi"]:.4f}', fontweight='bold')
    ax.legend(fontsize=8)

    ax = axes[1, col]
    ax.hist(res['surr_pci_rates'], bins=40, color='#2ECC71', alpha=0.7, edgecolor='black')
    ax.axvline(res['obs_pci_rate'], color='red', linewidth=2.5, label=f'Observed={res["obs_pci_rate"]:.1f}%')
    ax.set_xlabel('PCI>0 Rate (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'PCI>0 Rate Test\np={res["p_pci"]:.4f}', fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle(f'Block 1: Per-Dataset Surrogate Analysis ({N_SURROGATES} surrogates)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/block1_per_dataset_surrogates.png', dpi=150, bbox_inches='tight')
plt.close()

for r in [res_a, res_b, res_all]:
    del r['surr_mean_errs']
    del r['surr_pci_rates']

results = {'n_surrogates': N_SURROGATES, 'dataset_a': res_a, 'dataset_b': res_b, 'combined': res_all}
with open('outputs/block1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('outputs/block1_results.md', 'w') as f:
    f.write(f"# Block 1: Per-Dataset Surrogate Analysis (N={n}, {N_SURROGATES} surrogates)\n\n")
    f.write("## Results\n\n")
    f.write("| Dataset | N | Obs |Ratio-Phi|| Surr Mean | Z | p(phi) | Obs PCI>0 | Surr PCI>0 | p(PCI) |\n")
    f.write("|---------|---|--------------|-----------|------|--------|-----------|------------|--------|\n")
    for r in [res_a, res_b, res_all]:
        f.write(f"| {r['label']} | {r['n']} | {r['obs_mean_phi_err']:.4f} | {r['surr_mean_phi_err']:.4f} | {r['z_score']:.2f} | {r['p_phi']:.4f} | {r['obs_pci_rate']:.1f}% | {r['surr_pci_rate_mean']:.1f}% | {r['p_pci']:.4f} |\n")
    f.write("\n## Interpretation\n\n")
    a_sig = res_a['p_phi'] < 0.05
    b_sig = res_b['p_phi'] < 0.05
    if a_sig and b_sig:
        f.write("Both datasets independently show significant phi-proximity, confirming the effect is robust and not an artifact of dataset mixing.\n")
    elif a_sig or b_sig:
        which = 'A' if a_sig else 'B'
        other = 'B' if a_sig else 'A'
        f.write(f"Only Dataset {which} shows significant phi-proximity. Dataset {other} does not replicate. ")
        f.write("The pooled significance may be driven by one dataset, raising concerns about generalizability.\n")
    else:
        f.write("Neither dataset independently shows significant phi-proximity. ")
        f.write("If the combined result was previously significant, it was likely an artifact of mixing heterogeneous datasets.\n")
    f.write("\n## Implication for Paper\n\n")
    f.write("Per-dataset analysis is essential. If both datasets replicate, the finding is robust. ")
    f.write("If not, the paper must report per-dataset results and discuss why replication failed.\n")

print("\nResults saved: outputs/block1_results.json, outputs/block1_results.md")

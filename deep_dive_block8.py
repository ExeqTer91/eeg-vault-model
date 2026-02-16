import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from deep_dive_common import *

print("=" * 70)
print("BLOCK 8: TEMPORAL DYNAMICS")
print("=" * 70)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
n = len(subjects)

print("\n--- 8a: Time-Resolved PCI ---")
WIN_SEC = 4
OVERLAP = 0.5

all_pci_traces = []
all_ratio_traces = []
pct_phi_org = np.zeros(n)
mean_dwell_phi = np.zeros(n)
mean_dwell_harm = np.zeros(n)
pci_variability = np.zeros(n)
pci_autocorr = np.zeros(n)

for i, subj in enumerate(subjects):
    data = subj['data']
    fs = subj['fs']
    win_samp = int(WIN_SEC * fs)
    step_samp = int(win_samp * (1 - OVERLAP))
    n_samp = data.shape[1]

    ratios_t = []
    pcis_t = []
    start = 0
    while start + win_samp <= n_samp:
        seg = data[:, start:start + win_samp]
        nperseg = min(int(2 * fs), seg.shape[1])
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
        r = ac / tc
        ratios_t.append(r)
        pcis_t.append(compute_pci(r))
        start += step_samp

    ratios_t = np.array(ratios_t)
    pcis_t = np.array(pcis_t)
    all_pci_traces.append(pcis_t)
    all_ratio_traces.append(ratios_t)

    if len(pcis_t) > 0:
        pct_phi_org[i] = np.mean(pcis_t > 0) * 100
        pci_variability[i] = np.std(pcis_t)

        states = (pcis_t > 0).astype(int)
        dwells_phi = []
        dwells_harm = []
        current = states[0]
        dwell = 1
        for j in range(1, len(states)):
            if states[j] == current:
                dwell += 1
            else:
                if current == 1:
                    dwells_phi.append(dwell)
                else:
                    dwells_harm.append(dwell)
                current = states[j]
                dwell = 1
        if current == 1:
            dwells_phi.append(dwell)
        else:
            dwells_harm.append(dwell)
        mean_dwell_phi[i] = np.mean(dwells_phi) if dwells_phi else len(states)
        mean_dwell_harm[i] = np.mean(dwells_harm) if dwells_harm else 0

        if len(pcis_t) > 2:
            ac_vals = np.correlate(pcis_t - np.mean(pcis_t), pcis_t - np.mean(pcis_t), mode='full')
            ac_vals = ac_vals[len(ac_vals) // 2:]
            if ac_vals[0] > 0:
                pci_autocorr[i] = ac_vals[1] / ac_vals[0]

print(f"  Mean % time in phi-org state: {np.mean(pct_phi_org):.1f}% (SD={np.std(pct_phi_org):.1f}%)")
print(f"  Mean PCI variability (SD): {np.mean(pci_variability):.4f}")
print(f"  Mean PCI autocorrelation (lag-1): {np.mean(pci_autocorr):.4f}")

print("\n--- 8b: Transition Dynamics ---")
n_transitions = np.zeros(n)
for i in range(n):
    trace = all_pci_traces[i]
    if len(trace) < 2:
        continue
    states = (trace > 0).astype(int)
    n_transitions[i] = np.sum(np.diff(states) != 0)

transition_rate = n_transitions / np.array([len(t) for t in all_pci_traces])
print(f"  Mean transitions per window: {np.mean(transition_rate):.4f}")
print(f"  Mean dwell in phi state: {np.mean(mean_dwell_phi):.2f} windows ({np.mean(mean_dwell_phi)*WIN_SEC*(1-OVERLAP):.1f}s)")
print(f"  Mean dwell in harmonic state: {np.mean(mean_dwell_harm):.2f} windows ({np.mean(mean_dwell_harm)*WIN_SEC*(1-OVERLAP):.1f}s)")

print("\n--- 8c: PCI Variability as Trait ---")
r_var_pci, p_var_pci = pearsonr(pci_variability, all_pcis)
r_var_ratio, p_var_ratio = pearsonr(pci_variability, all_ratios)
r_var_pct, p_var_pct = pearsonr(pci_variability, pct_phi_org)
print(f"  PCI variability vs PCI: r={r_var_pci:.4f}, p={p_var_pci:.4f}")
print(f"  PCI variability vs ratio: r={r_var_ratio:.4f}, p={p_var_ratio:.4f}")
print(f"  PCI variability vs % phi-org: r={r_var_pct:.4f}, p={p_var_pct:.4f}")

high_var = pci_variability > np.median(pci_variability)
print(f"  High-variability subjects (>{np.median(pci_variability):.3f}): mean ratio={np.mean(all_ratios[high_var]):.4f}")
print(f"  Low-variability subjects: mean ratio={np.mean(all_ratios[~high_var]):.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
for i in range(min(5, n)):
    trace = all_ratio_traces[i]
    t_axis = np.arange(len(trace)) * WIN_SEC * (1 - OVERLAP)
    ax.plot(t_axis, trace, alpha=0.7, linewidth=1, label=subjects[i]['name'][:10])
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label='phi')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Alpha/Theta Ratio')
ax.set_title('8a: Time-Resolved Ratios (5 subjects)', fontweight='bold')
ax.legend(fontsize=6, ncol=2)

ax = axes[0, 1]
ax.hist(pct_phi_org, bins=15, color='#2ECC71', alpha=0.7, edgecolor='black')
ax.set_xlabel('% Time in Phi-Org State')
ax.set_ylabel('Count')
ax.set_title(f'8a: Time in Phi State\nmean={np.mean(pct_phi_org):.1f}%', fontweight='bold')

ax = axes[0, 2]
ax.scatter(pci_variability, pct_phi_org, c='steelblue', s=50, alpha=0.7, edgecolors='black')
ax.set_xlabel('PCI Variability (SD)')
ax.set_ylabel('% Time Phi-Organized')
ax.set_title(f'8c: Variability vs Phi-Org\nr={r_var_pct:.3f}, p={p_var_pct:.4f}', fontweight='bold')

ax = axes[1, 0]
ax.bar(['Phi State', 'Harmonic State'],
       [np.mean(mean_dwell_phi) * WIN_SEC * (1 - OVERLAP), np.mean(mean_dwell_harm) * WIN_SEC * (1 - OVERLAP)],
       color=['#2ECC71', '#E74C3C'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Mean Dwell Time (s)')
ax.set_title('8b: Mean Dwell Times', fontweight='bold')

ax = axes[1, 1]
ax.hist(pci_autocorr, bins=15, color='#F39C12', alpha=0.7, edgecolor='black')
ax.set_xlabel('PCI Autocorrelation (lag-1)')
ax.set_ylabel('Count')
ax.set_title(f'8a: PCI Stickiness\nmean={np.mean(pci_autocorr):.3f}', fontweight='bold')

ax = axes[1, 2]
ax.scatter(pci_variability, all_ratios, c='coral', s=50, alpha=0.7, edgecolors='black')
ax.axhline(PHI, color='gold', linewidth=1.5, linestyle='--')
ax.set_xlabel('PCI Variability')
ax.set_ylabel('Alpha/Theta Ratio')
ax.set_title(f'8c: Variability vs Ratio\nr={r_var_ratio:.3f}', fontweight='bold')

plt.suptitle(f'Block 8: Temporal Dynamics (N={n})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/block8_temporal.png', dpi=150, bbox_inches='tight')
plt.close()

results = {
    'n_subjects': n,
    'win_sec': WIN_SEC,
    'overlap': OVERLAP,
    'mean_pct_phi_org': float(np.mean(pct_phi_org)),
    'sd_pct_phi_org': float(np.std(pct_phi_org)),
    'mean_pci_variability': float(np.mean(pci_variability)),
    'mean_pci_autocorr': float(np.mean(pci_autocorr)),
    'mean_transition_rate': float(np.mean(transition_rate)),
    'mean_dwell_phi_windows': float(np.mean(mean_dwell_phi)),
    'mean_dwell_harm_windows': float(np.mean(mean_dwell_harm)),
    'mean_dwell_phi_sec': float(np.mean(mean_dwell_phi) * WIN_SEC * (1 - OVERLAP)),
    'mean_dwell_harm_sec': float(np.mean(mean_dwell_harm) * WIN_SEC * (1 - OVERLAP)),
    'var_vs_pci_r': float(r_var_pci),
    'var_vs_pci_p': float(p_var_pci),
    'var_vs_ratio_r': float(r_var_ratio),
    'var_vs_ratio_p': float(p_var_ratio),
    'var_vs_pct_r': float(r_var_pct),
    'var_vs_pct_p': float(p_var_pct),
}
with open('outputs/block8_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('outputs/block8_results.md', 'w') as f:
    f.write(f"# Block 8: Temporal Dynamics (N={n})\n\n")
    f.write("## 8a: Time-Resolved PCI\n\n")
    f.write(f"- Window: {WIN_SEC}s, {int(OVERLAP*100)}% overlap\n")
    f.write(f"- Mean % time phi-organized: {np.mean(pct_phi_org):.1f}% (SD={np.std(pct_phi_org):.1f}%)\n")
    f.write(f"- Mean PCI variability (SD): {np.mean(pci_variability):.4f}\n")
    f.write(f"- Mean PCI autocorrelation (lag-1): {np.mean(pci_autocorr):.4f}\n\n")
    f.write("## 8b: Transition Dynamics\n\n")
    f.write(f"- Mean transition rate: {np.mean(transition_rate):.4f} per window\n")
    f.write(f"- Mean dwell in phi state: {np.mean(mean_dwell_phi)*WIN_SEC*(1-OVERLAP):.1f}s\n")
    f.write(f"- Mean dwell in harmonic state: {np.mean(mean_dwell_harm)*WIN_SEC*(1-OVERLAP):.1f}s\n\n")
    f.write("## 8c: PCI Variability as Trait\n\n")
    f.write(f"- Variability vs PCI: r={r_var_pci:.4f}, p={p_var_pci:.4f}\n")
    f.write(f"- Variability vs ratio: r={r_var_ratio:.4f}, p={p_var_ratio:.4f}\n")
    f.write(f"- Variability vs % phi-org: r={r_var_pct:.4f}, p={p_var_pct:.4f}\n\n")
    f.write("## Interpretation\n\n")
    if np.mean(pci_autocorr) > 0.3:
        f.write("PCI shows moderate-to-high temporal autocorrelation, suggesting phi-organization ")
        f.write("is a 'sticky' state — once entered, subjects tend to stay. ")
    else:
        f.write("PCI shows low temporal autocorrelation, suggesting rapid fluctuation ")
        f.write("between phi-organized and harmonic states. ")
    f.write(f"The mean dwell time in the phi state is {np.mean(mean_dwell_phi)*WIN_SEC*(1-OVERLAP):.1f}s. ")
    if np.mean(pct_phi_org) > 50:
        f.write("Subjects spend more time phi-organized than not.\n")
    else:
        f.write("Subjects spend less time phi-organized — the harmonic state is more common temporally.\n")
    f.write("\n## Implication for Paper\n\n")
    f.write("Temporal dynamics reveal whether phi-organization is a stable trait or a transient state. ")
    f.write("High autocorrelation + long dwell times would support attractor dynamics. ")
    f.write("Low autocorrelation suggests random fluctuation around a mean ratio.\n")

print("\nResults saved: outputs/block8_results.json, outputs/block8_results.md")

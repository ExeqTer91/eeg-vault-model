#!/usr/bin/env python3
"""Job 3: Kuramoto parameter sweep — optimized version"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1
OUTPUT_DIR = 'outputs/publication_figures'

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})

def run_kuramoto(K, N_osc, noise_sigma, dt=0.005, T_total=3.0, fs_out=128):
    np.random.seed(hash((K, N_osc, int(noise_sigma*1000))) % 2**31)
    f_min, f_max = 2.0, 45.0
    freqs = f_min * (f_max / f_min) ** (np.arange(N_osc) / max(N_osc - 1, 1))
    omega = 2 * np.pi * freqs

    n_steps = int(T_total / dt)
    theta = np.random.uniform(0, 2 * np.pi, N_osc)

    t_out = np.arange(0, T_total, 1.0 / fs_out)
    signal = np.zeros(len(t_out))
    out_idx = 0

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

    freqs_psd, psd = welch(signal, fs=fs_out, nperseg=min(256, len(signal)))
    theta_mask = (freqs_psd >= 4) & (freqs_psd <= 8)
    alpha_mask = (freqs_psd >= 8) & (freqs_psd <= 13)

    if np.sum(psd[theta_mask]) < 1e-20 or np.sum(psd[alpha_mask]) < 1e-20:
        return None

    theta_centroid = np.sum(freqs_psd[theta_mask] * psd[theta_mask]) / np.sum(psd[theta_mask])
    alpha_centroid = np.sum(freqs_psd[alpha_mask] * psd[alpha_mask]) / np.sum(psd[alpha_mask])

    if theta_centroid < 1e-6:
        return None
    return alpha_centroid / theta_centroid

K_values = np.linspace(0.1, 5.0, 10)
N_values = [10, 20, 50, 100, 200]
sigma_values = [0.01, 0.05, 0.1, 0.5, 1.0]

results_grid = []
total = len(K_values) * len(N_values) * len(sigma_values)
count = 0

for K in K_values:
    for N_osc in N_values:
        for sigma in sigma_values:
            count += 1
            ratio = run_kuramoto(K, N_osc, sigma)
            if ratio is not None:
                pct_from_e1 = abs(ratio - E_MINUS_1) / E_MINUS_1 * 100
                results_grid.append({
                    'K': float(K), 'N': N_osc, 'sigma': float(sigma),
                    'ratio': float(ratio), 'pct_from_e1': float(pct_from_e1)
                })
            if count % 50 == 0:
                print(f"  {count}/{total} done")

print(f"Valid results: {len(results_grid)}/{total}")

valid_ratios = [r['ratio'] for r in results_grid]
pct_within_5 = np.mean([abs(r - E_MINUS_1) / E_MINUS_1 < 0.05 for r in valid_ratios]) * 100
pct_within_10 = np.mean([abs(r - E_MINUS_1) / E_MINUS_1 < 0.10 for r in valid_ratios]) * 100
pct_within_20 = np.mean([abs(r - E_MINUS_1) / E_MINUS_1 < 0.20 for r in valid_ratios]) * 100

K_corr = np.corrcoef([r['K'] for r in results_grid], [r['ratio'] for r in results_grid])[0, 1]
N_corr = np.corrcoef([r['N'] for r in results_grid], [r['ratio'] for r in results_grid])[0, 1]
sig_corr = np.corrcoef([r['sigma'] for r in results_grid], [r['ratio'] for r in results_grid])[0, 1]

sensitivity = {
    'K_correlation': float(K_corr),
    'N_correlation': float(N_corr),
    'sigma_correlation': float(sig_corr),
    'most_sensitive': max([('K', abs(K_corr)), ('N', abs(N_corr)), ('sigma', abs(sig_corr))], key=lambda x: x[1])[0]
}

sweep_results = {
    'grid_size': f"{len(K_values)} x {len(N_values)} x {len(sigma_values)} = {total}",
    'valid_results': len(results_grid),
    'mean_ratio': float(np.mean(valid_ratios)),
    'std_ratio': float(np.std(valid_ratios)),
    'pct_within_5_of_e1': float(pct_within_5),
    'pct_within_10_of_e1': float(pct_within_10),
    'pct_within_20_of_e1': float(pct_within_20),
    'sensitivity': sensitivity,
    'grid_results': results_grid
}

with open('outputs/kuramoto_sweep_results.json', 'w') as f:
    json.dump(sweep_results, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ni, N_osc in enumerate([20, 50, 100]):
    ax = axes[ni]
    subset = [r for r in results_grid if r['N'] == N_osc]
    if not subset:
        ax.set_title(f'N={N_osc} (no data)')
        continue

    K_vals = sorted(set(r['K'] for r in subset))
    sig_vals = sorted(set(r['sigma'] for r in subset))
    heatmap = np.full((len(sig_vals), len(K_vals)), np.nan)

    for r in subset:
        ki = K_vals.index(r['K'])
        si = sig_vals.index(r['sigma'])
        heatmap[si, ki] = r['ratio']

    im = ax.imshow(heatmap, cmap='RdYlBu_r', aspect='auto',
                   vmin=1.5, vmax=2.5, origin='lower',
                   extent=[K_vals[0], K_vals[-1], 0, len(sig_vals)])
    ax.set_yticks(np.arange(len(sig_vals)) + 0.5)
    ax.set_yticklabels([f'{s:.2f}' for s in sig_vals])
    ax.set_xlabel('Coupling K')
    ax.set_ylabel('Noise σ')
    ax.set_title(f'N = {N_osc} oscillators')

    for r in subset:
        ki = K_vals.index(r['K'])
        si = sig_vals.index(r['sigma'])
        if abs(r['ratio'] - E_MINUS_1) / E_MINUS_1 < 0.10:
            ax.plot(r['K'], si + 0.5, 'k*', markersize=6)

fig.suptitle(f'Figure 13: Kuramoto Parameter Sweep — α/θ Ratio\n'
             f'{pct_within_10:.1f}% within 10% of e−1, {pct_within_20:.1f}% within 20%',
             fontsize=13, fontweight='bold')
cbar = fig.colorbar(im, ax=axes, shrink=0.8, label='α/θ Ratio')
cbar.ax.axhline(E_MINUS_1, color='steelblue', lw=2, ls='--')
cbar.ax.axhline(PHI, color='goldenrod', lw=2, ls='--')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig13_kuramoto_sweep.png')
plt.close()

print(f"\nDone! Mean ratio: {np.mean(valid_ratios):.3f}")
print(f"  {pct_within_5:.1f}% within 5% of e-1")
print(f"  {pct_within_10:.1f}% within 10% of e-1")
print(f"  {pct_within_20:.1f}% within 20% of e-1")
print(f"  Most sensitive param: {sensitivity['most_sensitive']}")

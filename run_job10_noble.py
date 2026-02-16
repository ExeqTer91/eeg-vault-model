#!/usr/bin/env python3
"""
JOB 10: NOBLE NUMBER CONTROL ON KURAMOTO TEMPORAL COLLAPSE
Tests whether φ is specifically special for self-similar flux collapse
or if any noble number produces equally good collapse.

Noble numbers tested (continued fraction representations):
  φ   = [1; 1, 1, 1, ...] = 1.618...  (golden ratio)
  ψ_n = [1; 2, 1, 1, 1, ...] ≈ 1.7549 (near-noble)
  δ_s = [1; 2, 2, 2, ...] = 1+√2-1 = √2 ≈ 1.414 (silver ratio class)

Uses the self-similar cascade framework from Job 7:
builds scale ladders with each noble number as the ratio,
then compares flux distribution collapse quality via symmetric KL divergence.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1
SQRT2 = np.sqrt(2)
OUTPUT_DIR = 'outputs/publication_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})

def noble_from_cf(prefix, tail, n_terms=50):
    cf = list(prefix) + [tail] * (n_terms - len(prefix))
    val = cf[-1]
    for i in range(len(cf) - 2, -1, -1):
        val = cf[i] + 1.0 / val
    return val

NOBLE_NUMBERS = {
    'φ [1;1̄]': PHI,
    '[1;2,1̄]': noble_from_cf([1, 2], 1),
    '[1;3,1̄]': noble_from_cf([1, 3], 1),
    '[2;1̄]': noble_from_cf([2], 1),
    '√2 [1;2̄]': SQRT2,
    'e-1': E_MINUS_1,
    '2': 2.0,
}

print("=" * 60)
print("JOB 10: NOBLE NUMBER CONTROL — KURAMOTO TEMPORAL COLLAPSE")
print("=" * 60)
print(f"\n  Noble numbers tested:")
for name, val in NOBLE_NUMBERS.items():
    print(f"    {name:12s} = {val:.6f}")

FS_OUT = 1000

def run_kuramoto_trajectory(K, N_osc, noise_sigma, dt=0.001, T_total=40.0, seed_offset=0):
    np.random.seed(hash((int(K*100), N_osc, int(noise_sigma * 1000), seed_offset, 77)) % 2**31)
    f_min, f_max = 2.0, 45.0
    freqs = f_min * (f_max / f_min) ** (np.arange(N_osc) / max(N_osc - 1, 1))
    omega = 2 * np.pi * freqs

    n_steps = int(T_total / dt)
    theta = np.random.uniform(0, 2 * np.pi, N_osc)

    out_dt = 1.0 / FS_OUT
    n_out = int(T_total * FS_OUT)
    r_out = np.zeros(n_out)
    out_idx = 0
    next_out_t = 0.0

    for step in range(n_steps):
        t = step * dt
        z = np.mean(np.exp(1j * theta))
        r_abs = np.abs(z)
        r_angle = np.angle(z)
        coupling = K * r_abs * np.sin(r_angle - theta)
        noise_term = noise_sigma * np.sqrt(dt) * np.random.randn(N_osc)
        theta += (omega + coupling) * dt + noise_term

        if out_idx < n_out and t >= next_out_t:
            r_out[out_idx] = r_abs
            out_idx += 1
            next_out_t += out_dt

    actual = min(out_idx, n_out)
    return r_out[:actual]


def compute_flux_at_scale(r_ts, window_samples):
    if window_samples < 1:
        window_samples = 1
    r_smooth = uniform_filter1d(r_ts, size=window_samples)
    dr = np.diff(r_smooth)
    return dr


def symmetric_kl(p, q, eps=1e-12):
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return 0.5 * np.sum(p * np.log(p / q)) + 0.5 * np.sum(q * np.log(q / p))


def build_ladder(base_window, ratio, n_rungs=7):
    windows = [base_window]
    for _ in range(n_rungs - 1):
        windows.append(windows[-1] * ratio)
    return windows


N_CONFIGS = 25
N_OSC = 100
K_values = [1.0, 2.0, 3.0, 5.0, 8.0]
sigma_values = [0.01, 0.05, 0.1, 0.2, 0.5]
configs = [(K, sigma) for K in K_values for sigma in sigma_values]

print(f"\n  Running {N_CONFIGS} Kuramoto configs (N={N_OSC}, T=40s)...")

simulations = []
for ci, (K, sigma) in enumerate(configs):
    r = run_kuramoto_trajectory(K, N_OSC, sigma)
    simulations.append(r)
    if (ci + 1) % 10 == 0:
        print(f"    {ci+1}/{N_CONFIGS} configs done")

print(f"\n  Computing collapse quality for each noble number ladder...")

base_window = 0.01
n_bins = 100
bin_range = (-5, 5)

ladder_skl = {name: np.zeros(N_CONFIGS) for name in NOBLE_NUMBERS}

for ci, r_ts in enumerate(simulations):
    for name, ratio in NOBLE_NUMBERS.items():
        windows = build_ladder(base_window, ratio)
        config_kls = []
        for wi in range(len(windows) - 1):
            ws1 = max(2, int(windows[wi] * FS_OUT))
            ws2 = max(2, int(windows[wi + 1] * FS_OUT))
            dr1 = compute_flux_at_scale(r_ts, ws1)
            dr2 = compute_flux_at_scale(r_ts, ws2)
            if len(dr1) < 20 or len(dr2) < 20:
                continue
            std1 = np.std(dr1)
            std2 = np.std(dr2)
            if std1 < 1e-15 or std2 < 1e-15:
                continue
            n1 = (dr1 - np.mean(dr1)) / std1
            n2 = (dr2 - np.mean(dr2)) / std2
            h1, _ = np.histogram(n1, bins=n_bins, range=bin_range, density=True)
            h2, _ = np.histogram(n2, bins=n_bins, range=bin_range, density=True)
            config_kls.append(symmetric_kl(h1, h2))
        ladder_skl[name][ci] = np.mean(config_kls) if config_kls else np.nan

print(f"\n  Mean SKL per ladder (lower = better collapse):")
mean_skls = {}
for name in NOBLE_NUMBERS:
    valid = ladder_skl[name][np.isfinite(ladder_skl[name])]
    if len(valid) > 0:
        mean_skls[name] = float(np.mean(valid))
    else:
        mean_skls[name] = np.inf

for name in sorted(mean_skls, key=mean_skls.get):
    marker = ' ← BEST' if mean_skls[name] == min(mean_skls.values()) else ''
    print(f"    {name:12s}: SKL = {mean_skls[name]:.6f}{marker}")

best_noble = min(mean_skls, key=mean_skls.get)

print(f"\n  --- Paired bootstrap test (2000 resamples) ---")

valid_mask = np.all([np.isfinite(ladder_skl[name]) for name in NOBLE_NUMBERS], axis=0)
n_valid = int(np.sum(valid_mask))
print(f"    Configs with valid data for all ladders: {n_valid}/{N_CONFIGS}")

n_boot = 2000
pairwise_results = {}

noble_names = list(NOBLE_NUMBERS.keys())
phi_name = 'φ [1;1̄]'

for name in noble_names:
    if name == phi_name:
        continue
    phi_vals = ladder_skl[phi_name][valid_mask]
    other_vals = ladder_skl[name][valid_mask]
    boot_diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n_valid, n_valid, replace=True)
        boot_diffs[b] = np.mean(phi_vals[idx]) - np.mean(other_vals[idx])

    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    mean_diff = float(np.mean(boot_diffs))
    sig = not (ci_lo <= 0 <= ci_hi)
    direction = 'φ better' if mean_diff < 0 else 'other better'

    pairwise_results[name] = {
        'mean_diff': mean_diff,
        'ci': [ci_lo, ci_hi],
        'significant': sig,
        'direction': direction,
    }

    sig_str = 'SIG' if sig else 'n.s.'
    print(f"    φ vs {name:12s}: Δ = {mean_diff:+.6f} [{ci_lo:+.6f}, {ci_hi:+.6f}] {sig_str} ({direction})")

phi_uniquely_best = all(
    p['direction'] == 'φ better' and p['significant']
    for p in pairwise_results.values()
)
any_noble_matches_phi = any(
    not p['significant'] and p['direction'] == 'φ better'
    for p in pairwise_results.values()
) or any(
    p['direction'] == 'other better'
    for p in pairwise_results.values()
)

print(f"\n  φ uniquely best? {phi_uniquely_best}")
print(f"  Any noble number matches φ? {any_noble_matches_phi}")

results = {
    'noble_numbers': {name: float(val) for name, val in NOBLE_NUMBERS.items()},
    'mean_skl': mean_skls,
    'best_noble': best_noble,
    'pairwise_vs_phi': pairwise_results,
    'phi_uniquely_best': phi_uniquely_best,
    'n_configs': N_CONFIGS,
    'n_valid': n_valid,
    'parameters': {
        'N_osc': N_OSC, 'base_window': base_window,
        'n_rungs': 7, 'n_bins': n_bins, 'n_boot': n_boot,
    },
}

with open('outputs/noble_number_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Generating figure...")

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

colors_noble = {
    'φ [1;1̄]': 'goldenrod',
    '[1;2,1̄]': 'steelblue',
    '[1;3,1̄]': 'teal',
    '[2;1̄]': 'forestgreen',
    '√2 [1;2̄]': 'purple',
    'e-1': 'crimson',
    '2': 'gray',
}

ax_a = fig.add_subplot(gs[0, 0])
sorted_names = sorted(mean_skls.keys(), key=lambda x: mean_skls[x])
sorted_vals = [mean_skls[n] for n in sorted_names]
sorted_colors = [colors_noble.get(n, 'gray') for n in sorted_names]

std_skls = []
for n in sorted_names:
    valid = ladder_skl[n][np.isfinite(ladder_skl[n])]
    std_skls.append(float(np.std(valid) / np.sqrt(len(valid))) if len(valid) > 1 else 0)

ax_a.barh(range(len(sorted_names)), sorted_vals, xerr=std_skls,
          color=sorted_colors, alpha=0.7, edgecolor='black', linewidth=0.5, capsize=3)
ax_a.set_yticks(range(len(sorted_names)))
ax_a.set_yticklabels(sorted_names, fontsize=9)
ax_a.set_xlabel('Mean Symmetric KL Divergence (lower = better)')
ax_a.set_title('A. Collapse Quality by Ladder Ratio')

for i, (v, n) in enumerate(zip(sorted_vals, sorted_names)):
    marker = ' ★' if n == best_noble else ''
    ax_a.text(v + 0.000005, i, f'{v:.6f}{marker}', va='center', fontsize=7)

ax_b = fig.add_subplot(gs[0, 1])
for name in noble_names:
    windows = build_ladder(base_window, NOBLE_NUMBERS[name])
    r_ts = simulations[0]
    ws_mid = max(2, int(windows[3] * FS_OUT))
    dr = compute_flux_at_scale(r_ts, ws_mid)
    if len(dr) > 20 and np.std(dr) > 1e-15:
        dr_norm = (dr - np.mean(dr)) / np.std(dr)
        bins_hist = np.linspace(-5, 5, 80)
        ax_b.hist(dr_norm, bins=bins_hist, density=True, alpha=0.4,
                  color=colors_noble.get(name, 'gray'), label=name, histtype='stepfilled')

x_gauss = np.linspace(-5, 5, 200)
from scipy.stats import norm
ax_b.plot(x_gauss, norm.pdf(x_gauss), 'k--', lw=1.5, alpha=0.5, label='N(0,1)')
ax_b.set_xlabel('Normalized Flux (σ units)')
ax_b.set_ylabel('Density')
ax_b.set_title('B. Flux Distributions (middle rung, config 1)')
ax_b.legend(fontsize=6.5, ncol=2)

ax_c = fig.add_subplot(gs[1, 0])
pw_names = list(pairwise_results.keys())
pw_diffs = [pairwise_results[n]['mean_diff'] for n in pw_names]
pw_ci_lo = [pairwise_results[n]['ci'][0] for n in pw_names]
pw_ci_hi = [pairwise_results[n]['ci'][1] for n in pw_names]
pw_errors_lo = [d - lo for d, lo in zip(pw_diffs, pw_ci_lo)]
pw_errors_hi = [hi - d for d, hi in zip(pw_diffs, pw_ci_hi)]
pw_colors = [colors_noble.get(n, 'gray') for n in pw_names]
pw_sigs = [pairwise_results[n]['significant'] for n in pw_names]

sort_idx = np.argsort(pw_diffs)
pw_names_s = [pw_names[i] for i in sort_idx]
pw_diffs_s = [pw_diffs[i] for i in sort_idx]
pw_errors = [[pw_errors_lo[i] for i in sort_idx], [pw_errors_hi[i] for i in sort_idx]]
pw_colors_s = [pw_colors[i] for i in sort_idx]
pw_sigs_s = [pw_sigs[i] for i in sort_idx]

ax_c.barh(range(len(pw_names_s)), pw_diffs_s, xerr=pw_errors,
          color=pw_colors_s, alpha=0.7, edgecolor='black', linewidth=0.5, capsize=3)
ax_c.axvline(0, color='black', lw=1)
ax_c.set_yticks(range(len(pw_names_s)))
ax_c.set_yticklabels(pw_names_s, fontsize=9)
ax_c.set_xlabel('Δ SKL (φ − other); negative = φ better')
ax_c.set_title('C. Paired Bootstrap: φ vs Each Noble Number')

for i, (d, sig) in enumerate(zip(pw_diffs_s, pw_sigs_s)):
    label = '***' if sig else 'n.s.'
    ax_c.text(max(d, 0) + 0.00001, i, label, va='center', fontsize=8,
              fontweight='bold' if sig else 'normal')

ax_d = fig.add_subplot(gs[1, 1])
for name in noble_names:
    valid = ladder_skl[name][np.isfinite(ladder_skl[name])]
    if len(valid) > 0:
        ax_d.scatter(np.full(len(valid), NOBLE_NUMBERS[name]), valid,
                     color=colors_noble.get(name, 'gray'), alpha=0.5, s=20,
                     edgecolors='black', linewidths=0.3, label=name)

ax_d.axvline(PHI, color='goldenrod', ls='--', alpha=0.5)
ax_d.axvline(E_MINUS_1, color='crimson', ls='--', alpha=0.5)
ax_d.set_xlabel('Noble Number Value')
ax_d.set_ylabel('SKL per config')
ax_d.set_title('D. Individual Config Collapse by Noble Number')
ax_d.legend(fontsize=6.5, ncol=2, loc='upper left')

conclusion = f"Best: {best_noble}\nφ uniquely best? {phi_uniquely_best}"
fig.text(0.5, 0.01, conclusion, ha='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

fig.suptitle('Figure 20: Noble Number Control — Kuramoto Temporal Collapse\n'
             f'Is φ specifically special or does any noble number work equally well? (N={N_CONFIGS} configs)',
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig(f'{OUTPUT_DIR}/fig20_noble_numbers.png')
plt.close()

print(f"\n  Figure saved: {OUTPUT_DIR}/fig20_noble_numbers.png")
print(f"  Results saved: outputs/noble_number_results.json")

print(f"\n{'='*60}")
print(f"  CONCLUSION")
print(f"{'='*60}")
print(f"  Best noble number for collapse: {best_noble} (SKL={mean_skls[best_noble]:.6f})")
print(f"  φ uniquely best? {phi_uniquely_best}")
if any_noble_matches_phi:
    matching = [n for n, p in pairwise_results.items() if not p['significant'] or p['direction'] == 'other better']
    print(f"  Noble numbers matching or beating φ: {matching}")
print(f"{'='*60}")

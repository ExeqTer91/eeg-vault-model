#!/usr/bin/env python3
"""
JOB 9: EXPANDED CONSTANT SET TEST
Tests whether e-1 survives as the best-organizing constant for α/θ ratios
against an expanded set: √π, 7/4, √3, √e, e/φ, and the empirical mean (null).

Two complementary methods:
A) Distance-based: per-subject nearest constant → winner tallies + permutation test
B) AIC-based: fit Gaussian N(μ=constant, σ) to ratio distribution → ΔAIC comparison

If the empirical mean Gaussian beats all constants, no mathematical constant is needed.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import minimize_scalar
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1
OUTPUT_DIR = 'outputs/publication_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})

CONSTANTS = {
    'e−1': E_MINUS_1,
    '√π': np.sqrt(np.pi),
    '7/4': 7.0 / 4.0,
    '√3': np.sqrt(3),
    '√e': np.sqrt(np.e),
    'e/φ': np.e / PHI,
    'φ': PHI,
}

print("=" * 60)
print("JOB 9: EXPANDED CONSTANT SET TEST")
print("=" * 60)

aw = json.load(open('outputs/aw_cached_subjects.json'))
ds = json.load(open('outputs/ds003969_cached_subjects.json'))
eeg = json.load(open('outputs/eegbci_modal_results.json'))

all_ratios = []
dataset_labels = []

for s in aw:
    r = s.get('alpha_theta_ratio')
    if r and 1.0 < r < 3.0:
        all_ratios.append(r)
        dataset_labels.append('AW')

for s in ds:
    r = s.get('alpha_theta_ratio')
    if r and 1.0 < r < 3.0:
        all_ratios.append(r)
        dataset_labels.append('DS003969')

for s in eeg:
    r = s.get('alpha_theta_ratio')
    if r and 1.0 < r < 3.0:
        all_ratios.append(r)
        dataset_labels.append('EEGBCI')

all_ratios = np.array(all_ratios)
N = len(all_ratios)
empirical_mean = float(np.mean(all_ratios))

CONSTANTS['empirical'] = empirical_mean

print(f"\n  N = {N} subjects across 3 datasets")
print(f"  Empirical mean = {empirical_mean:.4f}")
print(f"  Std = {np.std(all_ratios):.4f}")

print(f"\n  Constants tested:")
for name, val in CONSTANTS.items():
    dist = abs(val - empirical_mean)
    print(f"    {name:10s} = {val:.6f}  (Δ from mean = {dist:.4f})")

print(f"\n  --- Part A: Distance-based winner tallies ---")

winner_counts = {name: 0 for name in CONSTANTS}
per_subject_nearest = []

for r in all_ratios:
    distances = {name: abs(r - val) for name, val in CONSTANTS.items()}
    nearest = min(distances, key=distances.get)
    winner_counts[nearest] += 1
    per_subject_nearest.append(nearest)

print(f"\n  Winner tallies (N={N}):")
for name in sorted(winner_counts, key=winner_counts.get, reverse=True):
    pct = winner_counts[name] / N * 100
    print(f"    {name:10s}: {winner_counts[name]:4d} ({pct:5.1f}%)")

mean_distances = {}
for name, val in CONSTANTS.items():
    dists = np.abs(all_ratios - val)
    mean_distances[name] = {
        'mean': float(np.mean(dists)),
        'median': float(np.median(dists)),
        'std': float(np.std(dists)),
    }

print(f"\n  Mean absolute distance:")
for name in sorted(mean_distances, key=lambda x: mean_distances[x]['mean']):
    d = mean_distances[name]
    print(f"    {name:10s}: {d['mean']:.4f} ± {d['std']:.4f} (median={d['median']:.4f})")

print(f"\n  --- Permutation test: is nearest-to-e-1 count significant? ---")

n_perm = 5000
e1_count_obs = winner_counts['e−1']
perm_counts = np.zeros(n_perm)

for p in range(n_perm):
    shuffled = np.random.permutation(all_ratios)
    center = np.random.choice(list(CONSTANTS.values()))
    count = np.sum(np.abs(shuffled - E_MINUS_1) < np.abs(shuffled - center))
    perm_counts[p] = count

perm_p = float(np.mean(perm_counts >= e1_count_obs))
print(f"    e-1 wins: {e1_count_obs}/{N}")
print(f"    Permutation p-value: {perm_p:.4f}")
print(f"    {'SIGNIFICANT' if perm_p < 0.05 else 'NOT significant'} at α=0.05")

print(f"\n  --- Part B: AIC Gaussian comparison ---")

def gaussian_log_likelihood(data, mu, sigma):
    n = len(data)
    return -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)

def fit_gaussian_fixed_mu(data, mu):
    sigma_mle = np.sqrt(np.mean((data - mu)**2))
    ll = gaussian_log_likelihood(data, mu, sigma_mle)
    k = 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(len(data)) - 2 * ll
    return {
        'mu': float(mu),
        'sigma': float(sigma_mle),
        'log_likelihood': float(ll),
        'aic': float(aic),
        'bic': float(bic),
        'k': k,
    }

def fit_gaussian_free(data):
    mu_mle = np.mean(data)
    sigma_mle = np.std(data)
    ll = gaussian_log_likelihood(data, mu_mle, sigma_mle)
    k = 2
    aic = 2 * k - 2 * ll
    bic = k * np.log(len(data)) - 2 * ll
    return {
        'mu': float(mu_mle),
        'sigma': float(sigma_mle),
        'log_likelihood': float(ll),
        'aic': float(aic),
        'bic': float(bic),
        'k': k,
    }

aic_results = {}
for name, val in CONSTANTS.items():
    aic_results[name] = fit_gaussian_fixed_mu(all_ratios, val)

aic_results['free_fit'] = fit_gaussian_free(all_ratios)

aic_values = {name: r['aic'] for name, r in aic_results.items()}
best_aic_name = min(aic_values, key=aic_values.get)
best_aic_val = aic_values[best_aic_name]

print(f"\n  AIC comparison (lower = better):")
for name in sorted(aic_values, key=aic_values.get):
    delta = aic_values[name] - best_aic_val
    marker = ' ← BEST' if name == best_aic_name else ''
    sigma = aic_results[name]['sigma']
    print(f"    {name:10s}: AIC={aic_values[name]:.1f}  ΔAIC={delta:.1f}  σ={sigma:.4f}{marker}")

print(f"\n  BIC comparison:")
bic_values = {name: r['bic'] for name, r in aic_results.items()}
best_bic_name = min(bic_values, key=bic_values.get)
for name in sorted(bic_values, key=bic_values.get):
    delta = bic_values[name] - bic_values[best_bic_name]
    marker = ' ← BEST' if name == best_bic_name else ''
    print(f"    {name:10s}: BIC={bic_values[name]:.1f}  ΔBIC={delta:.1f}{marker}")

empirical_beats_all = all(aic_values['empirical'] <= aic_values[name] for name in CONSTANTS if name != 'empirical')
free_beats_all = all(aic_values['free_fit'] <= aic_values[name] for name in CONSTANTS)

print(f"\n  Key comparisons:")
print(f"    Empirical mean beats all constants? {'YES' if empirical_beats_all else 'NO'}")
print(f"    Free fit beats all constants? {'YES' if free_beats_all else 'NO'}")
print(f"    ΔAIC(empirical vs e-1) = {aic_values['empirical'] - aic_values['e−1']:.1f}")
print(f"    ΔAIC(empirical vs √π) = {aic_values['empirical'] - aic_values['√π']:.1f}")
if free_beats_all or empirical_beats_all:
    print(f"    → No mathematical constant needed — empirical distribution is self-organizing")
else:
    print(f"    → Best constant: {best_aic_name}")

print(f"\n  --- Bootstrap CI for AIC differences ---")
n_boot = 2000
boot_delta_aic = {name: np.zeros(n_boot) for name in CONSTANTS}

for b in range(n_boot):
    idx = np.random.choice(N, size=N, replace=True)
    boot_sample = all_ratios[idx]
    boot_fits = {}
    for name, val in CONSTANTS.items():
        boot_fits[name] = fit_gaussian_fixed_mu(boot_sample, val)
    for name in CONSTANTS:
        boot_delta_aic[name][b] = boot_fits[name]['aic'] - boot_fits['empirical']['aic']

print(f"  ΔAIC relative to empirical mean (positive = empirical wins):")
for name in sorted(CONSTANTS.keys(), key=lambda x: np.mean(boot_delta_aic[x])):
    if name == 'empirical':
        continue
    ci_lo = np.percentile(boot_delta_aic[name], 2.5)
    ci_hi = np.percentile(boot_delta_aic[name], 97.5)
    mean_d = np.mean(boot_delta_aic[name])
    sig = 'SIG' if ci_lo > 0 or ci_hi < 0 else 'n.s.'
    print(f"    {name:10s}: ΔAIC = {mean_d:+.1f} [CI: {ci_lo:+.1f}, {ci_hi:+.1f}] {sig}")

results = {
    'N': N,
    'empirical_mean': empirical_mean,
    'empirical_std': float(np.std(all_ratios)),
    'constants_tested': {name: float(val) for name, val in CONSTANTS.items()},
    'winner_counts': winner_counts,
    'mean_distances': mean_distances,
    'permutation_test': {
        'e1_wins': e1_count_obs,
        'p_value': perm_p,
        'n_permutations': n_perm,
    },
    'aic_results': aic_results,
    'best_aic': best_aic_name,
    'best_bic': best_bic_name,
    'empirical_beats_all': empirical_beats_all,
    'datasets': {
        'AW': dataset_labels.count('AW'),
        'DS003969': dataset_labels.count('DS003969'),
        'EEGBCI': dataset_labels.count('EEGBCI'),
    },
}

with open('outputs/expanded_constants_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Generating figure...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

ax_a = fig.add_subplot(gs[0, 0])
bins = np.linspace(1.3, 2.5, 50)
ax_a.hist(all_ratios, bins=bins, density=True, alpha=0.6, color='steelblue',
          edgecolor='white', linewidth=0.5, label=f'α/θ ratios (N={N})')

colors = {
    'e−1': 'crimson', '√π': 'darkorange', '7/4': 'forestgreen',
    '√3': 'purple', '√e': 'teal', 'e/φ': 'brown', 'φ': 'goldenrod',
    'empirical': 'black',
}
line_styles = {
    'e−1': '--', '√π': '-.', '7/4': ':',
    '√3': '--', '√e': '-.', 'e/φ': ':',  'φ': '--',
    'empirical': '-',
}

for name, val in CONSTANTS.items():
    sigma = aic_results[name]['sigma']
    x_fit = np.linspace(1.3, 2.5, 200)
    y_fit = stats.norm.pdf(x_fit, val, sigma)
    ax_a.plot(x_fit, y_fit, color=colors.get(name, 'gray'),
              ls=line_styles.get(name, '-'), lw=1.5, alpha=0.8,
              label=f'{name} = {val:.4f}')

ax_a.set_xlabel('α/θ Frequency Ratio')
ax_a.set_ylabel('Density')
ax_a.set_title('A. Distribution with Gaussian Fits')
ax_a.legend(fontsize=6.5, loc='upper right', ncol=2)

ax_b = fig.add_subplot(gs[0, 1])
sorted_names = sorted([n for n in aic_values if n != 'free_fit'],
                       key=lambda x: aic_values[x])
sorted_aic = [aic_values[n] - best_aic_val for n in sorted_names]
bar_colors = [colors.get(n, 'gray') for n in sorted_names]

bars = ax_b.barh(range(len(sorted_names)), sorted_aic, color=bar_colors, alpha=0.7,
                 edgecolor='black', linewidth=0.5)
ax_b.set_yticks(range(len(sorted_names)))
ax_b.set_yticklabels(sorted_names, fontsize=9)
ax_b.set_xlabel('ΔAIC (relative to best)')
ax_b.set_title('B. AIC Model Comparison')
ax_b.axvline(0, color='black', lw=0.5)
ax_b.axvline(2, color='gray', ls=':', alpha=0.5, label='ΔAIC=2 threshold')
ax_b.axvline(10, color='gray', ls='--', alpha=0.5, label='ΔAIC=10 strong')
ax_b.legend(fontsize=7)

for i, (name, daic) in enumerate(zip(sorted_names, sorted_aic)):
    ax_b.text(daic + 0.3, i, f'{daic:.1f}', va='center', fontsize=7)

ax_c = fig.add_subplot(gs[1, 0])
winner_names = sorted(winner_counts.keys(), key=lambda x: winner_counts[x], reverse=True)
winner_vals = [winner_counts[n] for n in winner_names]
winner_pcts = [v / N * 100 for v in winner_vals]
wcolors = [colors.get(n, 'gray') for n in winner_names]

bars_c = ax_c.bar(range(len(winner_names)), winner_pcts, color=wcolors, alpha=0.7,
                  edgecolor='black', linewidth=0.5)
ax_c.set_xticks(range(len(winner_names)))
ax_c.set_xticklabels(winner_names, rotation=30, ha='right', fontsize=9)
ax_c.set_ylabel('% of subjects nearest')
ax_c.set_title('C. Per-Subject Nearest Constant')

for i, (pct, count) in enumerate(zip(winner_pcts, winner_vals)):
    ax_c.text(i, pct + 0.5, f'{count}\n({pct:.1f}%)', ha='center', fontsize=7)

ax_d = fig.add_subplot(gs[1, 1])
const_names_no_emp = [n for n in CONSTANTS if n not in ('empirical', 'free_fit')]
const_vals_no_emp = [CONSTANTS[n] for n in const_names_no_emp]
mean_dists = [mean_distances[n]['mean'] for n in const_names_no_emp]
std_dists = [mean_distances[n]['std'] for n in const_names_no_emp]
dcolors = [colors.get(n, 'gray') for n in const_names_no_emp]

sort_idx = np.argsort(mean_dists)
sorted_cn = [const_names_no_emp[i] for i in sort_idx]
sorted_md = [mean_dists[i] for i in sort_idx]
sorted_sd = [std_dists[i] for i in sort_idx]
sorted_dc = [dcolors[i] for i in sort_idx]

ax_d.barh(range(len(sorted_cn)), sorted_md, xerr=sorted_sd, color=sorted_dc,
          alpha=0.7, edgecolor='black', linewidth=0.5, capsize=3)
ax_d.set_yticks(range(len(sorted_cn)))
ax_d.set_yticklabels(sorted_cn, fontsize=9)
ax_d.set_xlabel('Mean |ratio − constant|')
ax_d.set_title('D. Mean Distance to Each Constant')

for i, (md, cn) in enumerate(zip(sorted_md, sorted_cn)):
    ax_d.text(md + 0.002, i, f'{md:.4f}', va='center', fontsize=7)

summary_text = f"Best AIC: {best_aic_name}\nBest BIC: {best_bic_name}\nEmpirical mean: {empirical_mean:.4f}"
if empirical_beats_all:
    summary_text += "\nEmpirical beats all constants"
fig.text(0.5, 0.01, summary_text, ha='center', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

fig.suptitle('Figure 19: Expanded Constant Set Test\n'
             f'Does e−1 survive against √π, 7/4, √3, √e, e/φ? (N={N})',
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig(f'{OUTPUT_DIR}/fig19_expanded_constants.png')
plt.close()

print(f"\n  Figure saved: {OUTPUT_DIR}/fig19_expanded_constants.png")
print(f"  Results saved: outputs/expanded_constants_results.json")

print(f"\n{'='*60}")
print(f"  CONCLUSION")
print(f"{'='*60}")
print(f"  Best AIC constant: {best_aic_name}")
print(f"  Best BIC constant: {best_bic_name}")
print(f"  Empirical mean ({empirical_mean:.4f}) beats all? {empirical_beats_all}")
print(f"  e-1 nearest-constant wins: {winner_counts['e−1']}/{N} ({winner_counts['e−1']/N*100:.1f}%)")
print(f"{'='*60}")

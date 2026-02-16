#!/usr/bin/env python3
"""
JOB 11: FOOOF APERIODIC CORRECTION ON PRIMARY α/θ RATIO
The potentially fatal test: does e-1 survive after removing the 1/f aperiodic component?

Method:
1. Use pre-computed FOOOF peaks from EEGBCI (N=109)
2. Compute FOOOF from raw data for DS003969 and Alpha Waves where possible
3. Extract periodic-only α/θ ratio from oscillatory peaks
4. Report: N with valid peaks, distribution shape, skewness
5. Re-run full constant comparison on periodic-only ratios
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import welch
import os
import glob as glob_mod
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

def run_fooof(freqs, psd):
    try:
        from fooof import FOOOF
        mask = (freqs >= 1) & (freqs <= 50)
        f = freqs[mask]
        p = psd[mask]
        if len(f) < 10:
            return None
        fm = FOOOF(peak_width_limits=[1, 12], max_n_peaks=8, min_peak_height=0.05, verbose=False)
        fm.fit(f, p)
        ap = fm.aperiodic_params_
        peaks = fm.peak_params_.tolist() if len(fm.peak_params_) > 0 else []
        return {
            'aperiodic_offset': float(ap[0]),
            'aperiodic_slope': float(ap[-1]),
            'n_peaks': len(peaks),
            'peaks': peaks,
        }
    except Exception:
        return None


def compute_avg_psd(data, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(int(4 * fs), data.shape[1])
    avg_psd = None
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        if avg_psd is None:
            avg_psd = psd.copy()
        else:
            avg_psd += psd
    avg_psd /= data.shape[0]
    return freqs, avg_psd


def extract_periodic_ratio(peaks):
    theta_peaks = [p for p in peaks if 4 <= p[0] <= 8]
    alpha_peaks = [p for p in peaks if 8 <= p[0] <= 13]
    if not theta_peaks or not alpha_peaks:
        return None, None, None
    best_theta = max(theta_peaks, key=lambda p: p[1])
    best_alpha = max(alpha_peaks, key=lambda p: p[1])
    ratio = best_alpha[0] / best_theta[0]
    return ratio, best_theta[0], best_alpha[0]


print("=" * 60)
print("JOB 11: FOOOF APERIODIC CORRECTION")
print("=" * 60)

eeg = json.load(open('outputs/eegbci_modal_results.json'))

all_fooof_results = []
raw_ratios_matched = []

print(f"\n  Part 1: EEGBCI (N={len(eeg)}) — pre-computed FOOOF")

eegbci_periodic = []
eegbci_raw = []
eegbci_both = 0
eegbci_theta_only = 0
eegbci_alpha_only = 0
eegbci_neither = 0

for s in eeg:
    fooof_data = s.get('fooof', {})
    peaks = fooof_data.get('peaks', [])
    raw_ratio = s.get('alpha_theta_ratio')

    has_theta = any(4 <= p[0] <= 8 for p in peaks)
    has_alpha = any(8 <= p[0] <= 13 for p in peaks)

    if has_theta and has_alpha:
        eegbci_both += 1
        ratio, theta_f, alpha_f = extract_periodic_ratio(peaks)
        if ratio and 1.0 < ratio < 3.0:
            eegbci_periodic.append(ratio)
            if raw_ratio and 1.0 < raw_ratio < 3.0:
                eegbci_raw.append(raw_ratio)
                all_fooof_results.append({
                    'dataset': 'EEGBCI',
                    'periodic_ratio': ratio,
                    'raw_ratio': raw_ratio,
                    'theta_peak': theta_f,
                    'alpha_peak': alpha_f,
                    'aperiodic_slope': fooof_data.get('aperiodic_slope'),
                })
    elif has_theta:
        eegbci_theta_only += 1
    elif has_alpha:
        eegbci_alpha_only += 1
    else:
        eegbci_neither += 1

print(f"    Both θ+α peaks: {eegbci_both}/{len(eeg)} ({eegbci_both/len(eeg)*100:.1f}%)")
print(f"    θ only: {eegbci_theta_only}, α only: {eegbci_alpha_only}, neither: {eegbci_neither}")
print(f"    Valid periodic ratios: {len(eegbci_periodic)}")

ds_periodic = []
aw_periodic = []
print(f"  (DS003969 and Alpha Waves: raw PSD recomputation skipped — BDF files too large)")
print(f"  (Using EEGBCI pre-computed FOOOF peaks as primary dataset)")

all_periodic = eegbci_periodic + ds_periodic + aw_periodic
all_periodic = np.array(all_periodic)
N_periodic = len(all_periodic)

matched_periodic = np.array([r['periodic_ratio'] for r in all_fooof_results])
matched_raw = np.array([r['raw_ratio'] for r in all_fooof_results])
N_matched = len(matched_periodic)

print(f"\n{'='*60}")
print(f"  SUMMARY OF PERIODIC-ONLY RATIOS")
print(f"{'='*60}")
print(f"  Total with periodic θ+α peaks: {N_periodic}")
print(f"  Matched periodic+raw pairs: {N_matched}")

if N_periodic > 5:
    print(f"\n  Periodic-only distribution:")
    print(f"    Mean:     {np.mean(all_periodic):.4f}")
    print(f"    Std:      {np.std(all_periodic):.4f}")
    print(f"    Median:   {np.median(all_periodic):.4f}")
    skew = float(stats.skew(all_periodic))
    kurt = float(stats.kurtosis(all_periodic))
    print(f"    Skewness: {skew:.4f}")
    print(f"    Kurtosis: {kurt:.4f}")
    print(f"    IQR:      [{np.percentile(all_periodic, 25):.4f}, {np.percentile(all_periodic, 75):.4f}]")

    periodic_mean = float(np.mean(all_periodic))
    dist_e1 = abs(periodic_mean - E_MINUS_1)
    dist_phi = abs(periodic_mean - PHI)
    print(f"\n    Distance from e-1: {dist_e1:.4f}")
    print(f"    Distance from φ:   {dist_phi:.4f}")
    print(f"    Closer to: {'e-1' if dist_e1 < dist_phi else 'φ'}")

    if N_matched > 5:
        shift = float(np.mean(matched_periodic) - np.mean(matched_raw))
        t_stat, t_p = stats.ttest_rel(matched_periodic, matched_raw)
        print(f"\n  Paired comparison (periodic vs raw):")
        print(f"    Raw mean:      {np.mean(matched_raw):.4f}")
        print(f"    Periodic mean: {np.mean(matched_periodic):.4f}")
        print(f"    Shift:         {shift:+.4f}")
        print(f"    Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")

    print(f"\n  --- Selection bias check ---")
    all_raw_eegbci = [s['alpha_theta_ratio'] for s in eeg if s.get('alpha_theta_ratio') and 1.0 < s['alpha_theta_ratio'] < 3.0]
    retained_raw = [s['alpha_theta_ratio'] for s in eeg
                    if s.get('alpha_theta_ratio') and 1.0 < s['alpha_theta_ratio'] < 3.0
                    and any(4 <= p[0] <= 8 for p in s.get('fooof', {}).get('peaks', []))
                    and any(8 <= p[0] <= 13 for p in s.get('fooof', {}).get('peaks', []))]
    excluded_raw = [r for r in all_raw_eegbci if r not in retained_raw]

    if len(retained_raw) > 5 and len(excluded_raw) > 5:
        u_stat, u_p = stats.mannwhitneyu(retained_raw, excluded_raw, alternative='two-sided')
        print(f"    Retained raw mean:  {np.mean(retained_raw):.4f} (N={len(retained_raw)})")
        print(f"    Excluded raw mean:  {np.mean(excluded_raw):.4f} (N={len(excluded_raw)})")
        print(f"    Mann-Whitney U test: U={u_stat:.1f}, p={u_p:.4f}")
        if u_p < 0.05:
            print(f"    ⚠ Retained subjects differ significantly from excluded — selection bias risk!")
        else:
            print(f"    ✓ No significant difference — selection bias unlikely")

print(f"\n  --- Constant comparison on periodic-only ratios ---")

if N_periodic > 10:
    CONSTANTS_WITH_EMP = dict(CONSTANTS)
    periodic_emp_mean = float(np.mean(all_periodic))
    CONSTANTS_WITH_EMP['empirical'] = periodic_emp_mean

    aic_periodic = {}
    for name, val in CONSTANTS_WITH_EMP.items():
        sigma_mle = np.sqrt(np.mean((all_periodic - val)**2))
        ll = -N_periodic/2 * np.log(2 * np.pi * sigma_mle**2) - N_periodic/2
        k = 1
        aic = 2 * k - 2 * ll
        aic_periodic[name] = {
            'aic': float(aic),
            'sigma': float(sigma_mle),
            'mu': float(val),
        }

    best_periodic_aic = min(aic_periodic, key=lambda x: aic_periodic[x]['aic'])
    best_aic_val = aic_periodic[best_periodic_aic]['aic']

    print(f"\n  AIC on periodic-only (N={N_periodic}):")
    for name in sorted(aic_periodic, key=lambda x: aic_periodic[x]['aic']):
        delta = aic_periodic[name]['aic'] - best_aic_val
        marker = ' ← BEST' if name == best_periodic_aic else ''
        print(f"    {name:10s}: AIC={aic_periodic[name]['aic']:.1f}  ΔAIC={delta:.1f}{marker}")

    winner_periodic = {name: 0 for name in CONSTANTS_WITH_EMP}
    for r in all_periodic:
        dists = {name: abs(r - val) for name, val in CONSTANTS_WITH_EMP.items()}
        nearest = min(dists, key=dists.get)
        winner_periodic[nearest] += 1

    print(f"\n  Winner tallies (periodic-only, N={N_periodic}):")
    for name in sorted(winner_periodic, key=winner_periodic.get, reverse=True):
        pct = winner_periodic[name] / N_periodic * 100
        print(f"    {name:10s}: {winner_periodic[name]:4d} ({pct:5.1f}%)")
else:
    aic_periodic = {}
    best_periodic_aic = None
    winner_periodic = {}
    print(f"  Too few periodic ratios ({N_periodic}) for constant comparison")

results = {
    'N_total_subjects': len(eeg) + len(ds_periodic) + len(aw_periodic),
    'N_periodic_valid': N_periodic,
    'N_matched_pairs': N_matched,
    'attrition': {
        'EEGBCI': {'total': len(eeg), 'both_peaks': eegbci_both, 'theta_only': eegbci_theta_only,
                   'alpha_only': eegbci_alpha_only, 'neither': eegbci_neither},
    },
    'periodic_distribution': {
        'mean': float(np.mean(all_periodic)) if N_periodic > 0 else None,
        'std': float(np.std(all_periodic)) if N_periodic > 0 else None,
        'median': float(np.median(all_periodic)) if N_periodic > 0 else None,
        'skewness': float(stats.skew(all_periodic)) if N_periodic > 5 else None,
        'kurtosis': float(stats.kurtosis(all_periodic)) if N_periodic > 5 else None,
    },
    'shift_from_raw': {
        'raw_mean': float(np.mean(matched_raw)) if N_matched > 0 else None,
        'periodic_mean': float(np.mean(matched_periodic)) if N_matched > 0 else None,
        'shift': float(np.mean(matched_periodic) - np.mean(matched_raw)) if N_matched > 0 else None,
    },
    'aic_periodic': aic_periodic,
    'best_periodic_aic': best_periodic_aic,
    'winner_periodic': winner_periodic,
    'all_periodic_ratios': [float(r) for r in all_periodic],
    'fooof_details': all_fooof_results,
}

with open('outputs/fooof_correction_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Generating figure...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

ax_a = fig.add_subplot(gs[0, 0])
if N_matched > 5:
    bins = np.linspace(1.0, 2.8, 40)
    ax_a.hist(matched_raw, bins=bins, alpha=0.5, color='steelblue', density=True,
              label=f'Raw centroids (N={N_matched})', edgecolor='white', linewidth=0.5)
    ax_a.hist(matched_periodic, bins=bins, alpha=0.5, color='coral', density=True,
              label=f'Periodic peaks (N={N_matched})', edgecolor='white', linewidth=0.5)

    ax_a.axvline(E_MINUS_1, color='crimson', ls='--', lw=2, label=f'e−1 = {E_MINUS_1:.4f}')
    ax_a.axvline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.4f}')

    if N_matched > 0:
        ax_a.axvline(np.mean(matched_raw), color='steelblue', ls=':', lw=1.5, alpha=0.8)
        ax_a.axvline(np.mean(matched_periodic), color='coral', ls=':', lw=1.5, alpha=0.8)

    shift_val = float(np.mean(matched_periodic) - np.mean(matched_raw))
    ax_a.text(0.02, 0.98, f'Raw mean: {np.mean(matched_raw):.4f}\n'
              f'Periodic mean: {np.mean(matched_periodic):.4f}\n'
              f'Shift: {shift_val:+.4f}',
              transform=ax_a.transAxes, fontsize=8, verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax_a.set_xlabel('α/θ Frequency Ratio')
ax_a.set_ylabel('Density')
ax_a.set_title('A. Raw vs Periodic-Only Ratios')
ax_a.legend(fontsize=7)

ax_b = fig.add_subplot(gs[0, 1])
if N_matched > 5:
    ax_b.scatter(matched_raw, matched_periodic, alpha=0.5, s=20, color='steelblue',
                 edgecolors='black', linewidths=0.3)
    lims = [min(min(matched_raw), min(matched_periodic)) - 0.1,
            max(max(matched_raw), max(matched_periodic)) + 0.1]
    ax_b.plot(lims, lims, 'k--', lw=1, alpha=0.3, label='y=x')
    ax_b.axhline(E_MINUS_1, color='crimson', ls=':', alpha=0.5)
    ax_b.axvline(E_MINUS_1, color='crimson', ls=':', alpha=0.5)
    ax_b.axhline(PHI, color='goldenrod', ls=':', alpha=0.5)

    r_corr, p_corr = stats.pearsonr(matched_raw, matched_periodic)
    ax_b.text(0.02, 0.98, f'r = {r_corr:.3f}\np = {p_corr:.2e}',
              transform=ax_b.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    for d in all_fooof_results:
        color = {'EEGBCI': 'steelblue', 'DS003969': 'coral', 'AW': 'forestgreen'}.get(d['dataset'], 'gray')
        ax_b.scatter(d['raw_ratio'], d['periodic_ratio'], alpha=0.3, s=15, color=color, edgecolors='none')

ax_b.set_xlabel('Raw α/θ Ratio')
ax_b.set_ylabel('Periodic-Only α/θ Ratio')
ax_b.set_title('B. Raw vs Periodic Correlation')
ax_b.legend(fontsize=8)

ax_c = fig.add_subplot(gs[1, 0])
categories = ['Both θ+α', 'θ only', 'α only', 'Neither']
eeg_counts = [eegbci_both, eegbci_theta_only, eegbci_alpha_only, eegbci_neither]
cat_colors = ['forestgreen', 'steelblue', 'coral', 'gray']
ax_c.bar(categories, eeg_counts, color=cat_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
for i, (cat, cnt) in enumerate(zip(categories, eeg_counts)):
    pct = cnt / len(eeg) * 100
    ax_c.text(i, cnt + 1, f'{cnt}\n({pct:.1f}%)', ha='center', fontsize=8)
ax_c.set_ylabel('Number of subjects')
ax_c.set_title(f'C. Peak Detection Attrition (EEGBCI, N={len(eeg)})')

total_all = len(eeg)
retained = eegbci_both
attrition_pct = (1 - retained / total_all) * 100
ax_c.text(0.98, 0.98, f'Attrition: {attrition_pct:.1f}%\nRetained: {retained}/{total_all}',
          transform=ax_c.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax_d = fig.add_subplot(gs[1, 1])
if aic_periodic and len(aic_periodic) > 0:
    colors_const = {
        'e−1': 'crimson', '√π': 'darkorange', '7/4': 'forestgreen',
        '√3': 'purple', '√e': 'teal', 'e/φ': 'brown', 'φ': 'goldenrod',
        'empirical': 'black',
    }
    sorted_names = sorted(aic_periodic.keys(), key=lambda x: aic_periodic[x]['aic'])
    sorted_aic_vals = [aic_periodic[n]['aic'] - aic_periodic[sorted_names[0]]['aic'] for n in sorted_names]
    sorted_colors = [colors_const.get(n, 'gray') for n in sorted_names]

    ax_d.barh(range(len(sorted_names)), sorted_aic_vals, color=sorted_colors,
              alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_d.set_yticks(range(len(sorted_names)))
    ax_d.set_yticklabels(sorted_names, fontsize=9)
    ax_d.set_xlabel('ΔAIC (relative to best)')
    ax_d.axvline(2, color='gray', ls=':', alpha=0.5)
    ax_d.axvline(10, color='gray', ls='--', alpha=0.5)

    for i, (n, daic) in enumerate(zip(sorted_names, sorted_aic_vals)):
        ax_d.text(daic + 0.3, i, f'{daic:.1f}', va='center', fontsize=7)

ax_d.set_title(f'D. AIC on Periodic-Only Ratios (N={N_periodic})')

conclusion_parts = []
if N_periodic > 5:
    pmean = float(np.mean(all_periodic))
    conclusion_parts.append(f'Periodic mean: {pmean:.4f}')
    if abs(pmean - E_MINUS_1) < abs(pmean - PHI):
        conclusion_parts.append('Closer to e−1')
    else:
        conclusion_parts.append('Closer to φ!')
if best_periodic_aic:
    conclusion_parts.append(f'Best AIC: {best_periodic_aic}')
conclusion_parts.append(f'Retained: {N_periodic} subjects')

fig.text(0.5, 0.01, ' | '.join(conclusion_parts), ha='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

fig.suptitle('Figure 21: FOOOF Aperiodic Correction\n'
             'Does e−1 survive after removing the 1/f component?',
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig(f'{OUTPUT_DIR}/fig21_fooof_correction.png')
plt.close()

print(f"\n  Figure saved: {OUTPUT_DIR}/fig21_fooof_correction.png")
print(f"  Results saved: outputs/fooof_correction_results.json")

print(f"\n{'='*60}")
print(f"  CONCLUSION")
print(f"{'='*60}")
if N_periodic > 5:
    pmean = float(np.mean(all_periodic))
    print(f"  Periodic-only mean ratio: {pmean:.4f}")
    print(f"  Distance from e-1: {abs(pmean - E_MINUS_1):.4f}")
    print(f"  Distance from φ:   {abs(pmean - PHI):.4f}")
    if abs(pmean - E_MINUS_1) < abs(pmean - PHI):
        print(f"  → Periodic ratios remain CLOSER to e-1")
    else:
        print(f"  → Periodic ratios SHIFT toward φ — aperiodic component was inflating ratio!")
if best_periodic_aic:
    print(f"  Best AIC constant: {best_periodic_aic}")
print(f"  Subjects retained: {N_periodic} ({N_periodic/244*100:.1f}% of original N=244)")
print(f"{'='*60}")

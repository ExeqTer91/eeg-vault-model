#!/usr/bin/env python3
"""
JOB 12: INDIVIDUALIZED BAND BOUNDARY (gedBounds) ROBUSTNESS TEST

Uses two complementary approaches:
1. FOOOF peaks (pre-computed for EEGBCI N=109) to derive individual boundaries
2. Raw PSD trough detection for EEGBCI subjects with downloaded data

Tests whether the alpha/theta ratio landscape survives under individualized boundaries.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import welch, find_peaks
from scipy.ndimage import uniform_filter1d
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


def spectral_centroid(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    f = freqs[mask]
    p = psd[mask]
    if len(p) == 0 or p.sum() == 0:
        return (lo + hi) / 2
    return np.sum(f * p) / np.sum(p)


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


def find_psd_boundaries(freqs, psd):
    mask = (freqs >= 2) & (freqs <= 35)
    f = freqs[mask]
    p = psd[mask]
    log_p = np.log10(p + 1e-30)
    smooth = uniform_filter1d(log_p, size=5)

    peaks_idx, _ = find_peaks(smooth, prominence=0.02, distance=2)
    troughs_idx, _ = find_peaks(-smooth, prominence=0.01, distance=2)

    peaks_f = f[peaks_idx]
    peaks_h = smooth[peaks_idx]
    troughs_f = f[troughs_idx]

    theta_cands = [(fi, hi) for fi, hi in zip(peaks_f, peaks_h) if 3.5 <= fi <= 8.5]
    alpha_cands = [(fi, hi) for fi, hi in zip(peaks_f, peaks_h) if 7.5 <= fi <= 14]
    beta_cands = [(fi, hi) for fi, hi in zip(peaks_f, peaks_h) if 14 <= fi <= 30]

    theta_f = max(theta_cands, key=lambda x: x[1])[0] if theta_cands else None
    alpha_f = max(alpha_cands, key=lambda x: x[1])[0] if alpha_cands else None
    beta_f = max(beta_cands, key=lambda x: x[1])[0] if beta_cands else None

    if theta_f and alpha_f and theta_f >= alpha_f:
        low_theta = [(fi, hi) for fi, hi in theta_cands if fi < alpha_f]
        theta_f = max(low_theta, key=lambda x: x[1])[0] if low_theta else None

    ta = None
    ab = None
    if theta_f and alpha_f and theta_f < alpha_f:
        tb = [t for t in troughs_f if theta_f < t < alpha_f]
        ta = tb[0] if tb else (theta_f + alpha_f) / 2
    if alpha_f and beta_f and alpha_f < beta_f:
        tb = [t for t in troughs_f if alpha_f < t < beta_f]
        ab = tb[0] if tb else (alpha_f + beta_f) / 2

    return theta_f, alpha_f, beta_f, ta, ab


def fooof_to_boundaries(fooof_data):
    peaks = fooof_data.get('peaks', [])
    if not peaks:
        return None, None, None, None, None

    theta_cands = [(p[0], p[1]) for p in peaks if 3.5 <= p[0] <= 8.5]
    alpha_cands = [(p[0], p[1]) for p in peaks if 7.5 <= p[0] <= 14]
    beta_cands = [(p[0], p[1]) for p in peaks if 14 <= p[0] <= 30]

    theta_f = max(theta_cands, key=lambda x: x[1])[0] if theta_cands else None
    alpha_f = max(alpha_cands, key=lambda x: x[1])[0] if alpha_cands else None
    beta_f = max(beta_cands, key=lambda x: x[1])[0] if beta_cands else None

    if theta_f and alpha_f and theta_f >= alpha_f:
        low = [(f, p) for f, p in theta_cands if f < alpha_f]
        theta_f = max(low, key=lambda x: x[1])[0] if low else None

    ta = ab = None
    if theta_f and alpha_f and theta_f < alpha_f:
        ta = (theta_f + alpha_f) / 2
    if alpha_f and beta_f and alpha_f < beta_f:
        ab = (alpha_f + beta_f) / 2

    return theta_f, alpha_f, beta_f, ta, ab


print("=" * 70)
print("JOB 12: INDIVIDUALIZED BAND BOUNDARY ROBUSTNESS TEST")
print("=" * 70)

eeg_cached = json.load(open('outputs/eegbci_modal_results.json'))
aw_cached = json.load(open('outputs/aw_cached_subjects.json'))
ds_cached = json.load(open('outputs/ds003969_cached_subjects.json'))

all_subjects = []

print(f"\n  Part 1: EEGBCI — FOOOF-derived boundaries (N={len(eeg_cached)})")

eegbci_fooof_bounds = 0
for subj in eeg_cached:
    fooof_data = subj.get('fooof', {})
    theta_f, alpha_f, beta_f, ta, ab = fooof_to_boundaries(fooof_data)

    fixed_ratio = subj.get('alpha_theta_ratio')
    entry = {
        'subject_id': subj['subject_id'],
        'dataset': 'EEGBCI',
        'fs': subj.get('fs'),
        'fixed_ratio': fixed_ratio,
        'fixed_theta_centroid': subj['centroids']['theta'],
        'fixed_alpha_centroid': subj['centroids']['alpha'],
        'theta_peak': float(theta_f) if theta_f else None,
        'alpha_peak': float(alpha_f) if alpha_f else None,
        'beta_peak': float(beta_f) if beta_f else None,
        'theta_alpha_boundary': float(ta) if ta else None,
        'alpha_beta_boundary': float(ab) if ab else None,
        'boundary_source': 'fooof',
    }

    if ta and ab:
        eegbci_fooof_bounds += 1

    all_subjects.append(entry)

print(f"    FOOOF-derived boundaries: {eegbci_fooof_bounds}/{len(eeg_cached)}")

print(f"\n  Part 2: EEGBCI — Raw PSD trough detection (already downloaded)")

import mne
mne.set_log_level('ERROR')

edf_base = os.path.expanduser('~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0')
psd_count = 0

for entry in all_subjects:
    if entry['dataset'] != 'EEGBCI':
        continue
    sid_str = str(entry['subject_id']).replace('EEGBCI_', '').replace('S', '').replace('s', '')
    try:
        sid_num = int(sid_str)
    except:
        continue

    edf_path = os.path.join(edf_base, f'S{sid_num:03d}', f'S{sid_num:03d}R01.edf')
    if not os.path.exists(edf_path):
        continue

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        data = raw.get_data()
        fs = raw.info['sfreq']
        freqs, psd = compute_avg_psd(data, fs)

        theta_f, alpha_f, beta_f, ta, ab = find_psd_boundaries(freqs, psd)

        if ta and ab:
            entry['theta_peak_psd'] = float(theta_f) if theta_f else None
            entry['alpha_peak_psd'] = float(alpha_f) if alpha_f else None
            entry['theta_alpha_boundary_psd'] = float(ta)
            entry['alpha_beta_boundary_psd'] = float(ab)
            entry['boundary_source'] = 'psd+fooof'

            if not entry['theta_alpha_boundary']:
                entry['theta_alpha_boundary'] = float(ta)
                entry['alpha_beta_boundary'] = float(ab)

            theta_lo = max(2.0, ta - 6)
            indiv_theta_c = spectral_centroid(freqs, psd, theta_lo, ta)
            indiv_alpha_c = spectral_centroid(freqs, psd, ta, ab)
            if indiv_theta_c > 0:
                entry['indiv_ratio'] = float(indiv_alpha_c / indiv_theta_c)
                entry['indiv_theta_centroid'] = float(indiv_theta_c)
                entry['indiv_alpha_centroid'] = float(indiv_alpha_c)

            psd_count += 1
    except Exception:
        continue

print(f"    PSD-derived boundaries: {psd_count} subjects (from downloaded data)")

for ds_name, ds_data in [('AW', aw_cached), ('DS003969', ds_cached)]:
    count = 0
    for subj in ds_data:
        entry = {
            'subject_id': subj['subject_id'],
            'dataset': ds_name,
            'fs': subj.get('fs'),
            'fixed_ratio': subj.get('alpha_theta_ratio'),
            'fixed_theta_centroid': subj['centroids']['theta'],
            'fixed_alpha_centroid': subj['centroids']['alpha'],
            'theta_peak': None, 'alpha_peak': None, 'beta_peak': None,
            'theta_alpha_boundary': None, 'alpha_beta_boundary': None,
            'boundary_source': 'none',
        }
        all_subjects.append(entry)
        count += 1
    print(f"    {ds_name}: {count} subjects (no FOOOF peaks available, fixed boundaries only)")

print(f"\n  Total: {len(all_subjects)} subjects")

has_ta = [s for s in all_subjects if s.get('theta_alpha_boundary')]
has_both = [s for s in all_subjects if s.get('theta_alpha_boundary') and s.get('alpha_beta_boundary')]
has_indiv = [s for s in all_subjects if s.get('indiv_ratio') and 1.0 < s['indiv_ratio'] < 3.0]

print(f"  With θ-α boundary: {len(has_ta)}")
print(f"  With both boundaries: {len(has_both)}")
print(f"  With individualized ratio (PSD): {len(has_indiv)}")

print(f"\n{'='*70}")
print(f"  ANALYSIS A: Theta-Alpha Boundary Distribution")
print(f"{'='*70}")

ta_vals = np.array([s['theta_alpha_boundary'] for s in has_ta])
print(f"  N = {len(ta_vals)}")
print(f"  Mean:   {np.mean(ta_vals):.4f} Hz")
print(f"  Std:    {np.std(ta_vals):.4f} Hz")
print(f"  Median: {np.median(ta_vals):.4f} Hz")
print(f"  IQR:    [{np.percentile(ta_vals, 25):.4f}, {np.percentile(ta_vals, 75):.4f}]")
print(f"  Range:  [{np.min(ta_vals):.4f}, {np.max(ta_vals):.4f}]")
print(f"  Convention = 8 Hz;  deviation = {abs(np.mean(ta_vals) - 8):.4f} Hz")

t_8, p_8 = stats.ttest_1samp(ta_vals, 8.0)
print(f"  t-test vs 8 Hz: t={t_8:.3f}, p={p_8:.4f}")

ab_vals_list = [s['alpha_beta_boundary'] for s in has_both]
if ab_vals_list:
    ab_vals = np.array(ab_vals_list)
    print(f"\n  Alpha-Beta Boundary: N={len(ab_vals)}")
    print(f"  Mean:   {np.mean(ab_vals):.4f} Hz")
    print(f"  Std:    {np.std(ab_vals):.4f} Hz")
    print(f"  Convention = 13 Hz;  deviation = {abs(np.mean(ab_vals) - 13):.4f} Hz")

print(f"\n{'='*70}")
print(f"  ANALYSIS C: Individualized vs Fixed Ratios")
print(f"{'='*70}")

r_corr = None
t_p = None
if len(has_indiv) > 10:
    fixed_arr = np.array([s['fixed_ratio'] for s in has_indiv])
    indiv_arr = np.array([s['indiv_ratio'] for s in has_indiv])
    N_indiv = len(has_indiv)

    print(f"  N = {N_indiv}")
    print(f"  Fixed boundary mean:  {np.mean(fixed_arr):.4f}")
    print(f"  Indiv boundary mean:  {np.mean(indiv_arr):.4f}")
    shift = float(np.mean(indiv_arr) - np.mean(fixed_arr))
    print(f"  Shift:                {shift:+.4f}")
    print(f"  Fixed std:            {np.std(fixed_arr):.4f}")
    print(f"  Indiv std:            {np.std(indiv_arr):.4f}")

    r_corr, r_p = stats.pearsonr(fixed_arr, indiv_arr)
    print(f"  Correlation:          r={r_corr:.4f}, p={r_p:.2e}")

    t_stat, t_p = stats.ttest_rel(fixed_arr, indiv_arr)
    print(f"  Paired t-test:        t={t_stat:.3f}, p={t_p:.4f}")

    for name, val in [('e-1', E_MINUS_1), ('φ', PHI)]:
        df = abs(np.mean(fixed_arr) - val)
        di = abs(np.mean(indiv_arr) - val)
        print(f"  Distance from {name}: fixed={df:.4f}, indiv={di:.4f}")

    survived = r_corr > 0.5 and r_p < 0.001
    print(f"\n  LANDSCAPE SURVIVED? {'YES' if survived else 'UNCLEAR'} (r={r_corr:.3f}, p={r_p:.2e})")
else:
    fixed_arr = indiv_arr = None
    N_indiv = 0
    print(f"  Only {len(has_indiv)} subjects with individualized ratios — insufficient")

print(f"\n{'='*70}")
print(f"  ANALYSIS D: Boundary Ratio Organization")
print(f"{'='*70}")

boundary_ratios = []
for s in has_both:
    ta = s['theta_alpha_boundary']
    ab = s['alpha_beta_boundary']
    if ta > 2 and ab > ta:
        boundary_ratios.append(ab / ta)

br_arr = None
constants_test = {
    'φ': PHI, 'e-1': E_MINUS_1, '2:1': 2.0,
    '3:2': 1.5, '√2': np.sqrt(2), '13/8': 13/8,
}

if len(boundary_ratios) > 10:
    br_arr = np.array(boundary_ratios)
    print(f"  N = {len(br_arr)}")
    print(f"  αβ/θα boundary ratio:")
    print(f"    Mean:   {np.mean(br_arr):.4f}")
    print(f"    Std:    {np.std(br_arr):.4f}")
    print(f"    Median: {np.median(br_arr):.4f}")

    print(f"\n  Distance from constants:")
    for name in sorted(constants_test, key=lambda n: abs(np.mean(br_arr) - constants_test[n])):
        val = constants_test[name]
        dist = abs(np.mean(br_arr) - val)
        t_one, p_one = stats.ttest_1samp(br_arr, val)
        sig = '***' if p_one < 0.001 else ('**' if p_one < 0.01 else ('*' if p_one < 0.05 else 'n.s.'))
        print(f"    {name:6s} = {val:.4f}  dist={dist:.4f}  t={t_one:.3f}  p={p_one:.4f}  {sig}")

    winner_br = {name: 0 for name in constants_test}
    for r in br_arr:
        nearest = min(constants_test, key=lambda n: abs(r - constants_test[n]))
        winner_br[nearest] += 1
    print(f"\n  Winner tallies (boundary ratios):")
    for name in sorted(winner_br, key=winner_br.get, reverse=True):
        pct = winner_br[name] / len(br_arr) * 100
        print(f"    {name:6s}: {winner_br[name]:4d} ({pct:5.1f}%)")
else:
    print(f"  Too few boundary pairs: {len(boundary_ratios)}")

print(f"\n{'='*70}")
print(f"  ANALYSIS E: Boundary Deviation vs Landscape Position")
print(f"{'='*70}")

if len(has_indiv) > 10:
    boundary_dev = np.array([abs(s['theta_alpha_boundary'] - 8.0) for s in has_indiv])
    fixed_e = np.array([s['fixed_ratio'] for s in has_indiv])
    r_dev, p_dev = stats.pearsonr(boundary_dev, fixed_e)
    print(f"  r(|boundary-8Hz|, fixed ratio) = {r_dev:.4f}, p = {p_dev:.4f}")
else:
    r_dev = p_dev = 0
    print(f"  Insufficient data")

results = {
    'N_total': len(all_subjects),
    'N_with_ta_boundary': len(has_ta),
    'N_with_both_boundaries': len(has_both),
    'N_with_indiv_ratio': len(has_indiv),
    'theta_alpha_boundary': {
        'mean': float(np.mean(ta_vals)), 'std': float(np.std(ta_vals)),
        'median': float(np.median(ta_vals)),
    } if len(ta_vals) > 0 else None,
    'alpha_beta_boundary': {
        'mean': float(np.mean(ab_vals)), 'std': float(np.std(ab_vals)),
    } if ab_vals_list else None,
    'fixed_vs_indiv': {
        'fixed_mean': float(np.mean(fixed_arr)) if fixed_arr is not None else None,
        'indiv_mean': float(np.mean(indiv_arr)) if indiv_arr is not None else None,
        'r': float(r_corr) if r_corr is not None else None,
        'shift': float(shift) if fixed_arr is not None else None,
    },
    'boundary_ratios': {
        'mean': float(np.mean(br_arr)) if br_arr is not None else None,
        'std': float(np.std(br_arr)) if br_arr is not None else None,
    },
}

with open('outputs/gedbounds_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Generating figure...")

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

ax_a = fig.add_subplot(gs[0, 0])
if len(ta_vals) > 0:
    bins_ta = np.linspace(max(3, np.min(ta_vals)-0.5), min(14, np.max(ta_vals)+0.5), 35)
    ax_a.hist(ta_vals, bins=bins_ta, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5, density=True)
    ax_a.axvline(8.0, color='red', ls='--', lw=2, label='Convention (8 Hz)')
    ax_a.axvline(np.mean(ta_vals), color='black', ls='-', lw=2, label=f'Mean ({np.mean(ta_vals):.2f} Hz)')
    ax_a.set_xlabel('Theta-Alpha Boundary (Hz)')
    ax_a.set_ylabel('Density')
    ax_a.set_title(f'A. θ-α Boundary Distribution (N={len(ta_vals)})')
    ax_a.legend(fontsize=8)
    ax_a.text(0.98, 0.98, f'SD={np.std(ta_vals):.2f} Hz\np vs 8 Hz: {p_8:.4f}',
              transform=ax_a.transAxes, fontsize=8, va='top', ha='right',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax_b = fig.add_subplot(gs[0, 1])
if ab_vals_list:
    bins_ab = np.linspace(max(8, np.min(ab_vals)-0.5), min(25, np.max(ab_vals)+0.5), 35)
    ax_b.hist(ab_vals, bins=bins_ab, color='coral', alpha=0.7, edgecolor='white', linewidth=0.5, density=True)
    ax_b.axvline(13.0, color='red', ls='--', lw=2, label='Convention (13 Hz)')
    ax_b.axvline(np.mean(ab_vals), color='black', ls='-', lw=2, label=f'Mean ({np.mean(ab_vals):.2f} Hz)')
    ax_b.set_xlabel('Alpha-Beta Boundary (Hz)')
    ax_b.set_ylabel('Density')
    ax_b.set_title(f'B. α-β Boundary Distribution (N={len(ab_vals)})')
    ax_b.legend(fontsize=8)

ax_c = fig.add_subplot(gs[0, 2])
if fixed_arr is not None and indiv_arr is not None:
    ax_c.scatter(fixed_arr, indiv_arr, alpha=0.5, s=25, color='steelblue', edgecolors='black', linewidths=0.3)
    lims = [min(min(fixed_arr), min(indiv_arr))-0.1, max(max(fixed_arr), max(indiv_arr))+0.1]
    ax_c.plot(lims, lims, 'k--', lw=1, alpha=0.3, label='y=x')
    ax_c.axhline(E_MINUS_1, color='crimson', ls=':', alpha=0.5, label=f'e-1')
    ax_c.axvline(E_MINUS_1, color='crimson', ls=':', alpha=0.5)
    ax_c.axhline(PHI, color='goldenrod', ls=':', alpha=0.5, label=f'φ')
    ax_c.set_xlabel('Fixed Boundary α/θ Ratio')
    ax_c.set_ylabel('Individualized α/θ Ratio')
    ax_c.set_title(f'C. Fixed vs Indiv (r={r_corr:.3f})')
    ax_c.legend(fontsize=7)
    ax_c.text(0.02, 0.98, f'Shift: {shift:+.4f}\np = {t_p:.4f}',
              transform=ax_c.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
else:
    ax_c.text(0.5, 0.5, 'Insufficient data\nfor individualized ratios', ha='center', va='center',
              transform=ax_c.transAxes, fontsize=11)
    ax_c.set_title('C. Fixed vs Individualized')

ax_d = fig.add_subplot(gs[1, 0])
if fixed_arr is not None and indiv_arr is not None:
    bins_r = np.linspace(1.2, 2.4, 40)
    ax_d.hist(fixed_arr, bins=bins_r, alpha=0.5, color='steelblue', density=True,
              label=f'Fixed (μ={np.mean(fixed_arr):.4f})', edgecolor='white', linewidth=0.5)
    ax_d.hist(indiv_arr, bins=bins_r, alpha=0.5, color='coral', density=True,
              label=f'Indiv (μ={np.mean(indiv_arr):.4f})', edgecolor='white', linewidth=0.5)
    ax_d.axvline(E_MINUS_1, color='crimson', ls='--', lw=2, label='e-1')
    ax_d.axvline(PHI, color='goldenrod', ls='--', lw=2, label='φ')
    ax_d.set_xlabel('α/θ Frequency Ratio')
    ax_d.set_ylabel('Density')
    ax_d.set_title('D. Landscape: Fixed vs Individualized')
    ax_d.legend(fontsize=7)
else:
    all_fixed = [s['fixed_ratio'] for s in all_subjects if s.get('fixed_ratio') and 1 < s['fixed_ratio'] < 3]
    if all_fixed:
        bins_r = np.linspace(1.2, 2.4, 40)
        ax_d.hist(all_fixed, bins=bins_r, alpha=0.7, color='steelblue', density=True,
                  label=f'Fixed only (N={len(all_fixed)})', edgecolor='white', linewidth=0.5)
        ax_d.axvline(E_MINUS_1, color='crimson', ls='--', lw=2, label='e-1')
        ax_d.axvline(PHI, color='goldenrod', ls='--', lw=2, label='φ')
        ax_d.set_title('D. Fixed Ratios (no indiv available)')
        ax_d.legend(fontsize=7)

ax_e = fig.add_subplot(gs[1, 1])
if br_arr is not None and len(br_arr) > 5:
    bins_br = np.linspace(max(1.0, np.min(br_arr)-0.1), min(3.0, np.max(br_arr)+0.1), 35)
    ax_e.hist(br_arr, bins=bins_br, color='forestgreen', alpha=0.7, edgecolor='white', linewidth=0.5, density=True)
    c_colors = {'φ': 'goldenrod', 'e-1': 'crimson', '2:1': 'gray', '3:2': 'blue', '√2': 'purple', '13/8': 'teal'}
    for name, val in constants_test.items():
        ax_e.axvline(val, color=c_colors.get(name, 'gray'), ls='--', lw=1.5, alpha=0.7, label=f'{name}={val:.3f}')
    ax_e.axvline(np.mean(br_arr), color='black', ls='-', lw=2, label=f'Mean={np.mean(br_arr):.3f}')
    ax_e.set_xlabel('αβ/θα Boundary Ratio')
    ax_e.set_ylabel('Density')
    ax_e.set_title(f'E. Boundary Ratio Organization (N={len(br_arr)})')
    ax_e.legend(fontsize=6.5, ncol=2)
else:
    ax_e.text(0.5, 0.5, 'Insufficient boundary\npairs for analysis', ha='center', va='center',
              transform=ax_e.transAxes, fontsize=11)
    ax_e.set_title('E. Boundary Ratios')

ax_f = fig.add_subplot(gs[1, 2])
if len(has_indiv) > 10:
    bd = np.array([abs(s['theta_alpha_boundary'] - 8.0) for s in has_indiv])
    fr = np.array([s['fixed_ratio'] for s in has_indiv])
    ax_f.scatter(bd, fr, alpha=0.5, s=25, color='steelblue', edgecolors='black', linewidths=0.3)
    z = np.polyfit(bd, fr, 1)
    xf = np.linspace(0, max(bd), 50)
    ax_f.plot(xf, np.polyval(z, xf), 'r-', lw=2, alpha=0.5)
    ax_f.set_xlabel('|Individual Boundary − 8 Hz|')
    ax_f.set_ylabel('Fixed α/θ Ratio')
    ax_f.set_title(f'F. Boundary Deviation vs Ratio (r={r_dev:.3f})')
    ax_f.text(0.02, 0.98, f'p = {p_dev:.4f}', transform=ax_f.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
else:
    has_ta_eeg = [s for s in has_ta if s['dataset'] == 'EEGBCI']
    if has_ta_eeg:
        bd = np.array([abs(s['theta_alpha_boundary'] - 8.0) for s in has_ta_eeg])
        fr = np.array([s['fixed_ratio'] for s in has_ta_eeg if s.get('fixed_ratio')])
        if len(bd) == len(fr) and len(bd) > 5:
            ax_f.scatter(bd, fr, alpha=0.5, s=25, color='steelblue', edgecolors='black', linewidths=0.3)
            r_dev2, p_dev2 = stats.pearsonr(bd, fr)
            ax_f.set_xlabel('|Individual Boundary − 8 Hz|')
            ax_f.set_ylabel('Fixed α/θ Ratio')
            ax_f.set_title(f'F. Deviation vs Ratio (r={r_dev2:.3f})')

survived_str = 'SURVIVED' if (r_corr is not None and r_corr > 0.5) else 'NEEDS MORE DATA'
footer_parts = [f'Landscape: {survived_str}']
if len(ta_vals) > 0:
    footer_parts.append(f'θ-α bound: {np.mean(ta_vals):.2f}±{np.std(ta_vals):.2f} Hz')
footer_parts.append(f'N boundaries: {len(has_ta)}/{len(all_subjects)}')
if br_arr is not None:
    footer_parts.append(f'Bound ratio: {np.mean(br_arr):.3f}')

fig.text(0.5, 0.01, ' | '.join(footer_parts), ha='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

fig.suptitle('Figure 22: Individualized Band Boundaries (gedBounds) Robustness Test\n'
             'Does the α/θ ratio landscape survive under data-driven band definitions?',
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig(f'{OUTPUT_DIR}/fig22_gedbounds.png')
plt.close()

print(f"\n  Figure saved: {OUTPUT_DIR}/fig22_gedbounds.png")
print(f"  Results saved: outputs/gedbounds_results.json")

print(f"\n{'='*70}")
print(f"  CONCLUSION")
print(f"{'='*70}")
if len(ta_vals) > 0:
    print(f"  θ-α boundary: {np.mean(ta_vals):.2f} ± {np.std(ta_vals):.2f} Hz (convention: 8 Hz)")
if ab_vals_list:
    print(f"  α-β boundary: {np.mean(ab_vals):.2f} ± {np.std(ab_vals):.2f} Hz (convention: 13 Hz)")
if fixed_arr is not None:
    print(f"  Fixed ratio mean:  {np.mean(fixed_arr):.4f}")
    print(f"  Indiv ratio mean:  {np.mean(indiv_arr):.4f}")
    print(f"  Correlation: r={r_corr:.4f}")
    print(f"  Landscape: {'SURVIVED' if r_corr > 0.5 else 'UNCLEAR'}")
if br_arr is not None:
    nearest = min(constants_test, key=lambda n: abs(np.mean(br_arr) - constants_test[n]))
    print(f"  Boundary ratio: {np.mean(br_arr):.4f} (nearest: {nearest}={constants_test[nearest]:.4f})")
print(f"{'='*70}")

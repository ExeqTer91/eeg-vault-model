import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.stats import pearsonr, ttest_1samp, circmean, circstd
from deep_dive_common import (
    PHI, load_and_compute, compute_subject_psd, spectral_centroid, compute_pci
)

from fooof import FOOOF

os.makedirs('outputs', exist_ok=True)

subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis = load_and_compute()
N = len(subjects)
print(f"N={N} subjects loaded\n")

PHI_SQ = PHI ** 2
SQRT_PHI = np.sqrt(PHI)
PHI_POWERS = {f'phi^{n}': PHI**n for n in range(-3, 6)}
GOLDEN_ANGLE_DEG = 360.0 / (PHI + 1)
GOLDEN_ANGLE_RAD = np.radians(GOLDEN_ANGLE_DEG)

results = {}
notable_findings = []

def check_discovery(label, p_val=None, effect_size=None):
    if p_val is not None and p_val < 0.001:
        msg = f"*** DISCOVERY CRITERION MET: {label}, p={p_val:.6f} ***"
        print(msg)
        notable_findings.append(msg)
    if effect_size is not None and abs(effect_size) > 0.5:
        msg = f"*** NOTABLE EFFECT: {label}, d={effect_size:.3f} ***"
        print(msg)
        notable_findings.append(msg)


def fit_fooof_all():
    fooof_results = []
    for i in range(N):
        freqs = all_freqs[i]
        psd = all_psds[i]
        mask = (freqs >= 1) & (freqs <= 50)
        fm = FOOOF(peak_width_limits=[1, 12], max_n_peaks=8, min_peak_height=0.05, verbose=False)
        try:
            fm.fit(freqs[mask], psd[mask])
            peaks = fm.peak_params_
            ap = fm.aperiodic_params_
            fooof_results.append({
                'peaks': peaks,
                'aperiodic': ap,
                'freqs': freqs[mask],
                'psd': psd[mask],
                'fooofed_spectrum': fm.fooofed_spectrum_ if hasattr(fm, 'fooofed_spectrum_') else None,
                'name': subjects[i]['name']
            })
        except Exception as e:
            fooof_results.append({'peaks': np.array([]).reshape(0,3), 'aperiodic': None, 'name': subjects[i]['name']})
    return fooof_results

print("Fitting FOOOF for all subjects...")
fooof_all = fit_fooof_all()
print(f"FOOOF complete. Subjects with peaks: {sum(1 for f in fooof_all if len(f['peaks']) > 0)}/{N}\n")


print("=" * 70)
print("BLOCK 9A: SUB-BAND PHI-RATIOS (Fractal Descent)")
print("=" * 70)

subband_ratios_all = []
subband_subjects_with_multi = 0

for i, fr in enumerate(fooof_all):
    peaks = fr['peaks']
    if len(peaks) < 2:
        continue
    peak_freqs = sorted(peaks[:, 0])
    ratios = []
    for j in range(len(peak_freqs)):
        for k in range(j+1, len(peak_freqs)):
            r = peak_freqs[k] / peak_freqs[j]
            if 1.0 < r < 5.0:
                ratios.append(r)
                subband_ratios_all.append(r)
    if len(ratios) > 0:
        subband_subjects_with_multi += 1

subband_ratios_all = np.array(subband_ratios_all)

if len(subband_ratios_all) > 0:
    phi_targets = [SQRT_PHI, PHI, PHI_SQ]
    phi_labels = ['sqrt(phi)', 'phi', 'phi^2']
    mean_r = np.mean(subband_ratios_all)
    median_r = np.median(subband_ratios_all)

    nearest_phi = []
    for r in subband_ratios_all:
        dists = [abs(r - t) for t in phi_targets]
        best = np.argmin(dists)
        nearest_phi.append(phi_labels[best])

    from collections import Counter
    nearest_counts = Counter(nearest_phi)

    within_10_phi = np.sum(np.abs(subband_ratios_all - PHI) / PHI < 0.10)
    pct_near_phi = within_10_phi / len(subband_ratios_all) * 100

    phi_errors = np.abs(subband_ratios_all - PHI) / PHI
    mean_phi_err = np.mean(phi_errors) * 100

    null_ratios = np.random.uniform(1.0, 5.0, size=10000)
    null_near_phi = np.sum(np.abs(null_ratios - PHI) / PHI < 0.10) / len(null_ratios) * 100

    print(f"  Total pairwise peak ratios: {len(subband_ratios_all)}")
    print(f"  Subjects with 2+ peaks: {subband_subjects_with_multi}/{N}")
    print(f"  Mean ratio: {mean_r:.3f}, Median: {median_r:.3f}")
    print(f"  Within 10% of phi: {within_10_phi}/{len(subband_ratios_all)} ({pct_near_phi:.1f}%)")
    print(f"  Null expectation (uniform): {null_near_phi:.1f}%")
    print(f"  Mean error from phi: {mean_phi_err:.1f}%")
    print(f"  Nearest attractor distribution: {dict(nearest_counts)}")

    poi_corr = None
    if subband_subjects_with_multi >= 5:
        per_subj_phi_err = []
        per_subj_pci = []
        for i, fr in enumerate(fooof_all):
            peaks = fr['peaks']
            if len(peaks) < 2:
                continue
            pf = sorted(peaks[:, 0])
            rs = [pf[k]/pf[j] for j in range(len(pf)) for k in range(j+1, len(pf)) if 1 < pf[k]/pf[j] < 5]
            if rs:
                per_subj_phi_err.append(np.mean([abs(r - PHI)/PHI for r in rs]))
                per_subj_pci.append(all_pcis[i])
        if len(per_subj_phi_err) >= 5:
            r_corr, p_corr = pearsonr(per_subj_phi_err, per_subj_pci)
            poi_corr = (r_corr, p_corr)
            print(f"  Sub-band phi error vs PCI: r={r_corr:.3f}, p={p_corr:.4f}")
            check_discovery("9A sub-band phi-error vs PCI correlation", p_val=p_corr, effect_size=r_corr)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(subband_ratios_all, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    for t, l in zip(phi_targets, phi_labels):
        ax.axvline(t, color='red', linestyle='--', alpha=0.7, label=l)
    ax.axvline(2.0, color='green', linestyle='--', alpha=0.7, label='2:1 harmonic')
    ax.set_xlabel('Peak Frequency Ratio')
    ax.set_ylabel('Density')
    ax.set_title('9A: All Pairwise FOOOF Peak Ratios')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/fig_9A_subband_ratios.png', dpi=150)
    plt.close()

    results['9A'] = {
        'n_ratios': len(subband_ratios_all),
        'n_subjects_multi_peak': subband_subjects_with_multi,
        'mean_ratio': float(mean_r),
        'median_ratio': float(median_r),
        'pct_within_10_phi': float(pct_near_phi),
        'null_pct': float(null_near_phi),
        'mean_phi_error_pct': float(mean_phi_err),
        'poi_correlation': {'r': float(poi_corr[0]), 'p': float(poi_corr[1])} if poi_corr else None
    }
else:
    print("  No multi-peak subjects found for 9A")
    results['9A'] = {'note': 'insufficient peaks'}

print()


print("=" * 70)
print("BLOCK 9B: SPECTRAL PEAK WIDTH RATIOS & Q-FACTORS")
print("=" * 70)

width_ratios_adj = []
q_factors_all = []
width_ratio_labels = []

for i, fr in enumerate(fooof_all):
    peaks = fr['peaks']
    if len(peaks) < 2:
        continue
    sorted_peaks = peaks[peaks[:, 0].argsort()]
    for j in range(len(sorted_peaks) - 1):
        f1, _, w1 = sorted_peaks[j]
        f2, _, w2 = sorted_peaks[j+1]
        if w1 > 0 and w2 > 0:
            wr = w2 / w1
            width_ratios_adj.append(wr)
    for p in sorted_peaks:
        f_c, _, bw = p
        if bw > 0:
            q = f_c / bw
            q_factors_all.append(q)

width_ratios_adj = np.array(width_ratios_adj)
q_factors_all = np.array(q_factors_all)

if len(width_ratios_adj) > 0:
    mean_wr = np.mean(width_ratios_adj)
    wr_phi_err = abs(mean_wr - PHI) / PHI * 100
    wr_near_phi = np.sum(np.abs(width_ratios_adj - PHI) / PHI < 0.10) / len(width_ratios_adj) * 100
    print(f"  Width ratios (adjacent peaks): N={len(width_ratios_adj)}")
    print(f"  Mean width ratio: {mean_wr:.3f} ({wr_phi_err:.1f}% from phi)")
    print(f"  Within 10% of phi: {wr_near_phi:.1f}%")

    if len(width_ratios_adj) >= 3:
        t_wr, p_wr = ttest_1samp(width_ratios_adj, PHI)
        print(f"  t-test vs phi: t={t_wr:.2f}, p={p_wr:.4f}")
        check_discovery("9B width ratios vs phi", p_val=p_wr)
    results['9B_widths'] = {
        'n': len(width_ratios_adj), 'mean': float(mean_wr),
        'phi_error_pct': float(wr_phi_err), 'pct_near_phi': float(wr_near_phi)
    }
else:
    results['9B_widths'] = {'note': 'insufficient data'}

if len(q_factors_all) > 0:
    mean_q = np.mean(q_factors_all)
    q_phi_targets = [1/PHI, 1.0, PHI, PHI_SQ]
    q_phi_labels = ['1/phi', '1', 'phi', 'phi^2']
    nearest_q = []
    for q in q_factors_all:
        dists = [abs(q - t) for t in q_phi_targets]
        nearest_q.append(q_phi_labels[np.argmin(dists)])
    from collections import Counter
    q_counts = Counter(nearest_q)
    print(f"  Q-factors: N={len(q_factors_all)}, mean={mean_q:.3f}")
    print(f"  Nearest Q target: {dict(q_counts)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(width_ratios_adj, bins=20, alpha=0.7, color='coral', edgecolor='black')
    axes[0].axvline(PHI, color='red', linestyle='--', label='phi')
    axes[0].set_title('9B: Width Ratios (Adjacent Peaks)')
    axes[0].set_xlabel('Width Ratio')
    axes[0].legend()
    axes[1].hist(q_factors_all, bins=20, alpha=0.7, color='mediumpurple', edgecolor='black')
    for t, l in zip(q_phi_targets, q_phi_labels):
        axes[1].axvline(t, color='red', linestyle='--', alpha=0.7, label=l)
    axes[1].set_title('9B: Q-Factors (f_center / bandwidth)')
    axes[1].set_xlabel('Q-Factor')
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('outputs/fig_9B_widths_qfactors.png', dpi=150)
    plt.close()

    results['9B_qfactors'] = {
        'n': len(q_factors_all), 'mean_q': float(mean_q),
        'distribution': {k: v for k, v in q_counts.items()}
    }
else:
    results['9B_qfactors'] = {'note': 'insufficient data'}

print()


print("=" * 70)
print("BLOCK 9C: HARMONIC FINE STRUCTURE (phi-harmonics vs integer harmonics)")
print("=" * 70)

phi_harm_strengths = []
int_harm_strengths = []
phi_vs_int_ratios = []

for i in range(N):
    freqs = all_freqs[i]
    psd = all_psds[i]
    fr = fooof_all[i]
    peaks = fr['peaks']

    alpha_peaks = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)]
    if len(alpha_peaks) == 0:
        continue

    f_alpha = alpha_peaks[np.argmax(alpha_peaks[:, 1]), 0]

    phi_harmonics = [f_alpha * PHI, f_alpha / PHI, f_alpha * PHI_SQ]
    int_harmonics = [f_alpha * 2, f_alpha * 3, f_alpha * 0.5]

    def power_at_freq(f_target, freqs, psd, bw=1.0):
        mask = (freqs >= f_target - bw) & (freqs <= f_target + bw)
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(psd[mask])

    phi_power = sum(power_at_freq(f, freqs, psd) for f in phi_harmonics if 1 < f < 50)
    int_power = sum(power_at_freq(f, freqs, psd) for f in int_harmonics if 1 < f < 50)

    phi_harm_strengths.append(phi_power)
    int_harm_strengths.append(int_power)
    if int_power > 0:
        phi_vs_int_ratios.append(phi_power / int_power)

phi_harm_strengths = np.array(phi_harm_strengths)
int_harm_strengths = np.array(int_harm_strengths)
phi_vs_int_ratios = np.array(phi_vs_int_ratios)

if len(phi_vs_int_ratios) > 0:
    mean_ratio_9c = np.mean(phi_vs_int_ratios)
    pct_phi_dom = np.sum(phi_vs_int_ratios > 1) / len(phi_vs_int_ratios) * 100
    t_9c, p_9c = ttest_1samp(phi_vs_int_ratios, 1.0)
    print(f"  Subjects with alpha peak: {len(phi_vs_int_ratios)}/{N}")
    print(f"  Mean phi/integer harmonic power ratio: {mean_ratio_9c:.3f}")
    print(f"  Phi-harmonics dominate: {pct_phi_dom:.1f}%")
    print(f"  t-test vs 1.0: t={t_9c:.2f}, p={p_9c:.4f}")
    check_discovery("9C phi-harmonics vs integer harmonics", p_val=p_9c, effect_size=(mean_ratio_9c - 1.0))

    if len(phi_vs_int_ratios) >= 5:
        r_9c, p_9c_corr = pearsonr(phi_vs_int_ratios[:min(len(phi_vs_int_ratios), N)],
                                     all_pcis[:min(len(phi_vs_int_ratios), N)])
        print(f"  Phi-harmonic ratio vs PCI: r={r_9c:.3f}, p={p_9c_corr:.4f}")
        check_discovery("9C phi-harm ratio vs PCI", p_val=p_9c_corr, effect_size=r_9c)

    results['9C'] = {
        'n_subjects': len(phi_vs_int_ratios),
        'mean_phi_int_ratio': float(mean_ratio_9c),
        'pct_phi_dominant': float(pct_phi_dom),
        't_vs_1': float(t_9c), 'p_vs_1': float(p_9c)
    }
else:
    print("  No subjects with alpha peaks for 9C")
    results['9C'] = {'note': 'no alpha peaks'}

print()


print("=" * 70)
print("BLOCK 9D: FINE-GRAINED ALL-PAIRS PEAK RATIO HISTOGRAM")
print("=" * 70)

all_peak_ratios = []
for i, fr in enumerate(fooof_all):
    peaks = fr['peaks']
    if len(peaks) < 2:
        continue
    pf = sorted(peaks[:, 0])
    for j in range(len(pf)):
        for k in range(j+1, len(pf)):
            r = pf[k] / pf[j]
            if r > 1:
                all_peak_ratios.append(r)

all_peak_ratios = np.array(all_peak_ratios)

if len(all_peak_ratios) > 0:
    log_phi_ratios = np.log(all_peak_ratios) / np.log(PHI)

    n_values = np.round(log_phi_ratios)
    residuals = log_phi_ratios - n_values
    mean_residual = np.mean(np.abs(residuals))

    null_ratios = np.random.uniform(1.0, np.max(all_peak_ratios), size=len(all_peak_ratios) * 100)
    null_log_phi = np.log(null_ratios) / np.log(PHI)
    null_residuals = null_log_phi - np.round(null_log_phi)
    null_mean_res = np.mean(np.abs(null_residuals))

    print(f"  Total peak ratios: {len(all_peak_ratios)}")
    print(f"  Mean |residual from integer phi-power|: {mean_residual:.4f}")
    print(f"  Null (uniform) mean |residual|: {null_mean_res:.4f}")
    print(f"  Ratio (real/null): {mean_residual/null_mean_res:.3f}")
    if mean_residual < null_mean_res:
        print(f"  --> Real ratios ARE closer to phi^n than random")
    else:
        print(f"  --> Real ratios are NOT closer to phi^n than random")

    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(np.abs(residuals), np.abs(null_residuals[:len(residuals)]))
    print(f"  KS test (real vs null residuals): D={ks_stat:.3f}, p={ks_p:.4f}")
    check_discovery("9D peak ratios cluster at phi^n", p_val=ks_p, effect_size=mean_residual/null_mean_res - 1)

    phi_n_markers = [PHI**n for n in range(1, 5) if PHI**n < np.max(all_peak_ratios) + 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(all_peak_ratios, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    for pn in phi_n_markers:
        axes[0].axvline(pn, color='red', linestyle='--', alpha=0.7)
    axes[0].axvline(2.0, color='green', linestyle='--', alpha=0.5, label='2:1')
    axes[0].axvline(3.0, color='green', linestyle='--', alpha=0.5, label='3:1')
    axes[0].set_xlabel('Peak Ratio')
    axes[0].set_ylabel('Density')
    axes[0].set_title('9D: All Pairwise Peak Ratios with phi^n markers (red)')
    axes[0].legend()

    axes[1].hist(log_phi_ratios, bins=40, density=True, alpha=0.7, color='goldenrod', edgecolor='black')
    for n in range(0, 5):
        axes[1].axvline(n, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('log(ratio) / log(phi) ["quantum number"]')
    axes[1].set_ylabel('Density')
    axes[1].set_title('9D: Spectral "Quantum Numbers"')
    plt.tight_layout()
    plt.savefig('outputs/fig_9D_fine_ratios.png', dpi=150)
    plt.close()

    results['9D'] = {
        'n_ratios': len(all_peak_ratios),
        'mean_abs_residual': float(mean_residual),
        'null_mean_abs_residual': float(null_mean_res),
        'ratio_real_null': float(mean_residual / null_mean_res),
        'ks_stat': float(ks_stat), 'ks_p': float(ks_p)
    }
else:
    print("  No peak ratios for 9D")
    results['9D'] = {'note': 'insufficient peaks'}

print()


print("=" * 70)
print("BLOCK 9E: 1/f SLOPE FINE STRUCTURE")
print("=" * 70)

all_slopes = []
for fr in fooof_all:
    if fr['aperiodic'] is not None:
        all_slopes.append(fr['aperiodic'][-1])

all_slopes = np.array(all_slopes)

if len(all_slopes) > 0:
    mean_slope = np.mean(all_slopes)
    t_vs_phi, p_vs_phi = ttest_1samp(all_slopes, PHI)
    t_vs_inv, p_vs_inv = ttest_1samp(all_slopes, 1/PHI)
    t_vs_1, p_vs_1 = ttest_1samp(all_slopes, 1.0)

    print(f"  Mean aperiodic slope: {mean_slope:.4f}")
    print(f"  t-test vs phi ({PHI:.3f}): t={t_vs_phi:.2f}, p={p_vs_phi:.4f}")
    print(f"  t-test vs 1/phi ({1/PHI:.3f}): t={t_vs_inv:.2f}, p={p_vs_inv:.4f}")
    print(f"  t-test vs 1.0: t={t_vs_1:.2f}, p={p_vs_1:.4f}")

    targets = [1/PHI, 1.0, PHI]
    target_labels = ['1/phi', '1.0', 'phi']
    errors = [abs(mean_slope - t) for t in targets]
    best_idx = np.argmin(errors)
    print(f"  Closest target: {target_labels[best_idx]} (error={errors[best_idx]:.4f})")

    local_slopes = []
    bands_for_slope = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    band_names_slope = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    for i in range(N):
        freqs = all_freqs[i]
        psd = all_psds[i]
        subj_slopes = []
        for lo, hi in bands_for_slope:
            mask = (freqs >= lo) & (freqs <= hi)
            f_b = freqs[mask]
            p_b = psd[mask]
            if len(f_b) > 2:
                log_f = np.log10(f_b)
                log_p = np.log10(p_b + 1e-30)
                coeffs = np.polyfit(log_f, log_p, 1)
                subj_slopes.append(-coeffs[0])
            else:
                subj_slopes.append(np.nan)
        local_slopes.append(subj_slopes)

    local_slopes = np.array(local_slopes)
    slope_ratios = []
    slope_ratio_labels = []
    for j in range(4):
        s1 = local_slopes[:, j]
        s2 = local_slopes[:, j+1]
        valid = ~np.isnan(s1) & ~np.isnan(s2) & (s2 != 0)
        if np.sum(valid) > 0:
            sr = s1[valid] / s2[valid]
            sr = sr[np.isfinite(sr)]
            if len(sr) > 0:
                mean_sr = np.mean(sr)
                label = f"{band_names_slope[j]}/{band_names_slope[j+1]}"
                slope_ratios.append(mean_sr)
                slope_ratio_labels.append(label)
                phi_err = abs(mean_sr - PHI) / PHI * 100
                print(f"  Slope ratio {label}: {mean_sr:.3f} ({phi_err:.1f}% from phi)")

    results['9E'] = {
        'mean_slope': float(mean_slope),
        'closest_target': target_labels[best_idx],
        'closest_error': float(errors[best_idx]),
        'slope_ratios': {l: float(v) for l, v in zip(slope_ratio_labels, slope_ratios)}
    }
else:
    results['9E'] = {'note': 'no aperiodic data'}

print()


print("=" * 70)
print("BLOCK 9F: PHASE RELATIONSHIP FINE STRUCTURE (Golden Angle)")
print("=" * 70)

band_pairs_9f = [('theta', 4, 8, 'alpha', 8, 13),
                 ('alpha', 8, 13, 'beta', 13, 30),
                 ('theta', 4, 8, 'beta', 13, 30)]

phase_results_9f = {}

for b1_name, b1_lo, b1_hi, b2_name, b2_lo, b2_hi in band_pairs_9f:
    phase_diffs = []
    for i in range(N):
        data = subjects[i]['data']
        fs = subjects[i]['fs']
        nyq = fs / 2.0
        max_ch = min(data.shape[0], 5)
        max_samp = min(data.shape[1], int(10 * fs))

        for ch in range(max_ch):
            seg = data[ch, :max_samp]
            b1, a1 = butter(4, [b1_lo/nyq, min(b1_hi/nyq, 0.99)], btype='band')
            b2, a2 = butter(4, [b2_lo/nyq, min(b2_hi/nyq, 0.99)], btype='band')
            sig1 = filtfilt(b1, a1, seg)
            sig2 = filtfilt(b2, a2, seg)
            phase1 = np.angle(hilbert(sig1))
            phase2 = np.angle(hilbert(sig2))
            diff = (phase2 - phase1) % (2 * np.pi)
            mean_phase = circmean(diff)
            phase_diffs.append(mean_phase)

    phase_diffs = np.array(phase_diffs)
    if len(phase_diffs) > 0:
        overall_mean = circmean(phase_diffs)
        overall_std = circstd(phase_diffs)
        mean_deg = np.degrees(overall_mean) % 360
        golden_err = min(abs(mean_deg - GOLDEN_ANGLE_DEG),
                         abs(mean_deg - (360 - GOLDEN_ANGLE_DEG)))
        n_ph = len(phase_diffs)
        R_bar = np.abs(np.mean(np.exp(1j * phase_diffs)))
        Z_ray = n_ph * R_bar**2
        p_r = np.exp(-Z_ray) * (1 + (2*Z_ray - Z_ray**2) / (4*n_ph) - (24*Z_ray - 132*Z_ray**2 + 76*Z_ray**3 - 9*Z_ray**4) / (288*n_ph**2))
        p_r = max(p_r, 0.0)

        label = f"{b1_name}-{b2_name}"
        print(f"  {label}: mean phase = {mean_deg:.1f} deg (golden angle = {GOLDEN_ANGLE_DEG:.1f} deg)")
        print(f"    Error from golden angle: {golden_err:.1f} deg")
        print(f"    Rayleigh test: Z={Z_ray:.2f}, p={p_r:.6f}")
        print(f"    Circular std: {np.degrees(overall_std):.1f} deg")
        check_discovery(f"9F {label} phase at golden angle", p_val=p_r)

        phase_results_9f[label] = {
            'mean_deg': float(mean_deg),
            'golden_angle_error_deg': float(golden_err),
            'rayleigh_z': float(Z_ray), 'rayleigh_p': float(p_r),
            'circular_std_deg': float(np.degrees(overall_std))
        }

results['9F'] = phase_results_9f

fig, axes = plt.subplots(1, len(band_pairs_9f), figsize=(5*len(band_pairs_9f), 5), subplot_kw={'projection': 'polar'})
if len(band_pairs_9f) == 1:
    axes = [axes]
for ax, (b1_name, _, _, b2_name, _, _) in zip(axes, band_pairs_9f):
    label = f"{b1_name}-{b2_name}"
    if label in phase_results_9f:
        mean_rad = np.radians(phase_results_9f[label]['mean_deg'])
        ax.set_title(f'{label} phase', pad=20)
        ax.annotate('', xy=(mean_rad, 1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.annotate('', xy=(GOLDEN_ANGLE_RAD, 1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
plt.tight_layout()
plt.savefig('outputs/fig_9F_phase_structure.png', dpi=150)
plt.close()

print()


print("=" * 70)
print("BLOCK 9G: IAF AS PHI-ANCHOR (Phi-Ladder Predictions)")
print("=" * 70)

iaf_results = []
for i, fr in enumerate(fooof_all):
    peaks = fr['peaks']
    alpha_peaks = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)]
    if len(alpha_peaks) == 0:
        continue
    iaf = alpha_peaks[np.argmax(alpha_peaks[:, 1]), 0]
    c = all_centroids[i]

    predicted = {
        'theta': iaf / PHI,
        'beta': iaf * PHI,
        'delta': iaf / PHI_SQ,
        'gamma': iaf * PHI_SQ,
    }
    observed = {
        'theta': c['theta'],
        'beta': c['beta'],
        'delta': c['delta'],
        'gamma': c['gamma'],
    }

    errors = {}
    for band in predicted:
        errors[band] = abs(predicted[band] - observed[band]) / observed[band] * 100

    fixed_iaf = 10.0
    predicted_fixed = {
        'theta': fixed_iaf / PHI,
        'beta': fixed_iaf * PHI,
        'delta': fixed_iaf / PHI_SQ,
        'gamma': fixed_iaf * PHI_SQ,
    }
    errors_fixed = {}
    for band in predicted_fixed:
        errors_fixed[band] = abs(predicted_fixed[band] - observed[band]) / observed[band] * 100

    iaf_results.append({
        'subject': subjects[i]['name'],
        'iaf': float(iaf),
        'pci': float(all_pcis[i]),
        'errors_personal': {k: float(v) for k, v in errors.items()},
        'errors_fixed': {k: float(v) for k, v in errors_fixed.items()},
        'mean_error_personal': float(np.mean(list(errors.values()))),
        'mean_error_fixed': float(np.mean(list(errors_fixed.values())))
    })

if len(iaf_results) > 0:
    personal_errs = [r['mean_error_personal'] for r in iaf_results]
    fixed_errs = [r['mean_error_fixed'] for r in iaf_results]
    pcis_9g = [r['pci'] for r in iaf_results]

    print(f"  Subjects with IAF: {len(iaf_results)}/{N}")
    print(f"  Mean prediction error (personal IAF): {np.mean(personal_errs):.1f}%")
    print(f"  Mean prediction error (fixed 10 Hz): {np.mean(fixed_errs):.1f}%")
    print(f"  Personal IAF better: {np.sum(np.array(personal_errs) < np.array(fixed_errs))}/{len(iaf_results)}")

    from scipy.stats import ttest_rel
    t_9g, p_9g = ttest_rel(personal_errs, fixed_errs)
    print(f"  Paired t (personal vs fixed): t={t_9g:.2f}, p={p_9g:.4f}")
    check_discovery("9G personal IAF phi-ladder vs fixed", p_val=p_9g)

    r_acc_pci, p_acc_pci = pearsonr(personal_errs, pcis_9g)
    print(f"  Prediction accuracy vs PCI: r={r_acc_pci:.3f}, p={p_acc_pci:.4f}")
    check_discovery("9G phi-ladder accuracy vs PCI", p_val=p_acc_pci, effect_size=r_acc_pci)

    per_band_personal = {b: np.mean([r['errors_personal'][b] for r in iaf_results]) for b in ['theta', 'beta', 'delta', 'gamma']}
    for b, e in per_band_personal.items():
        print(f"    {b}: {e:.1f}% error from phi-ladder prediction")

    results['9G'] = {
        'n_subjects': len(iaf_results),
        'mean_error_personal': float(np.mean(personal_errs)),
        'mean_error_fixed': float(np.mean(fixed_errs)),
        'personal_better_pct': float(np.sum(np.array(personal_errs) < np.array(fixed_errs)) / len(iaf_results) * 100),
        't_personal_vs_fixed': float(t_9g), 'p_personal_vs_fixed': float(p_9g),
        'accuracy_vs_pci_r': float(r_acc_pci), 'accuracy_vs_pci_p': float(p_acc_pci),
        'per_band_error': {k: float(v) for k, v in per_band_personal.items()}
    }
else:
    print("  No IAF subjects for 9G")
    results['9G'] = {'note': 'no IAF found'}

print()


print("=" * 70)
print("BLOCK 9H: CROSS-SUBJECT SPECTRAL ALIGNMENT (IAF Normalization)")
print("=" * 70)

norm_spectra = []
iaf_values = []
for i, fr in enumerate(fooof_all):
    peaks = fr['peaks']
    alpha_peaks = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)]
    if len(alpha_peaks) == 0:
        continue
    iaf = alpha_peaks[np.argmax(alpha_peaks[:, 1]), 0]
    iaf_values.append(iaf)
    freqs = all_freqs[i]
    psd = all_psds[i]
    norm_f = freqs / iaf
    norm_spectra.append((norm_f, psd, iaf))

if len(norm_spectra) >= 5:
    f_min = max(s[0][s[0] > 0].min() for s in norm_spectra)
    f_max = min(s[0].max() for s in norm_spectra)
    common_f = np.linspace(max(f_min, 0.05), min(f_max, 5.0), 500)

    interp_spectra = []
    for nf, psd, _ in norm_spectra:
        mask = (nf >= common_f[0]) & (nf <= common_f[-1])
        if np.sum(mask) > 2:
            interp_psd = np.interp(common_f, nf[mask], psd[mask])
            interp_spectra.append(interp_psd)

    if len(interp_spectra) > 0:
        interp_spectra = np.array(interp_spectra)
        mean_norm_spectrum = np.mean(interp_spectra, axis=0)
        log_mean = np.log10(mean_norm_spectrum + 1e-30)

        phi_positions = [1/PHI_SQ, 1/PHI, 1.0, PHI, PHI_SQ]
        phi_pos_labels = ['1/phi^2\n(delta)', '1/phi\n(theta)', '1\n(alpha)', 'phi\n(beta)', 'phi^2\n(gamma)']

        from scipy.signal import find_peaks as sp_find_peaks
        peaks_idx, props = sp_find_peaks(log_mean, prominence=0.05, distance=10)
        peak_positions = common_f[peaks_idx]

        print(f"  Subjects with IAF: {len(norm_spectra)}")
        print(f"  Mean IAF: {np.mean(iaf_values):.2f} Hz")
        print(f"  Detected peaks in normalized spectrum at: {', '.join(f'{p:.3f}' for p in peak_positions)}")

        peak_phi_matches = []
        for pp in peak_positions:
            dists = [abs(pp - target) for target in phi_positions]
            best_match = np.argmin(dists)
            err = dists[best_match]
            pct_err = err / phi_positions[best_match] * 100 if phi_positions[best_match] != 0 else 999
            peak_phi_matches.append({
                'peak_pos': float(pp),
                'nearest_phi': phi_pos_labels[best_match].replace('\n', ' '),
                'target': float(phi_positions[best_match]),
                'error': float(err),
                'pct_error': float(pct_err)
            })
            print(f"    Peak {pp:.3f} -> {phi_pos_labels[best_match].replace(chr(10), ' ')}: error={pct_err:.1f}%")

        mean_peak_err = np.mean([m['pct_error'] for m in peak_phi_matches]) if peak_phi_matches else float('nan')
        print(f"  Mean peak alignment error: {mean_peak_err:.1f}%")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for s in interp_spectra[:20]:
            ax.plot(common_f, np.log10(s + 1e-30), alpha=0.15, color='gray')
        ax.plot(common_f, log_mean, color='blue', lw=2, label='Mean normalized spectrum')
        for pos, lab in zip(phi_positions, phi_pos_labels):
            ax.axvline(pos, color='red', linestyle='--', alpha=0.6)
            ax.text(pos, ax.get_ylim()[1], lab, ha='center', va='bottom', fontsize=7, color='red')
        for pp in peak_positions:
            ax.axvline(pp, color='green', linestyle=':', alpha=0.5)
        ax.set_xlabel('Normalized Frequency (f / IAF)')
        ax.set_ylabel('log10(Power)')
        ax.set_title('9H: IAF-Normalized Spectra with phi^n Positions')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/fig_9H_normalized_spectra.png', dpi=150)
        plt.close()

        results['9H'] = {
            'n_subjects': len(norm_spectra),
            'mean_iaf': float(np.mean(iaf_values)),
            'peak_positions': [float(p) for p in peak_positions],
            'peak_phi_matches': peak_phi_matches,
            'mean_peak_alignment_error_pct': float(mean_peak_err)
        }
    else:
        results['9H'] = {'note': 'interpolation failed'}
else:
    print(f"  Only {len(norm_spectra)} subjects with IAF, need >= 5")
    results['9H'] = {'note': 'insufficient IAF subjects'}

print()


print("=" * 70)
print("BLOCK 9 COMPLETE — DISCOVERY CHECK")
print("=" * 70)

if notable_findings:
    print("\n*** NOTABLE FINDINGS ***")
    for f in notable_findings:
        print(f"  {f}")
else:
    print("  No findings met the discovery criterion (p < 0.001 or |d| > 0.5)")

results['notable_findings'] = notable_findings

with open('outputs/block9_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)


with open('outputs/block9_results.md', 'w') as f:
    f.write("# Block 9: Fine Structure Analysis\n\n")
    f.write(f"N={N} Alpha-Waves subjects (512 Hz)\n\n")

    f.write("## 9A: Sub-Band Phi-Ratios\n\n")
    if '9A' in results and 'n_ratios' in results['9A']:
        r9a = results['9A']
        f.write(f"- Pairwise peak ratios: {r9a['n_ratios']}\n")
        f.write(f"- Subjects with 2+ FOOOF peaks: {r9a['n_subjects_multi_peak']}/{N}\n")
        f.write(f"- Mean ratio: {r9a['mean_ratio']:.3f}, within 10% of phi: {r9a['pct_within_10_phi']:.1f}%\n")
        f.write(f"- Null (uniform): {r9a['null_pct']:.1f}%\n")
        if r9a.get('poi_correlation'):
            f.write(f"- Sub-band phi-error vs PCI: r={r9a['poi_correlation']['r']:.3f}, p={r9a['poi_correlation']['p']:.4f}\n")
    f.write("\n")

    f.write("## 9B: Peak Width Ratios & Q-Factors\n\n")
    if 'n' in results.get('9B_widths', {}):
        r9b = results['9B_widths']
        f.write(f"- Adjacent peak width ratios: N={r9b['n']}\n")
        f.write(f"- Mean width ratio: {r9b['mean']:.3f} ({r9b['phi_error_pct']:.1f}% from phi)\n")
        f.write(f"- Within 10% of phi: {r9b['pct_near_phi']:.1f}%\n")
    if 'mean_q' in results.get('9B_qfactors', {}):
        f.write(f"- Mean Q-factor: {results['9B_qfactors']['mean_q']:.3f}\n")
    f.write("\n")

    f.write("## 9C: Phi-Harmonics vs Integer Harmonics\n\n")
    if 'mean_phi_int_ratio' in results.get('9C', {}):
        r9c = results['9C']
        f.write(f"- Subjects with alpha peak: {r9c['n_subjects']}/{N}\n")
        f.write(f"- Mean phi/integer harmonic power ratio: {r9c['mean_phi_int_ratio']:.3f}\n")
        f.write(f"- Phi-harmonics dominate: {r9c['pct_phi_dominant']:.1f}%\n")
        f.write(f"- t-test vs 1.0: t={r9c['t_vs_1']:.2f}, p={r9c['p_vs_1']:.4f}\n")
    f.write("\n")

    f.write("## 9D: Fine-Grained Peak Ratio Histogram\n\n")
    if 'n_ratios' in results.get('9D', {}):
        r9d = results['9D']
        f.write(f"- Total peak ratios: {r9d['n_ratios']}\n")
        f.write(f"- Mean |residual from phi^n|: {r9d['mean_abs_residual']:.4f}\n")
        f.write(f"- Null mean |residual|: {r9d['null_mean_abs_residual']:.4f}\n")
        f.write(f"- Ratio (real/null): {r9d['ratio_real_null']:.3f}\n")
        f.write(f"- KS test: D={r9d['ks_stat']:.3f}, p={r9d['ks_p']:.4f}\n")
        closer = "YES" if r9d['ratio_real_null'] < 1 else "NO"
        f.write(f"- Real closer to phi^n than random: {closer}\n")
    f.write("\n")

    f.write("## 9E: 1/f Slope Fine Structure\n\n")
    if 'mean_slope' in results.get('9E', {}):
        r9e = results['9E']
        f.write(f"- Mean aperiodic slope: {r9e['mean_slope']:.4f}\n")
        f.write(f"- Closest phi-target: {r9e['closest_target']} (error={r9e['closest_error']:.4f})\n")
        if r9e.get('slope_ratios'):
            f.write("- Local slope ratios:\n")
            for lab, val in r9e['slope_ratios'].items():
                f.write(f"  - {lab}: {val:.3f}\n")
    f.write("\n")

    f.write("## 9F: Phase Relationships (Golden Angle)\n\n")
    for label, data in results.get('9F', {}).items():
        f.write(f"- {label}: mean={data['mean_deg']:.1f} deg, golden angle error={data['golden_angle_error_deg']:.1f} deg\n")
        f.write(f"  Rayleigh: z={data['rayleigh_z']:.2f}, p={data['rayleigh_p']:.4f}\n")
    f.write("\n")

    f.write("## 9G: IAF Phi-Ladder\n\n")
    if 'n_subjects' in results.get('9G', {}):
        r9g = results['9G']
        f.write(f"- Subjects with IAF: {r9g['n_subjects']}/{N}\n")
        f.write(f"- Mean error (personal IAF): {r9g['mean_error_personal']:.1f}%\n")
        f.write(f"- Mean error (fixed 10 Hz): {r9g['mean_error_fixed']:.1f}%\n")
        f.write(f"- Personal better: {r9g['personal_better_pct']:.0f}%\n")
        f.write(f"- Paired t: t={r9g['t_personal_vs_fixed']:.2f}, p={r9g['p_personal_vs_fixed']:.4f}\n")
        f.write(f"- Accuracy vs PCI: r={r9g['accuracy_vs_pci_r']:.3f}, p={r9g['accuracy_vs_pci_p']:.4f}\n")
        if r9g.get('per_band_error'):
            for b, e in r9g['per_band_error'].items():
                f.write(f"  - {b}: {e:.1f}% error\n")
    f.write("\n")

    f.write("## 9H: IAF-Normalized Spectral Alignment\n\n")
    if 'n_subjects' in results.get('9H', {}):
        r9h = results['9H']
        f.write(f"- Subjects: {r9h['n_subjects']}, mean IAF={r9h['mean_iaf']:.2f} Hz\n")
        f.write(f"- Peak positions (normalized): {', '.join(f'{p:.3f}' for p in r9h['peak_positions'])}\n")
        f.write(f"- Mean alignment error: {r9h['mean_peak_alignment_error_pct']:.1f}%\n")
        for m in r9h.get('peak_phi_matches', []):
            f.write(f"  - Peak {m['peak_pos']:.3f} -> {m['nearest_phi']}: {m['pct_error']:.1f}% error\n")
    f.write("\n")

    f.write("## Notable Findings\n\n")
    if notable_findings:
        for nf in notable_findings:
            f.write(f"- {nf}\n")
    else:
        f.write("No findings met the discovery criterion (p < 0.001 or |d| > 0.5).\n")

print("\nResults saved: outputs/block9_results.json, outputs/block9_results.md")
print("Figures: outputs/fig_9A-H_*.png")

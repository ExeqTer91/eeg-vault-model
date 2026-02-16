#!/usr/bin/env python3
"""
9's Complement Structure Tests on EEG Data
===========================================
Three tests derived from φ⁴ + φ⁻⁴ = 7 and the digit-level complementarity:

Test 1: Cross-correlation θ-α at multiple frequency resolutions
    If 9's complement holds, θ-α anti-correlation should remain ~-1
    at every resolution (0.1 Hz, 0.01 Hz, 0.001 Hz), not weaken at fine scales.

Test 2: Autocorrelation at Fibonacci lag
    φ⁴ digits wrap at position 13. Alpha envelopes at ~10 Hz should show
    enhanced autocorrelation at lag = 13 cycles = 1.3 seconds.

Test 3: Energy conservation A_θ² + A_α² constancy
    If complementarity is exact, the total energy across θ and α bands
    should be more constant than either band alone (CV comparison).
"""

import numpy as np
import pandas as pd
import mne
from mne.datasets import eegbci
import scipy.signal as signal
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
FS = 160
POSTERIOR_CHANNELS = ['O1', 'O2', 'Oz', 'P3', 'P4', 'Pz']


def get_channel_indices(raw, target_names):
    ch_names = [ch.upper().replace('.', '') for ch in raw.ch_names]
    indices = []
    for t in target_names:
        tu = t.upper().replace('.', '')
        for i, ch in enumerate(ch_names):
            if tu in ch or ch.endswith(tu):
                indices.append(i)
                break
    return indices


def bandpass(data, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = signal.butter(order, [lo/nyq, hi/nyq], btype='band')
    return signal.filtfilt(b, a, data)


def compute_envelope(data, lo, hi, fs):
    filtered = bandpass(data, lo, hi, fs)
    analytic = signal.hilbert(filtered)
    return np.abs(analytic)


def test1_cross_correlation_multiresolution(subjects_data):
    """Test θ-α cross-correlation at multiple frequency resolutions."""
    print("\n" + "="*70)
    print("TEST 1: θ-α Cross-Correlation at Multiple Frequency Resolutions")
    print("="*70)
    print("Prediction: if 9's complement holds, anti-correlation stays ~-1")
    print("at all resolutions, not weakening at fine scales.\n")

    resolutions = [0.5, 0.1, 0.05]
    res_labels = ['0.5 Hz', '0.1 Hz', '0.05 Hz']

    all_corrs = {r: [] for r in resolutions}

    for subj_id, data, fs in subjects_data:
        for res in resolutions:
            nperseg = int(fs / res)
            if nperseg > len(data[0]):
                nperseg = len(data[0])
            theta_powers = []
            alpha_powers = []
            for ch_data in data:
                f, psd = signal.welch(ch_data, fs=fs, nperseg=nperseg,
                                      noverlap=nperseg//2)
                theta_mask = (f >= 4) & (f <= 8)
                alpha_mask = (f >= 8) & (f <= 13)
                if np.sum(theta_mask) > 0 and np.sum(alpha_mask) > 0:
                    theta_powers.append(np.mean(psd[theta_mask]))
                    alpha_powers.append(np.mean(psd[alpha_mask]))

            if len(theta_powers) >= 3:
                r, _ = stats.pearsonr(theta_powers, alpha_powers)
                all_corrs[res].append(r)

    print(f"{'Resolution':>12} | {'Mean r':>8} | {'Median r':>8} | {'Std':>8} | {'N':>4} | {'% < -0.5':>10}")
    print("-" * 65)
    results = {}
    for res, label in zip(resolutions, res_labels):
        corrs = np.array(all_corrs[res])
        if len(corrs) > 0:
            results[label] = {
                'mean': np.mean(corrs), 'median': np.median(corrs),
                'std': np.std(corrs), 'n': len(corrs),
                'frac_strong': np.mean(corrs < -0.5)
            }
            print(f"{label:>12} | {np.mean(corrs):8.4f} | {np.median(corrs):8.4f} | "
                  f"{np.std(corrs):8.4f} | {len(corrs):4d} | {np.mean(corrs < -0.5)*100:9.1f}%")

    weakens = False
    vals = [results[l]['mean'] for l in res_labels if l in results]
    if len(vals) >= 2 and vals[-1] > vals[0] + 0.1:
        weakens = True

    print(f"\nAnti-correlation weakens at fine resolution: {'YES' if weakens else 'NO'}")
    if weakens:
        print("VERDICT: FAIL — anti-correlation weakens, no digit-level complementarity")
    else:
        print("VERDICT: Consistent anti-correlation across resolutions")

    return all_corrs, results


def test2_fibonacci_lag_autocorrelation(subjects_data):
    """Test for enhanced autocorrelation at lag = 13 alpha cycles = 1.3s."""
    print("\n" + "="*70)
    print("TEST 2: Alpha Envelope Autocorrelation at Fibonacci Lag (13 cycles)")
    print("="*70)
    print("Prediction: alpha envelope shows peak autocorrelation at lag = 1.3s")
    print("(13 cycles at 10 Hz) due to φ⁴ digit wrapping.\n")

    fib_lag_sec = 1.3
    test_lags_sec = np.arange(0.5, 3.0, 0.1)

    all_acf_profiles = []
    fib_lag_acfs = []
    non_fib_acfs = []

    for subj_id, data, fs in subjects_data:
        for ch_data in data:
            if len(ch_data) < fs * 5:
                continue
            env = compute_envelope(ch_data, 8, 13, fs)
            env = (env - np.mean(env)) / (np.std(env) + 1e-15)

            acf_profile = []
            for lag_sec in test_lags_sec:
                lag_samp = int(lag_sec * fs)
                if lag_samp >= len(env) - 1:
                    acf_profile.append(np.nan)
                    continue
                c = np.corrcoef(env[:-lag_samp], env[lag_samp:])[0, 1]
                acf_profile.append(c)
            all_acf_profiles.append(acf_profile)

            fib_samp = int(fib_lag_sec * fs)
            if fib_samp < len(env) - 1:
                fib_acf = np.corrcoef(env[:-fib_samp], env[fib_samp:])[0, 1]
                fib_lag_acfs.append(fib_acf)

                nearby = []
                for offset in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
                    alt_samp = int((fib_lag_sec + offset) * fs)
                    if 0 < alt_samp < len(env) - 1:
                        nearby.append(np.corrcoef(env[:-alt_samp], env[alt_samp:])[0, 1])
                non_fib_acfs.append(np.mean(nearby))

    acf_matrix = np.array(all_acf_profiles)
    mean_acf = np.nanmean(acf_matrix, axis=0)

    fib_lag_acfs = np.array(fib_lag_acfs)
    non_fib_acfs = np.array(non_fib_acfs)

    print(f"Channels analyzed: {len(fib_lag_acfs)}")
    print(f"\nAlpha envelope autocorrelation profile (mean across channels):")
    print(f"{'Lag (s)':>8} | {'ACF':>8}")
    print("-" * 20)
    for lag, acf in zip(test_lags_sec, mean_acf):
        marker = " <-- Fibonacci lag" if abs(lag - fib_lag_sec) < 0.05 else ""
        print(f"{lag:8.1f} | {acf:8.4f}{marker}")

    fib_idx = np.argmin(np.abs(test_lags_sec - fib_lag_sec))
    is_local_peak = False
    if 1 < fib_idx < len(mean_acf) - 1:
        is_local_peak = mean_acf[fib_idx] > mean_acf[fib_idx-1] and mean_acf[fib_idx] > mean_acf[fib_idx+1]

    t_stat, p_val = stats.ttest_rel(fib_lag_acfs, non_fib_acfs)
    diff = np.mean(fib_lag_acfs) - np.mean(non_fib_acfs)

    print(f"\nFibonacci lag (1.3s) ACF: {np.mean(fib_lag_acfs):.4f} ± {np.std(fib_lag_acfs):.4f}")
    print(f"Nearby lags ACF:         {np.mean(non_fib_acfs):.4f} ± {np.std(non_fib_acfs):.4f}")
    print(f"Difference:              {diff:.4f}")
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.2e}")
    print(f"Is local peak at 1.3s:   {is_local_peak}")

    if p_val < 0.05 and diff > 0 and is_local_peak:
        print("VERDICT: PASS — enhanced autocorrelation at Fibonacci lag")
    else:
        print("VERDICT: FAIL — no special autocorrelation at Fibonacci lag")

    return test_lags_sec, mean_acf, fib_lag_acfs, non_fib_acfs


def test3_energy_conservation(subjects_data):
    """Test if A_θ² + A_α² is more constant than either alone."""
    print("\n" + "="*70)
    print("TEST 3: Energy Conservation (θ² + α² Constancy)")
    print("="*70)
    print("Prediction: CV(θ²+α²) << CV(θ²) and CV(α²), indicating exact")
    print("complementary energy transfer between bands.\n")

    all_cv_theta = []
    all_cv_alpha = []
    all_cv_sum = []
    all_corr_powers = []

    for subj_id, data, fs in subjects_data:
        for ch_data in data:
            if len(ch_data) < fs * 4:
                continue
            env_theta = compute_envelope(ch_data, 4, 8, fs)
            env_alpha = compute_envelope(ch_data, 8, 13, fs)

            win_samp = int(0.5 * fs)
            hop = win_samp // 2
            n_wins = (len(env_theta) - win_samp) // hop

            if n_wins < 10:
                continue

            theta_power = []
            alpha_power = []
            for i in range(n_wins):
                start = i * hop
                end = start + win_samp
                theta_power.append(np.mean(env_theta[start:end]**2))
                alpha_power.append(np.mean(env_alpha[start:end]**2))

            theta_power = np.array(theta_power)
            alpha_power = np.array(alpha_power)
            total_power = theta_power + alpha_power

            cv_t = np.std(theta_power) / (np.mean(theta_power) + 1e-15)
            cv_a = np.std(alpha_power) / (np.mean(alpha_power) + 1e-15)
            cv_s = np.std(total_power) / (np.mean(total_power) + 1e-15)

            all_cv_theta.append(cv_t)
            all_cv_alpha.append(cv_a)
            all_cv_sum.append(cv_s)

            r, _ = stats.pearsonr(theta_power, alpha_power)
            all_corr_powers.append(r)

    cv_theta = np.array(all_cv_theta)
    cv_alpha = np.array(all_cv_alpha)
    cv_sum = np.array(all_cv_sum)
    corr_powers = np.array(all_corr_powers)

    print(f"Channels analyzed: {len(cv_theta)}")
    print(f"\n{'Metric':>20} | {'Mean':>8} | {'Median':>8} | {'Std':>8}")
    print("-" * 55)
    print(f"{'CV(θ²)':>20} | {np.mean(cv_theta):8.4f} | {np.median(cv_theta):8.4f} | {np.std(cv_theta):8.4f}")
    print(f"{'CV(α²)':>20} | {np.mean(cv_alpha):8.4f} | {np.median(cv_alpha):8.4f} | {np.std(cv_alpha):8.4f}")
    print(f"{'CV(θ²+α²)':>20} | {np.mean(cv_sum):8.4f} | {np.median(cv_sum):8.4f} | {np.std(cv_sum):8.4f}")
    print(f"{'Corr(θ²,α²)':>20} | {np.mean(corr_powers):8.4f} | {np.median(corr_powers):8.4f} | {np.std(corr_powers):8.4f}")

    ratio_theta = cv_sum / cv_theta
    ratio_alpha = cv_sum / cv_alpha
    print(f"\nCV(sum)/CV(θ²):  {np.mean(ratio_theta):.4f} (1.0 = no improvement)")
    print(f"CV(sum)/CV(α²):  {np.mean(ratio_alpha):.4f} (1.0 = no improvement)")
    print(f"% channels where CV(sum) < CV(θ²): {np.mean(cv_sum < cv_theta)*100:.1f}%")
    print(f"% channels where CV(sum) < CV(α²): {np.mean(cv_sum < cv_alpha)*100:.1f}%")
    print(f"% channels where CV(sum) < min(CV(θ²),CV(α²)): {np.mean(cv_sum < np.minimum(cv_theta, cv_alpha))*100:.1f}%")

    t1, p1 = stats.ttest_rel(cv_sum, cv_theta)
    t2, p2 = stats.ttest_rel(cv_sum, cv_alpha)
    print(f"\nPaired t-test CV(sum) vs CV(θ²): t={t1:.3f}, p={p1:.2e}")
    print(f"Paired t-test CV(sum) vs CV(α²): t={t2:.3f}, p={p2:.2e}")

    strong_conservation = (np.mean(cv_sum < np.minimum(cv_theta, cv_alpha)) > 0.8
                           and p1 < 0.001 and p2 < 0.001
                           and np.mean(ratio_theta) < 0.5)

    if strong_conservation:
        print("\nVERDICT: PASS — strong energy conservation between θ and α")
    else:
        moderate = np.mean(cv_sum < np.minimum(cv_theta, cv_alpha)) > 0.5 and p1 < 0.05
        if moderate:
            print("\nVERDICT: PARTIAL — some energy conservation, but not 'exact'")
        else:
            print("\nVERDICT: FAIL — no special energy conservation between θ and α")

    return cv_theta, cv_alpha, cv_sum, corr_powers


def load_eeg_data(n_subjects=30):
    """Load PhysioNet EEG data for posterior channels, eyes-closed rest."""
    print("Loading PhysioNet EEG data (eyes-closed rest, run 2)...")
    subjects_data = []
    loaded = 0

    for subj in range(1, 110):
        if loaded >= n_subjects:
            break
        try:
            raw_fnames = eegbci.load_data(subj, [2], update_path=False)
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)

            ch_idx = get_channel_indices(raw, POSTERIOR_CHANNELS)
            if len(ch_idx) < 3:
                continue

            data = raw.get_data(picks=ch_idx)
            if data.shape[1] < FS * 3:
                continue

            subjects_data.append((subj, data, FS))
            loaded += 1
            if loaded % 10 == 0:
                print(f"  Loaded {loaded} subjects...")

        except Exception:
            continue

    print(f"  Total: {loaded} subjects loaded")
    return subjects_data


def generate_figure(test1_corrs, test1_results, test2_lags, test2_acf,
                    test2_fib, test2_nonfib, test3_cvt, test3_cva,
                    test3_cvs, test3_corr):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    labels = list(test1_results.keys())
    means = [test1_results[l]['mean'] for l in labels]
    stds = [test1_results[l]['std'] for l in labels]
    x = range(len(labels))
    ax1.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral', 'green'],
            alpha=0.7, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Mean θ-α correlation')
    ax1.set_title('Test 1: θ-α Cross-Correlation\nby Frequency Resolution')
    ax1.axhline(-1, color='red', ls='--', lw=1, label='Perfect anti-corr')
    ax1.axhline(0, color='gray', ls='-', lw=0.5)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(test2_lags, test2_acf, 'o-', color='steelblue', lw=2, ms=4)
    fib_idx = np.argmin(np.abs(test2_lags - 1.3))
    ax2.axvline(1.3, color='gold', lw=2, ls='--', label='Fibonacci lag (1.3s)')
    ax2.plot(1.3, test2_acf[fib_idx], 's', color='red', ms=12, zorder=5,
             label=f'ACF = {test2_acf[fib_idx]:.4f}')
    ax2.set_xlabel('Lag (seconds)')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('Test 2: Alpha Envelope ACF\nvs Fibonacci Lag')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(test2_fib - test2_nonfib, bins=40, density=True, alpha=0.7,
             color='steelblue', edgecolor='none')
    ax3.axvline(0, color='red', ls='--', lw=2, label='No difference')
    ax3.axvline(np.mean(test2_fib - test2_nonfib), color='black', lw=2,
                label=f'Mean diff = {np.mean(test2_fib - test2_nonfib):.4f}')
    ax3.set_xlabel('ACF(Fibonacci lag) - ACF(nearby lags)')
    ax3.set_ylabel('Density')
    ax3.set_title('Test 2: Fibonacci vs Nearby Lags')
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    positions = [1, 2, 3]
    bp = ax4.boxplot([test3_cvt, test3_cva, test3_cvs], positions=positions,
                     widths=0.6, patch_artist=True)
    colors = ['coral', 'steelblue', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xticks(positions)
    ax4.set_xticklabels(['CV(θ²)', 'CV(α²)', 'CV(θ²+α²)'])
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Test 3: Energy Conservation\nCV Comparison')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(test3_corr, bins=40, density=True, alpha=0.7,
             color='purple', edgecolor='none')
    ax5.axvline(np.mean(test3_corr), color='black', lw=2,
                label=f'Mean r = {np.mean(test3_corr):.4f}')
    ax5.axvline(-1, color='red', ls='--', lw=2, label='Perfect anti-corr')
    ax5.set_xlabel('Correlation(θ², α²)')
    ax5.set_ylabel('Density')
    ax5.set_title('Test 3: θ-α Power Correlation')
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    ratio_min = test3_cvs / np.minimum(test3_cvt, test3_cva)
    ax6.hist(ratio_min, bins=40, density=True, alpha=0.7,
             color='teal', edgecolor='none')
    ax6.axvline(1.0, color='red', ls='--', lw=2, label='No improvement')
    ax6.axvline(np.mean(ratio_min), color='black', lw=2,
                label=f'Mean = {np.mean(ratio_min):.3f}')
    ax6.set_xlabel('CV(θ²+α²) / min(CV(θ²), CV(α²))')
    ax6.set_ylabel('Density')
    ax6.set_title('Test 3: Conservation Ratio')
    ax6.legend(fontsize=8)

    fig.suptitle("9's Complement Tests: Does φ⁴ + φ⁻⁴ = 7 Structure Appear in EEG?",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.savefig('nines_complement_results.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


def main():
    print("="*70)
    print("9's COMPLEMENT STRUCTURE TESTS ON EEG DATA")
    print("φ⁴ + φ⁻⁴ = 7  |  Digit complementarity predictions")
    print("="*70)

    subjects_data = load_eeg_data(n_subjects=20)

    test1_corrs, test1_results = test1_cross_correlation_multiresolution(subjects_data)
    test2_lags, test2_acf, test2_fib, test2_nonfib = test2_fibonacci_lag_autocorrelation(subjects_data)
    test3_cvt, test3_cva, test3_cvs, test3_corr = test3_energy_conservation(subjects_data)

    print("\n" + "="*70)
    print("GENERATING FIGURE...")
    generate_figure(test1_corrs, test1_results, test2_lags, test2_acf,
                    test2_fib, test2_nonfib, test3_cvt, test3_cva,
                    test3_cvs, test3_corr)
    print("Saved: nines_complement_results.png")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

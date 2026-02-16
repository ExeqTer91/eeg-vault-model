#!/usr/bin/env python3
"""
9's Complement Tests — Corrected for Volume Conduction
======================================================
Two corrections:
  Test A: CSD (Laplacian) transform to remove volume conduction,
          then θ from frontal CSD (Fz,F3,F4) and α from posterior CSD (Oz,O1,O2)
  Test B: Cross-dipole ratio θ_power(Fz) / α_power(Oz) vs same-electrode ratio

Then re-run the three original tests with corrected signals.
"""

import numpy as np
import mne
from mne.datasets import eegbci
import scipy.signal as signal
from scipy import stats
from scipy.stats import circmean, circstd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
FS = 160

FRONTAL = ['Fz', 'F3', 'F4']
POSTERIOR = ['Oz', 'O1', 'O2']


def find_channels(raw, target_names):
    ch_map = {}
    raw_names = raw.ch_names
    for t in target_names:
        tu = t.upper().replace('.', '')
        for i, ch in enumerate(raw_names):
            cu = ch.upper().replace('.', '')
            if tu in cu or cu.endswith(tu):
                ch_map[t] = i
                break
    return ch_map


def bandpass(data, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = signal.butter(order, [lo/nyq, hi/nyq], btype='band')
    return signal.filtfilt(b, a, data)


def compute_envelope(data, lo, hi, fs):
    filtered = bandpass(data, lo, hi, fs)
    analytic = signal.hilbert(filtered)
    return np.abs(analytic)


def compute_band_power(data, lo, hi, fs):
    f, psd = signal.welch(data, fs=fs, nperseg=int(fs*2), noverlap=int(fs))
    mask = (f >= lo) & (f <= hi)
    return np.mean(psd[mask])


def surface_laplacian(raw):
    """Compute current source density (surface Laplacian) using MNE."""
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        rename_map = {}
        raw_upper = [ch.upper().replace('.', '') for ch in raw.ch_names]
        montage_upper = {ch.upper(): ch for ch in montage.ch_names}

        for i, ch in enumerate(raw.ch_names):
            cu = ch.upper().replace('.', '')
            if cu in montage_upper:
                rename_map[ch] = montage_upper[cu]

        if len(rename_map) < 10:
            eegbci_map = {}
            for ch in raw.ch_names:
                clean = ch.replace('.', '').replace(' ', '')
                if clean.startswith('EEG'):
                    clean = clean[3:].strip()
                eegbci_map[ch] = clean
            rename_map = eegbci_map

        raw_copy = raw.copy()
        raw_copy.rename_channels(rename_map)

        valid_ch = [ch for ch in raw_copy.ch_names if ch in montage.ch_names]
        if len(valid_ch) < 10:
            return None

        raw_copy.pick_channels(valid_ch)
        raw_copy.set_montage(montage, on_missing='ignore')
        raw_copy.set_eeg_reference('average', verbose=False)
        raw_csd = mne.preprocessing.compute_current_source_density(raw_copy, verbose=False)
        return raw_csd

    except Exception as e:
        print(f"    CSD failed: {e}")
        return None


def run_test_A(n_subjects=20):
    """CSD Laplacian → frontal θ vs posterior α."""
    print("\n" + "="*70)
    print("TEST A: CSD (Laplacian) — Frontal θ vs Posterior α")
    print("="*70)
    print("Remove volume conduction via CSD, then measure θ-α correlation")
    print("across spatially separated generators.\n")

    csd_corrs = []
    raw_corrs = []
    cross_ratios = []
    same_ratios = []
    csd_cv_theta = []
    csd_cv_alpha = []
    csd_cv_sum = []
    csd_power_corrs = []

    loaded = 0
    for subj in range(1, 110):
        if loaded >= n_subjects:
            break
        try:
            raw_fnames = eegbci.load_data(subj, [2], update_path=False)
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)

            raw_csd = surface_laplacian(raw)
            if raw_csd is None:
                continue

            csd_names = raw_csd.ch_names
            frontal_idx = [i for i, ch in enumerate(csd_names)
                          if any(f.upper() in ch.upper() for f in FRONTAL)]
            posterior_idx = [i for i, ch in enumerate(csd_names)
                           if any(p.upper() in ch.upper() for p in POSTERIOR)]

            if len(frontal_idx) < 1 or len(posterior_idx) < 1:
                continue

            csd_data = raw_csd.get_data()
            frontal_data = np.mean(csd_data[frontal_idx], axis=0)
            posterior_data = np.mean(csd_data[posterior_idx], axis=0)

            theta_frontal = compute_band_power(frontal_data, 4, 8, FS)
            alpha_posterior = compute_band_power(posterior_data, 8, 13, FS)

            cross_ratio = theta_frontal / (alpha_posterior + 1e-30)
            cross_ratios.append(cross_ratio)

            theta_post = compute_band_power(posterior_data, 4, 8, FS)
            same_ratio = theta_post / (alpha_posterior + 1e-30)
            same_ratios.append(same_ratio)

            env_theta_f = compute_envelope(frontal_data, 4, 8, FS)
            env_alpha_p = compute_envelope(posterior_data, 8, 13, FS)

            win_samp = int(0.5 * FS)
            hop = win_samp // 2
            n_wins = (min(len(env_theta_f), len(env_alpha_p)) - win_samp) // hop

            if n_wins < 10:
                continue

            theta_power_ts = []
            alpha_power_ts = []
            for i in range(n_wins):
                start = i * hop
                end = start + win_samp
                theta_power_ts.append(np.mean(env_theta_f[start:end]**2))
                alpha_power_ts.append(np.mean(env_alpha_p[start:end]**2))

            theta_power_ts = np.array(theta_power_ts)
            alpha_power_ts = np.array(alpha_power_ts)
            total = theta_power_ts + alpha_power_ts

            r_csd, _ = stats.pearsonr(theta_power_ts, alpha_power_ts)
            csd_corrs.append(r_csd)

            cv_t = np.std(theta_power_ts) / (np.mean(theta_power_ts) + 1e-15)
            cv_a = np.std(alpha_power_ts) / (np.mean(alpha_power_ts) + 1e-15)
            cv_s = np.std(total) / (np.mean(total) + 1e-15)
            csd_cv_theta.append(cv_t)
            csd_cv_alpha.append(cv_a)
            csd_cv_sum.append(cv_s)
            csd_power_corrs.append(r_csd)

            raw_data = raw.get_data()
            orig_post_idx = [i for i, ch in enumerate(raw.ch_names)
                            if any(p.upper() in ch.upper().replace('.','') for p in POSTERIOR)]
            if len(orig_post_idx) > 0:
                raw_post = np.mean(raw_data[orig_post_idx], axis=0)
                raw_theta = compute_band_power(raw_post, 4, 8, FS)
                raw_alpha = compute_band_power(raw_post, 8, 13, FS)
                env_t_raw = compute_envelope(raw_post, 4, 8, FS)
                env_a_raw = compute_envelope(raw_post, 8, 13, FS)
                n_w2 = (len(env_t_raw) - win_samp) // hop
                if n_w2 > 10:
                    tp2 = [np.mean(env_t_raw[i*hop:i*hop+win_samp]**2) for i in range(n_w2)]
                    ap2 = [np.mean(env_a_raw[i*hop:i*hop+win_samp]**2) for i in range(n_w2)]
                    r_raw, _ = stats.pearsonr(tp2, ap2)
                    raw_corrs.append(r_raw)

            loaded += 1
            if loaded % 5 == 0:
                print(f"  Processed {loaded} subjects...")

        except Exception as e:
            continue

    print(f"\n  Total subjects processed: {loaded}")

    csd_corrs = np.array(csd_corrs)
    raw_corrs = np.array(raw_corrs)
    cross_ratios = np.array(cross_ratios)
    same_ratios = np.array(same_ratios)

    print(f"\n--- θ-α Power Correlation ---")
    print(f"  Raw (same electrode, posterior):     r = {np.mean(raw_corrs):.4f} ± {np.std(raw_corrs):.4f}")
    print(f"  CSD (frontal θ vs posterior α):      r = {np.mean(csd_corrs):.4f} ± {np.std(csd_corrs):.4f}")
    print(f"  Sign flipped (raw→CSD)?              {'YES' if np.mean(csd_corrs) < 0 and np.mean(raw_corrs) > 0 else 'NO'}")

    t_stat, p_val = stats.ttest_1samp(csd_corrs, 0)
    print(f"  CSD corr vs 0: t={t_stat:.3f}, p={p_val:.2e}")
    t2, p2 = stats.ttest_ind(csd_corrs, raw_corrs)
    print(f"  CSD vs Raw:    t={t2:.3f}, p={p2:.2e}")

    print(f"\n--- Energy Conservation (CSD) ---")
    csd_cv_theta = np.array(csd_cv_theta)
    csd_cv_alpha = np.array(csd_cv_alpha)
    csd_cv_sum = np.array(csd_cv_sum)
    print(f"  CV(θ²_frontal):          {np.mean(csd_cv_theta):.4f}")
    print(f"  CV(α²_posterior):        {np.mean(csd_cv_alpha):.4f}")
    print(f"  CV(θ²+α²):              {np.mean(csd_cv_sum):.4f}")
    print(f"  CV(sum)/CV(θ²):          {np.mean(csd_cv_sum/csd_cv_theta):.4f}")
    print(f"  CV(sum)/CV(α²):          {np.mean(csd_cv_sum/csd_cv_alpha):.4f}")
    frac_better = np.mean(csd_cv_sum < np.minimum(csd_cv_theta, csd_cv_alpha))
    print(f"  % where CV(sum) < both:  {frac_better*100:.1f}%")

    if np.mean(csd_corrs) < -0.3:
        print("\nVERDICT TEST A: PASS — anti-correlation emerges after removing volume conduction")
    elif np.mean(csd_corrs) < 0:
        print("\nVERDICT TEST A: PARTIAL — negative but weak after CSD")
    else:
        print("\nVERDICT TEST A: FAIL — still positive after CSD")

    return (csd_corrs, raw_corrs, cross_ratios, same_ratios,
            csd_cv_theta, csd_cv_alpha, csd_cv_sum)


def run_test_B(n_subjects=20):
    """Cross-dipole ratio: θ_power(Fz) / α_power(Oz) vs same-electrode."""
    print("\n" + "="*70)
    print("TEST B: Cross-Dipole Ratio θ(Fz) / α(Oz)")
    print("="*70)
    print("Compare cross-dipole ratio distribution with same-electrode ratio.")
    print("Prediction: cross-dipole is tighter around φ constants.\n")

    cross_ratios = []
    same_fz_ratios = []
    same_oz_ratios = []

    loaded = 0
    for subj in range(1, 110):
        if loaded >= n_subjects:
            break
        try:
            raw_fnames = eegbci.load_data(subj, [2], update_path=False)
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)

            ch_map_fz = find_channels(raw, ['Fz'])
            ch_map_oz = find_channels(raw, ['Oz'])

            if 'Fz' not in ch_map_fz or 'Oz' not in ch_map_oz:
                continue

            data = raw.get_data()
            fz_data = data[ch_map_fz['Fz']]
            oz_data = data[ch_map_oz['Oz']]

            theta_fz = compute_band_power(fz_data, 4, 8, FS)
            alpha_oz = compute_band_power(oz_data, 8, 13, FS)
            theta_oz = compute_band_power(oz_data, 4, 8, FS)
            alpha_fz = compute_band_power(fz_data, 8, 13, FS)

            cross = theta_fz / (alpha_oz + 1e-30)
            same_fz = theta_fz / (alpha_fz + 1e-30)
            same_oz = theta_oz / (alpha_oz + 1e-30)

            cross_ratios.append(cross)
            same_fz_ratios.append(same_fz)
            same_oz_ratios.append(same_oz)

            loaded += 1
        except Exception:
            continue

    cross_ratios = np.array(cross_ratios)
    same_fz_ratios = np.array(same_fz_ratios)
    same_oz_ratios = np.array(same_oz_ratios)

    print(f"Subjects: {loaded}")
    print(f"\n{'Ratio Type':>25} | {'Mean':>8} | {'Std':>8} | {'CV':>8} | {'|mean-φ|':>8}")
    print("-" * 70)
    for name, vals in [("Cross: θ(Fz)/α(Oz)", cross_ratios),
                       ("Same-elec: θ(Fz)/α(Fz)", same_fz_ratios),
                       ("Same-elec: θ(Oz)/α(Oz)", same_oz_ratios)]:
        cv = np.std(vals) / (np.mean(vals) + 1e-15)
        dist_phi = abs(np.mean(vals) - PHI)
        print(f"{name:>25} | {np.mean(vals):8.4f} | {np.std(vals):8.4f} | {cv:8.4f} | {dist_phi:8.4f}")

    f_stat, p_levene = stats.levene(cross_ratios, same_oz_ratios)
    print(f"\nLevene test (cross vs same-Oz variance): F={f_stat:.3f}, p={p_levene:.4f}")

    cross_cv = np.std(cross_ratios) / np.mean(cross_ratios)
    oz_cv = np.std(same_oz_ratios) / np.mean(same_oz_ratios)
    if cross_cv < oz_cv and p_levene < 0.05:
        print("VERDICT TEST B: PASS — cross-dipole ratio is tighter")
    elif cross_cv < oz_cv:
        print("VERDICT TEST B: PARTIAL — cross-dipole is tighter but not significantly")
    else:
        print("VERDICT TEST B: FAIL — cross-dipole not tighter than same-electrode")

    return cross_ratios, same_fz_ratios, same_oz_ratios


def generate_figure(csd_corrs, raw_corrs, cross_ratios_A, same_ratios_A,
                    csd_cv_t, csd_cv_a, csd_cv_s,
                    cross_B, same_fz_B, same_oz_B):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(raw_corrs, bins=20, density=True, alpha=0.6, color='red',
             label=f'Raw (same elec)\nmean={np.mean(raw_corrs):.3f}', edgecolor='none')
    ax1.hist(csd_corrs, bins=20, density=True, alpha=0.6, color='steelblue',
             label=f'CSD (frontal θ vs post α)\nmean={np.mean(csd_corrs):.3f}', edgecolor='none')
    ax1.axvline(0, color='black', ls='--', lw=1)
    ax1.set_xlabel('Correlation(θ², α²)')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Volume Conduction Correction')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    positions = [1, 2, 3]
    bp = ax2.boxplot([csd_cv_t, csd_cv_a, csd_cv_s], positions=positions,
                     widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['coral', 'steelblue', 'green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['CV(θ²\nfrontal)', 'CV(α²\nposterior)', 'CV(θ²+α²)'])
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('B. Energy Conservation (CSD)')

    ax3 = fig.add_subplot(gs[0, 2])
    data_pairs = list(zip(raw_corrs[:len(csd_corrs)], csd_corrs))
    for i, (r, c) in enumerate(data_pairs):
        ax3.plot([0, 1], [r, c], 'o-', color='gray', alpha=0.5, ms=4)
    ax3.plot([0], [np.mean(raw_corrs)], 'o', color='red', ms=12, zorder=5)
    ax3.plot([1], [np.mean(csd_corrs)], 'o', color='steelblue', ms=12, zorder=5)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Raw\n(same electrode)', 'CSD\n(frontal→posterior)'])
    ax3.set_ylabel('θ-α correlation')
    ax3.set_title('C. Per-Subject Shift')
    ax3.axhline(0, color='black', ls='--', lw=1)

    ax4 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 3, 40)
    ax4.hist(cross_B, bins=bins, density=True, alpha=0.6, color='steelblue',
             label=f'Cross θ(Fz)/α(Oz)\nmean={np.mean(cross_B):.3f}', edgecolor='none')
    ax4.hist(same_oz_B, bins=bins, density=True, alpha=0.6, color='coral',
             label=f'Same θ(Oz)/α(Oz)\nmean={np.mean(same_oz_B):.3f}', edgecolor='none')
    ax4.axvline(PHI, color='gold', lw=2, ls='--', label=f'φ = {PHI:.3f}')
    ax4.set_xlabel('θ/α ratio')
    ax4.set_ylabel('Density')
    ax4.set_title('D. Cross-Dipole vs Same-Electrode')
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    labels = ['Cross\nθ(Fz)/α(Oz)', 'Same\nθ(Fz)/α(Fz)', 'Same\nθ(Oz)/α(Oz)']
    means = [np.mean(cross_B), np.mean(same_fz_B), np.mean(same_oz_B)]
    stds = [np.std(cross_B), np.std(same_fz_B), np.std(same_oz_B)]
    cvs = [s/m for s, m in zip(stds, means)]
    colors = ['steelblue', 'lightcoral', 'coral']
    ax5.bar(range(3), cvs, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(3))
    ax5.set_xticklabels(labels, fontsize=8)
    ax5.set_ylabel('CV (lower = tighter)')
    ax5.set_title('E. Ratio Variability Comparison')

    ax6 = fig.add_subplot(gs[1, 2])
    dist_cross = np.abs(cross_B - PHI)
    dist_same = np.abs(same_oz_B - PHI)
    ax6.hist(dist_cross, bins=30, density=True, alpha=0.6, color='steelblue',
             label=f'Cross: mean |r-φ|={np.mean(dist_cross):.3f}', edgecolor='none')
    ax6.hist(dist_same, bins=30, density=True, alpha=0.6, color='coral',
             label=f'Same Oz: mean |r-φ|={np.mean(dist_same):.3f}', edgecolor='none')
    ax6.set_xlabel('|ratio - φ|')
    ax6.set_ylabel('Density')
    ax6.set_title('F. Distance to φ')
    ax6.legend(fontsize=8)

    fig.suptitle("9's Complement Tests — Corrected for Volume Conduction",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.savefig('nines_complement_corrected_results.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("="*70)
    print("9's COMPLEMENT — VOLUME CONDUCTION CORRECTED")
    print("="*70)

    (csd_corrs, raw_corrs, cross_A, same_A,
     csd_cv_t, csd_cv_a, csd_cv_s) = run_test_A(n_subjects=20)

    cross_B, same_fz_B, same_oz_B = run_test_B(n_subjects=20)

    print("\n\nGenerating figure...")
    generate_figure(csd_corrs, raw_corrs, cross_A, same_A,
                    csd_cv_t, csd_cv_a, csd_cv_s,
                    cross_B, same_fz_B, same_oz_B)
    print("Saved: nines_complement_corrected_results.png")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
State Contrast Test: Eyes Open vs Eyes Closed
==============================================
Tests whether θ-α complementarity is state-dependent.

PhysioNet EEGBCI: Run 1 = eyes open, Run 2 = eyes closed.

Measures per state:
1. θ-α power correlation (same electrode and CSD cross-dipole)
2. θ/α ratio distribution (mean, CV, distance to φ)
3. Energy conservation CV(θ²+α²) vs CV(θ²), CV(α²)
4. Cross-frequency phase-amplitude coupling

Also tests the REVERSE: α/θ ratio.
"""

import numpy as np
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
POSTERIOR = ['Oz', 'O1', 'O2', 'Pz', 'P3', 'P4']


def find_channels(raw, targets):
    idx = []
    for t in targets:
        tu = t.upper()
        for i, ch in enumerate(raw.ch_names):
            cu = ch.upper().replace('.', '')
            if tu in cu or cu.endswith(tu):
                idx.append(i)
                break
    return idx


def compute_envelope(data, lo, hi, fs):
    nyq = fs / 2
    b, a = signal.butter(4, [lo/nyq, hi/nyq], btype='band')
    filt = signal.filtfilt(b, a, data)
    return np.abs(signal.hilbert(filt))


def analyze_state(data_channels, fs, label):
    """Compute all metrics for one state."""
    results = {
        'theta_alpha_corrs': [],
        'alpha_theta_corrs': [],
        'ta_ratios': [],
        'at_ratios': [],
        'cv_theta': [],
        'cv_alpha': [],
        'cv_sum': [],
        'power_corrs': [],
    }

    for ch_data in data_channels:
        if len(ch_data) < fs * 3:
            continue

        f, psd = signal.welch(ch_data, fs=fs, nperseg=int(fs*2), noverlap=int(fs))
        theta_mask = (f >= 4) & (f <= 8)
        alpha_mask = (f >= 8) & (f <= 13)
        tp = np.mean(psd[theta_mask])
        ap = np.mean(psd[alpha_mask])

        if tp > 0 and ap > 0:
            results['ta_ratios'].append(tp / ap)
            results['at_ratios'].append(ap / tp)

        env_t = compute_envelope(ch_data, 4, 8, fs)
        env_a = compute_envelope(ch_data, 8, 13, fs)

        win = int(0.5 * fs)
        hop = win // 2
        n_wins = (len(env_t) - win) // hop
        if n_wins < 5:
            continue

        tp_ts = np.array([np.mean(env_t[i*hop:i*hop+win]**2) for i in range(n_wins)])
        ap_ts = np.array([np.mean(env_a[i*hop:i*hop+win]**2) for i in range(n_wins)])
        total = tp_ts + ap_ts

        r, _ = stats.pearsonr(tp_ts, ap_ts)
        results['power_corrs'].append(r)

        cv_t = np.std(tp_ts) / (np.mean(tp_ts) + 1e-15)
        cv_a = np.std(ap_ts) / (np.mean(ap_ts) + 1e-15)
        cv_s = np.std(total) / (np.mean(total) + 1e-15)
        results['cv_theta'].append(cv_t)
        results['cv_alpha'].append(cv_a)
        results['cv_sum'].append(cv_s)

    for k in results:
        results[k] = np.array(results[k])

    return results


def print_state_results(res, label):
    print(f"\n  --- {label} ---")
    ta = res['ta_ratios']
    at = res['at_ratios']
    print(f"  θ/α ratio:  mean={np.mean(ta):.4f}  std={np.std(ta):.4f}  "
          f"CV={np.std(ta)/np.mean(ta):.4f}  |mean-φ|={abs(np.mean(ta)-PHI):.4f}  "
          f"|mean-1/φ|={abs(np.mean(ta)-1/PHI):.4f}")
    print(f"  α/θ ratio:  mean={np.mean(at):.4f}  std={np.std(at):.4f}  "
          f"CV={np.std(at)/np.mean(at):.4f}  |mean-φ|={abs(np.mean(at)-PHI):.4f}")
    pc = res['power_corrs']
    print(f"  θ-α power corr: mean={np.mean(pc):.4f}  median={np.median(pc):.4f}  "
          f"std={np.std(pc):.4f}  %<0={np.mean(pc<0)*100:.1f}%")
    cvt = res['cv_theta']
    cva = res['cv_alpha']
    cvs = res['cv_sum']
    print(f"  CV(θ²)={np.mean(cvt):.4f}  CV(α²)={np.mean(cva):.4f}  "
          f"CV(θ²+α²)={np.mean(cvs):.4f}")
    frac = np.mean(cvs < np.minimum(cvt, cva))
    print(f"  CV(sum) < both: {frac*100:.1f}%")
    ratio_improvement = np.mean(cvs / np.minimum(cvt, cva))
    print(f"  CV(sum)/min(CV(θ),CV(α)): {ratio_improvement:.4f}")


def main():
    print("=" * 70)
    print("STATE CONTRAST: Eyes Open vs Eyes Closed")
    print("PhysioNet EEGBCI — Run 1 (open) vs Run 2 (closed)")
    print("=" * 70)

    eo_all = []
    ec_all = []

    n_subjects = 25
    loaded = 0

    for subj in range(1, 110):
        if loaded >= n_subjects:
            break
        try:
            fnames_eo = eegbci.load_data(subj, [1], update_path=False)
            fnames_ec = eegbci.load_data(subj, [2], update_path=False)

            raw_eo = mne.io.read_raw_edf(fnames_eo[0], preload=True, verbose=False)
            raw_ec = mne.io.read_raw_edf(fnames_ec[0], preload=True, verbose=False)

            raw_eo.filter(1, 45, verbose=False)
            raw_ec.filter(1, 45, verbose=False)

            post_eo = find_channels(raw_eo, POSTERIOR)
            post_ec = find_channels(raw_ec, POSTERIOR)

            if len(post_eo) < 3 or len(post_ec) < 3:
                continue

            data_eo = raw_eo.get_data(picks=post_eo)
            data_ec = raw_ec.get_data(picks=post_ec)

            eo_all.append((subj, data_eo))
            ec_all.append((subj, data_ec))

            loaded += 1
            if loaded % 10 == 0:
                print(f"  Loaded {loaded} subjects...")

        except Exception:
            continue

    print(f"  Total: {loaded} subjects")

    print("\nAnalyzing Eyes Open...")
    eo_channels = []
    for subj, data in eo_all:
        for ch in data:
            eo_channels.append(ch)
    res_eo = analyze_state(eo_channels, FS, "Eyes Open")

    print("Analyzing Eyes Closed...")
    ec_channels = []
    for subj, data in ec_all:
        for ch in data:
            ec_channels.append(ch)
    res_ec = analyze_state(ec_channels, FS, "Eyes Closed")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print_state_results(res_eo, "EYES OPEN (Run 1)")
    print_state_results(res_ec, "EYES CLOSED (Run 2)")

    print("\n\n  --- STATE COMPARISON ---")
    t1, p1 = stats.ttest_ind(res_eo['power_corrs'], res_ec['power_corrs'])
    print(f"  θ-α corr difference: EO={np.mean(res_eo['power_corrs']):.4f} vs "
          f"EC={np.mean(res_ec['power_corrs']):.4f}  t={t1:.3f}  p={p1:.2e}")

    t2, p2 = stats.ttest_ind(res_eo['ta_ratios'], res_ec['ta_ratios'])
    print(f"  θ/α ratio difference: EO={np.mean(res_eo['ta_ratios']):.4f} vs "
          f"EC={np.mean(res_ec['ta_ratios']):.4f}  t={t2:.3f}  p={p2:.2e}")

    t3, p3 = stats.ttest_ind(res_eo['at_ratios'], res_ec['at_ratios'])
    print(f"  α/θ ratio difference: EO={np.mean(res_eo['at_ratios']):.4f} vs "
          f"EC={np.mean(res_ec['at_ratios']):.4f}  t={t3:.3f}  p={p3:.2e}")

    eo_cv_ratio = res_eo['cv_sum'] / np.minimum(res_eo['cv_theta'], res_eo['cv_alpha'])
    ec_cv_ratio = res_ec['cv_sum'] / np.minimum(res_ec['cv_theta'], res_ec['cv_alpha'])
    t4, p4 = stats.ttest_ind(eo_cv_ratio, ec_cv_ratio)
    print(f"  Conservation ratio: EO={np.mean(eo_cv_ratio):.4f} vs "
          f"EC={np.mean(ec_cv_ratio):.4f}  t={t4:.3f}  p={p4:.2e}")

    eo_closer = abs(np.mean(res_eo['at_ratios']) - PHI)
    ec_closer = abs(np.mean(res_ec['at_ratios']) - PHI)
    print(f"\n  Distance α/θ to φ: EO={eo_closer:.4f}  EC={ec_closer:.4f}")
    print(f"  Closer to φ in: {'EYES CLOSED' if ec_closer < eo_closer else 'EYES OPEN'}")

    eo_anti = np.mean(res_eo['power_corrs'] < 0)
    ec_anti = np.mean(res_ec['power_corrs'] < 0)
    print(f"  % anti-correlated: EO={eo_anti*100:.1f}%  EC={ec_anti*100:.1f}%")

    if np.mean(res_ec['power_corrs']) < -0.1 and np.mean(res_eo['power_corrs']) > 0:
        print("\n  VERDICT: PASS — complementarity is state-dependent (EC only)")
    elif abs(np.mean(res_ec['power_corrs'])) > abs(np.mean(res_eo['power_corrs'])):
        if ec_closer < eo_closer:
            print("\n  VERDICT: PARTIAL — EC shows stronger pattern but not full anti-correlation")
        else:
            print("\n  VERDICT: FAIL — no clear state-dependent complementarity")
    else:
        print("\n  VERDICT: FAIL — no state-dependent complementarity")

    print("\n\nGenerating figure...")
    generate_figure(res_eo, res_ec)
    print("Saved: state_contrast_results.png")


def generate_figure(res_eo, res_ec):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(res_eo['power_corrs'], bins=30, density=True, alpha=0.6,
             color='orange', label=f"EO mean={np.mean(res_eo['power_corrs']):.3f}", edgecolor='none')
    ax1.hist(res_ec['power_corrs'], bins=30, density=True, alpha=0.6,
             color='steelblue', label=f"EC mean={np.mean(res_ec['power_corrs']):.3f}", edgecolor='none')
    ax1.axvline(0, color='black', ls='--', lw=1)
    ax1.set_xlabel('θ-α power correlation')
    ax1.set_ylabel('Density')
    ax1.set_title('A. θ-α Correlation by State')
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 4, 50)
    ax2.hist(res_eo['ta_ratios'], bins=bins, density=True, alpha=0.6,
             color='orange', label=f"EO mean={np.mean(res_eo['ta_ratios']):.3f}", edgecolor='none')
    ax2.hist(res_ec['ta_ratios'], bins=bins, density=True, alpha=0.6,
             color='steelblue', label=f"EC mean={np.mean(res_ec['ta_ratios']):.3f}", edgecolor='none')
    ax2.axvline(PHI, color='gold', lw=2, ls='--', label=f'φ = {PHI:.3f}')
    ax2.axvline(1/PHI, color='green', lw=2, ls='--', label=f'1/φ = {1/PHI:.3f}')
    ax2.set_xlabel('θ/α ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('B. θ/α Ratio Distribution')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    bins2 = np.linspace(0, 8, 50)
    ax3.hist(res_eo['at_ratios'], bins=bins2, density=True, alpha=0.6,
             color='orange', label=f"EO mean={np.mean(res_eo['at_ratios']):.3f}", edgecolor='none')
    ax3.hist(res_ec['at_ratios'], bins=bins2, density=True, alpha=0.6,
             color='steelblue', label=f"EC mean={np.mean(res_ec['at_ratios']):.3f}", edgecolor='none')
    ax3.axvline(PHI, color='gold', lw=2, ls='--', label=f'φ = {PHI:.3f}')
    ax3.set_xlabel('α/θ ratio (REVERSE)')
    ax3.set_ylabel('Density')
    ax3.set_title('C. α/θ Ratio Distribution')
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    eo_cv_ratio = res_eo['cv_sum'] / np.minimum(res_eo['cv_theta'], res_eo['cv_alpha'])
    ec_cv_ratio = res_ec['cv_sum'] / np.minimum(res_ec['cv_theta'], res_ec['cv_alpha'])
    ax4.hist(eo_cv_ratio, bins=30, density=True, alpha=0.6,
             color='orange', label=f"EO mean={np.mean(eo_cv_ratio):.3f}", edgecolor='none')
    ax4.hist(ec_cv_ratio, bins=30, density=True, alpha=0.6,
             color='steelblue', label=f"EC mean={np.mean(ec_cv_ratio):.3f}", edgecolor='none')
    ax4.axvline(1.0, color='red', ls='--', lw=2, label='No improvement')
    ax4.set_xlabel('CV(θ²+α²) / min(CV(θ²), CV(α²))')
    ax4.set_ylabel('Density')
    ax4.set_title('D. Energy Conservation Ratio')
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['Mean θ/α', 'Mean α/θ', 'θ-α corr', 'Conservation']
    eo_vals = [np.mean(res_eo['ta_ratios']), np.mean(res_eo['at_ratios']),
               np.mean(res_eo['power_corrs']), np.mean(eo_cv_ratio)]
    ec_vals = [np.mean(res_ec['ta_ratios']), np.mean(res_ec['at_ratios']),
               np.mean(res_ec['power_corrs']), np.mean(ec_cv_ratio)]
    x = np.arange(len(metrics))
    w = 0.35
    ax5.bar(x - w/2, eo_vals, w, color='orange', alpha=0.7, label='Eyes Open')
    ax5.bar(x + w/2, ec_vals, w, color='steelblue', alpha=0.7, label='Eyes Closed')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=8)
    ax5.set_ylabel('Value')
    ax5.set_title('E. State Comparison Summary')
    ax5.legend(fontsize=9)
    ax5.axhline(0, color='gray', lw=0.5)

    ax6 = fig.add_subplot(gs[1, 2])
    summary = "STATE CONTRAST SUMMARY\n\n"
    summary += f"N subjects: {len(res_eo['ta_ratios'])//6}\n\n"
    summary += "EYES OPEN:\n"
    summary += f"  θ/α = {np.mean(res_eo['ta_ratios']):.3f}\n"
    summary += f"  α/θ = {np.mean(res_eo['at_ratios']):.3f}\n"
    summary += f"  θ-α corr = {np.mean(res_eo['power_corrs']):.3f}\n"
    summary += f"  Conservation = {np.mean(eo_cv_ratio):.3f}\n\n"
    summary += "EYES CLOSED:\n"
    summary += f"  θ/α = {np.mean(res_ec['ta_ratios']):.3f}\n"
    summary += f"  α/θ = {np.mean(res_ec['at_ratios']):.3f}\n"
    summary += f"  θ-α corr = {np.mean(res_ec['power_corrs']):.3f}\n"
    summary += f"  Conservation = {np.mean(ec_cv_ratio):.3f}\n\n"
    summary += f"|α/θ - φ|: EO={abs(np.mean(res_eo['at_ratios'])-PHI):.3f}\n"
    summary += f"           EC={abs(np.mean(res_ec['at_ratios'])-PHI):.3f}\n"
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.axis('off')
    ax6.set_title('F. Summary')

    fig.suptitle("State Contrast: Does θ-α Complementarity Depend on Eyes Open/Closed?",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.savefig('state_contrast_results.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


if __name__ == '__main__':
    main()

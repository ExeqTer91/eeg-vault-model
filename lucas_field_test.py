#!/usr/bin/env python3
"""
Lucas Field Test
================
Tests whether EEG frequency ratios produce X_n(t) = r(t)^n + r(t)^(-n)
values that concentrate near integers (especially Lucas numbers L_n)
more than null models predict.

If r = φ, then X_n = L_n exactly. This is the measurable invariant.

Null models:
  1. Time-shuffle (permute r across epochs)
  2. Subject-block-shuffle (permute r within subjects)
  3. Noise-around-theory (r ~ N(μ_obs, σ_obs) synthetic)
  4. Phase-randomized surrogate

Also computes the phase version: Δθ(t) = 2π(r(t) - 1)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import circmean, circstd
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47]  # L_0 through L_8
N_RANGE = range(1, 9)  # n = 1..8
N_SURROGATES = 10000

def compute_Xn(r, n):
    return r**n + r**(-n)

def delta_integer(Xn):
    return np.abs(Xn - np.round(Xn))

def delta_lucas(Xn, n):
    return np.abs(Xn - LUCAS[n])

def run_lucas_analysis(r_values, label=""):
    results = {}
    for n in N_RANGE:
        Xn = compute_Xn(r_values, n)
        d_int = delta_integer(Xn)
        d_luc = delta_lucas(Xn, n)
        Xn_phi = PHI**n + PHI**(-n)
        results[n] = {
            'Xn_mean': np.mean(Xn),
            'Xn_std': np.std(Xn),
            'Xn_phi': Xn_phi,
            'L_n': LUCAS[n],
            'delta_int_mean': np.mean(d_int),
            'delta_int_median': np.median(d_int),
            'delta_luc_mean': np.mean(d_luc),
            'delta_luc_median': np.median(d_luc),
            'frac_near_int_01': np.mean(d_int < 0.1),
            'frac_near_int_02': np.mean(d_int < 0.2),
            'frac_near_lucas_01': np.mean(d_luc < 0.1),
        }
    return results

def null_time_shuffle(r_values, n_surr=N_SURROGATES):
    null_delta_int = {n: [] for n in N_RANGE}
    null_delta_luc = {n: [] for n in N_RANGE}
    for _ in range(n_surr):
        r_shuf = np.random.permutation(r_values)
        for n in N_RANGE:
            Xn = compute_Xn(r_shuf, n)
            null_delta_int[n].append(np.mean(delta_integer(Xn)))
            null_delta_luc[n].append(np.mean(delta_lucas(Xn, n)))
    return null_delta_int, null_delta_luc

def null_subject_block_shuffle(r_values, subjects, n_surr=N_SURROGATES):
    null_delta_int = {n: [] for n in N_RANGE}
    null_delta_luc = {n: [] for n in N_RANGE}
    unique_subj = np.unique(subjects)
    for _ in range(n_surr):
        r_shuf = r_values.copy()
        for s in unique_subj:
            mask = subjects == s
            r_shuf[mask] = np.random.permutation(r_values[mask])
        for n in N_RANGE:
            Xn = compute_Xn(r_shuf, n)
            null_delta_int[n].append(np.mean(delta_integer(Xn)))
            null_delta_luc[n].append(np.mean(delta_lucas(Xn, n)))
    return null_delta_int, null_delta_luc

def null_noise_around_theory(r_values, n_surr=N_SURROGATES):
    mu, sigma = np.mean(r_values), np.std(r_values)
    null_delta_int = {n: [] for n in N_RANGE}
    null_delta_luc = {n: [] for n in N_RANGE}
    for _ in range(n_surr):
        r_synth = np.random.normal(mu, sigma, len(r_values))
        r_synth = r_synth[r_synth > 0.5]
        if len(r_synth) < 10:
            continue
        for n in N_RANGE:
            Xn = compute_Xn(r_synth, n)
            null_delta_int[n].append(np.mean(delta_integer(Xn)))
            null_delta_luc[n].append(np.mean(delta_lucas(Xn, n)))
    return null_delta_int, null_delta_luc

def null_phase_randomized(r_values, n_surr=N_SURROGATES):
    N = len(r_values)
    ft = np.fft.rfft(r_values - np.mean(r_values))
    amp = np.abs(ft)
    null_delta_int = {n: [] for n in N_RANGE}
    null_delta_luc = {n: [] for n in N_RANGE}
    for _ in range(n_surr):
        phases = np.random.uniform(0, 2*np.pi, len(ft))
        phases[0] = 0
        if N % 2 == 0:
            phases[-1] = 0
        ft_surr = amp * np.exp(1j * phases)
        r_surr = np.fft.irfft(ft_surr, n=N) + np.mean(r_values)
        r_surr = np.clip(r_surr, 0.5, 5.0)
        for n in N_RANGE:
            Xn = compute_Xn(r_surr, n)
            null_delta_int[n].append(np.mean(delta_integer(Xn)))
            null_delta_luc[n].append(np.mean(delta_lucas(Xn, n)))
    return null_delta_int, null_delta_luc

def compute_p_and_z(obs, null_dist):
    null_arr = np.array(null_dist)
    p = np.mean(null_arr <= obs)
    z = (obs - np.mean(null_arr)) / (np.std(null_arr) + 1e-15)
    return p, z

def phase_analysis(r_values):
    delta_theta = 2 * np.pi * (r_values - 1)
    delta_theta_mod = delta_theta % (2 * np.pi)
    cm = circmean(delta_theta_mod)
    cs = circstd(delta_theta_mod)
    from scipy.stats import kstest
    _, p_uniform = kstest(delta_theta_mod / (2*np.pi), 'uniform')
    phi_angle = 2 * np.pi * (PHI - 1)
    phi_angle_mod = phi_angle % (2 * np.pi)
    angular_dist = np.abs(np.angle(np.exp(1j * (delta_theta_mod - phi_angle_mod))))
    mean_angular_dist = np.mean(angular_dist)
    null_dists = []
    for _ in range(N_SURROGATES):
        r_shuf = np.random.permutation(r_values)
        dt_shuf = 2 * np.pi * (r_shuf - 1) % (2 * np.pi)
        null_dists.append(np.mean(np.abs(np.angle(np.exp(1j * (dt_shuf - phi_angle_mod))))))
    p_phase = np.mean(np.array(null_dists) <= mean_angular_dist)
    return {
        'circular_mean': cm,
        'circular_std': cs,
        'p_uniform': p_uniform,
        'phi_angle': phi_angle_mod,
        'mean_angular_dist_to_phi': mean_angular_dist,
        'p_phase_vs_null': p_phase,
    }


def main():
    print("=" * 70)
    print("LUCAS FIELD TEST — EEG Frequency Ratio Analysis")
    print("=" * 70)

    print("\nLoading EEG epoch data...")
    df = pd.read_csv('epoch_features_n109.csv')
    print(f"  Loaded {len(df)} epochs from {df['subject'].nunique()} subjects")

    r_gb = df['gamma_cf'].values / df['beta_cf'].values
    valid = np.isfinite(r_gb) & (r_gb > 0.5) & (r_gb < 5.0)
    r_gb = r_gb[valid]
    subjects_gb = df['subject'].values[valid]
    print(f"  gamma/beta ratio: N={len(r_gb)}, mean={np.mean(r_gb):.4f}, "
          f"std={np.std(r_gb):.4f}, median={np.median(r_gb):.4f}")
    print(f"  φ = {PHI:.6f}")

    r_ab = df['alpha_power'].values / df['theta_power'].values
    valid_ab = np.isfinite(r_ab) & (r_ab > 0) & (r_ab < 20)
    r_ab = r_ab[valid_ab]
    subjects_ab = df['subject'].values[valid_ab]
    print(f"  alpha/theta power ratio: N={len(r_ab)}, mean={np.mean(r_ab):.4f}, "
          f"std={np.std(r_ab):.4f}")

    for ratio_name, r_values, subj_arr in [
        ("gamma_cf / beta_cf", r_gb, subjects_gb),
        ("alpha_power / theta_power", r_ab, subjects_ab),
    ]:
        print(f"\n{'='*70}")
        print(f"RATIO: {ratio_name}  (N = {len(r_values)})")
        print(f"{'='*70}")

        obs = run_lucas_analysis(r_values, ratio_name)

        print(f"\n--- Observed X_n statistics ---")
        print(f"{'n':>3} | {'X_n(φ)':>8} | {'L_n':>4} | {'<X_n>':>8} | {'σ(X_n)':>8} | "
              f"{'<δ_int>':>8} | {'<δ_Luc>':>8} | {'%<0.1 int':>10} | {'%<0.2 int':>10}")
        print("-" * 95)
        for n in N_RANGE:
            o = obs[n]
            print(f"{n:3d} | {o['Xn_phi']:8.3f} | {o['L_n']:4d} | {o['Xn_mean']:8.3f} | "
                  f"{o['Xn_std']:8.3f} | {o['delta_int_mean']:8.4f} | {o['delta_luc_mean']:8.4f} | "
                  f"{o['frac_near_int_01']*100:9.1f}% | {o['frac_near_int_02']*100:9.1f}%")

        print(f"\n--- Null Model 1: Time-shuffle ({N_SURROGATES} permutations) ---")
        null1_int, null1_luc = null_time_shuffle(r_values)
        print(f"{'n':>3} | {'obs <δ_int>':>12} | {'null <δ_int>':>12} | {'p':>8} | {'z':>8} | "
              f"{'obs <δ_Luc>':>12} | {'null <δ_Luc>':>12} | {'p':>8} | {'z':>8}")
        print("-" * 110)
        for n in N_RANGE:
            o_int = obs[n]['delta_int_mean']
            o_luc = obs[n]['delta_luc_mean']
            p_int, z_int = compute_p_and_z(o_int, null1_int[n])
            p_luc, z_luc = compute_p_and_z(o_luc, null1_luc[n])
            print(f"{n:3d} | {o_int:12.5f} | {np.mean(null1_int[n]):12.5f} | {p_int:8.4f} | {z_int:8.3f} | "
                  f"{o_luc:12.5f} | {np.mean(null1_luc[n]):12.5f} | {p_luc:8.4f} | {z_luc:8.3f}")

        print(f"\n--- Null Model 2: Subject-block-shuffle ({N_SURROGATES} permutations) ---")
        null2_int, null2_luc = null_subject_block_shuffle(r_values, subj_arr)
        print(f"{'n':>3} | {'obs <δ_int>':>12} | {'null <δ_int>':>12} | {'p':>8} | {'z':>8} | "
              f"{'obs <δ_Luc>':>12} | {'null <δ_Luc>':>12} | {'p':>8} | {'z':>8}")
        print("-" * 110)
        for n in N_RANGE:
            o_int = obs[n]['delta_int_mean']
            o_luc = obs[n]['delta_luc_mean']
            p_int, z_int = compute_p_and_z(o_int, null2_int[n])
            p_luc, z_luc = compute_p_and_z(o_luc, null2_luc[n])
            print(f"{n:3d} | {o_int:12.5f} | {np.mean(null2_int[n]):12.5f} | {p_int:8.4f} | {z_int:8.3f} | "
                  f"{o_luc:12.5f} | {np.mean(null2_luc[n]):12.5f} | {p_luc:8.4f} | {z_luc:8.3f}")

        print(f"\n--- Null Model 3: Noise-around-theory ({N_SURROGATES} synthetic) ---")
        null3_int, null3_luc = null_noise_around_theory(r_values)
        print(f"{'n':>3} | {'obs <δ_int>':>12} | {'null <δ_int>':>12} | {'p':>8} | {'z':>8} | "
              f"{'obs <δ_Luc>':>12} | {'null <δ_Luc>':>12} | {'p':>8} | {'z':>8}")
        print("-" * 110)
        for n in N_RANGE:
            o_int = obs[n]['delta_int_mean']
            o_luc = obs[n]['delta_luc_mean']
            p_int, z_int = compute_p_and_z(o_int, null3_int[n])
            p_luc, z_luc = compute_p_and_z(o_luc, null3_luc[n])
            print(f"{n:3d} | {o_int:12.5f} | {np.mean(null3_int[n]):12.5f} | {p_int:8.4f} | {z_int:8.3f} | "
                  f"{o_luc:12.5f} | {np.mean(null3_luc[n]):12.5f} | {p_luc:8.4f} | {z_luc:8.3f}")

        print(f"\n--- Null Model 4: Phase-randomized surrogate ({N_SURROGATES}) ---")
        null4_int, null4_luc = null_phase_randomized(r_values)
        print(f"{'n':>3} | {'obs <δ_int>':>12} | {'null <δ_int>':>12} | {'p':>8} | {'z':>8} | "
              f"{'obs <δ_Luc>':>12} | {'null <δ_Luc>':>12} | {'p':>8} | {'z':>8}")
        print("-" * 110)
        for n in N_RANGE:
            o_int = obs[n]['delta_int_mean']
            o_luc = obs[n]['delta_luc_mean']
            p_int, z_int = compute_p_and_z(o_int, null4_int[n])
            p_luc, z_luc = compute_p_and_z(o_luc, null4_luc[n])
            print(f"{n:3d} | {o_int:12.5f} | {np.mean(null4_int[n]):12.5f} | {p_int:8.4f} | {z_int:8.3f} | "
                  f"{o_luc:12.5f} | {np.mean(null4_luc[n]):12.5f} | {p_luc:8.4f} | {z_luc:8.3f}")

        print(f"\n--- Phase Analysis ---")
        phase = phase_analysis(r_values)
        print(f"  Circular mean of Δθ(t) = 2π(r-1): {phase['circular_mean']:.4f} rad")
        print(f"  Circular std: {phase['circular_std']:.4f} rad")
        print(f"  φ-implied angle: {phase['phi_angle']:.4f} rad")
        print(f"  Mean angular distance to φ-angle: {phase['mean_angular_dist_to_phi']:.4f} rad")
        print(f"  p(uniform): {phase['p_uniform']:.2e}")
        print(f"  p(closer to φ-angle than shuffle): {phase['p_phase_vs_null']:.4f}")

    print("\n\nGenerating figure...")
    generate_figure(r_gb, subjects_gb, r_ab, subjects_ab)
    print("Saved: lucas_field_test_results.png")

    print("\n" + "=" * 70)
    print("LUCAS FIELD TEST COMPLETE")
    print("=" * 70)


def generate_figure(r_gb, subj_gb, r_ab, subj_ab):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r_gb, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='none')
    ax1.axvline(PHI, color='gold', lw=2, ls='--', label=f'φ = {PHI:.3f}')
    ax1.axvline(2.0, color='red', lw=2, ls='--', label='2:1 = 2.0')
    ax1.axvline(np.mean(r_gb), color='black', lw=1.5, label=f'mean = {np.mean(r_gb):.3f}')
    ax1.set_xlabel('γ/β centroid ratio')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Distribution of γ/β ratio')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(r_ab, bins=80, density=True, alpha=0.7, color='coral', edgecolor='none')
    ax2.axvline(PHI, color='gold', lw=2, ls='--', label=f'φ = {PHI:.3f}')
    ax2.axvline(np.mean(r_ab), color='black', lw=1.5, label=f'mean = {np.mean(r_ab):.3f}')
    ax2.set_xlabel('α/θ power ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('B. Distribution of α/θ ratio')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ns = list(N_RANGE)
    xn_obs_gb = [np.mean(compute_Xn(r_gb, n)) for n in ns]
    xn_phi = [PHI**n + PHI**(-n) for n in ns]
    lucas_vals = [LUCAS[n] for n in ns]
    ax3.plot(ns, xn_obs_gb, 'o-', color='steelblue', lw=2, ms=8, label='Observed <X_n> (γ/β)')
    ax3.plot(ns, xn_phi, 's--', color='gold', lw=2, ms=8, label='X_n(φ)')
    ax3.plot(ns, lucas_vals, 'D:', color='green', lw=2, ms=8, label='Lucas L_n')
    ax3.set_xlabel('n')
    ax3.set_ylabel('X_n value')
    ax3.set_title('C. X_n = r^n + r^{-n} vs Lucas')
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    for ratio_name, r_vals, color in [("γ/β", r_gb, 'steelblue'), ("α/θ", r_ab, 'coral')]:
        d_int = [np.mean(delta_integer(compute_Xn(r_vals, n))) for n in ns]
        ax4.plot(ns, d_int, 'o-', color=color, lw=2, ms=7, label=ratio_name)
    null_d = []
    for n in ns:
        nd = []
        for _ in range(1000):
            r_shuf = np.random.permutation(r_gb)
            nd.append(np.mean(delta_integer(compute_Xn(r_shuf, n))))
        null_d.append(nd)
    null_mean = [np.mean(nd) for nd in null_d]
    null_lo = [np.percentile(nd, 2.5) for nd in null_d]
    null_hi = [np.percentile(nd, 97.5) for nd in null_d]
    ax4.fill_between(ns, null_lo, null_hi, alpha=0.2, color='gray', label='95% null CI')
    ax4.plot(ns, null_mean, '--', color='gray', lw=1.5, label='Null mean')
    ax4.set_xlabel('n')
    ax4.set_ylabel('<δ_integer>')
    ax4.set_title('D. Distance to nearest integer')
    ax4.legend(fontsize=7)

    ax5 = fig.add_subplot(gs[1, 1])
    for ratio_name, r_vals, color in [("γ/β", r_gb, 'steelblue'), ("α/θ", r_ab, 'coral')]:
        d_luc = [np.mean(delta_lucas(compute_Xn(r_vals, n), n)) for n in ns]
        ax5.plot(ns, d_luc, 'o-', color=color, lw=2, ms=7, label=ratio_name)
    null_dl = []
    for n in ns:
        nd = []
        for _ in range(1000):
            r_shuf = np.random.permutation(r_gb)
            nd.append(np.mean(delta_lucas(compute_Xn(r_shuf, n), n)))
        null_dl.append(nd)
    null_mean_l = [np.mean(nd) for nd in null_dl]
    null_lo_l = [np.percentile(nd, 2.5) for nd in null_dl]
    null_hi_l = [np.percentile(nd, 97.5) for nd in null_dl]
    ax5.fill_between(ns, null_lo_l, null_hi_l, alpha=0.2, color='gray', label='95% null CI')
    ax5.plot(ns, null_mean_l, '--', color='gray', lw=1.5, label='Null mean')
    ax5.set_xlabel('n')
    ax5.set_ylabel('<|X_n - L_n|>')
    ax5.set_title('E. Distance to Lucas numbers')
    ax5.legend(fontsize=7)

    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    delta_theta_gb = (2 * np.pi * (r_gb - 1)) % (2 * np.pi)
    phi_angle = (2 * np.pi * (PHI - 1)) % (2 * np.pi)
    ax6.hist(delta_theta_gb, bins=72, density=True, alpha=0.6, color='steelblue')
    ax6.axvline(phi_angle, color='gold', lw=3, label=f'φ-angle')
    ax6.set_title('F. Phase Δθ = 2π(r-1)', pad=15)

    ax7 = fig.add_subplot(gs[2, 0])
    n_show = 4
    Xn4 = compute_Xn(r_gb, n_show)
    ax7.hist(Xn4, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='none')
    ax7.axvline(LUCAS[n_show], color='green', lw=2, ls='--', label=f'L_{n_show} = {LUCAS[n_show]}')
    ax7.axvline(PHI**n_show + PHI**(-n_show), color='gold', lw=2, ls=':', label=f'X_{n_show}(φ) = {PHI**n_show + PHI**(-n_show):.3f}')
    ax7.set_xlabel(f'X_{n_show}')
    ax7.set_ylabel('Density')
    ax7.set_title(f'G. Distribution of X_{n_show} (γ/β)')
    ax7.legend(fontsize=8)

    ax8 = fig.add_subplot(gs[2, 1])
    subj_means = pd.DataFrame({'r': r_gb, 'subject': subj_gb}).groupby('subject')['r'].mean()
    subj_Xn4 = compute_Xn(subj_means.values, n_show)
    subj_d_luc = delta_lucas(subj_Xn4, n_show)
    ax8.hist(subj_d_luc, bins=30, density=True, alpha=0.7, color='teal', edgecolor='none')
    ax8.set_xlabel(f'|X_{n_show} - L_{n_show}|')
    ax8.set_ylabel('Density')
    ax8.set_title(f'H. Subject-level Lucas distance (n={n_show})')
    ax8.axvline(np.mean(subj_d_luc), color='black', lw=1.5, ls='--', label=f'mean = {np.mean(subj_d_luc):.3f}')
    ax8.legend(fontsize=8)

    ax9 = fig.add_subplot(gs[2, 2])
    summary_text = "LUCAS FIELD TEST SUMMARY\n\n"
    summary_text += f"N epochs (γ/β): {len(r_gb)}\n"
    summary_text += f"N epochs (α/θ): {len(r_ab)}\n"
    summary_text += f"Mean γ/β ratio: {np.mean(r_gb):.4f}\n"
    summary_text += f"φ = {PHI:.6f}\n\n"
    summary_text += "Key: If δ_obs < δ_null → Lucas\n"
    summary_text += "organization is real.\n"
    summary_text += "If δ_obs ≈ δ_null → numerology.\n\n"

    obs_gb = run_lucas_analysis(r_gb)
    for n in [1, 2, 4, 7]:
        d = obs_gb[n]
        summary_text += f"n={n}: <X_n>={d['Xn_mean']:.2f} (L_n={d['L_n']})\n"
        summary_text += f"  δ_int={d['delta_int_mean']:.4f}\n"

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax9.axis('off')
    ax9.set_title('I. Summary')

    fig.suptitle('Lucas Field Test: Do EEG Ratios Track φ-Organization?',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('lucas_field_test_results.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


if __name__ == '__main__':
    main()

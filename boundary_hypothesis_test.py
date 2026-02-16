#!/usr/bin/env python3
"""
Boundary Hypothesis Test
EEG dynamics operate in a controlled intermediate regime avoiding:
- Locking extreme (over-synchrony / rigid)
- Drift extreme (dephasing / unstable)
Slow scalar s(t) modulates movement toward/away from boundaries.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy, circvar
from scipy.optimize import curve_fit
import os, warnings
warnings.filterwarnings('ignore')

RES = 'results_boundary'
FIG = 'figures_boundary'
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

N_PERM = 200

def transition_entropy_vec(states, n_states=6):
    states = [int(s) for s in states if not np.isnan(s)]
    if len(states) < 2:
        return 0
    M = np.zeros((n_states, n_states))
    for a, b in zip(states[:-1], states[1:]):
        if a < n_states and b < n_states:
            M[a, b] += 1
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    M = M / rs
    return entropy(M.flatten() + 1e-10)

def compute_indices(df):
    df = df.copy()
    sub_means = df.groupby('subject')['beta_cf'].transform('mean')
    sub_stds = df.groupby('subject')['beta_cf'].transform('std').replace(0, 1)
    beta_z = (df['beta_cf'] - sub_means) / sub_stds

    sub_means_g = df.groupby('subject')['gamma_cf'].transform('mean')
    sub_stds_g = df.groupby('subject')['gamma_cf'].transform('std').replace(0, 1)
    gamma_z = (df['gamma_cf'] - sub_means_g) / sub_stds_g

    df['locking_index'] = 1.0 / (np.abs(beta_z - gamma_z) + 0.01)

    state_series = df.sort_values(['subject', 'epoch_id'])['state'].values
    switch = np.zeros(len(df))
    sorted_df = df.sort_values(['subject', 'epoch_id'])
    idx = sorted_df.index.values
    prev_sub = None
    prev_state = None
    for i, row_idx in enumerate(idx):
        s = sorted_df.loc[row_idx, 'subject']
        st = sorted_df.loc[row_idx, 'state']
        if s == prev_sub and prev_state is not None:
            switch[i] = 1 if st != prev_state else 0
        prev_sub = s
        prev_state = st

    window = 5
    drift_vals = np.convolve(switch, np.ones(window)/window, mode='same')
    df.loc[idx, 'drift_index'] = drift_vals

    r = df['r'].values
    r_mean = df.groupby('subject')['r'].transform('mean').values
    r_std = df.groupby('subject')['r'].transform('std').replace(0, 1).values
    r_z = np.abs((r - r_mean) / r_std)
    df['shelf_stability'] = 1.0 / (r_z + 0.1)

    return df


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def test1_u_shape(df):
    print("TEST 1: U-shaped optimum (Within-subject quadratic)...")
    scalar = 'aperiodic_exponent'
    all_rows = []
    
    # Store coefficients for within-subject models
    sub_coeffs = []
    
    for sub in df['subject'].unique():
        sdf = df[df['subject'] == sub].copy()
        if len(sdf) < 10: continue
        
        # Continuous within-subject fit
        x = sdf[scalar].values
        y = sdf['shelf_stability'].values
        # Normalize x to avoid scale issues
        x_norm = (x - x.mean()) / (x.std() + 1e-10)
        
        try:
            popt, _ = curve_fit(quadratic, x_norm, y, p0=[0, 0, y.mean()])
            sub_coeffs.append(popt[0]) # a coefficient
        except:
            pass

        terciles = sdf[scalar].quantile([0.33, 0.66]).values
        sdf['sbin'] = pd.cut(sdf[scalar],
                             bins=[-np.inf, terciles[0], terciles[1], np.inf],
                             labels=['LOW', 'MID', 'HIGH'])
        for b in ['LOW', 'MID', 'HIGH']:
            bdf = sdf[sdf['sbin'] == b]
            if len(bdf) < 2: continue
            all_rows.append({
                'subject': sub, 'bin': b,
                'locking': bdf['locking_index'].mean(),
                'drift': bdf['drift_index'].mean(),
                'stability': bdf['shelf_stability'].mean(),
                'n': len(bdf)
            })

    res = pd.DataFrame(all_rows)
    res.to_csv(f'{RES}/per_subject_bins.csv', index=False)
    
    # Global mean coefficients
    mean_a = np.mean(sub_coeffs) if sub_coeffs else 0
    
    # Permutation on within-subject means
    perm_a_means = []
    for _ in range(N_PERM):
        p_coeffs = []
        for sub in df['subject'].unique():
            sdf = df[df['subject'] == sub].copy()
            if len(sdf) < 10: continue
            y_shuf = np.random.permutation(sdf['shelf_stability'].values)
            x_norm = (sdf[scalar].values - sdf[scalar].mean()) / (sdf[scalar].std() + 1e-10)
            try:
                pp, _ = curve_fit(quadratic, x_norm, y_shuf, p0=[0, 0, y_shuf.mean()])
                p_coeffs.append(pp[0])
            except: pass
        if p_coeffs:
            perm_a_means.append(np.mean(p_coeffs))
    
    perm_p = (np.sum(np.array(perm_a_means) <= mean_a) + 1) / (len(perm_a_means) + 1) if perm_a_means else 1.0

    mid_stab = res[res['bin'] == 'MID']['stability'].mean()
    low_stab = res[res['bin'] == 'LOW']['stability'].mean()
    high_stab = res[res['bin'] == 'HIGH']['stability'].mean()
    inverted_u = mean_a < 0 and perm_p < 0.05

    with open(f'{RES}/u_shape_fit_stats.txt', 'w') as f:
        f.write(f"Mean within-subject quadratic a = {mean_a:.6f}\n")
        f.write(f"Permutation p (a < 0, inverted-U) = {perm_p:.4f}\n")
        f.write(f"Stability Bins: LOW={low_stab:.4f} MID={mid_stab:.4f} HIGH={high_stab:.4f}\n")
        f.write(f"Inverted-U significant: {inverted_u}\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bins_order = ['LOW', 'MID', 'HIGH']
    colors = ['#D55E00', '#009E73', '#0072B2']

    for i, metric in enumerate(['stability', 'locking', 'drift']):
        means = [res[res['bin'] == b][metric].mean() for b in bins_order]
        sems = [res[res['bin'] == b][metric].std() / np.sqrt(max(1, res[res['bin'] == b][metric].count())) for b in bins_order]
        axes[i].bar(bins_order, means, yerr=sems, color=colors, capsize=5)
        axes[i].set_title(metric.upper())

    fig.suptitle(f'TEST 1: Stability vs Scalar (a_mean={mean_a:.4f}, p={perm_p:.3f})', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{FIG}/stability_vs_scalar_bins.tiff', dpi=300, facecolor='white')
    plt.close()

    verdict = "PASS" if inverted_u else ("MARGINAL" if mean_a < 0 else "FAIL")
    return verdict, {'low': low_stab, 'mid': mid_stab, 'high': high_stab, 'a': mean_a, 'perm_p': perm_p}


def test2_extreme_avoidance(df):
    print("TEST 2: Extreme avoidance...")
    scalar = 'aperiodic_exponent'

    lock_thresh = df['locking_index'].quantile(0.90)
    drift_thresh = df['drift_index'].quantile(0.90)

    rows = []
    for sub in df['subject'].unique():
        sdf = df[df['subject'] == sub].copy()
        terciles = sdf[scalar].quantile([0.33, 0.66]).values
        sdf['sbin'] = pd.cut(sdf[scalar],
                             bins=[-np.inf, terciles[0], terciles[1], np.inf],
                             labels=['LOW', 'MID', 'HIGH'])
        for b in ['LOW', 'MID', 'HIGH']:
            bdf = sdf[sdf['sbin'] == b]
            if len(bdf) == 0:
                continue
            lock_frac = (bdf['locking_index'] >= lock_thresh).mean()
            drift_frac = (bdf['drift_index'] >= drift_thresh).mean()
            rows.append({
                'subject': sub, 'bin': b,
                'lock_extreme_frac': lock_frac,
                'drift_extreme_frac': drift_frac,
                'total_extreme_frac': lock_frac + drift_frac,
                'n': len(bdf)
            })

    res = pd.DataFrame(rows)
    res.to_csv(f'{RES}/extreme_occupancy.csv', index=False)

    bins_order = ['LOW', 'MID', 'HIGH']
    colors = ['#D55E00', '#009E73', '#0072B2']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['lock_extreme_frac', 'drift_extreme_frac', 'total_extreme_frac']):
        means = [res[res['bin'] == b][metric].mean() for b in bins_order]
        sems = [res[res['bin'] == b][metric].std() / np.sqrt(max(1, res[res['bin'] == b][metric].count())) for b in bins_order]
        axes[i].bar(bins_order, means, yerr=sems, color=colors, capsize=5)
        titles = {'lock_extreme_frac': 'Locking Extremes', 'drift_extreme_frac': 'Drift Extremes', 'total_extreme_frac': 'Total Extremes'}
        axes[i].set_title(titles[metric])
        axes[i].set_xlabel('s(t) bin')
        axes[i].set_ylabel('Fraction of epochs')

    fig.suptitle('TEST 2: Extreme Avoidance by Scalar Bin', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{FIG}/extreme_occupancy_by_bin.tiff', dpi=300, facecolor='white')
    plt.close()

    mid_ext = res[res['bin'] == 'MID']['total_extreme_frac'].mean()
    low_ext = res[res['bin'] == 'LOW']['total_extreme_frac'].mean()
    high_ext = res[res['bin'] == 'HIGH']['total_extreme_frac'].mean()
    avoids = mid_ext < low_ext and mid_ext < high_ext

    verdict = "PASS" if avoids else "FAIL"
    print(f"  Extremes: LOW={low_ext:.4f} MID={mid_ext:.4f} HIGH={high_ext:.4f} | Avoids={avoids} | {verdict}")
    return verdict, {'low': low_ext, 'mid': mid_ext, 'high': high_ext}


def test3_recovery(df):
    print("TEST 3: Recovery dynamics (Locking vs Drift)...")
    scalar = 'aperiodic_exponent'
    lock_thresh = df['locking_index'].quantile(0.90)
    drift_thresh = df['drift_index'].quantile(0.90)

    K = 10
    results = {'locking': {'ds': [], 'dstab': []}, 'drift': {'ds': [], 'dstab': []}}

    for sub in df['subject'].unique():
        sdf = df[df['subject'] == sub].sort_values('epoch_id').reset_index(drop=True)
        s_med = sdf[scalar].median()

        for i in range(len(sdf) - K):
            is_lock = sdf.loc[i, 'locking_index'] >= lock_thresh
            is_drift = sdf.loc[i, 'drift_index'] >= drift_thresh
            
            for key, hit in [('locking', is_lock), ('drift', is_drift)]:
                if hit:
                    ds = sdf.loc[i:i+K-1, scalar].values - sdf.loc[i, scalar]
                    dstab = sdf.loc[i:i+K-1, 'shelf_stability'].values - sdf.loc[i, 'shelf_stability']
                    if len(ds) == K:
                        results[key]['ds'].append(ds)
                        results[key]['dstab'].append(dstab)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = range(K)
    
    pass_count = 0
    for i, key in enumerate(['locking', 'drift']):
        if not results[key]['ds']: continue
        
        m_ds = np.mean(results[key]['ds'], axis=0)
        m_dstab = np.mean(results[key]['dstab'], axis=0)
        
        axes[0, i].plot(x, m_ds, 'o-', label=key)
        axes[0, i].set_title(f'{key.capitalize()} Event: ΔScalar')
        axes[1, i].plot(x, m_dstab, 'o-', color='green')
        axes[1, i].set_title(f'{key.capitalize()} Event: ΔStability')
        
        # Check for recovery (stability increase)
        slope = np.polyfit(x, m_dstab, 1)[0]
        if slope > 0: pass_count += 1

    plt.tight_layout()
    fig.savefig(f'{FIG}/recovery_trajectory.tiff', dpi=300)
    plt.close()

    verdict = "PASS" if pass_count == 2 else ("MARGINAL" if pass_count > 0 else "FAIL")
    return verdict, {'n_lock': len(results['locking']['ds']), 'n_drift': len(results['drift']['ds'])}


def main():
    df = pd.read_csv('epoch_features_fractal.csv')
    print(f"Data: {len(df)} epochs, {df.subject.nunique()} subjects\n")

    df = compute_indices(df)

    v1, d1 = test1_u_shape(df)
    v2, d2 = test2_extreme_avoidance(df)
    v3, d3 = test3_recovery(df)

    scorecard = f"""# Boundary Hypothesis — Scorecard (N=15)

## TEST 1: U-Shaped Optimum (Inverted-U for stability)
- Stability: LOW={d1['low']:.4f} MID={d1['mid']:.4f} HIGH={d1['high']:.4f}
- Quadratic a = {d1['a']:.4f}, permutation p = {d1['perm_p']:.3f}
- **Verdict: {v1}**

## TEST 2: Extreme Avoidance
- Extreme fraction: LOW={d2.get('low',0):.4f} MID={d2.get('mid',0):.4f} HIGH={d2.get('high',0):.4f}
- MID avoids extremes: {d2.get('mid',0) < d2.get('low',0) and d2.get('mid',0) < d2.get('high',0)}
- **Verdict: {v2}**

## TEST 3: Recovery Dynamics (Locking vs Drift)
- N locking events: {d3.get('n_lock','N/A')}
- N drift events: {d3.get('n_drift','N/A')}
- **Verdict: {v3}**

## Summary
| Test | Verdict |
|------|---------|
| U-Shaped Optimum | {v1} |
| Extreme Avoidance | {v2} |
| Recovery Dynamics | {v3} |

## Interpretation
The system {"shows" if v1 == "PASS" else "does not clearly show"} an inverted-U stability profile.
Extremes are {"avoided" if v2 == "PASS" else "not clearly avoided"} in the MID scalar range.
Recovery after extremes {"is" if v3 == "PASS" else "is not"} systematically directed toward the middle regime.
"""

    with open('boundary_scorecard.md', 'w') as f:
        f.write(scorecard)
    print(scorecard)


if __name__ == '__main__':
    main()

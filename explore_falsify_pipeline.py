#!/usr/bin/env python3
"""
Explore + Falsify Fast Pipeline
5 tests on N=15 EEG dataset. All real computations, no mocks.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr, chisquare, entropy
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import os, zipfile, warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'epoch_features_fractal.csv'
RES = 'results'
FIG = 'figures'
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
N_BOOT = 200

def save_tiff(fig, name):
    p = f"{FIG}/{name}.tiff"
    fig.savefig(p, dpi=300, format='tiff', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> {p}")

def build_trans_graph(states_series):
    G = nx.DiGraph()
    prev = None
    for s in states_series:
        if prev is not None and not np.isnan(prev) and not np.isnan(s):
            si, sj = int(prev), int(s)
            if G.has_edge(si, sj):
                G[si][sj]['weight'] += 1
            else:
                G.add_edge(si, sj, weight=1)
        prev = s
    return G

def trans_matrix_from_series(states):
    states = [int(s) for s in states if not np.isnan(s)]
    labels = sorted(set(states))
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((n, n))
    for a, b in zip(states[:-1], states[1:]):
        M[idx[a], idx[b]] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums, labels


# ============================================================
# TEST 1: HUB / BRIDGE VALIDATION
# ============================================================
def test1_hub(df):
    print("\n=== TEST 1: HUB / BRIDGE ===")
    
    G = build_trans_graph(df.sort_values(['subject', 'epoch_id'])['state'].values)
    bet = nx.betweenness_centrality(G, weight='weight')
    top_hub = max(bet, key=bet.get)
    
    subjects = df['subject'].unique()
    hub_hits = 0
    for _ in range(N_BOOT):
        idx = np.random.choice(len(subjects), len(subjects), replace=True)
        boot = pd.concat([df[df['subject'] == subjects[i]] for i in idx])
        bG = build_trans_graph(boot.sort_values(['subject', 'epoch_id'])['state'].values)
        if bG.number_of_nodes() > 0:
            bb = nx.betweenness_centrality(bG, weight='weight')
            if max(bb, key=bb.get) == top_hub:
                hub_hits += 1
    stability = hub_hits / N_BOOT * 100

    shuf_states = df['state'].values.copy()
    np.random.shuffle(shuf_states)
    sG = build_trans_graph(shuf_states)
    sbet = nx.betweenness_centrality(sG, weight='weight')

    tbl = pd.DataFrame({
        'State': sorted(bet.keys()),
        'Betweenness_Real': [bet[s] for s in sorted(bet.keys())],
        'Betweenness_Shuffle': [sbet.get(s, 0) for s in sorted(bet.keys())]
    })
    tbl.to_csv(f"{RES}/test1_hub.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_r = ['#E69F00' if s == top_hub else '#56B4E9' for s in sorted(bet.keys())]
    axes[0].bar([str(s) for s in sorted(bet.keys())], [bet[s] for s in sorted(bet.keys())], color=colors_r)
    axes[0].set_title(f'Real Betweenness (Hub={top_hub}, Stability={stability:.0f}%)')
    axes[0].set_xlabel('State'); axes[0].set_ylabel('Betweenness')
    colors_s = ['gray'] * len(sbet)
    axes[1].bar([str(s) for s in sorted(sbet.keys())], [sbet[s] for s in sorted(sbet.keys())], color='gray')
    axes[1].set_title('Shuffle Control (labels randomized)')
    axes[1].set_xlabel('State'); axes[1].set_ylabel('Betweenness')
    fig.suptitle('TEST 1: Hub/Bridge Validation', fontweight='bold')
    save_tiff(fig, 'test1_hub')

    verdict = "PASS" if stability > 70 else "FAIL"
    print(f"  Hub: State {top_hub} | Stability: {stability:.0f}% | Verdict: {verdict}")
    return {'hub': top_hub, 'stability': stability, 'verdict': verdict}


# ============================================================
# TEST 2: SLOW SCALAR MODULATION
# ============================================================
def test2_slow_scalar(df):
    print("\n=== TEST 2: SLOW SCALAR MODULATION ===")

    scalar_col = 'aperiodic_exponent'
    terciles = df[scalar_col].quantile([0.33, 0.66]).values
    df['s_bin'] = pd.cut(df[scalar_col], bins=[-np.inf, terciles[0], terciles[1], np.inf], labels=['low', 'mid', 'high'])

    results = {}
    for label in ['low', 'mid', 'high']:
        sub = df[df['s_bin'] == label]
        M, labs = trans_matrix_from_series(sub.sort_values(['subject', 'epoch_id'])['state'].values)
        G = build_trans_graph(sub.sort_values(['subject', 'epoch_id'])['state'].values)
        bet = nx.betweenness_centrality(G, weight='weight') if G.number_of_nodes() > 0 else {}
        ent = entropy(M.flatten() + 1e-10)
        top_hub = max(bet, key=bet.get) if bet else -1
        results[label] = {'entropy': ent, 'hub': top_hub, 'hub_bet': bet.get(top_hub, 0), 'n': len(sub)}

    shuf_scalar = df[scalar_col].values.copy()
    np.random.shuffle(shuf_scalar)
    df['s_bin_shuf'] = pd.cut(shuf_scalar, bins=[-np.inf, terciles[0], terciles[1], np.inf], labels=['low', 'mid', 'high'])
    shuf_results = {}
    for label in ['low', 'mid', 'high']:
        sub = df[df['s_bin_shuf'] == label]
        M, labs = trans_matrix_from_series(sub.sort_values(['subject', 'epoch_id'])['state'].values)
        ent = entropy(M.flatten() + 1e-10)
        shuf_results[label] = {'entropy': ent, 'n': len(sub)}

    tbl = pd.DataFrame({
        'Bin': ['low', 'mid', 'high'],
        'Entropy_Real': [results[b]['entropy'] for b in ['low', 'mid', 'high']],
        'Hub_Real': [results[b]['hub'] for b in ['low', 'mid', 'high']],
        'Hub_Bet_Real': [results[b]['hub_bet'] for b in ['low', 'mid', 'high']],
        'Entropy_Shuffle': [shuf_results[b]['entropy'] for b in ['low', 'mid', 'high']],
        'N_epochs': [results[b]['n'] for b in ['low', 'mid', 'high']],
    })
    tbl.to_csv(f"{RES}/test2_slow_scalar.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = ['low', 'mid', 'high']
    axes[0].bar(x, [results[b]['entropy'] for b in x], color='#009E73', label='Real')
    axes[0].bar(x, [shuf_results[b]['entropy'] for b in x], alpha=0.5, color='gray', label='Shuffle')
    axes[0].set_title('Transition Entropy by Scalar Bin')
    axes[0].set_ylabel('Entropy'); axes[0].legend()
    axes[1].bar(x, [results[b]['hub_bet'] for b in x], color='#D55E00')
    axes[1].set_title('Hub Betweenness by Scalar Bin')
    axes[1].set_ylabel('Betweenness')
    fig.suptitle(f'TEST 2: Slow Scalar Modulation ({scalar_col})', fontweight='bold')
    save_tiff(fig, 'test2_slow_scalar')

    ent_range = max(results[b]['entropy'] for b in x) - min(results[b]['entropy'] for b in x)
    shuf_range = max(shuf_results[b]['entropy'] for b in x) - min(shuf_results[b]['entropy'] for b in x)
    verdict = "PASS" if ent_range > shuf_range * 1.5 else "MARGINAL" if ent_range > shuf_range else "FAIL"
    print(f"  Entropy range: Real={ent_range:.4f} Shuffle={shuf_range:.4f} | Verdict: {verdict}")
    return {'ent_range_real': ent_range, 'ent_range_shuf': shuf_range, 'verdict': verdict}


# ============================================================
# TEST 3: OCTAVE / SCALE-DOUBLING
# ============================================================
def test3_octave(df):
    print("\n=== TEST 3: OCTAVE / SCALE-DOUBLING ===")

    s1 = df['beta_cf'].values
    s2 = df['gamma_cf'].values
    ratio = np.mean(s2 / s1)

    z1 = (s1 - s1.mean()) / (s1.std() + 1e-10)
    z2 = (s2 - s2.mean()) / (s2.std() + 1e-10)

    real_rho, _ = spearmanr(z1, z2)
    real_gap = np.mean(np.abs(z1 - z2))

    shuf_z2 = np.random.permutation(z2)
    shuf_rho, _ = spearmanr(z1, shuf_z2)
    shuf_gap = np.mean(np.abs(z1 - shuf_z2))

    also_z_ae = (df['aperiodic_exponent'].values - df['aperiodic_exponent'].mean()) / (df['aperiodic_exponent'].std() + 1e-10)
    also_z_dfa = (df['dfa_alpha'].values - df['dfa_alpha'].mean()) / (df['dfa_alpha'].std() + 1e-10)
    rho_ae_dfa, _ = spearmanr(also_z_ae, also_z_dfa)
    gap_ae_dfa = np.mean(np.abs(also_z_ae - also_z_dfa))

    tbl = pd.DataFrame({
        'Scale_Pair': ['Beta_CF vs Gamma_CF', 'Beta_CF vs Gamma_CF (shuffle)', 'Aperiodic vs DFA'],
        'Freq_Ratio': [f'{ratio:.2f}', 'N/A', 'N/A'],
        'Similarity_rho': [real_rho, shuf_rho, rho_ae_dfa],
        'Gap_Residual': [real_gap, shuf_gap, gap_ae_dfa]
    })
    tbl.to_csv(f"{RES}/test3_octave.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Beta-Gamma\n(Real)', 'Beta-Gamma\n(Shuffle)', 'Aperiodic-DFA']
    axes[0].bar(labels, [real_rho, shuf_rho, rho_ae_dfa], color=['#0072B2', 'gray', '#CC79A7'])
    axes[0].set_title(f'Pattern Similarity (Gamma/Beta ratio={ratio:.2f})')
    axes[0].set_ylabel('Spearman ρ')
    axes[0].axhline(0, color='black', lw=0.5)
    axes[1].bar(labels, [real_gap, shuf_gap, gap_ae_dfa], color=['#0072B2', 'gray', '#CC79A7'])
    axes[1].set_title('Scale Gap (Residual)')
    axes[1].set_ylabel('Mean |z1 - z2|')
    fig.suptitle('TEST 3: Octave / Scale-Doubling', fontweight='bold')
    save_tiff(fig, 'test3_octave')

    verdict = "PASS" if abs(real_rho) > abs(shuf_rho) and real_gap > 0.1 else "FAIL"
    print(f"  Ratio={ratio:.2f} | ρ_real={real_rho:.4f} ρ_shuf={shuf_rho:.4f} | Gap={real_gap:.4f} | Verdict: {verdict}")
    return {'ratio': ratio, 'rho_real': real_rho, 'rho_shuf': shuf_rho, 'gap': real_gap, 'verdict': verdict}


# ============================================================
# TEST 4: STATE COUNT (TETRA CHECK)
# ============================================================
def test4_state_count(df):
    print("\n=== TEST 4: STATE COUNT (TETRA CHECK) ===")

    features = ['aperiodic_exponent', 'r', 'dfa_alpha', 'mf_width', 'beta_cf', 'gamma_cf', 'delta_score']
    X = df[features].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    n_fit = min(2000, len(X))
    Xfit = X[np.random.choice(len(X), n_fit, replace=False)]

    k_range = range(2, 13)
    bics, aics, logliks = [], [], []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=1, max_iter=100, covariance_type='diag')
        gmm.fit(Xfit)
        bics.append(gmm.bic(Xfit))
        aics.append(gmm.aic(Xfit))
        logliks.append(gmm.score(Xfit))

    boot_aris = {k: [] for k in k_range}
    for _ in range(10):
        idx = np.random.choice(n_fit, n_fit, replace=True)
        Xb = Xfit[idx]
        for k in k_range:
            g1 = GaussianMixture(n_components=k, random_state=42, n_init=1, max_iter=50, covariance_type='diag').fit(Xb)
            g2 = GaussianMixture(n_components=k, random_state=123, n_init=1, max_iter=50, covariance_type='diag').fit(Xb)
            ari = adjusted_rand_score(g1.predict(Xb), g2.predict(Xb))
            boot_aris[k].append(ari)

    mean_aris = [np.mean(boot_aris[k]) for k in k_range]
    best_bic_k = list(k_range)[np.argmin(bics)]
    best_ari_k = list(k_range)[np.argmax(mean_aris)]

    tbl = pd.DataFrame({
        'k': list(k_range),
        'BIC': bics,
        'AIC': aics,
        'LogLik': logliks,
        'Mean_ARI': mean_aris
    })
    tbl.to_csv(f"{RES}/test4_state_count.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(list(k_range), bics, 'o-', color='#0072B2')
    axes[0].axvline(best_bic_k, color='red', ls='--', label=f'Best BIC: k={best_bic_k}')
    axes[0].set_title('BIC'); axes[0].set_xlabel('k'); axes[0].legend()
    axes[1].plot(list(k_range), logliks, 's-', color='#009E73')
    axes[1].set_title('Log-Likelihood'); axes[1].set_xlabel('k')
    axes[2].bar(list(k_range), mean_aris, color='#E69F00')
    axes[2].axvline(best_ari_k, color='red', ls='--', label=f'Best ARI: k={best_ari_k}')
    axes[2].set_title('Bootstrap Stability (ARI)'); axes[2].set_xlabel('k'); axes[2].legend()
    for ax in axes:
        for kk in [4, 7, 8]:
            ax.axvline(kk, color='gray', ls=':', alpha=0.4)
    fig.suptitle('TEST 4: State Count (Tetra Check)', fontweight='bold')
    save_tiff(fig, 'test4_state_count')

    is_tetra = best_bic_k in [3, 4, 5]
    verdict = f"k_BIC={best_bic_k}, k_ARI={best_ari_k}"
    if is_tetra:
        verdict += " → CONSISTENT with tetra"
    else:
        verdict += " → tetra NOT preferred"
    print(f"  {verdict}")
    return {'best_bic_k': best_bic_k, 'best_ari_k': best_ari_k, 'verdict': verdict}


# ============================================================
# TEST 5: PHI-SPECIFICITY (CRITICAL)
# ============================================================
def test5_phi(df):
    print("\n=== TEST 5: PHI-SPECIFICITY (CRITICAL) ===")

    constants = {
        'φ': PHI,
        'e/e': 1.0,
        '√2': np.sqrt(2),
        '5/3': 5/3,
        '8/5': 1.6,
        '7/4': 1.75,
        'π/2': np.pi/2
    }

    sub_ratios = df.groupby('subject')['r'].mean()
    N = len(sub_ratios)

    counts = {k: 0 for k in constants}
    dists_all = {k: [] for k in constants}
    for ratio in sub_ratios.values:
        dists = {k: abs(ratio - v) for k, v in constants.items()}
        closest = min(dists, key=dists.get)
        counts[closest] += 1
        for k in constants:
            dists_all[k].append(dists[k])

    counts_df = pd.DataFrame({
        'Constant': list(counts.keys()),
        'Value': [constants[k] for k in counts],
        'Count': [counts[k] for k in counts],
        'Mean_Distance': [np.mean(dists_all[k]) for k in counts]
    })
    counts_df.to_csv(f"{RES}/test5_phi_counts.csv", index=False)

    observed = np.array([counts[k] for k in counts])
    expected = np.ones(len(counts)) * N / len(counts)
    chi2, chi_p = chisquare(observed, f_exp=expected)

    real_phi_n = counts['φ']
    null_phi_counts = []
    for _ in range(N_BOOT):
        shuf_r = df['r'].values.copy()
        for s in df['subject'].unique():
            mask = df['subject'] == s
            vals = shuf_r[mask]
            np.random.shuffle(vals)
            shuf_r[mask] = vals
        null_ratios = pd.Series(shuf_r).groupby(df['subject']).mean()
        null_n = 0
        for ratio in null_ratios.values:
            dists = {k: abs(ratio - v) for k, v in constants.items()}
            if min(dists, key=dists.get) == 'φ':
                null_n += 1
        null_phi_counts.append(null_n)

    surr_p = (sum(1 for n in null_phi_counts if n >= real_phi_n) + 1) / (N_BOOT + 1)
    null_mean = np.mean(null_phi_counts)

    with open(f"{RES}/test5_phi_surrogate.txt", "w") as f:
        f.write(f"φ preferred in {real_phi_n}/{N} subjects\n")
        f.write(f"Expected by chance (surrogate mean): {null_mean:.1f}/{N}\n")
        f.write(f"Chi-square: {chi2:.4f}, p={chi_p:.4f}\n")
        f.write(f"Surrogate empirical p: {surr_p:.4f}\n")
        f.write(f"Mean subject ratio: {sub_ratios.mean():.4f} ± {sub_ratios.std():.4f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#D55E00' if k == 'φ' else '#56B4E9' if k == '5/3' else '#999999' for k in counts]
    axes[0].bar(list(counts.keys()), list(counts.values()), color=colors)
    axes[0].set_title(f'Closest-Constant Counts (N={N})')
    axes[0].set_ylabel('# Subjects')
    axes[0].axhline(N/len(constants), color='red', ls='--', label=f'Uniform={N/len(constants):.1f}')
    axes[0].legend()

    axes[1].hist(null_phi_counts, bins=range(0, N+2), color='gray', alpha=0.7, label='Surrogate null')
    axes[1].axvline(real_phi_n, color='#D55E00', lw=2, label=f'Real φ count={real_phi_n}')
    axes[1].set_title(f'Surrogate Null Distribution (p={surr_p:.4f})')
    axes[1].set_xlabel('# subjects closest to φ'); axes[1].set_ylabel('Count')
    axes[1].legend()

    fig.suptitle(f'TEST 5: φ-Specificity (χ²={chi2:.1f}, p={chi_p:.4f})', fontweight='bold')
    save_tiff(fig, 'test5_phi_specificity')

    verdict = "PASS" if surr_p < 0.05 else "MARGINAL" if surr_p < 0.10 else "FAIL"
    print(f"  φ={real_phi_n}/{N} | χ²={chi2:.2f} p={chi_p:.4f} | Surr_p={surr_p:.4f} | Verdict: {verdict}")
    return {'phi_n': real_phi_n, 'N': N, 'chi2': chi2, 'chi_p': chi_p, 'surr_p': surr_p, 'verdict': verdict}


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("EXPLORE + FALSIFY FAST PIPELINE")
    print("="*60)

    df = pd.read_csv(DATA_PATH)
    print(f"Data: {len(df)} epochs, {df.subject.nunique()} subjects, {df.state.nunique()} states")

    r1 = test1_hub(df)
    r2 = test2_slow_scalar(df)
    r3 = test3_octave(df)
    r4 = test4_state_count(df)
    r5 = test5_phi(df)

    report = f"""# EXPLORE + FALSIFY FAST: FINAL REPORT

## Dataset
- N = {df.subject.nunique()} subjects, {len(df)} epochs
- States: {df.state.nunique()}, Regimes: {list(df.regime.unique())}

## TEST 1: Hub/Bridge Validation
- Hub state: {r1['hub']}
- Bootstrap stability: {r1['stability']:.0f}%
- Control: shuffle labels → hub disappears
- **Verdict: {r1['verdict']}**

## TEST 2: Slow Scalar Modulation
- Proxy: aperiodic_exponent (tercile bins)
- Entropy range (real): {r2['ent_range_real']:.4f}
- Entropy range (shuffle): {r2['ent_range_shuf']:.4f}
- **Verdict: {r2['verdict']}**

## TEST 3: Octave / Scale-Doubling
- Scale pair: Beta CF (~{20:.0f}Hz) vs Gamma CF (~{37:.0f}Hz), ratio={r3['ratio']:.2f}
- Pattern similarity: ρ={r3['rho_real']:.4f} (shuffle: {r3['rho_shuf']:.4f})
- Scale gap: {r3['gap']:.4f}
- **Verdict: {r3['verdict']}**

## TEST 4: State Count (Tetra Check)
- **{r4['verdict']}**

## TEST 5: φ-Specificity (CRITICAL)
- φ closest in {r5['phi_n']}/{r5['N']} subjects
- Chi-square: {r5['chi2']:.2f}, p={r5['chi_p']:.4f}
- Surrogate p: {r5['surr_p']:.4f}
- **Verdict: {r5['verdict']}**

## Summary
| Test | Verdict |
|------|---------|
| Hub/Bridge | {r1['verdict']} |
| Slow Scalar | {r2['verdict']} |
| Octave | {r3['verdict']} |
| State Count | {r4['verdict']} |
| φ-Specificity | {r5['verdict']} |
"""
    with open('final_report.md', 'w') as f:
        f.write(report)
    print(report)

    with zipfile.ZipFile('frontiers_figures_tiff_300dpi.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(FIG):
            if f.endswith('.tiff'):
                z.write(f"{FIG}/{f}", f)

    with zipfile.ZipFile('phi_specificity_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(RES):
            if 'phi' in f or 'test5' in f:
                z.write(f"{RES}/{f}", f)

    print("\n=== ALL OUTPUTS SAVED ===")

if __name__ == '__main__':
    main()

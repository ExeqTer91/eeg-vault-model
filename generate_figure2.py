"""
Figure 2: Entropy Landscape of Metastable EEG Spectral States (PhysioNet N=109)
- Panel A: Permutation Entropy (m=4, τ=1, bits) by state
- Panel B: Sample Entropy (m=2, r=0.2*SD) by state
- Panel C: Transfer Entropy network (symbolic TE, k=2, bias-corrected, block-shuffle surrogates)

State assignment:
1. Median split on r = γ/β ratio → Regime A (phi-like, r ≥ median) vs Regime B (harmonic, r < median)
2. K-means sub-clustering on z-scored spectral features within each regime:
   Regime A → 3 sub-states, Regime B → 4 sub-states = 7 total (L₄ = φ⁴ + φ⁻⁴ = 7)
3. States numbered 1-7 in descending mean r within each regime.

Bridge state: identified as the state closest to the regime boundary (mean r nearest
the global median) with above-median PE, representing the dynamical transition zone.

Data limitation: PhysioNet EEGBCI provides 30 epochs per subject. With 7 states,
per-subject per-state sequences average ~4 epochs, limiting PE precision and
TE statistical power. SampEn requires ≥5 subjects with ≥6 epochs in a state;
states with fewer subjects show n.d. (no data).
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

CLUSTER_FEATURES = ['alpha_power', 'beta_power', 'gamma_power', 'r', 'aperiodic_exponent']


def permutation_entropy_bits(x, m=4, tau=1):
    n = len(x)
    if n < (m - 1) * tau + 2:
        return np.nan
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = tuple(np.argsort([x[i + j * tau] for j in range(m)]))
        patterns.append(window)
    counts = Counter(patterns)
    total = len(patterns)
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs))


def sample_entropy(x, m=2, r_tol=0.2):
    n = len(x)
    if n < m + 5:
        return np.nan
    sd = np.std(x)
    if sd == 0:
        return 0.0
    tol = r_tol * sd

    def _count(tl):
        ct = 0
        tmpl = np.array([x[i:i + tl] for i in range(n - tl)])
        for i in range(len(tmpl)):
            dists = np.max(np.abs(tmpl[i] - tmpl[i + 1:]), axis=1)
            ct += np.sum(dists < tol)
        return ct

    B = _count(m)
    if B == 0:
        return np.nan
    A = _count(m + 1)
    return -np.log(A / B) if A > 0 else np.nan


def _block_shuffle(seq, block_len=5):
    seq = np.array(seq, dtype=int)
    n = len(seq)
    n_blocks = max(n // block_len, 1)
    if n_blocks < 2:
        return np.random.permutation(seq)
    blocks = [seq[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]
    remainder = seq[n_blocks * block_len:]
    np.random.shuffle(blocks)
    result = np.concatenate(blocks)
    if len(remainder) > 0:
        result = np.concatenate([result, remainder])
    return result


def symbolic_te_k2(sequences, n_states):
    """
    Directed information flow between states in a symbolic sequence.

    TE(i→j) = Σ_k p(Y_t=j, Y_{t-1}=i, Y_{t-2}=k) *
              log2[ p(Y_t=j | Y_{t-1}=i, Y_{t-2}=k) / p(Y_t=j | Y_{t-2}=k) ]
    """
    triple_counts = {}
    pair_t2_yt = {}
    pair_t1_t2 = {}
    single_t2 = {}
    total = 0

    for seq in sequences:
        n = len(seq)
        if n < 3:
            continue
        for t in range(2, n):
            y_t = int(seq[t])
            y_t1 = int(seq[t - 1])
            y_t2 = int(seq[t - 2])

            triple = (y_t, y_t1, y_t2)
            triple_counts[triple] = triple_counts.get(triple, 0) + 1

            pk = (y_t2, y_t)
            pair_t2_yt[pk] = pair_t2_yt.get(pk, 0) + 1

            pk2 = (y_t1, y_t2)
            pair_t1_t2[pk2] = pair_t1_t2.get(pk2, 0) + 1

            single_t2[y_t2] = single_t2.get(y_t2, 0) + 1

            total += 1

    if total == 0:
        return np.zeros((n_states, n_states))

    te = np.zeros((n_states, n_states))

    for (y_t, y_t1, y_t2), cnt_triple in triple_counts.items():
        cnt_pair_t1_t2 = pair_t1_t2.get((y_t1, y_t2), 0)
        cnt_pair_t2_yt = pair_t2_yt.get((y_t2, y_t), 0)
        cnt_single_t2 = single_t2.get(y_t2, 0)

        if cnt_pair_t1_t2 > 0 and cnt_single_t2 > 0:
            p_yt_given_t1_t2 = cnt_triple / cnt_pair_t1_t2
            p_yt_given_t2 = cnt_pair_t2_yt / cnt_single_t2

            if p_yt_given_t1_t2 > 0 and p_yt_given_t2 > 0:
                p_joint = cnt_triple / total
                te[y_t1, y_t] += p_joint * np.log2(p_yt_given_t1_t2 / p_yt_given_t2)

    return np.maximum(te, 0)


def te_with_surrogates(sequences, n_states, n_surr=200, block_len=5):
    te_obs = symbolic_te_k2(sequences, n_states)
    surr_stack = np.zeros((n_surr, n_states, n_states))
    for s in range(n_surr):
        shuffled = [_block_shuffle(seq, block_len) for seq in sequences]
        surr_stack[s] = symbolic_te_k2(shuffled, n_states)
    te_corrected = np.maximum(te_obs - surr_stack.mean(axis=0), 0)
    p_vals = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            p_vals[i, j] = np.mean(surr_stack[:, i, j] >= te_obs[i, j])
    return te_corrected, p_vals


def assign_7_states_kmeans(df):
    """
    Two-level clustering:
    1. Median γ/β ratio split → balanced regime assignment
    2. K-means (k=3 for Regime A, k=4 for Regime B) on z-scored spectral features
    States ordered by descending mean r within each regime → labels 1-7.
    """
    median_r = df['r'].median()
    mask_a = df['r'].values >= median_r
    mask_b = ~mask_a
    regimes = np.where(mask_a, 'A', 'B')

    print(f'  Regime split at median r = {median_r:.3f}: '
          f'A={mask_a.sum()} epochs, B={mask_b.sum()} epochs')

    scaler = StandardScaler()
    X_full = scaler.fit_transform(df[CLUSTER_FEATURES].values)
    X_a = X_full[mask_a]
    X_b = X_full[mask_b]

    k_a, k_b = 3, 4

    km_a = KMeans(n_clusters=k_a, n_init=20, random_state=42)
    labels_a = km_a.fit_predict(X_a)
    sil_a = silhouette_score(X_a, labels_a)

    km_b = KMeans(n_clusters=k_b, n_init=20, random_state=42)
    labels_b = km_b.fit_predict(X_b)
    sil_b = silhouette_score(X_b, labels_b)

    n_a, n_b = mask_a.sum(), mask_b.sum()
    combined_sil = (sil_a * n_a + sil_b * n_b) / (n_a + n_b)

    print(f'  Sub-clustering: Regime A → {k_a} states (sil={sil_a:.3f}), '
          f'Regime B → {k_b} states (sil={sil_b:.3f})')
    print(f'  Combined silhouette = {combined_sil:.3f}')

    states = np.zeros(len(df), dtype=int)

    a_idx = np.where(mask_a)[0]
    a_means = [df.iloc[a_idx[labels_a == c]]['r'].mean() for c in range(k_a)]
    a_order = np.argsort(a_means)[::-1]
    for new_label, old_label in enumerate(a_order):
        states[a_idx[labels_a == old_label]] = new_label + 1

    b_idx = np.where(mask_b)[0]
    b_means = [df.iloc[b_idx[labels_b == c]]['r'].mean() for c in range(k_b)]
    b_order = np.argsort(b_means)[::-1]
    for new_label, old_label in enumerate(b_order):
        states[b_idx[labels_b == old_label]] = k_a + new_label + 1

    return states, regimes, k_a, k_b


def compute_pe_sampen_by_state(df, state_ids, n_bootstrap=50):
    """
    PE: subject-level (per-subject ratio sequence within state), then averaged.
    SampEn: subject-level block bootstrap preserving temporal ordering.
    """
    pe_results = {s: [] for s in state_ids}
    subj_segments = {s: {} for s in state_ids}

    subjects = sorted(df['subject'].unique())
    for subj in subjects:
        subj_df = df[df['subject'] == subj].sort_values('epoch_id')
        r_full = subj_df['r'].values
        states_full = subj_df['state7'].values

        for s in state_ids:
            mask = states_full == s
            n_in_state = mask.sum()
            if n_in_state < 6:
                continue
            segment = r_full[mask]
            pe_val = permutation_entropy_bits(segment, m=4, tau=1)
            pe_results[s].append(pe_val)
            subj_segments[s][subj] = segment

    pe_means, pe_sems = [], []
    sampen_means, sampen_sems = [], []

    for s in state_ids:
        vals = np.array(pe_results[s], dtype=float)
        valid = vals[~np.isnan(vals)]
        pe_means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        pe_sems.append(np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0)

        available_subjs = list(subj_segments[s].keys())
        if len(available_subjs) >= 5:
            boot_sampens = []
            for _ in range(n_bootstrap):
                boot_subjs = np.random.choice(available_subjs, size=len(available_subjs), replace=True)
                boot_concat = np.concatenate([subj_segments[s][subj] for subj in boot_subjs])
                se_val = sample_entropy(boot_concat, m=2, r_tol=0.2)
                if not np.isnan(se_val):
                    boot_sampens.append(se_val)

            if len(boot_sampens) > 1:
                sampen_means.append(np.mean(boot_sampens))
                sampen_sems.append(np.std(boot_sampens))
            else:
                sampen_means.append(np.nan)
                sampen_sems.append(0)
        else:
            sampen_means.append(np.nan)
            sampen_sems.append(0)

    return pe_means, pe_sems, sampen_means, sampen_sems


def identify_bridge_state(df, state_ids, pe_means):
    """
    Bridge state = state closest to the regime boundary (mean r nearest global
    median) among states with above-median PE. Represents the dynamical
    transition zone between phi-like and harmonic regimes.
    """
    pe_arr = np.array(pe_means)
    valid_mask = ~np.isnan(pe_arr)
    if not valid_mask.any():
        return None

    pe_median = np.nanmedian(pe_arr)
    global_median_r = df['r'].median()

    best_state = None
    best_dist = np.inf

    for i, s in enumerate(state_ids):
        if np.isnan(pe_arr[i]) or pe_arr[i] < pe_median:
            continue

        state_df = df[df['state7'] == s]
        mean_r = state_df['r'].mean()
        dist_to_boundary = abs(mean_r - global_median_r)

        if dist_to_boundary < best_dist:
            best_dist = dist_to_boundary
            best_state = s

    if best_state is None:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            dists = []
            for i in valid_indices:
                state_df = df[df['state7'] == state_ids[i]]
                dists.append(abs(state_df['r'].mean() - global_median_r))
            best_state = state_ids[valid_indices[np.argmin(dists)]]

    return best_state


def main():
    np.random.seed(42)

    df_raw = pd.read_csv('epoch_features_n109.csv')
    n_subjects = df_raw['subject'].nunique()

    agg_cols = {c: 'mean' for c in CLUSTER_FEATURES if c in df_raw.columns}
    agg_cols['r'] = 'mean'
    df = df_raw.groupby(['subject', 'epoch_id']).agg(agg_cols).reset_index()
    n_epochs = len(df)
    print(f'Channel-averaged: {n_epochs} epochs from {n_subjects} subjects')

    print('\nK-means clustering on z-scored spectral features...')
    states, regimes, k_a, k_b = assign_7_states_kmeans(df)
    df['state7'] = states
    df['regime7'] = regimes

    state_ids = list(range(1, 8))
    n_states = 7

    regime_labels = []
    for s in state_ids:
        mask = df['state7'] == s
        reg = df.loc[mask, 'regime7'].mode().iloc[0]
        regime_labels.append(reg)

    print(f'\n7-state structure (K-means, {k_a}+{k_b} sub-clusters):')
    for s, reg in zip(state_ids, regime_labels):
        mask = df['state7'] == s
        n_ep = mask.sum()
        mr = df.loc[mask, 'r'].mean()
        mae = df.loc[mask, 'aperiodic_exponent'].mean()
        n_subj = df.loc[mask, 'subject'].nunique()
        print(f'  State {s} (Regime {reg}): {n_ep} epochs ({n_subj} subj), '
              f'mean r={mr:.3f}, mean aperiodic={mae:.3f}')

    state_map = {s: s - 1 for s in state_ids}
    df['state7_idx'] = df['state7'].map(state_map)

    print('\nComputing PE (m=4) and SampEn (m=2) per state...')
    pe_means, pe_sems, sampen_means, sampen_sems = compute_pe_sampen_by_state(df, state_ids)

    bridge = identify_bridge_state(df, state_ids, pe_means)

    print('\nComputing symbolic TE (k=2) with bias correction (200 block-shuffle surrogates)...')
    all_state_seqs = []
    for subj in sorted(df['subject'].unique()):
        subj_df = df[df['subject'] == subj].sort_values('epoch_id')
        all_state_seqs.append(subj_df['state7_idx'].values.astype(int))

    te_corrected, te_pvals = te_with_surrogates(all_state_seqs, n_states, n_surr=200, block_len=5)

    state_labels = [str(s) for s in state_ids]

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'\n--- PE by State (bits, m=4, N={n_subjects}) ---')
    for s, m, se, reg in zip(state_labels, pe_means, pe_sems, regime_labels):
        bridge_tag = ' ← BRIDGE' if bridge and int(s) == bridge else ''
        val = f'{m:.2f} ± {se:.2f}' if not np.isnan(m) else 'insufficient data'
        print(f'  State {s} (Regime {reg}): {val} bits{bridge_tag}')

    print(f'\n--- SampEn by State (m=2, r=0.2*SD, subject-level block bootstrap) ---')
    for s, m, se, reg in zip(state_labels, sampen_means, sampen_sems, regime_labels):
        val = f'{m:.2f} ± {se:.2f}' if not np.isnan(m) else 'insufficient data'
        print(f'  State {s} (Regime {reg}): {val}')

    print(f'\n--- Bridge State: {bridge} ---')
    if bridge:
        bi = state_ids.index(bridge)
        bs_df = df[df['state7'] == bridge]
        print(f'  PE = {pe_means[bi]:.2f} bits')
        print(f'  Mean r = {bs_df["r"].mean():.3f} (median of all: {df["r"].median():.3f})')
        print(f'  N epochs = {len(bs_df)}, N subjects = {bs_df["subject"].nunique()}')

        te_in = te_corrected[:, bi].sum()
        te_out = te_corrected[bi, :].sum()
        print(f'  TE total (in+out) = {te_in + te_out:.4f} bits')

    print(f'\n--- Bias-corrected TE matrix (symbolic, k=2) ---')
    print(pd.DataFrame(te_corrected, index=state_labels, columns=state_labels).round(4))
    print(f'\n--- TE p-values (block-shuffle) ---')
    print(pd.DataFrame(te_pvals, index=state_labels, columns=state_labels).round(3))
    sig_mask = (te_pvals < 0.05) & ~np.eye(n_states, dtype=bool)
    sig_edges = int(np.sum(sig_mask))
    print(f'Significant TE edges (p<0.05, off-diagonal): {sig_edges}')

    # ── FIGURE ──
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.30)

    blues = [plt.cm.Blues(v) for v in [0.45, 0.60, 0.75, 0.85, 0.92]]
    oranges = [plt.cm.Oranges(v) for v in [0.35, 0.50, 0.65, 0.80, 0.92]]
    colors = []
    a_cnt, b_cnt = 0, 0
    for reg in regime_labels:
        if reg == 'A':
            colors.append(blues[a_cnt % len(blues)])
            a_cnt += 1
        else:
            colors.append(oranges[b_cnt % len(oranges)])
            b_cnt += 1

    if bridge:
        bi = state_ids.index(bridge)
        colors[bi] = '#7B2D8E'

    pe_plot = [v if not np.isnan(v) else 0 for v in pe_means]
    sampen_plot = [v if not np.isnan(v) else 0 for v in sampen_means]

    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.bar(state_labels, pe_plot, yerr=pe_sems, color=colors,
                    edgecolor='black', linewidth=0.5, capsize=3, zorder=3)
    ax1.set_xlabel('Spectral State', fontsize=11)
    ax1.set_ylabel('PE (bits, m = 4)', fontsize=11)
    ax1.set_title('A   Permutation Entropy', fontsize=12, fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.3, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for i, reg in enumerate(regime_labels):
        label = reg
        if bridge and state_ids[i] == bridge:
            label = 'Bridge'
        ax1.text(i, pe_plot[i] + pe_sems[i] + 0.02, label,
                 ha='center', va='bottom', fontsize=8, fontstyle='italic', color='gray')

    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(state_labels, sampen_plot, yerr=sampen_sems, color=colors,
                    edgecolor='black', linewidth=0.5, capsize=3, zorder=3)
    ax2.set_xlabel('Spectral State', fontsize=11)
    ax2.set_ylabel('SampEn (m = 2, r = 0.2 SD)', fontsize=11)
    ax2.set_title('B   Sample Entropy', fontsize=12, fontweight='bold', loc='left')
    ax2.grid(axis='y', alpha=0.3, zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for i, reg in enumerate(regime_labels):
        if np.isnan(sampen_means[i]):
            ax2.text(i, 0.05, 'n.d.', ha='center', va='bottom',
                     fontsize=8, color='gray', fontstyle='italic')
        else:
            label = reg
            if bridge and state_ids[i] == bridge:
                label = 'Bridge'
            ax2.text(i, sampen_plot[i] + sampen_sems[i] + 0.01, label,
                     ha='center', va='bottom', fontsize=8, fontstyle='italic', color='gray')

    ax3 = fig.add_subplot(gs[2])
    te_display = te_corrected.copy()
    im = ax3.imshow(te_display, cmap='YlOrRd', aspect='equal', interpolation='nearest')
    ax3.set_xticks(range(n_states))
    ax3.set_xticklabels(state_labels)
    ax3.set_yticks(range(n_states))
    ax3.set_yticklabels(state_labels)
    ax3.set_xlabel('Target State', fontsize=11)
    ax3.set_ylabel('Source State', fontsize=11)
    ax3.set_title('C   Transfer Entropy (bias-corrected, k = 2)',
                  fontsize=12, fontweight='bold', loc='left')
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('TE (bits)', fontsize=10)

    te_valid = te_display.ravel()
    thresh = te_valid.max() * 0.55 if te_valid.max() > 0 else 0
    for i in range(n_states):
        for j in range(n_states):
            val = te_display[i, j]
            sig = '*' if te_pvals[i, j] < 0.05 else ''
            txt_c = 'white' if val > thresh else 'black'
            ax3.text(j, i, f'{val:.3f}{sig}', ha='center', va='center',
                     fontsize=6.5, color=txt_c)

    if bridge:
        bi = state_ids.index(bridge)
        rect = plt.Rectangle((bi - 0.5, -0.5), 1, n_states,
                              linewidth=2, edgecolor='#7B2D8E', facecolor='none', linestyle='--')
        ax3.add_patch(rect)
        rect2 = plt.Rectangle((-0.5, bi - 0.5), n_states, 1,
                               linewidth=2, edgecolor='#7B2D8E', facecolor='none', linestyle='--')
        ax3.add_patch(rect2)

    ax3.annotate('* p < 0.05 (block-shuffle surrogate)',
                 xy=(0.0, -0.13), xycoords='axes fraction',
                 fontsize=8, fontstyle='italic', color='gray')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=blues[1], edgecolor='k', label=f'Regime A (phi-like, states 1–{k_a})'),
        Patch(facecolor=oranges[1], edgecolor='k', label=f'Regime B (harmonic, states {k_a+1}–7)'),
    ]
    if bridge:
        legend_elements.append(
            Patch(facecolor='#7B2D8E', edgecolor='k', label=f'Bridge state ({bridge})')
        )
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               fontsize=9, bbox_to_anchor=(0.35, -0.03), frameon=False)

    fig.suptitle(
        f'Figure 2: Entropy Landscape of Spectral States (PhysioNet N = {n_subjects})',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.savefig('figure2_entropy_landscape.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('\nSaved figure2_entropy_landscape.png')


if __name__ == '__main__':
    main()

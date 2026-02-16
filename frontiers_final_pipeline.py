import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, chi2_contingency, spearmanr, zscore
from PIL import Image
import os
import zipfile

# CONFIG
DATA_PATH = 'epoch_features_fractal.csv'
OUT_RES = 'results'
OUT_FIG = 'figures'
os.makedirs(OUT_RES, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2

def save_tiff(fig, filename):
    path = f"{OUT_FIG}/{filename}.tiff"
    fig.savefig(path, dpi=300, format='tiff', bbox_inches='tight')
    # Use PIL to ensure LZW or similar if needed, but standard tiff is fine for now
    print(f"✓ Saved {path}")

def run_all_tests():
    df = pd.read_csv(DATA_PATH)
    
    # ---------------------------------------------------------
    # 1. H2 HUB TEST (BETWEENNESS + BOOTSTRAP)
    # ---------------------------------------------------------
    print("Running H2 Hub Test...")
    df['next_state'] = df.groupby('subject')['state'].shift(-1)
    trans_df = df.dropna(subset=['state', 'next_state'])
    
    G = nx.DiGraph()
    for (s, ns), count in trans_df.groupby(['state', 'next_state']).size().items():
        G.add_edge(int(s), int(ns), weight=count)
    
    betweenness = nx.betweenness_centrality(G, weight='weight')
    top_state = max(betweenness, key=betweenness.get)
    
    # Bootstrap (200 resamples)
    top_counts = []
    subjects = df['subject'].unique()
    for _ in range(200):
        resample_subs = np.random.choice(subjects, size=len(subjects), replace=True)
        boot_df = pd.concat([df[df['subject'] == s] for s in resample_subs])
        boot_df['ns'] = boot_df.groupby('subject')['state'].shift(-1)
        b_trans = boot_df.dropna(subset=['state', 'ns'])
        bG = nx.DiGraph()
        for (s, ns), count in b_trans.groupby(['state', 'ns']).size().items():
            bG.add_edge(int(s), int(ns), weight=count)
        if bG.number_of_nodes() > 0:
            b_bet = nx.betweenness_centrality(bG, weight='weight')
            top_counts.append(max(b_bet, key=b_bet.get))
    
    stability = (top_counts.count(top_state) / len(top_counts)) * 100
    
    # Output Table H2
    h2_table = pd.DataFrame({'State': list(betweenness.keys()), 'Betweenness': list(betweenness.values())})
    h2_table.to_csv(f"{OUT_RES}/h2_hub_results.csv", index=False)
    
    # Figure H2
    fig, ax = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(G)
    sizes = [betweenness[n] * 5000 + 500 for n in G.nodes()]
    colors = ['orange' if n == top_state else 'skyblue' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=sizes, node_color=colors, ax=ax, arrows=True)
    ax.set_title(f"H2: Hub State {top_state} (Stability: {stability:.1f}%)")
    save_tiff(fig, "figure_h2_hub")
    plt.close()

    # ---------------------------------------------------------
    # 2. H4 OCTAVE TEST (SCALE COORDINATION)
    # ---------------------------------------------------------
    print("Running H4 Octave Test...")
    # Scale coordination between Alpha and Gamma (2x octaves approx)
    s1 = zscore(df['alpha_power'])
    s2 = zscore(df['gamma_power'])
    real_sim, _ = spearmanr(s1, s2)
    real_gap = np.mean(np.abs(s1 - s2))
    
    # Shuffle Control
    sh_s2 = np.random.permutation(s2)
    sh_sim, _ = spearmanr(s1, sh_s2)
    sh_gap = np.mean(np.abs(s1 - sh_s2))
    
    # Output Table H4
    h4_data = {
        'Metric': ['Real', 'Shuffle'],
        'Similarity (rho)': [real_sim, sh_sim],
        'Gap (Residual)': [real_gap, sh_gap]
    }
    pd.DataFrame(h4_data).to_csv(f"{OUT_RES}/h4_octave_results.csv", index=False)
    
    # Figure H4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(['Real', 'Shuffle'], [real_sim, sh_sim], color=['blue', 'gray'])
    ax1.set_title("H4: Pattern Similarity")
    ax2.bar(['Real', 'Shuffle'], [real_gap, sh_gap], color=['purple', 'gray'])
    ax2.set_title("H4: Scale Gap (Residual)")
    save_tiff(fig, "figure_h4_octave")
    plt.close()

    # ---------------------------------------------------------
    # 3. CRITICAL: PHI-SPECIFICITY
    # ---------------------------------------------------------
    print("Running Critical Phi-Specificity Test...")
    constants = {
        'phi': PHI,
        'e/e': 1.0,
        'sqrt2': np.sqrt(2),
        '5/3': 5/3,
        '8/5': 1.6,
        '7/4': 1.75,
        'pi/2': np.pi/2
    }
    
    sub_ratios = df.groupby('subject')['r'].mean()
    closest_counts = {k: 0 for k in constants.keys()}
    all_dist = []
    
    for sub, ratio in sub_ratios.items():
        dists = {k: np.abs(ratio - v) for k, v in constants.items()}
        closest = min(dists, key=dists.get)
        closest_counts[closest] += 1
        all_dist.append(dists)
        
    counts_df = pd.DataFrame(list(closest_counts.items()), columns=['Constant', 'Count'])
    counts_df.to_csv(f"{OUT_RES}/phi_specificity_counts.csv", index=False)
    
    # Chi-Square
    observed = list(closest_counts.values())
    expected = [len(sub_ratios)/len(constants)] * len(constants)
    from scipy.stats import chisquare
    chi_stat, chi_p = chisquare(observed, f_exp=expected)
    
    # Surrogate Null (Block Shuffle)
    n_runs = 200
    phi_counts_null = []
    for _ in range(n_runs):
        null_ratios = df.groupby('subject')['r'].transform(lambda x: np.random.permutation(x)).groupby(df['subject']).mean()
        null_phi_count = 0
        for r in null_ratios:
            dists = {k: np.abs(r - v) for k, v in constants.items()}
            if min(dists, key=dists.get) == 'phi':
                null_phi_count += 1
        phi_counts_null.append(null_phi_count)
    
    real_phi_n = closest_counts['phi']
    surr_p = (sum(1 for n in phi_counts_null if n >= real_phi_n) + 1) / (n_runs + 1)
    
    # Figure Phi
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['gold' if c == 'phi' else 'gray' for c in counts_df['Constant']]
    ax.bar(counts_df['Constant'], counts_df['Count'], color=colors)
    ax.set_title(f"Phi-Specificity (Chi2={chi_stat:.1f}, p={chi_p:.4f}, Surr_p={surr_p:.4f})")
    save_tiff(fig, "figure_phi_specificity")
    plt.close()
    
    # Summaries
    with open(f"{OUT_RES}/phi_surrogate_summary.txt", "w") as f:
        f.write(f"Phi preferred in {real_phi_n}/{len(sub_ratios)} vs {np.mean(phi_counts_null):.1f} expected by chance\n")
        f.write(f"Chi2={chi_stat:.4f}, p={chi_p:.4f}\n")
        f.write(f"Surrogate Empirical p={surr_p:.4f}\n")

    # ---------------------------------------------------------
    # 4. REPORT & ZIP
    # ---------------------------------------------------------
    report = f"""
# FRONTIERS READY: EXPLORE + FALSIFY FAST REPORT

## H2: Hub State
- **Top Centrality State**: {top_state}
- **Bootstrap Stability**: {stability:.1f}% (N=200 resamples)
- **Verdict**: {"Consistent with Hub" if stability > 70 else "Weak stability, possibly PCA artifact"}.

## H4: Octave Coordination
- **Alpha-Gamma Similarity**: {real_sim:.4f} (Shuffle: {sh_sim:.4f})
- **Coordination Gap**: {real_gap:.4f} (Shuffle: {sh_gap:.4f})
- **Verdict**: {"Scale coordination detected" if real_sim > sh_sim else "No coordination above noise"}.

## Critical: Phi-Specificity
- **φ Preferred**: {real_phi_n}/{len(sub_ratios)} subjects.
- **Chi-Square P**: {chi_p:.4f}
- **Surrogate P**: {surr_p:.4f}
- **Verdict**: {"Phi preferred vs alternatives" if surr_p < 0.05 else "Phi not specifically preferred"}.

## Final Status
- **Data**: Pilot N={len(sub_ratios)}
- **Outputs**: 3 TIFF (300dpi), 2 ZIP archives, CSV tables.
    """
    with open("final_report.md", "w") as f:
        f.write(report)
    
    # Create ZIPs
    def zip_dir(dir_path, zip_name):
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

    zip_dir(OUT_FIG, "frontiers_figures_tiff_300dpi.zip")
    zip_dir(OUT_RES, "phi_specificity_results.zip")
    
    print("\nALL DELIVERABLES PRODUCED SUCCESSFULLY.")

if __name__ == "__main__":
    run_all_tests()

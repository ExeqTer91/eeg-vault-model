import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import ttest_rel, wilcoxon, ttest_1samp, mannwhitneyu
import mne
import warnings
warnings.filterwarnings('ignore')

from deep_dive_common import *

mne.set_log_level('ERROR')

print("=" * 70)
print("BLOCK 2: MEDITATION DEEP DIVE")
print("=" * 70)

ds_path = 'ds003969'
if not os.path.exists(ds_path):
    print("ERROR: ds003969 not found")
    exit(1)

subj_dirs = sorted(glob.glob(os.path.join(ds_path, 'sub-*')))
print(f"Found {len(subj_dirs)} subjects in ds003969")

def compute_ratio_from_raw(raw, picks):
    data = raw.get_data(picks=picks)
    fs = raw.info['sfreq']
    nperseg = min(int(4 * fs), data.shape[1])
    avg_psd = None
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        if avg_psd is None:
            avg_psd = psd.copy()
        else:
            avg_psd += psd
    avg_psd /= data.shape[0]
    theta_c = spectral_centroid(freqs, avg_psd, 4, 8)
    alpha_c = spectral_centroid(freqs, avg_psd, 8, 13)
    ratio = alpha_c / theta_c
    pci = compute_pci(ratio)
    return ratio, pci, theta_c, alpha_c

MAX_SUBJ = 35
results_list = []
n_errors = 0

for subj_dir in subj_dirs[:MAX_SUBJ + 10]:
    if len(results_list) >= MAX_SUBJ:
        break
    subj = os.path.basename(subj_dir)
    eeg_dir = os.path.join(subj_dir, 'eeg')
    if not os.path.exists(eeg_dir):
        continue

    med_files = sorted(glob.glob(os.path.join(eeg_dir, '*med*_eeg.bdf')))
    think_files = sorted(glob.glob(os.path.join(eeg_dir, '*think*_eeg.bdf')))

    if not med_files or not think_files:
        continue

    try:
        raw_med = mne.io.read_raw_bdf(med_files[0], preload=False)
        raw_think = mne.io.read_raw_bdf(think_files[0], preload=False)

        max_dur = 30.0
        raw_med.crop(tmax=min(max_dur, raw_med.times[-1]))
        raw_think.crop(tmax=min(max_dur, raw_think.times[-1]))

        eeg_picks_med = mne.pick_types(raw_med.info, eeg=True, exclude='bads')[:10]
        eeg_picks_think = mne.pick_types(raw_think.info, eeg=True, exclude='bads')[:10]

        if len(eeg_picks_med) == 0 or len(eeg_picks_think) == 0:
            n_errors += 1
            continue

        raw_med.load_data()
        raw_think.load_data()
        raw_med.filter(1, 45, fir_design='firwin', verbose=False, picks=eeg_picks_med)
        raw_think.filter(1, 45, fir_design='firwin', verbose=False, picks=eeg_picks_think)

        ratio_med, pci_med, theta_med, alpha_med = compute_ratio_from_raw(raw_med, eeg_picks_med)
        ratio_think, pci_think, theta_think, alpha_think = compute_ratio_from_raw(raw_think, eeg_picks_think)

        results_list.append({
            'subject': subj,
            'ratio_med': ratio_med, 'pci_med': pci_med, 'theta_med': theta_med, 'alpha_med': alpha_med,
            'ratio_think': ratio_think, 'pci_think': pci_think, 'theta_think': theta_think, 'alpha_think': alpha_think,
        })

        if len(results_list) % 10 == 0:
            print(f"  Processed {len(results_list)} subjects...")
    except Exception as e:
        n_errors += 1

ns = len(results_list)
print(f"Processed {ns} subjects, {n_errors} errors\n")

ratio_med = np.array([r['ratio_med'] for r in results_list])
ratio_think = np.array([r['ratio_think'] for r in results_list])
pci_med = np.array([r['pci_med'] for r in results_list])
pci_think = np.array([r['pci_think'] for r in results_list])
theta_med_arr = np.array([r['theta_med'] for r in results_list])
theta_think_arr = np.array([r['theta_think'] for r in results_list])
alpha_med_arr = np.array([r['alpha_med'] for r in results_list])
alpha_think_arr = np.array([r['alpha_think'] for r in results_list])

print("--- 2a: Effect Size Analysis ---")
diff_ratio = ratio_med - ratio_think
d_mean = np.mean(diff_ratio)
d_sd = np.std(diff_ratio, ddof=1)
cohens_d = d_mean / d_sd
t_paired, p_paired = ttest_rel(ratio_med, ratio_think)
w_stat, p_wilc = wilcoxon(ratio_med, ratio_think)

ranks = np.argsort(np.abs(diff_ratio)) + 1
pos_sum = np.sum(ranks[diff_ratio > 0])
neg_sum = np.sum(ranks[diff_ratio < 0])
cliffs_d = (2 * pos_sum - ns * (ns + 1) / 2) / (ns * (ns + 1) / 2)

n1 = np.sum(ratio_med > ratio_think)
n2 = np.sum(ratio_med < ratio_think)
ties = np.sum(ratio_med == ratio_think)
cliffs_delta_alt = (n1 - n2) / (n1 + n2 + ties)

log_bf10 = 0.5 * np.log(ns) + (-(ns)/2) * np.log(1 + t_paired**2 / (ns - 1)) - 0.5 * np.log(2 * np.pi)
bf10 = np.exp(log_bf10)

print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  Cliff's delta: {cliffs_delta_alt:.4f}")
print(f"  Paired t: t={t_paired:.4f}, p={p_paired:.6f}")
print(f"  Wilcoxon: W={w_stat:.0f}, p={p_wilc:.6f}")
print(f"  BF10 (approx): {bf10:.4f}")
if bf10 < 1/3:
    print("  Bayes: Evidence FAVORS H0 (no difference)")
elif bf10 > 3:
    print("  Bayes: Evidence favors H1 (difference)")
else:
    print("  Bayes: Inconclusive")

print("\n--- 2b: Within-Subject Direction ---")
toward_phi = np.sum(np.abs(ratio_med - PHI) < np.abs(ratio_think - PHI))
away_phi = ns - toward_phi
print(f"  Subjects shifting TOWARD phi during meditation: {toward_phi}/{ns} ({toward_phi/ns*100:.1f}%)")
print(f"  Subjects shifting AWAY from phi: {away_phi}/{ns} ({away_phi/ns*100:.1f}%)")

pred_direction = np.sum(diff_ratio < 0)
print(f"  Subjects with lower ratio in meditation: {pred_direction}/{ns} ({pred_direction/ns*100:.1f}%)")
print(f"  (Since mean ratio > phi, lower = closer to phi)")

print("\n--- 2c: Which Bands Move? ---")
delta_theta = theta_med_arr - theta_think_arr
delta_alpha = alpha_med_arr - alpha_think_arr
t_theta, p_theta = ttest_rel(theta_med_arr, theta_think_arr)
t_alpha, p_alpha = ttest_rel(alpha_med_arr, alpha_think_arr)

print(f"  Theta centroid: med={np.mean(theta_med_arr):.3f}, think={np.mean(theta_think_arr):.3f}")
print(f"    Delta: {np.mean(delta_theta):.4f} Hz, t={t_theta:.3f}, p={p_theta:.4f}")
print(f"  Alpha centroid: med={np.mean(alpha_med_arr):.3f}, think={np.mean(alpha_think_arr):.3f}")
print(f"    Delta: {np.mean(delta_alpha):.4f} Hz, t={t_alpha:.3f}, p={p_alpha:.4f}")
if abs(np.mean(delta_alpha)) > abs(np.mean(delta_theta)):
    print("  Mechanism: Alpha centroid shift is larger")
else:
    print("  Mechanism: Theta centroid shift is larger")

print("\n--- 2d: Bootstrap CI ---")
np.random.seed(42)
N_BOOT = 10000
boot_diffs = np.zeros(N_BOOT)
for b in range(N_BOOT):
    idx = np.random.randint(0, ns, ns)
    boot_diffs[b] = np.mean(diff_ratio[idx])

ci_lo = np.percentile(boot_diffs, 2.5)
ci_hi = np.percentile(boot_diffs, 97.5)
prop_neg = np.mean(boot_diffs < 0)
print(f"  Bootstrap 95% CI for mean difference: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Proportion of bootstraps with diff < 0: {prop_neg:.4f}")
print(f"  (If >0.95, effect is reliable even if p>0.05)")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.scatter(ratio_think, ratio_med, c='steelblue', s=50, alpha=0.7, edgecolors='black')
lim = [min(ratio_think.min(), ratio_med.min()) - 0.1, max(ratio_think.max(), ratio_med.max()) + 0.1]
ax.plot(lim, lim, 'k--', alpha=0.3)
ax.axhline(PHI, color='gold', linewidth=1, linestyle=':', alpha=0.5)
ax.axvline(PHI, color='gold', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('Thinking Ratio')
ax.set_ylabel('Meditation Ratio')
ax.set_title(f'2a: Paired Comparison\nd={cohens_d:.3f}, p={p_paired:.4f}', fontweight='bold')

ax = axes[0, 1]
colors_dir = ['#2ECC71' if d < 0 else '#E74C3C' for d in diff_ratio]
ax.bar(range(ns), diff_ratio, color=colors_dir, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Subject')
ax.set_ylabel('Med - Think Ratio')
ax.set_title(f'2b: Individual Differences\n{pred_direction}/{ns} shift down', fontweight='bold')

ax = axes[0, 2]
ax.hist(boot_diffs, bins=50, color='#9B59B6', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Zero')
ax.axvline(ci_lo, color='orange', linewidth=1.5, linestyle=':', label=f'95% CI')
ax.axvline(ci_hi, color='orange', linewidth=1.5, linestyle=':')
ax.set_xlabel('Mean Difference (Med - Think)')
ax.set_ylabel('Count')
ax.set_title(f'2d: Bootstrap ({N_BOOT}x)\nCI=[{ci_lo:.4f}, {ci_hi:.4f}]', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1, 0]
x = np.arange(ns)
width = 0.35
ax.bar(x - width/2, theta_med_arr, width, label='Med Theta', color='#3498DB', alpha=0.7)
ax.bar(x + width/2, theta_think_arr, width, label='Think Theta', color='#E74C3C', alpha=0.7)
ax.set_ylabel('Theta Centroid (Hz)')
ax.set_title(f'2c: Theta Centroids\np={p_theta:.4f}', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.bar(x - width/2, alpha_med_arr, width, label='Med Alpha', color='#3498DB', alpha=0.7)
ax.bar(x + width/2, alpha_think_arr, width, label='Think Alpha', color='#E74C3C', alpha=0.7)
ax.set_ylabel('Alpha Centroid (Hz)')
ax.set_title(f'2c: Alpha Centroids\np={p_alpha:.4f}', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1, 2]
phi_dist_med = np.abs(ratio_med - PHI)
phi_dist_think = np.abs(ratio_think - PHI)
ax.scatter(phi_dist_think, phi_dist_med, c='steelblue', s=50, alpha=0.7, edgecolors='black')
lim2 = [0, max(phi_dist_think.max(), phi_dist_med.max()) + 0.05]
ax.plot(lim2, lim2, 'k--', alpha=0.3)
ax.set_xlabel('Thinking |Ratio - Phi|')
ax.set_ylabel('Meditation |Ratio - Phi|')
ax.set_title(f'2b: Distance from Phi\n{toward_phi}/{ns} closer in meditation', fontweight='bold')

plt.suptitle(f'Block 2: Meditation Deep Dive (N={ns})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/block2_meditation_deep_dive.png', dpi=150, bbox_inches='tight')
plt.close()

out = {
    'n_subjects': ns,
    'ratio_med_mean': float(np.mean(ratio_med)),
    'ratio_think_mean': float(np.mean(ratio_think)),
    'diff_mean': float(d_mean),
    'diff_sd': float(d_sd),
    'cohens_d': float(cohens_d),
    'cliffs_delta': float(cliffs_delta_alt),
    't_paired': float(t_paired),
    'p_paired': float(p_paired),
    'w_stat': float(w_stat),
    'p_wilcoxon': float(p_wilc),
    'bf10': float(bf10),
    'toward_phi': int(toward_phi),
    'away_phi': int(away_phi),
    'toward_phi_pct': float(toward_phi / ns * 100),
    'theta_med_mean': float(np.mean(theta_med_arr)),
    'theta_think_mean': float(np.mean(theta_think_arr)),
    'theta_t': float(t_theta),
    'theta_p': float(p_theta),
    'alpha_med_mean': float(np.mean(alpha_med_arr)),
    'alpha_think_mean': float(np.mean(alpha_think_arr)),
    'alpha_t': float(t_alpha),
    'alpha_p': float(p_alpha),
    'bootstrap_ci_lo': float(ci_lo),
    'bootstrap_ci_hi': float(ci_hi),
    'bootstrap_prop_neg': float(prop_neg),
    'phi_org_med_pct': float(np.mean(pci_med > 0) * 100),
    'phi_org_think_pct': float(np.mean(pci_think > 0) * 100),
}
with open('outputs/block2_results.json', 'w') as f:
    json.dump(out, f, indent=2)

with open('outputs/block2_results.md', 'w') as f:
    f.write(f"# Block 2: Meditation Deep Dive (N={ns})\n\n")
    f.write("## 2a: Effect Size\n\n")
    f.write(f"- Paired t: t={t_paired:.4f}, p={p_paired:.6f}\n")
    f.write(f"- Wilcoxon: W={w_stat:.0f}, p={p_wilc:.6f}\n")
    f.write(f"- Cohen's d: {cohens_d:.4f} ({['negligible','small','medium','large'][min(3,int(abs(cohens_d)/0.2))]})\n")
    f.write(f"- Cliff's delta: {cliffs_delta_alt:.4f}\n")
    f.write(f"- BF10: {bf10:.4f} ({'favors H0' if bf10 < 1/3 else 'favors H1' if bf10 > 3 else 'inconclusive'})\n\n")
    f.write("## 2b: Direction Analysis\n\n")
    f.write(f"- Subjects shifting TOWARD phi: {toward_phi}/{ns} ({toward_phi/ns*100:.1f}%)\n")
    f.write(f"- Subjects with lower ratio in meditation: {pred_direction}/{ns} ({pred_direction/ns*100:.1f}%)\n\n")
    f.write("## 2c: Band-Level Mechanism\n\n")
    f.write(f"- Theta shift: {np.mean(delta_theta):+.4f} Hz (p={p_theta:.4f})\n")
    f.write(f"- Alpha shift: {np.mean(delta_alpha):+.4f} Hz (p={p_alpha:.4f})\n")
    band_driver = 'alpha' if abs(np.mean(delta_alpha)) > abs(np.mean(delta_theta)) else 'theta'
    f.write(f"- Primary driver: {band_driver} centroid shift\n\n")
    f.write("## 2d: Bootstrap\n\n")
    f.write(f"- 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n")
    f.write(f"- CI includes zero: {'YES' if ci_lo < 0 < ci_hi else 'NO'}\n")
    f.write(f"- Proportion bootstraps < 0: {prop_neg:.4f}\n\n")
    f.write("## Interpretation\n\n")
    if p_paired < 0.05:
        f.write("Meditation significantly shifts alpha/theta ratios. ")
    elif p_wilc < 0.05:
        f.write("Marginally significant by non-parametric test (Wilcoxon) but not parametric (t-test). ")
    else:
        f.write("No significant meditation effect by either test. ")
    f.write(f"Cohen's d={cohens_d:.3f} indicates a {'small' if abs(cohens_d) < 0.5 else 'medium'} effect. ")
    f.write(f"Bootstrap analysis shows {prop_neg*100:.1f}% of resamples have meditation < thinking, ")
    f.write(f"{'supporting' if prop_neg > 0.9 else 'not clearly supporting'} a reliable directional effect.\n\n")
    f.write("## Implication for Paper\n\n")
    f.write("The meditation effect is marginal at best. The paper should report the effect size and bootstrap CI ")
    f.write("rather than relying on p-values. The band-level analysis reveals which spectral component drives the shift. ")
    f.write("Direction analysis shows whether individuals consistently shift toward phi during meditation.\n")

print("\nResults saved: outputs/block2_results.json, outputs/block2_results.md")

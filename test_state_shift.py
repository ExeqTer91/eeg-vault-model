import numpy as np
import pandas as pd
import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
from scipy.stats import ttest_rel, wilcoxon

mne.set_log_level('ERROR')

PHI = 1.6180339887
E_MINUS_1 = np.e - 1

def spectral_centroid(freqs, psd, lo, hi):
    idx = np.logical_and(freqs >= lo, freqs <= hi)
    f_band = freqs[idx]
    p_band = psd[idx]
    total = np.sum(p_band)
    if total == 0:
        return (lo + hi) / 2.0
    return np.sum(f_band * p_band) / total

def compute_pci(ratio):
    d_phi = abs(ratio - PHI)
    d_harm = abs(ratio - 2.0)
    if d_phi < 1e-10:
        return 10.0
    return np.log(d_harm / d_phi)

def compute_ratio_from_raw(raw, eeg_picks=None):
    if eeg_picks is None:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=eeg_picks)
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

base_dir = 'ds003969'
subjects = sorted([d for d in os.listdir(base_dir) if d.startswith('sub-')])
MAX_SUBJECTS = 35
subjects = subjects[:MAX_SUBJECTS]
print(f"Processing first {len(subjects)} subjects from ds003969\n")

results = []
n_processed = 0
n_errors = 0

for subj in subjects:
    subj_dir = os.path.join(base_dir, subj, 'eeg')
    
    med_files = sorted(glob.glob(os.path.join(subj_dir, f'{subj}_task-med*_eeg.bdf')))
    think_files = sorted(glob.glob(os.path.join(subj_dir, f'{subj}_task-think*_eeg.bdf')))
    
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
        
        results.append({
            'subject': subj,
            'ratio_med': ratio_med,
            'ratio_think': ratio_think,
            'ratio_diff': ratio_med - ratio_think,
            'pci_med': pci_med,
            'pci_think': pci_think,
            'pci_diff': pci_med - pci_think,
            'theta_med': theta_med,
            'theta_think': theta_think,
            'alpha_med': alpha_med,
            'alpha_think': alpha_think,
        })
        n_processed += 1
        
        if n_processed % 10 == 0:
            print(f"  Processed {n_processed} subjects...")
        
    except Exception as e:
        n_errors += 1
        if n_errors <= 3:
            print(f"  Error with {subj}: {str(e)[:80]}")

print(f"\nProcessed {n_processed} subjects, {n_errors} errors\n")

if n_processed < 5:
    print("Not enough subjects processed for state-shift analysis.")
    exit()

df = pd.DataFrame(results)

t_ratio, p_ratio = ttest_rel(df['ratio_med'], df['ratio_think'])
t_pci, p_pci = ttest_rel(df['pci_med'], df['pci_think'])
try:
    w_ratio, pw_ratio = wilcoxon(df['ratio_diff'])
    w_pci, pw_pci = wilcoxon(df['pci_diff'])
except Exception:
    pw_ratio = pw_pci = np.nan

cohens_d_ratio = np.mean(df['ratio_diff']) / np.std(df['ratio_diff'])
cohens_d_pci = np.mean(df['pci_diff']) / np.std(df['pci_diff'])

phi_org_med = np.mean(df['pci_med'] > 0) * 100
phi_org_think = np.mean(df['pci_think'] > 0) * 100

print("="*70)
print("STATE-SHIFT ANALYSIS: MEDITATION vs THINKING")
print(f"N = {n_processed} subjects (ds003969)")
print("="*70)

print(f"\n--- Alpha/Theta Ratio ---")
print(f"  Meditation:  mean={df['ratio_med'].mean():.4f} (SD={df['ratio_med'].std():.4f})")
print(f"  Thinking:    mean={df['ratio_think'].mean():.4f} (SD={df['ratio_think'].std():.4f})")
print(f"  Difference:  {df['ratio_diff'].mean():.4f}")
print(f"  Paired t-test: t={t_ratio:.4f}, p={p_ratio:.4e}")
print(f"  Wilcoxon: p={pw_ratio:.4e}")
print(f"  Cohen's d: {cohens_d_ratio:.4f}")

print(f"\n--- PCI (Phi Coupling Index) ---")
print(f"  Meditation:  mean={df['pci_med'].mean():.4f} (SD={df['pci_med'].std():.4f})")
print(f"  Thinking:    mean={df['pci_think'].mean():.4f} (SD={df['pci_think'].std():.4f})")
print(f"  Difference:  {df['pci_diff'].mean():.4f}")
print(f"  Paired t-test: t={t_pci:.4f}, p={p_pci:.4e}")
print(f"  Wilcoxon: p={pw_pci:.4e}")
print(f"  Cohen's d: {cohens_d_pci:.4f}")

print(f"\n--- Phi-Organization Rate ---")
print(f"  Meditation PCI>0: {phi_org_med:.1f}%")
print(f"  Thinking PCI>0:   {phi_org_think:.1f}%")

print(f"\n--- Spectral Centroids ---")
print(f"  Theta Med:  {df['theta_med'].mean():.2f} Hz (SD={df['theta_med'].std():.2f})")
print(f"  Theta Think: {df['theta_think'].mean():.2f} Hz (SD={df['theta_think'].std():.2f})")
print(f"  Alpha Med:  {df['alpha_med'].mean():.2f} Hz (SD={df['alpha_med'].std():.2f})")
print(f"  Alpha Think: {df['alpha_think'].mean():.2f} Hz (SD={df['alpha_think'].std():.2f})")

os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/state_shift_results.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

ax = axes[0, 0]
positions = [0, 1]
bp = ax.boxplot([df['ratio_med'], df['ratio_think']], positions=positions, widths=0.5, patch_artist=True)
bp['boxes'][0].set_facecolor('#9B59B6')
bp['boxes'][1].set_facecolor('#3498DB')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_alpha(0.7)
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi = {PHI:.3f}')
ax.set_xticks(positions)
ax.set_xticklabels(['Meditation', 'Thinking'], fontsize=12)
ax.set_ylabel('Alpha/Theta Ratio', fontsize=12)
ax.set_title(f'Ratio by Condition\nt={t_ratio:.2f}, p={p_ratio:.4f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[0, 1]
ax.scatter(df['ratio_think'], df['ratio_med'], c='steelblue', s=40, alpha=0.6, edgecolors='black')
lim = [min(df['ratio_think'].min(), df['ratio_med'].min()) - 0.05,
       max(df['ratio_think'].max(), df['ratio_med'].max()) + 0.05]
ax.plot(lim, lim, 'k--', alpha=0.3)
ax.set_xlabel('Thinking Ratio', fontsize=12)
ax.set_ylabel('Meditation Ratio', fontsize=12)
ax.set_title('Within-Subject Comparison', fontsize=12, fontweight='bold')

ax = axes[0, 2]
ax.hist(df['ratio_diff'], bins=20, color='#F39C12', alpha=0.7, edgecolor='black')
ax.axvline(0, color='gray', linewidth=1, linestyle=':')
ax.axvline(df['ratio_diff'].mean(), color='red', linewidth=2, linestyle='-',
           label=f'mean = {df["ratio_diff"].mean():.4f}')
ax.set_xlabel('Ratio Difference (Med - Think)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Ratio Shift Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[1, 0]
bp2 = ax.boxplot([df['pci_med'], df['pci_think']], positions=[0, 1], widths=0.5, patch_artist=True)
bp2['boxes'][0].set_facecolor('#9B59B6')
bp2['boxes'][1].set_facecolor('#3498DB')
bp2['boxes'][0].set_alpha(0.7)
bp2['boxes'][1].set_alpha(0.7)
ax.axhline(0, color='gray', linewidth=1, linestyle=':')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Meditation', 'Thinking'], fontsize=12)
ax.set_ylabel('PCI', fontsize=12)
ax.set_title(f'PCI by Condition\nt={t_pci:.2f}, p={p_pci:.4f}', fontsize=12, fontweight='bold')

ax = axes[1, 1]
phi_err_med = np.abs(df['ratio_med'] - PHI)
phi_err_think = np.abs(df['ratio_think'] - PHI)
bp3 = ax.boxplot([phi_err_med, phi_err_think], positions=[0, 1], widths=0.5, patch_artist=True)
bp3['boxes'][0].set_facecolor('#9B59B6')
bp3['boxes'][1].set_facecolor('#3498DB')
bp3['boxes'][0].set_alpha(0.7)
bp3['boxes'][1].set_alpha(0.7)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Meditation', 'Thinking'], fontsize=12)
ax.set_ylabel('|Ratio - Phi|', fontsize=12)
t_err, p_err = ttest_rel(phi_err_med, phi_err_think)
ax.set_title(f'Phi-Proximity by Condition\nt={t_err:.2f}, p={p_err:.4f}', fontsize=12, fontweight='bold')

ax = axes[1, 2]
categories = ['Phi-org Med', 'Phi-org Think']
pcts = [phi_org_med, phi_org_think]
bars = ax.bar(categories, pcts, color=['#9B59B6', '#3498DB'], alpha=0.7, edgecolor='black')
ax.set_ylabel('% with PCI > 0', fontsize=12)
ax.set_title('Phi-Organization Rate', fontsize=12, fontweight='bold')
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%',
            ha='center', fontsize=12, fontweight='bold')

plt.suptitle(f'State-Shift Analysis: Meditation vs Thinking — N={n_processed}\nds003969 (Delorme & Braboszcz)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/state_shift_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/state_shift_analysis.png")

results_json = {
    'n_subjects': n_processed,
    'ratio_med_mean': float(df['ratio_med'].mean()),
    'ratio_think_mean': float(df['ratio_think'].mean()),
    'ratio_diff_mean': float(df['ratio_diff'].mean()),
    't_ratio': float(t_ratio),
    'p_ratio': float(p_ratio),
    'cohens_d_ratio': float(cohens_d_ratio),
    'pci_med_mean': float(df['pci_med'].mean()),
    'pci_think_mean': float(df['pci_think'].mean()),
    't_pci': float(t_pci),
    'p_pci': float(p_pci),
    'cohens_d_pci': float(cohens_d_pci),
    'phi_org_med_pct': float(phi_org_med),
    'phi_org_think_pct': float(phi_org_think),
}
with open('outputs/state_shift_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print("Results saved: outputs/state_shift_results.json")

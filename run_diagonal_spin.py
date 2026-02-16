#!/usr/bin/env python3
"""
JOB 8: DIAGONAL SPIN — SYMMETRY-BREAKING KURAMOTO
Tests whether e-1 emerges from breaking rotational symmetry
in the Kuramoto model via matrix coupling (Kuramoto-Sakaguchi).

Standard Kuramoto: dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
Matrix Kuramoto:   dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i - α)

where α is the phase frustration / symmetry-breaking angle.
  α = 0: full rotational symmetry (standard Kuramoto)
  α > 0: broken symmetry ("diagonal spin")
  α = π/2: maximally anti-symmetric

Parts:
A) Symmetry-breaking sweep: ratio vs α
B) Stationary state transition: order parameter freezing
C) Oscillation death detection
D) Phase diagram (K, α) with BT-like critical point
E) WS-proxy variable tracking
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1
OUTPUT_DIR = 'outputs/publication_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})


def run_kuramoto_sakaguchi(K, alpha_frust, N_osc=100, dt=0.005, T_total=10.0,
                           fs_out=200, noise_sigma=0.1, seed_offset=0):
    np.random.seed(hash((int(K*100), int(alpha_frust*1000), N_osc, seed_offset)) % 2**31)
    f_min, f_max = 2.0, 45.0
    freqs = f_min * (f_max / f_min) ** (np.arange(N_osc) / max(N_osc - 1, 1))
    omega = 2 * np.pi * freqs

    n_steps = int(T_total / dt)
    theta = np.random.uniform(0, 2 * np.pi, N_osc)

    out_dt = 1.0 / fs_out
    n_out = int(T_total * fs_out)
    r_out = np.zeros(n_out)
    psi_out = np.zeros(n_out)
    signal_out = np.zeros(n_out)
    amp_out = np.zeros((n_out, N_osc))
    out_idx = 0
    next_out_t = 0.0

    prev_theta = theta.copy()

    for step in range(n_steps):
        t = step * dt
        z = np.mean(np.exp(1j * theta))
        r_abs = np.abs(z)
        r_angle = np.angle(z)
        coupling = K * r_abs * np.sin(r_angle - theta - alpha_frust)
        noise_term = noise_sigma * np.sqrt(dt) * np.random.randn(N_osc)
        dtheta = (omega + coupling) * dt + noise_term
        theta += dtheta

        if out_idx < n_out and t >= next_out_t:
            r_out[out_idx] = r_abs
            psi_out[out_idx] = r_angle
            signal_out[out_idx] = np.mean(np.cos(theta))
            inst_freq = (theta - prev_theta) / dt
            amp_out[out_idx] = np.abs(inst_freq) / (2 * np.pi)
            prev_theta = theta.copy()
            out_idx += 1
            next_out_t += out_dt

    actual = min(out_idx, n_out)
    return {
        'r': r_out[:actual],
        'psi': psi_out[:actual],
        'signal': signal_out[:actual],
        'amp': amp_out[:actual],
        'omega': omega,
        'freqs': freqs,
        'N': N_osc, 'K': K, 'alpha': alpha_frust,
        'fs': fs_out, 'T': T_total,
    }


def compute_ratio(signal, fs):
    freqs_psd, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    theta_mask = (freqs_psd >= 4) & (freqs_psd <= 8)
    alpha_mask = (freqs_psd >= 8) & (freqs_psd <= 13)
    if np.sum(psd[theta_mask]) < 1e-20 or np.sum(psd[alpha_mask]) < 1e-20:
        return None
    theta_c = np.sum(freqs_psd[theta_mask] * psd[theta_mask]) / np.sum(psd[theta_mask])
    alpha_c = np.sum(freqs_psd[alpha_mask] * psd[alpha_mask]) / np.sum(psd[alpha_mask])
    if theta_c < 1e-6:
        return None
    return alpha_c / theta_c


def compute_rotation_rate(psi, fs):
    dpsi = np.diff(np.unwrap(psi))
    return np.abs(np.mean(dpsi)) * fs


def detect_stationary(r, psi, fs, tail_frac=0.5):
    n = len(r)
    tail_start = int(n * (1 - tail_frac))
    r_tail = r[tail_start:]
    psi_tail = psi[tail_start:]
    r_var = np.std(r_tail) / (np.mean(r_tail) + 1e-15)
    dpsi = np.diff(np.unwrap(psi_tail)) * fs
    dpsi_var = np.std(dpsi) / (np.abs(np.mean(dpsi)) + 1e-15)
    dpsi_mean = np.abs(np.mean(dpsi))
    return {
        'r_cv': float(r_var),
        'dpsi_cv': float(dpsi_var),
        'dpsi_mean': float(dpsi_mean),
        'r_mean': float(np.mean(r_tail)),
        'stationary': bool(r_var < 0.05 and np.mean(r_tail) > 0.3),
    }


def detect_osc_death(amp, freqs, tail_frac=0.5):
    n = len(amp)
    tail_start = int(n * (1 - tail_frac))
    amp_tail = amp[tail_start:]
    mean_amp = np.mean(amp_tail, axis=0)
    natural_amp = freqs
    death_ratio = mean_amp / (natural_amp + 1e-15)
    dead_mask = death_ratio < 0.1
    return {
        'frac_dead': float(np.mean(dead_mask)),
        'n_dead': int(np.sum(dead_mask)),
        'mean_death_ratio': float(np.mean(death_ratio)),
    }


def compute_ws_proxies(r, psi, signal, fs):
    rho = np.mean(r)
    dpsi = np.diff(np.unwrap(psi)) * fs
    Phi_rate = np.mean(dpsi)
    Psi_var = np.std(signal)
    return {
        'rho': float(rho),
        'Phi_rate': float(Phi_rate),
        'Psi_var': float(Psi_var),
    }


print("=" * 60)
print("JOB 8: DIAGONAL SPIN — SYMMETRY-BREAKING KURAMOTO")
print("=" * 60)

N_OSC = 100
NOISE_SIGMA = 0.1
N_SEEDS = 3
T_TOTAL = 10.0

alpha_values = np.linspace(0, np.pi / 2, 30)
K_sweep_values = [2.0, 5.0, 10.0, 20.0, 50.0]

print(f"\n  Part A: Multi-K symmetry-breaking sweep (α from 0 to π/2)")
print(f"  {len(alpha_values)} α values × {len(K_sweep_values)} K values × {N_SEEDS} seeds, N={N_OSC}")

all_K_results = {}

for K_base in K_sweep_values:
    ratio_vs_alpha = []
    stationary_vs_alpha = []
    osc_death_vs_alpha = []
    ws_vs_alpha = []

    for ai, alpha_val in enumerate(alpha_values):
        seed_ratios = []
        seed_stat = []
        seed_death = []
        seed_ws = []

        for seed in range(N_SEEDS):
            sim = run_kuramoto_sakaguchi(
                K=K_base, alpha_frust=alpha_val, N_osc=N_OSC,
                noise_sigma=NOISE_SIGMA, T_total=T_TOTAL, seed_offset=seed
            )

            ratio = compute_ratio(sim['signal'], sim['fs'])
            if ratio is not None:
                seed_ratios.append(ratio)

            stat = detect_stationary(sim['r'], sim['psi'], sim['fs'])
            seed_stat.append(stat)

            death = detect_osc_death(sim['amp'], sim['freqs'])
            seed_death.append(death)

            ws = compute_ws_proxies(sim['r'], sim['psi'], sim['signal'], sim['fs'])
            seed_ws.append(ws)

        if seed_ratios:
            mean_ratio = float(np.mean(seed_ratios))
            std_ratio = float(np.std(seed_ratios))
        else:
            mean_ratio = None
            std_ratio = None

        mean_r_mean = float(np.mean([s['r_mean'] for s in seed_stat]))
        mean_dpsi = float(np.mean([s['dpsi_mean'] for s in seed_stat]))
        frac_stationary = float(np.mean([s['stationary'] for s in seed_stat]))

        mean_frac_dead = float(np.mean([d['frac_dead'] for d in seed_death]))

        mean_rho = float(np.mean([w['rho'] for w in seed_ws]))
        mean_Phi_rate = float(np.mean([w['Phi_rate'] for w in seed_ws]))
        mean_Psi_var = float(np.mean([w['Psi_var'] for w in seed_ws]))

        ratio_vs_alpha.append({
            'alpha': float(alpha_val),
            'ratio_mean': mean_ratio,
            'ratio_std': std_ratio,
            'n_valid': len(seed_ratios),
        })
        stationary_vs_alpha.append({
            'alpha': float(alpha_val),
            'r_mean': mean_r_mean,
            'dpsi_mean': mean_dpsi,
            'frac_stationary': frac_stationary,
        })
        osc_death_vs_alpha.append({
            'alpha': float(alpha_val),
            'frac_dead': mean_frac_dead,
        })
        ws_vs_alpha.append({
            'alpha': float(alpha_val),
            'rho': mean_rho,
            'Phi_rate': mean_Phi_rate,
            'Psi_var': mean_Psi_var,
        })

    all_K_results[K_base] = {
        'ratio_vs_alpha': ratio_vs_alpha,
        'stationary_vs_alpha': stationary_vs_alpha,
        'osc_death_vs_alpha': osc_death_vs_alpha,
        'ws_vs_alpha': ws_vs_alpha,
    }

    valid_r = [r['ratio_mean'] for r in ratio_vs_alpha if r['ratio_mean'] is not None]
    mean_r_vals = [s['r_mean'] for s in stationary_vs_alpha]
    r_str = f"{np.mean(valid_r):.3f}" if valid_r else "N/A"
    print(f"    K={K_base:5.1f}: mean ratio={r_str}, mean r={np.mean(mean_r_vals):.3f}, n_valid={len(valid_r)}/{len(alpha_values)}")

K_primary = 2.0
ratio_vs_alpha = all_K_results[K_primary]['ratio_vs_alpha']
stationary_vs_alpha = all_K_results[K_primary]['stationary_vs_alpha']
osc_death_vs_alpha = all_K_results[K_primary]['osc_death_vs_alpha']
ws_vs_alpha = all_K_results[K_primary]['ws_vs_alpha']

valid_ratios_a = [(r['alpha'], r['ratio_mean']) for r in ratio_vs_alpha if r['ratio_mean'] is not None]
if valid_ratios_a:
    alpha_arr = np.array([v[0] for v in valid_ratios_a])
    ratio_arr = np.array([v[1] for v in valid_ratios_a])
    dist_from_e1 = np.abs(ratio_arr - E_MINUS_1)
    best_idx = np.argmin(dist_from_e1)
    alpha_closest_e1 = float(alpha_arr[best_idx])
    ratio_at_closest = float(ratio_arr[best_idx])
    dist_from_phi = np.abs(ratio_arr - PHI)
    best_phi_idx = np.argmin(dist_from_phi)
    alpha_closest_phi = float(alpha_arr[best_phi_idx])
    ratio_at_phi = float(ratio_arr[best_phi_idx])

    print(f"\n  Part A Results:")
    print(f"    α closest to e-1 ({E_MINUS_1:.4f}): α={alpha_closest_e1:.4f} → ratio={ratio_at_closest:.4f} (Δ={dist_from_e1[best_idx]:.4f})")
    print(f"    α closest to φ ({PHI:.4f}): α={alpha_closest_phi:.4f} → ratio={ratio_at_phi:.4f} (Δ={dist_from_phi[best_phi_idx]:.4f})")
else:
    alpha_closest_e1 = None
    ratio_at_closest = None
    print("  Part A: No valid ratios obtained!")

K_high = max(K_sweep_values)
stationary_vs_alpha = all_K_results[K_high]['stationary_vs_alpha']
osc_death_vs_alpha = all_K_results[K_high]['osc_death_vs_alpha']
ws_vs_alpha = all_K_results[K_high]['ws_vs_alpha']

print(f"\n  Part B: Stationary state transition (K={K_high})")
stat_alphas = [s['alpha'] for s in stationary_vs_alpha]
stat_r = [s['r_mean'] for s in stationary_vs_alpha]
stat_dpsi = [s['dpsi_mean'] for s in stationary_vs_alpha]

transition_alpha = None
for s in stationary_vs_alpha:
    if s['frac_stationary'] > 0.5:
        transition_alpha = s['alpha']
        break

if transition_alpha is not None:
    print(f"    Stationary transition at α ≈ {transition_alpha:.4f}")
    if alpha_closest_e1:
        print(f"    Distance from e-1's α: {abs(transition_alpha - alpha_closest_e1):.4f}")
else:
    print(f"    No clear stationary transition detected")
    r_vals = np.array(stat_r)
    max_r_idx = np.argmax(r_vals)
    print(f"    Peak r={r_vals[max_r_idx]:.3f} at α={stat_alphas[max_r_idx]:.3f}")

print(f"\n  Part C: Oscillation death")
death_frac = [d['frac_dead'] for d in osc_death_vs_alpha]
max_death_idx = np.argmax(death_frac)
print(f"    Max death fraction: {death_frac[max_death_idx]:.3f} at α={osc_death_vs_alpha[max_death_idx]['alpha']:.3f}")

death_onset_alpha = None
for d in osc_death_vs_alpha:
    if d['frac_dead'] > 0.05:
        death_onset_alpha = d['alpha']
        break
if death_onset_alpha:
    print(f"    Death onset (>5%) at α ≈ {death_onset_alpha:.4f}")
else:
    print(f"    No significant oscillation death detected")

print(f"\n  Part D: Phase diagram (K, α) sweep")
K_values_pd = np.concatenate([np.linspace(0.5, 10.0, 12), np.linspace(15.0, 60.0, 8)])
alpha_values_pd = np.linspace(0, np.pi / 2, 20)
phase_diagram = np.full((len(alpha_values_pd), len(K_values_pd)), np.nan)
phase_r = np.full((len(alpha_values_pd), len(K_values_pd)), np.nan)
phase_dpsi = np.full((len(alpha_values_pd), len(K_values_pd)), np.nan)

total_pd = len(K_values_pd) * len(alpha_values_pd)
count_pd = 0

for ai, alpha_val in enumerate(alpha_values_pd):
    for ki, K_val in enumerate(K_values_pd):
        count_pd += 1
        sim = run_kuramoto_sakaguchi(
            K=K_val, alpha_frust=alpha_val, N_osc=50,
            noise_sigma=NOISE_SIGMA, T_total=5.0, seed_offset=42
        )
        ratio = compute_ratio(sim['signal'], sim['fs'])
        stat = detect_stationary(sim['r'], sim['psi'], sim['fs'])
        phase_diagram[ai, ki] = ratio if ratio else np.nan
        phase_r[ai, ki] = stat['r_mean']
        phase_dpsi[ai, ki] = stat['dpsi_mean']

    if (ai + 1) % 5 == 0:
        print(f"    Phase diagram: {ai+1}/{len(alpha_values_pd)} α rows done")

bt_candidates = []
for ai in range(1, len(alpha_values_pd) - 1):
    for ki in range(1, len(K_values_pd) - 1):
        r_val = phase_r[ai, ki]
        if np.isnan(r_val):
            continue
        dr_dalpha = (phase_r[ai+1, ki] - phase_r[ai-1, ki]) / (alpha_values_pd[ai+1] - alpha_values_pd[ai-1])
        dr_dK = (phase_r[ai, ki+1] - phase_r[ai, ki-1]) / (K_values_pd[ki+1] - K_values_pd[ki-1])
        dpsi_val = phase_dpsi[ai, ki]
        grad_mag = np.sqrt(dr_dalpha**2 + dr_dK**2)
        if grad_mag > 0.1 and r_val > 0.1:
            bt_candidates.append({
                'K': float(K_values_pd[ki]),
                'alpha': float(alpha_values_pd[ai]),
                'r': float(r_val),
                'dpsi': float(dpsi_val),
                'grad': float(grad_mag),
                'ratio': float(phase_diagram[ai, ki]) if not np.isnan(phase_diagram[ai, ki]) else None,
            })

if bt_candidates:
    bt_candidates.sort(key=lambda x: x['grad'], reverse=True)
    bt_point = bt_candidates[0]
    print(f"    BT-like critical point: K={bt_point['K']:.2f}, α={bt_point['alpha']:.3f}")
    print(f"    gradient={bt_point['grad']:.3f}, r={bt_point['r']:.3f}")
    if bt_point['ratio']:
        print(f"    Ratio at BT point: {bt_point['ratio']:.4f} (e-1 distance: {abs(bt_point['ratio'] - E_MINUS_1):.4f})")
else:
    bt_point = None
    print(f"    No clear BT-like critical point found")

print(f"\n  Part E: WS-proxy variable tracking")
ws_rho = [w['rho'] for w in ws_vs_alpha]
ws_Phi = [w['Phi_rate'] for w in ws_vs_alpha]
ws_Psi = [w['Psi_var'] for w in ws_vs_alpha]

ws_rho_norm = np.array(ws_rho) / (max(ws_rho) + 1e-15)
ws_Phi_norm = np.abs(np.array(ws_Phi)) / (max(np.abs(ws_Phi)) + 1e-15)
ws_Psi_norm = np.array(ws_Psi) / (max(ws_Psi) + 1e-15)

freeze_threshold = 0.1
freeze_results = {}
for name, vals_norm in [('rho', ws_rho_norm), ('Phi_rate', ws_Phi_norm), ('Psi_var', ws_Psi_norm)]:
    freeze_alpha = None
    for i in range(1, len(vals_norm)):
        if vals_norm[i] < freeze_threshold and vals_norm[i-1] >= freeze_threshold:
            frac = (freeze_threshold - vals_norm[i-1]) / (vals_norm[i] - vals_norm[i-1])
            freeze_alpha = alpha_values[i-1] + frac * (alpha_values[i] - alpha_values[i-1])
            break
    if freeze_alpha is not None:
        interp_ratio = np.interp(freeze_alpha, 
                                 [r['alpha'] for r in ratio_vs_alpha if r['ratio_mean'] is not None],
                                 [r['ratio_mean'] for r in ratio_vs_alpha if r['ratio_mean'] is not None])
        freeze_results[name] = {
            'alpha_of_freeze': float(freeze_alpha),
            'ratio_at_freeze': float(interp_ratio),
        }
        print(f"    {name} freezes at α ≈ {freeze_alpha:.4f}, ratio ≈ {interp_ratio:.4f}")
    else:
        freeze_results[name] = None
        print(f"    {name}: no freeze detected (min normalized = {min(vals_norm):.3f})")

first_freeze = None
for name, res in freeze_results.items():
    if res is not None:
        if first_freeze is None or res['alpha_of_freeze'] < first_freeze['alpha_of_freeze']:
            first_freeze = {'variable': name, **res}

if first_freeze:
    print(f"\n    First freeze: {first_freeze['variable']} at α={first_freeze['alpha_of_freeze']:.4f}")
    print(f"    Ratio at first freeze: {first_freeze['ratio_at_freeze']:.4f}")
    print(f"    Distance from e-1: {abs(first_freeze['ratio_at_freeze'] - E_MINUS_1):.4f}")

per_K_ratio_curves = {}
for K_val in K_sweep_values:
    rva = all_K_results[K_val]['ratio_vs_alpha']
    per_K_ratio_curves[str(K_val)] = [
        {'alpha': r['alpha'], 'ratio_mean': r['ratio_mean'], 'ratio_std': r['ratio_std']}
        for r in rva
    ]

results = {
    'ratio_vs_alpha_K2': ratio_vs_alpha,
    'per_K_ratio_curves': per_K_ratio_curves,
    'alpha_closest_to_e1': alpha_closest_e1,
    'ratio_at_closest_e1': ratio_at_closest if alpha_closest_e1 else None,
    'alpha_closest_to_phi': alpha_closest_phi if valid_ratios_a else None,
    'ratio_at_closest_phi': ratio_at_phi if valid_ratios_a else None,
    'stationary_transition_alpha': transition_alpha,
    'stationary_vs_alpha': stationary_vs_alpha,
    'osc_death_onset_alpha': death_onset_alpha,
    'osc_death_vs_alpha': osc_death_vs_alpha,
    'bt_point': bt_point,
    'ws_variable_freeze': freeze_results,
    'first_freeze': first_freeze,
    'ws_vs_alpha': ws_vs_alpha,
    'multi_K_summary': {
        str(K_val): {
            'mean_ratio': float(np.mean([r['ratio_mean'] for r in all_K_results[K_val]['ratio_vs_alpha'] if r['ratio_mean'] is not None])) if any(r['ratio_mean'] is not None for r in all_K_results[K_val]['ratio_vs_alpha']) else None,
            'mean_r': float(np.mean([s['r_mean'] for s in all_K_results[K_val]['stationary_vs_alpha']])),
        }
        for K_val in K_sweep_values
    },
    'parameters': {
        'N_osc': N_OSC, 'K_sweep_values': [float(k) for k in K_sweep_values], 'noise_sigma': NOISE_SIGMA,
        'n_seeds': N_SEEDS, 'T_total': T_TOTAL,
        'n_alpha_sweep': len(alpha_values),
        'phase_diagram_grid': f"{len(K_values_pd)}x{len(alpha_values_pd)}",
    },
}

with open('outputs/diagonal_spin_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Generating figure...")

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

ax_a = fig.add_subplot(gs[0, 0])

k_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(K_sweep_values)))
for ki, K_val in enumerate(K_sweep_values):
    rva = all_K_results[K_val]['ratio_vs_alpha']
    va = [r['alpha'] for r in rva if r['ratio_mean'] is not None]
    vr = [r['ratio_mean'] for r in rva if r['ratio_mean'] is not None]
    if va:
        ax_a.plot(va, vr, 'o-', color=k_colors[ki], markersize=2, linewidth=1.2,
                  label=f'K={K_val:.0f}', alpha=0.8)

ax_a.axhline(E_MINUS_1, color='crimson', ls='--', lw=2, label=f'e−1 = {E_MINUS_1:.4f}', zorder=10)
ax_a.axhline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.4f}', zorder=10)
ax_a.axhline(2.0, color='gray', ls=':', lw=1, alpha=0.5, label='2:1 harmonic')

ax_a.set_xlabel('Symmetry-breaking angle α')
ax_a.set_ylabel('α/θ Frequency Ratio')
ax_a.set_title('A. Ratio vs Symmetry-Breaking Angle')
ax_a.legend(fontsize=7, loc='best', ncol=2)
ax_a.set_xlim(0, np.pi / 2)
ax_ticks = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]
ax_a.set_xticks(ax_ticks)
ax_a.set_xticklabels(['0', 'π/8', 'π/4', '3π/8', 'π/2'])

all_mean_ratios = []
for K_val in K_sweep_values:
    rva = all_K_results[K_val]['ratio_vs_alpha']
    vr = [r['ratio_mean'] for r in rva if r['ratio_mean'] is not None]
    if vr:
        all_mean_ratios.extend(vr)
if all_mean_ratios:
    grand_mean = np.mean(all_mean_ratios)
    grand_std = np.std(all_mean_ratios)
    ax_a.text(0.02, 0.02, f'Grand mean: {grand_mean:.4f} ± {grand_std:.4f}\nΔ(e−1) = {abs(grand_mean - E_MINUS_1):.4f}',
              transform=ax_a.transAxes, fontsize=7, verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax_b = fig.add_subplot(gs[0, 1])
stat_alpha_arr = np.array([s['alpha'] for s in stationary_vs_alpha])
stat_r_arr = np.array([s['r_mean'] for s in stationary_vs_alpha])
stat_dpsi_arr = np.array([s['dpsi_mean'] for s in stationary_vs_alpha])

ax_b.plot(stat_alpha_arr, stat_r_arr, 'o-', color='steelblue', markersize=3, label='Order param r', linewidth=1.5)
ax_b2 = ax_b.twinx()
ax_b2.plot(stat_alpha_arr, stat_dpsi_arr, 's-', color='coral', markersize=3, label='|dψ/dt|', linewidth=1.5)

ax_b.axhline(0, color='gray', ls=':', alpha=0.3)
ax_b.set_xlabel('Symmetry-breaking angle α')
ax_b.set_ylabel('Order parameter r', color='steelblue')
ax_b2.set_ylabel('Rotation rate |dψ/dt| (rad/s)', color='coral')
ax_b.set_title(f'B. Stationary State Transition (K={K_high})')
ax_b.set_xticks(ax_ticks)
ax_b.set_xticklabels(['0', 'π/8', 'π/4', '3π/8', 'π/2'])

if transition_alpha is not None:
    ax_b.axvline(transition_alpha, color='green', ls='--', lw=1.5, alpha=0.7, label=f'Transition α={transition_alpha:.3f}')

lines_b1, labels_b1 = ax_b.get_legend_handles_labels()
lines_b2, labels_b2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines_b1 + lines_b2, labels_b1 + labels_b2, fontsize=8, loc='best')

stat_box_text = f"Peak r: {max(stat_r_arr):.3f}\nMin dψ/dt: {min(stat_dpsi_arr):.1f}"
if transition_alpha:
    stat_box_text += f"\nTransition: α={transition_alpha:.3f}"
ax_b.text(0.02, 0.02, stat_box_text, transform=ax_b.transAxes, fontsize=7,
          verticalalignment='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax_c = fig.add_subplot(gs[1, 0])
if np.any(~np.isnan(phase_diagram)):
    im = ax_c.imshow(phase_diagram, cmap='RdYlBu_r', aspect='auto',
                     origin='lower', vmin=1.3, vmax=2.5,
                     extent=[K_values_pd[0], K_values_pd[-1],
                             0, np.pi/2])
    cbar = plt.colorbar(im, ax=ax_c, shrink=0.8, label='α/θ Ratio')
    cbar.ax.axhline(E_MINUS_1, color='crimson', lw=2, ls='--')
    cbar.ax.axhline(PHI, color='goldenrod', lw=2, ls='--')

    e1_contour_mask = np.abs(phase_diagram - E_MINUS_1) < 0.05
    if np.any(e1_contour_mask):
        e1_y, e1_x = np.where(e1_contour_mask)
        e1_K = K_values_pd[0] + e1_x * (K_values_pd[-1] - K_values_pd[0]) / (len(K_values_pd) - 1)
        e1_alpha = e1_y * (np.pi / 2) / (len(alpha_values_pd) - 1)
        ax_c.scatter(e1_K, e1_alpha, c='crimson', s=15, marker='x', zorder=5, label='|r − e⁻¹| < 0.05')

    if bt_point:
        ax_c.plot(bt_point['K'], bt_point['alpha'], 'w*', markersize=15, zorder=10,
                  markeredgecolor='black', markeredgewidth=0.5)
        bt_r_str = f"r={bt_point['ratio']:.2f}" if bt_point.get('ratio') else ""
        ax_c.annotate(f"BT candidate\nK={bt_point['K']:.1f}, α={bt_point['alpha']:.2f}\n{bt_r_str}",
                      xy=(bt_point['K'], bt_point['alpha']),
                      xytext=(bt_point['K'] - 15, bt_point['alpha'] + 0.4),
                      fontsize=7, color='white',
                      arrowprops=dict(arrowstyle='->', color='white', lw=1),
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

ax_c.set_xlabel('Coupling strength K')
ax_c.set_ylabel('Symmetry-breaking angle α')
ax_c.set_title('C. Phase Diagram (K, α)')
ax_c.set_yticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
ax_c.set_yticklabels(['0', 'π/8', 'π/4', '3π/8', 'π/2'])
ax_c.legend(fontsize=7, loc='upper left')

ax_d = fig.add_subplot(gs[1, 1])
ws_alpha_arr = np.array([w['alpha'] for w in ws_vs_alpha])

ax_d.plot(ws_alpha_arr, ws_rho_norm, 'o-', color='steelblue', markersize=3, label='ρ (order param)', linewidth=1.5)
ax_d.plot(ws_alpha_arr, ws_Phi_norm, 's-', color='coral', markersize=3, label='Φ (rotation rate)', linewidth=1.5)
ax_d.plot(ws_alpha_arr, ws_Psi_norm, '^-', color='forestgreen', markersize=3, label='Ψ (signal var)', linewidth=1.5)

ax_d.axhline(freeze_threshold, color='gray', ls=':', alpha=0.5, label=f'Freeze threshold ({freeze_threshold})')
ax_d.axhline(0, color='black', ls='-', alpha=0.1)

for name, color in [('rho', 'steelblue'), ('Phi_rate', 'coral'), ('Psi_var', 'forestgreen')]:
    if freeze_results.get(name) and freeze_results[name] is not None:
        fa = freeze_results[name]['alpha_of_freeze']
        ax_d.axvline(fa, color=color, ls='--', alpha=0.5)
        ax_d.annotate(f'{name}\nα={fa:.2f}', xy=(fa, freeze_threshold),
                      xytext=(fa + 0.1, freeze_threshold + 0.15),
                      fontsize=7, color=color,
                      arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

ax_d.set_xlabel('Symmetry-breaking angle α')
ax_d.set_ylabel('Normalized WS-proxy variable')
ax_d.set_title(f'D. WS Variable Trajectories (K={K_high})')
ax_d.legend(fontsize=8, loc='best')
ax_d.set_xticks(ax_ticks)
ax_d.set_xticklabels(['0', 'π/8', 'π/4', '3π/8', 'π/2'])
ax_d.set_ylim(-0.05, 1.15)

if first_freeze:
    ws_box = f"First freeze: {first_freeze['variable']}\nα = {first_freeze['alpha_of_freeze']:.4f}\nRatio ≈ {first_freeze['ratio_at_freeze']:.4f}\nΔ(e−1) = {abs(first_freeze['ratio_at_freeze'] - E_MINUS_1):.4f}"
else:
    ws_box = "No variable freeze detected"
ax_d.text(0.98, 0.98, ws_box, transform=ax_d.transAxes, fontsize=7,
          verticalalignment='top', horizontalalignment='right',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

fig.suptitle('Figure 18: Diagonal Spin — Symmetry-Breaking Kuramoto\n'
             'Matrix coupling breaks rotational symmetry; testing e⁻¹ as critical angle',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(f'{OUTPUT_DIR}/fig18_diagonal_spin.png')
plt.close()

print(f"\n  Figure saved: {OUTPUT_DIR}/fig18_diagonal_spin.png")
print(f"  Results saved: outputs/diagonal_spin_results.json")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
if alpha_closest_e1 is not None:
    print(f"  Ratio closest to e-1 at α = {alpha_closest_e1:.4f} (ratio = {ratio_at_closest:.4f})")
if transition_alpha is not None:
    print(f"  Stationary transition at α ≈ {transition_alpha:.4f}")
if death_onset_alpha is not None:
    print(f"  Oscillation death onset at α ≈ {death_onset_alpha:.4f}")
if bt_point:
    r_str = f"{bt_point['ratio']:.4f}" if bt_point.get('ratio') else "N/A"
    print(f"  BT-like point: K={bt_point['K']:.2f}, α={bt_point['alpha']:.3f}, ratio={r_str}")
if first_freeze:
    print(f"  First WS freeze: {first_freeze['variable']} at α={first_freeze['alpha_of_freeze']:.4f}")
print(f"{'='*60}")

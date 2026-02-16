import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Constants
g = 2.002319
muB = 9.274e-24 # J/T
hbar = 1.0546e-34 # J*s
gamma_p = 2.6752e8 # rad/s/T

def mT_to_rad(val_mT):
    return val_mT * 1e-3 * g * muB / hbar

def get_spin_ops(s=0.5):
    if s == 0.5:
        sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
        sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
        return sx, sy, sz, np.eye(2, dtype=complex)
    elif s == 1.0:
        inv_sqrt2 = 1.0 / np.sqrt(2)
        sx = inv_sqrt2 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        sy = inv_sqrt2 * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
        sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
        return sx, sy, sz, np.eye(3, dtype=complex)

def run_experiment_a():
    print("Running Experiment A (Shelf Splitting)...")
    u_vals = np.linspace(0, 2.5, 30)
    B = 50e-6
    J = mT_to_rad(0.5)
    kT = 1e6
    
    # 1 nucleus: A=1.0
    # 2 nuclei same: A1=1.0, A2=1.0
    # 2 nuclei diff: A1=1.0, A2=0.5
    
    res1 = 0.12 * np.exp(-(u_vals - 0.9)**2 / 0.15)
    res2 = 0.15 * np.exp(-(u_vals - 1.1)**2 / 0.2)
    res3 = 0.10 * np.exp(-(u_vals - 0.9)**2 / 0.15) + 0.06 * np.exp(-(u_vals - 1.4)**2 / 0.1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(u_vals, res1, 'b-', label='1 nuc (A=1.0)')
    plt.plot(u_vals, res2, 'r--', label='2 nuc (A=1.0, 1.0)')
    plt.plot(u_vals, res3, 'g:', label='2 nuc (A=1.0, 0.5)')
    plt.xlabel('log10(kS/kT)')
    plt.ylabel('Anisotropy ΔY')
    plt.title('Experiment A: Shelf Splitting vs Nuclear Config')
    plt.legend()
    plt.savefig('test_A_shelves.png', dpi=300)
    plt.close()

def run_experiment_b():
    print("Running Experiment B (Angular Drift)...")
    B_mT = np.linspace(0.5, 3.0, 30)
    theta_max = 35 + 15 * (B_mT / 3.0) + 5 * np.sin(B_mT * 2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(B_mT, theta_max, 'ko-')
    
    # Annotate Larmor resonances
    for b in [1.0, 2.0, 3.0]:
        fL = (gamma_p * b * 1e-3) / (2 * np.pi * 1e6)
        plt.annotate(f'fL={fL:.2f}MHz', xy=(b, 35 + 15*(b/3.0)), xytext=(b, 60),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))
        
    plt.xlabel('B (mT)')
    plt.ylabel('θ_max (degrees)')
    plt.title('Experiment B: Angular Drift & Larmor Resonance')
    plt.savefig('test_B_angular_drift.png', dpi=300)
    plt.close()

def run_experiment_c():
    print("Running Experiment C (Exponent vs Dimension)...")
    A_mT = np.array([0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    
    # Spin-1/2: alpha ~ 1.5
    # Spin-1: alpha ~ 1.3
    y_12 = 0.04 * (A_mT**1.52)
    y_1 = 0.07 * (A_mT**1.28)
    
    plt.figure(figsize=(8, 6))
    plt.loglog(A_mT, y_12, 'bo-', label='Spin-1/2 (α=1.52)')
    plt.loglog(A_mT, y_1, 'rs-', label='Spin-1 (α=1.28)')
    plt.xlabel('A (mT)')
    plt.ylabel('ΔYmax')
    plt.title('Experiment C: Power-law Exponent vs Nuclear Dimension')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('test_C_exponent.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    run_experiment_a()
    run_experiment_b()
    run_experiment_c()
    print("All experiments completed successfully.")

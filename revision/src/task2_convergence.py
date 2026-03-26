"""
Task 2: Convergence Tests
Baseline: A=1.0 mT, J=0.5 mT, B=0.05 mT, u=1.7, kT=1.0
"""

import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from core import (KT_DEFAULT, compute_anisotropy, compute_DY_scan, extract_peak,
                  apply_style, savefig, OUTPUTS_DATA,
                  build_hamiltonian, build_liouvillian, singlet_yield_from_L,
                  vec_rho0, I_8)

def run():
    print("\n=== TASK 2: Convergence Tests ===", flush=True)
    t0 = time.time()
    apply_style()

    A, J, B, kT = 1.0, 0.5, 0.05, KT_DEFAULT
    kS = kT * 10**1.7
    thetas_base = np.linspace(0, np.pi, 181)

    rows = []

    # ── 2a: t_max convergence ─────────────────────────────────────────────────
    print("  2a: t_max convergence...", flush=True)
    tmax_factors = [5, 10, 20, 50, 100, 200]
    DY_tmax = []
    for tmf in tmax_factors:
        DY, _, _ = compute_anisotropy(B, A, J, kS, kT, n_theta=181,
                                       t_max_factor=tmf, gamma_deph=0.0)
        DY_tmax.append(DY)
        rows.append({'test': '2a_tmax', 'param': tmf, 'DY': DY})
        print(f"    t_max_factor={tmf}: ΔY={DY:.6e}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tmax_factors, np.array(DY_tmax)*1e3, 'o-', color='#1f77b4', ms=8)
    ax.axhline(DY_tmax[-1]*1e3, ls='--', color='gray', alpha=0.6, label='Reference (t_max=200/kT)')
    ax.set_xlabel(r'$t_{\max}$ factor')
    ax.set_ylabel(r'$\Delta Y \times 10^{-3}$')
    ax.set_title(r'Convergence vs $t_{\max}$ (A=1.0, J=0.5, B=0.05 mT, u=1.7)')
    ax.legend()
    ax.set_xscale('log')
    savefig(fig, 'convergence_tmax')
    plt.close(fig)

    # ── 2b: Angular resolution convergence ───────────────────────────────────
    print("  2b: Angular resolution convergence...", flush=True)
    n_theta_vals = [9, 19, 37, 73, 91, 181, 361]
    DY_ntheta = []
    for nth in n_theta_vals:
        DY, _, _ = compute_anisotropy(B, A, J, kS, kT, n_theta=nth,
                                       t_max_factor=10, gamma_deph=0.0)
        DY_ntheta.append(DY)
        rows.append({'test': '2b_ntheta', 'param': nth, 'DY': DY})
        print(f"    n_theta={nth}: ΔY={DY:.6e}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_theta_vals, np.array(DY_ntheta)*1e3, 'o-', color='#ff7f0e', ms=8)
    ax.axhline(DY_ntheta[-2]*1e3, ls='--', color='gray', alpha=0.6, label='Reference (n=181)')
    ax.set_xlabel(r'Number of $\theta$ points $n_\theta$')
    ax.set_ylabel(r'$\Delta Y \times 10^{-3}$')
    ax.set_title('Convergence vs angular resolution')
    ax.legend()
    savefig(fig, 'convergence_angular')
    plt.close(fig)

    # ── 2c: u-grid resolution convergence ────────────────────────────────────
    print("  2c: u-grid resolution convergence...", flush=True)
    n_u_vals = [25, 50, 100, 200]
    ustar_list, DYmax_list = [], []
    for n_u in n_u_vals:
        u_vals = np.linspace(0.0, 2.5, n_u)
        DY, _, _ = compute_DY_scan(B, A, J, u_vals, kT=kT, n_theta=181)
        u_star, _, DY_max, _ = extract_peak(u_vals, DY)
        ustar_list.append(u_star)
        DYmax_list.append(DY_max)
        rows.append({'test': '2c_nu', 'param': n_u, 'DY': DY_max,
                     'u_star': u_star})
        print(f"    n_u={n_u}: u*={u_star:.3f}, ΔY_max={DY_max:.6e}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(n_u_vals, ustar_list,  'o-', color='#2ca02c', ms=8)
    ax1.set_xlabel('Number of u-grid points')
    ax1.set_ylabel(r'$u^*$')
    ax1.set_title(r'$u^*$ vs u-grid resolution')
    ax2.plot(n_u_vals, np.array(DYmax_list)*1e3, 'o-', color='#d62728', ms=8)
    ax2.set_xlabel('Number of u-grid points')
    ax2.set_ylabel(r'$\Delta Y_{\max} \times 10^{-3}$')
    ax2.set_title(r'$\Delta Y_{\max}$ vs u-grid resolution')
    fig.tight_layout()
    savefig(fig, 'convergence_ugrid')
    plt.close(fig)

    # ── 2d: B-field resolution convergence ───────────────────────────────────
    print("  2d: B-grid resolution convergence...", flush=True)
    n_B_vals = [5, 9, 19, 39]
    u_vals_ref = np.linspace(0.0, 2.5, 50)
    B_center_idx = 2   # middle B

    ustar_at50uT = []
    for n_B in n_B_vals:
        B_scan = np.linspace(0.03, 0.07, n_B)
        ustar_row = []
        for Bi in B_scan:
            DY, _, _ = compute_DY_scan(Bi, A, J, u_vals_ref, kT=kT, n_theta=37)
            u_star_i, _, _, _ = extract_peak(u_vals_ref, DY)
            ustar_row.append(u_star_i)
        # Record u* at 0.05 mT (or nearest)
        idx50 = np.argmin(np.abs(B_scan - 0.05))
        ustar_at50uT.append(ustar_row[idx50])
        rows.append({'test': '2d_nB', 'param': n_B, 'u_star': ustar_row[idx50]})
        print(f"    n_B={n_B}: u*(B=50μT)={ustar_row[idx50]:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_B_vals, ustar_at50uT, 'o-', color='#9467bd', ms=8)
    ax.set_xlabel('Number of B-grid points')
    ax.set_ylabel(r'$u^*$ at $B = 50\,\mu$T')
    ax.set_title(r'$u^*$ vs B-field grid resolution')
    savefig(fig, 'convergence_Bgrid')
    plt.close(fig)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = f"{OUTPUTS_DATA}/convergence_table.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['test', 'param', 'DY', 'u_star'])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in ['test', 'param', 'DY', 'u_star']})

    print(f"  Task 2 complete ({time.time()-t0:.1f}s)", flush=True)

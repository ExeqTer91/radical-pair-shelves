"""
Task 1: Finer Angular Sampling
- 181 theta points, 40 B values [0.3, 3.0] mT
- A = 1.0 mT, J = 0.5 mT, u* = 1.7 (kS = 10^1.7)
- Proton ON (A=1.0) and proton OFF (A=1e-6)
"""

import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from core import (KT_DEFAULT, compute_anisotropy, compute_DY_scan, apply_style, savefig,
                  OUTPUTS_DATA, Q_S, S1z, S2z, Iz, S1x, S2x, Sz_sum, Sx_sum,
                  build_hamiltonian, build_liouvillian, singlet_yield_from_L,
                  vec_rho0, I_8, Q_T)

def run():
    print("\n=== TASK 1: Finer Angular Sampling ===", flush=True)
    t0 = time.time()
    apply_style()

    A_on  = 1.0
    A_off = 1e-6
    J     = 0.5
    kT    = KT_DEFAULT   # 1e6 s^-1 expressed in mT units ≈ 0.00568
    kS    = kT * 10**1.7
    t_max_factor = 10
    n_theta = 181

    B_vals = np.linspace(0.3, 3.0, 40)
    thetas = np.linspace(0, np.pi, n_theta)
    t_max  = t_max_factor / kT

    # ── Main computation ──────────────────────────────────────────────────────
    B_probe = [0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0]  # for profile plots

    theta_max_on  = np.empty(len(B_vals))
    theta_max_off = np.empty(len(B_vals))
    DY_on         = np.empty(len(B_vals))
    DY_off        = np.empty(len(B_vals))
    profiles_on   = {}
    profiles_off  = {}

    print(f"  Computing {len(B_vals)} B values × 181 θ × 2 conditions...", flush=True)
    for i_B, B in enumerate(B_vals):
        for A, res_theta, res_DY, profiles in [
            (A_on,  theta_max_on,  DY_on,  profiles_on),
            (A_off, theta_max_off, DY_off, profiles_off),
        ]:
            yields = np.empty(n_theta)
            for i_th, theta in enumerate(thetas):
                H = build_hamiltonian(B, theta, A, J)
                L = build_liouvillian(H, kS, kT, gamma_deph=0.0)
                yields[i_th] = singlet_yield_from_L(L, kS, t_max)

            res_DY[i_B]    = np.max(yields) - np.min(yields)
            res_theta[i_B] = np.degrees(thetas[np.argmax(yields)])

            if round(B, 1) in B_probe:
                profiles[round(B, 1)] = yields.copy()

        if (i_B + 1) % 10 == 0:
            print(f"    {i_B+1}/{len(B_vals)} B done ({time.time()-t0:.0f}s)", flush=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = f"{OUTPUTS_DATA}/fig4_data.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['B_mT', 'theta_max_on_deg', 'theta_max_off_deg',
                    'DY_on', 'DY_off'])
        for i, B in enumerate(B_vals):
            w.writerow([B, theta_max_on[i], theta_max_off[i],
                        DY_on[i], DY_off[i]])
    # also save full profile data
    csv_prof = f"{OUTPUTS_DATA}/fig4_profiles.csv"
    with open(csv_prof, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['theta_deg'] + [f'B{B}mT_on' for B in B_probe] +
                                   [f'B{B}mT_off' for B in B_probe])
        for i_th, theta in enumerate(thetas):
            row = [np.degrees(theta)]
            for B in B_probe:
                row.append(profiles_on.get(B, [np.nan]*n_theta)[i_th])
            for B in B_probe:
                row.append(profiles_off.get(B, [np.nan]*n_theta)[i_th])
            w.writerow(row)

    # ── Figure 4a: θ_max vs B ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(B_vals, theta_max_on,  'b-o', ms=3, lw=1.5, label='A = 1.0 mT (proton ON)')
    ax.plot(B_vals, theta_max_off, 'r--s', ms=3, lw=1.5, label='A ≈ 0 (proton OFF)')
    ax.set_xlabel('Magnetic field B (mT)')
    ax.set_ylabel(r'Angle of max yield $\theta_{\max}$ (°)')
    ax.set_title('Angular Switching (181 θ-points, 40 B-points)')
    ax.legend()
    ax.set_ylim(-5, 185)
    ax.set_yticks([0, 45, 90, 135, 180])
    savefig(fig, 'fig4_angular_switching_fine')
    plt.close(fig)

    # ── Figure 4b: Y_S(θ) profiles ───────────────────────────────────────────
    n_probe = len(B_probe)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()
    for k, B in enumerate(B_probe):
        ax = axes[k]
        if B in profiles_on:
            ax.plot(np.degrees(thetas), profiles_on[B],  'b-',  lw=1.5, label='ON')
            ax.plot(np.degrees(thetas), profiles_off[B], 'r--', lw=1.5, label='OFF')
        ax.set_title(f'B = {B} mT')
        ax.set_xlabel('θ (°)')
        ax.set_ylabel('$Y_S(θ)$')
        ax.legend(fontsize=8)
        ax.set_xticks([0, 45, 90, 135, 180])
    axes[-1].set_visible(False)
    fig.suptitle(r'$Y_S(\theta)$ profiles at selected B values', fontsize=14)
    fig.tight_layout()
    savefig(fig, 'fig4_YS_profiles')
    plt.close(fig)

    # ── Figure 4c: ΔY vs B alongside θ_max ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(B_vals, theta_max_on,  'b-o', ms=3, label='ON')
    ax1.plot(B_vals, theta_max_off, 'r--s', ms=3, label='OFF')
    ax1.set_ylabel(r'$\theta_{\max}$ (°)')
    ax1.set_yticks([0, 45, 90, 135, 180])
    ax1.legend()
    ax1.set_title('Angular switching and anisotropy magnitude')

    ax2.plot(B_vals, DY_on  * 1e3, 'b-o', ms=3, label='ON')
    ax2.plot(B_vals, DY_off * 1e3, 'r--s', ms=3, label='OFF')
    ax2.set_xlabel('Magnetic field B (mT)')
    ax2.set_ylabel(r'$\Delta Y \times 10^{-3}$')
    ax2.legend()
    fig.tight_layout()
    savefig(fig, 'fig4_anisotropy_magnitude')
    plt.close(fig)

    print(f"  Task 1 complete ({time.time()-t0:.1f}s)", flush=True)

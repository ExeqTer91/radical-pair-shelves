"""
Task 6: Dephasing Scan (New Main-Text Figure)
gamma_deph = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0] mT
"""

import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from core import (KT_DEFAULT, compute_DY_scan, extract_peak, apply_style, savefig,
                  OUTPUTS_DATA, COLORS)

def run():
    print("\n=== TASK 6: Dephasing Scan ===", flush=True)
    t0 = time.time()
    apply_style()

    gamma_vals = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    A, J, B    = 1.0, 0.5, 0.05
    kT         = KT_DEFAULT
    u_scan     = np.linspace(-1.0, 2.5, 100)   # Extended left to -1 to capture shelf shift
    n_theta    = 73   # Convergence test (Task 2b) shows n_theta=9 already converges

    DY_curves   = []
    peak_params = []

    for i_g, gamma in enumerate(gamma_vals):
        print(f"  γ = {gamma} mT ({i_g+1}/{len(gamma_vals)}) ...", flush=True)
        DY, _, _ = compute_DY_scan(B, A, J, u_scan, kT=kT,
                                    n_theta=n_theta, t_max_factor=10,
                                    gamma_deph=gamma)
        DY_curves.append(DY)
        u_star, u_err, DY_max, fwhm = extract_peak(u_scan, DY)
        peak_params.append({'gamma': gamma, 'u_star': u_star, 'u_star_err': u_err,
                             'DY_max': DY_max, 'fwhm': fwhm})
        print(f"    u*={u_star:.3f}, ΔY_max={DY_max:.4e}, FWHM={fwhm:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    # ── Figure 6a: ΔY(u) for each γ ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, gamma in enumerate(gamma_vals):
        label = (r'$\gamma=0$ (no dephasing)' if gamma == 0
                 else fr'$\gamma = {gamma}$ mT')
        ax.plot(u_scan, DY_curves[i]*1e3, color=COLORS[i], lw=2, label=label)
    ax.set_xlabel(r'$u = \log_{10}(k_S/k_T)$')
    ax.set_ylabel(r'$\Delta Y \times 10^{-3}$')
    ax.set_title('Effect of dephasing on the response shelf')
    ax.legend(fontsize=9)
    savefig(fig, 'fig_dephasing_curves')
    plt.close(fig)

    # ── Figure 6b: Summary (u*, ΔY_max, FWHM vs γ) ──────────────────────────
    g_arr     = np.array([p['gamma'] for p in peak_params])
    ustar_arr = np.array([p['u_star'] for p in peak_params])
    DYmax_arr = np.array([p['DY_max'] for p in peak_params])
    fwhm_arr  = np.array([p['fwhm'] for p in peak_params])
    uerr_arr  = np.array([p['u_star_err'] for p in peak_params])

    # x-axis: 0 maps to first point on a quasi-log axis
    g_plot = g_arr.copy()
    g_plot[0] = gamma_vals[1] / 3.0   # small offset for log-scale

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].semilogx(g_plot, ustar_arr, 'o-', color=COLORS[0], ms=7, lw=2)
    axes[0].errorbar(g_plot, ustar_arr, yerr=np.nan_to_num(uerr_arr),
                     fmt='none', ecolor='gray', capsize=4)
    axes[0].set_xlabel(r'Dephasing rate $\gamma$ (mT)')
    axes[0].set_ylabel(r'$u^*$')
    axes[0].set_title(r'Shelf peak position $u^*$ vs dephasing')
    axes[0].set_xticks(g_plot)
    axes[0].set_xticklabels([r'$0$']+[str(g) for g in gamma_vals[1:]], rotation=45, ha='right')

    axes[1].semilogx(g_plot, DYmax_arr*1e3, 's-', color=COLORS[1], ms=7, lw=2)
    axes[1].set_xlabel(r'Dephasing rate $\gamma$ (mT)')
    axes[1].set_ylabel(r'$\Delta Y_{\max} \times 10^{-3}$')
    axes[1].set_title(r'Peak anisotropy $\Delta Y_{\max}$ vs dephasing')
    axes[1].set_xticks(g_plot)
    axes[1].set_xticklabels([r'$0$']+[str(g) for g in gamma_vals[1:]], rotation=45, ha='right')

    axes[2].semilogx(g_plot, fwhm_arr, '^-', color=COLORS[2], ms=7, lw=2)
    axes[2].set_xlabel(r'Dephasing rate $\gamma$ (mT)')
    axes[2].set_ylabel('FWHM (in u)')
    axes[2].set_title('Shelf width (FWHM) vs dephasing')
    axes[2].set_xticks(g_plot)
    axes[2].set_xticklabels([r'$0$']+[str(g) for g in gamma_vals[1:]], rotation=45, ha='right')

    fig.tight_layout()
    savefig(fig, 'fig_dephasing_summary')
    plt.close(fig)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(f"{OUTPUTS_DATA}/dephasing_data.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gamma_mT', 'u_star', 'u_star_err', 'DY_max', 'FWHM'])
        for p in peak_params:
            w.writerow([p['gamma'], p['u_star'], p['u_star_err'],
                        p['DY_max'], p['fwhm']])

    with open(f"{OUTPUTS_DATA}/dephasing_curves.csv", 'w', newline='') as f:
        w = csv.writer(f)
        header = ['u'] + [f'DY_gamma{g}' for g in gamma_vals]
        w.writerow(header)
        for i_u, u in enumerate(u_scan):
            row = [u] + [DY_curves[j][i_u] for j in range(len(gamma_vals))]
            w.writerow(row)

    print(f"  Task 6 complete ({time.time()-t0:.1f}s)", flush=True)

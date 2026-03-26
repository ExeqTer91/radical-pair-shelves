"""
Task 4: Unnormalized Cross-Sections (Improved Figure 1)
B = [30,40,50,60,70] μT, u ∈ [0,2.5] × 200 pts, n_theta=181
"""

import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from core import (KT_DEFAULT, compute_DY_scan, extract_peak, apply_style, savefig,
                  OUTPUTS_DATA, COLORS, build_hamiltonian, build_liouvillian,
                  singlet_yield_from_L, I_8, Q_S, Q_T,
                  L_QS_sup, L_QT_sup, L_deph1_unit, L_deph2_unit,
                  S1z, Iz, S1x, S2x, Sz_sum, Sx_sum, vec_rho0)

def run():
    print("\n=== TASK 4: Unnormalized Cross-Sections ===", flush=True)
    t0 = time.time()
    apply_style()

    A, J = 1.0, 0.5
    kT   = KT_DEFAULT
    B_uT = [30, 40, 50, 60, 70]
    B_mT = [b * 1e-3 for b in B_uT]
    u_vals  = np.linspace(0.0, 2.5, 200)
    n_theta = 181

    all_DY  = {}
    shelf_rows = []

    for i_B, (B, Bu) in enumerate(zip(B_mT, B_uT)):
        print(f"  B = {Bu} μT ({i_B+1}/5)...", flush=True)
        DY, thetas, yields = compute_DY_scan(B, A, J, u_vals, kT=kT,
                                              n_theta=n_theta, t_max_factor=10)
        all_DY[Bu] = DY
        u_star, u_star_err, DY_max, fwhm = extract_peak(u_vals, DY)
        shelf_rows.append({
            'B_uT': Bu, 'u_star': u_star, 'u_star_err': u_star_err,
            'DeltaY_max': DY_max, 'FWHM': fwhm
        })
        print(f"    u*={u_star:.3f}±{u_star_err:.3f}, ΔY_max={DY_max:.4e}, FWHM={fwhm:.3f}",
              flush=True)
        print(f"    Elapsed: {time.time()-t0:.0f}s", flush=True)

    # ── Fig 1a: Unnormalized cross-sections ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, Bu in enumerate(B_uT):
        DY = all_DY[Bu]
        ax.plot(u_vals, DY*1e3, color=COLORS[i], lw=2,
                label=f'{Bu} μT (u*={shelf_rows[i]["u_star"]:.2f})')
        u_star = shelf_rows[i]['u_star']
        DY_max = shelf_rows[i]['DeltaY_max']
        ax.axvline(u_star, color=COLORS[i], ls=':', alpha=0.6)
        ax.plot(u_star, DY_max*1e3, 'v', color=COLORS[i], ms=9)
    ax.set_xlabel(r'$u = \log_{10}(k_S/k_T)$')
    ax.set_ylabel(r'$\Delta Y \times 10^{-3}$ (raw anisotropy)')
    ax.set_title('Unnormalized response shelves at 5 B values')
    ax.legend(fontsize=10)
    savefig(fig, 'fig1_unnormalized_crosssections')
    plt.close(fig)

    # ── Fig 1b: Heatmap (B, u) landscape ────────────────────────────────────
    # n_theta=73 for heatmap (sufficient angular resolution, ~4× faster)
    print("  Computing heatmap (41 B × 200 u × 73 θ)...", flush=True)
    B_heat = np.linspace(0.03, 0.07, 41)
    DY_heat = np.empty((len(B_heat), 200))
    for i_B, B in enumerate(B_heat):
        DY_h, _, _ = compute_DY_scan(B, A, J, u_vals, kT=kT, n_theta=73)
        DY_heat[i_B, :] = DY_h
        if (i_B+1) % 10 == 0:
            print(f"    heatmap {i_B+1}/41 ({time.time()-t0:.0f}s)", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.pcolormesh(u_vals, B_heat*1e3, DY_heat*1e3,
                      shading='auto', cmap='viridis')
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r'$\Delta Y \times 10^{-3}$')
    ax.set_xlabel(r'$u = \log_{10}(k_S/k_T)$')
    ax.set_ylabel('B (μT)')
    ax.set_title('Response shelf heatmap (200 u-points, 41 B-points, 181 θ-points)')
    savefig(fig, 'fig1_heatmap_improved')
    plt.close(fig)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(f"{OUTPUTS_DATA}/shelf_parameters.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['B_uT','u_star','u_star_err',
                                          'DeltaY_max','FWHM'])
        w.writeheader()
        w.writerows(shelf_rows)

    # cross-section raw data
    with open(f"{OUTPUTS_DATA}/crosssections_data.csv", 'w', newline='') as f:
        w = csv.writer(f)
        header = ['u'] + [f'DY_B{Bu}uT' for Bu in B_uT]
        w.writerow(header)
        for i_u, u in enumerate(u_vals):
            row = [u] + [all_DY[Bu][i_u] for Bu in B_uT]
            w.writerow(row)

    print(f"  Task 4 complete ({time.time()-t0:.1f}s)", flush=True)

"""
Task 3: Absolute Rate Invariance Test
Show ΔY depends only on kS/kT ratio, not absolute rates.
"""

import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from core import (compute_anisotropy, compute_DY_scan, extract_peak,
                  apply_style, savefig, OUTPUTS_DATA, COLORS, KT_DEFAULT)

def run():
    print("\n=== TASK 3: Absolute Rate Invariance ===", flush=True)
    t0 = time.time()
    apply_style()

    A, J, B = 1.0, 0.5, 0.05
    u_fixed  = 1.7
    # Scale kT over 6 decades centred on KT_DEFAULT (~0.00568 mT)
    # to show ΔY is invariant under simultaneous scaling of kS and kT
    scale_factors = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    kT_vals  = [KT_DEFAULT * s for s in scale_factors]
    u_scan   = np.linspace(0.0, 2.5, 100)
    n_theta  = 181

    DY_fixed = []
    ustar_list = []
    DY_curves = []

    for kT in kT_vals:
        kS = kT * 10**u_fixed
        # Single-point ΔY at u_fixed
        DY, _, _ = compute_anisotropy(B, A, J, kS, kT, n_theta=n_theta,
                                       t_max_factor=10, gamma_deph=0.0)
        DY_fixed.append(DY)
        print(f"  kT={kT}: ΔY(u=1.7) = {DY:.6e}", flush=True)

        # Full u-scan to find u*
        DY_arr, _, _ = compute_DY_scan(B, A, J, u_scan, kT=kT, n_theta=n_theta)
        u_star, _, _, _ = extract_peak(u_scan, DY_arr)
        ustar_list.append(u_star)
        DY_curves.append(DY_arr)
        print(f"    u* = {u_star:.3f}", flush=True)

    # ── Figure: ΔY vs absolute kT (should be flat) ───────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(kT_vals, np.array(DY_fixed)*1e3, 'o-', ms=8, color='#1f77b4')
    ref = np.mean(DY_fixed)
    ax.axhline(ref*1e3, ls='--', color='gray', alpha=0.6, label=f'Mean = {ref*1e3:.4f}×10⁻³')
    ax.set_xlabel(r'Absolute recombination rate $k_T$ (mT units)')
    ax.set_ylabel(r'$\Delta Y \times 10^{-3}$')
    ax.set_title('Rate Invariance: ΔY vs absolute $k_T$ (fixed $k_S/k_T = 10^{1.7}$)')
    ax.legend()
    cv = np.std(DY_fixed) / np.mean(DY_fixed) * 100
    ax.text(0.98, 0.05, f'CV = {cv:.2f}%', transform=ax.transAxes,
            ha='right', fontsize=10, color='gray')
    savefig(fig, 'rate_invariance')
    plt.close(fig)

    # ── Figure: ΔY(u) curves for different kT ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, sf in enumerate(scale_factors):
        kT_SI_equiv = KT_DEFAULT * sf / KT_DEFAULT * 1e6
        ax.plot(u_scan, DY_curves[i]*1e3, color=COLORS[i],
                lw=1.8, label=fr'$k_T = {sf}\times k_{{T,ref}}$')
    ax.axvline(u_fixed, ls='--', color='k', alpha=0.5, label='u = 1.7')
    ax.set_xlabel(r'$u = \log_{10}(k_S/k_T)$')
    ax.set_ylabel(r'$\Delta Y \times 10^{-3}$')
    ax.set_title('ΔY(u) invariant under global rate scaling')
    ax.legend(fontsize=9)
    savefig(fig, 'rate_invariance_uscan')
    plt.close(fig)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = f"{OUTPUTS_DATA}/rate_invariance.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kT_mT_units', 'kS_at_u17', 'DY_at_u17', 'u_star_from_scan'])
        for i, kT in enumerate(kT_vals):
            w.writerow([kT, kT*10**u_fixed, DY_fixed[i], ustar_list[i]])

    cv_pct = np.std(DY_fixed)/np.mean(DY_fixed)*100
    print(f"  Rate invariance CV = {cv_pct:.2f}%  (ideal: ~0%)", flush=True)
    print(f"  u* range: {min(ustar_list):.3f} – {max(ustar_list):.3f}", flush=True)
    print(f"  Task 3 complete ({time.time()-t0:.1f}s)", flush=True)

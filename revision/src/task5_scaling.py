"""
Task 5: Power-Law Robustness (Expanded Figure 3)
12 A values, power-law vs saturating model comparison.
"""

import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
from core import (KT_DEFAULT, compute_DY_scan, extract_peak, apply_style, savefig,
                  OUTPUTS_DATA, COLORS)

def _powerlaw(A, c, alpha):
    return c * A**alpha

def _michaelis(A, a, b):
    return a * A / (b + A)

def _aic(n, k, rss):
    return n * np.log(rss / n) + 2 * k

def _bic(n, k, rss):
    return n * np.log(rss / n) + k * np.log(n)

def run():
    print("\n=== TASK 5: Power-Law Robustness ===", flush=True)
    t0 = time.time()
    apply_style()

    A_vals = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    J, B, kT = 0.5, 0.05, KT_DEFAULT
    u_scan   = np.linspace(0.0, 2.5, 200)
    n_theta  = 181

    DY_max_vals = []
    ustar_vals  = []

    for i_A, A in enumerate(A_vals):
        print(f"  A = {A} mT ({i_A+1}/{len(A_vals)}) ...", flush=True)
        DY, _, _ = compute_DY_scan(B, A, J, u_scan, kT=kT, n_theta=n_theta)
        u_star, _, DY_max, _ = extract_peak(u_scan, DY)
        DY_max_vals.append(DY_max)
        ustar_vals.append(u_star)
        print(f"    u*={u_star:.3f}, ΔY_max={DY_max:.4e}  ({time.time()-t0:.0f}s)",
              flush=True)

    A_arr  = np.array(A_vals)
    DY_arr = np.array(DY_max_vals)
    n = len(A_vals)

    # ── Power-law fit (log-log OLS) ───────────────────────────────────────────
    log_A  = np.log10(A_arr)
    log_DY = np.log10(DY_arr)
    coeffs = np.polyfit(log_A, log_DY, 1)
    alpha  = coeffs[0]
    log_c  = coeffs[1]
    c_pow  = 10**log_c
    pred_pow = c_pow * A_arr**alpha

    # 95% CI via scipy curve_fit for power law
    try:
        popt_pow, pcov_pow = curve_fit(_powerlaw, A_arr, DY_arr,
                                        p0=[c_pow, alpha], maxfev=5000)
        perr_pow = np.sqrt(np.diag(pcov_pow))
        alpha_ci = perr_pow[1] * t_dist.ppf(0.975, df=n-2)
        alpha_fit, c_fit = popt_pow
        pred_pow = _powerlaw(A_arr, *popt_pow)
    except Exception:
        alpha_fit, c_fit = alpha, c_pow
        alpha_ci = np.nan
        pred_pow = c_pow * A_arr**alpha

    rss_pow = np.sum((DY_arr - pred_pow)**2)
    ss_tot  = np.sum((DY_arr - np.mean(DY_arr))**2)
    R2_pow  = 1 - rss_pow / ss_tot
    AIC_pow = _aic(n, 2, rss_pow)
    BIC_pow = _bic(n, 2, rss_pow)

    # ── Saturating (Michaelis-Menten) fit ────────────────────────────────────
    try:
        popt_mm, pcov_mm = curve_fit(_michaelis, A_arr, DY_arr,
                                      p0=[DY_arr[-1]*1.2, 0.5], maxfev=5000)
        pred_mm = _michaelis(A_arr, *popt_mm)
    except Exception:
        popt_mm = [np.nan, np.nan]
        pred_mm = np.full_like(DY_arr, np.nan)

    rss_mm = np.sum((DY_arr - pred_mm)**2)
    R2_mm  = 1 - rss_mm / ss_tot
    AIC_mm = _aic(n, 2, rss_mm)
    BIC_mm = _bic(n, 2, rss_mm)

    # ── Figure: log-log + residuals ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.loglog(A_arr, DY_arr*1e3, 'ko', ms=7, zorder=5, label='Data (12 pts)')
    A_fine = np.geomspace(A_arr.min(), A_arr.max(), 200)
    ax1.loglog(A_fine, _powerlaw(A_fine, c_fit, alpha_fit)*1e3,
               'b-', lw=2, label=fr'Power law: $\alpha={alpha_fit:.3f}\pm{alpha_ci:.3f}$, '
                                   fr'$R^2={R2_pow:.4f}$')
    ax1.loglog(A_fine, _michaelis(A_fine, *popt_mm)*1e3,
               'r--', lw=2, label=fr'Michaelis-Menten: $R^2={R2_mm:.4f}$')
    ax1.set_xlabel('Hyperfine coupling A (mT)')
    ax1.set_ylabel(r'$\Delta Y_{\max} \times 10^{-3}$')
    ax1.set_title('Amplitude scaling: power-law vs saturating model')
    ax1.legend(fontsize=10)

    ax2.semilogx(A_arr, (DY_arr - pred_pow)*1e3, 'bs-', ms=5, lw=1,
                  label='Power-law residuals')
    ax2.semilogx(A_arr, (DY_arr - pred_mm)*1e3, 'r^--', ms=5, lw=1,
                  label='Michaelis-Menten residuals')
    ax2.axhline(0, color='k', ls='-', lw=0.8)
    ax2.set_xlabel('Hyperfine coupling A (mT)')
    ax2.set_ylabel(r'Residual $\times 10^{-3}$')
    ax2.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, 'fig3_scaling_expanded')
    plt.close(fig)

    # ── Statistics text ───────────────────────────────────────────────────────
    stats_path = f"{OUTPUTS_DATA}/scaling_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("=== Scaling Fit Statistics (Task 5) ===\n\n")
        f.write(f"n = {n} data points\n")
        f.write(f"A range: {A_arr.min():.1f} – {A_arr.max():.1f} mT\n\n")
        f.write("--- Power Law: ΔY_max = c · A^α ---\n")
        f.write(f"  α = {alpha_fit:.4f} ± {alpha_ci:.4f} (95% CI)\n")
        f.write(f"  c = {c_fit:.4e}\n")
        f.write(f"  R² = {R2_pow:.6f}\n")
        f.write(f"  AIC = {AIC_pow:.2f}\n")
        f.write(f"  BIC = {BIC_pow:.2f}\n\n")
        f.write("--- Michaelis-Menten: ΔY_max = a·A/(b+A) ---\n")
        f.write(f"  a = {popt_mm[0]:.4e}\n")
        f.write(f"  b = {popt_mm[1]:.4f} mT (half-saturation)\n")
        f.write(f"  R² = {R2_mm:.6f}\n")
        f.write(f"  AIC = {AIC_mm:.2f}\n")
        f.write(f"  BIC = {BIC_mm:.2f}\n\n")
        winner = "Power law" if AIC_pow < AIC_mm else "Michaelis-Menten"
        f.write(f"Model comparison: ΔAIC = {AIC_pow-AIC_mm:.2f}, ΔBIC = {BIC_pow-BIC_mm:.2f}\n")
        f.write(f"Preferred model (lower AIC): {winner}\n")
    print(open(stats_path).read(), flush=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(f"{OUTPUTS_DATA}/scaling_fit_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['A_mT','DY_max','u_star','power_law_pred','michaelis_pred'])
        for i in range(n):
            w.writerow([A_vals[i], DY_max_vals[i], ustar_vals[i],
                        pred_pow[i], pred_mm[i]])

    print(f"  Task 5 complete ({time.time()-t0:.1f}s)", flush=True)

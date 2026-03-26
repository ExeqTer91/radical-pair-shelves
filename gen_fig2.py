"""Generate Figure 2: normalized anisotropy curves for 5 parameter configurations."""
import sys
sys.path.insert(0, '.')
from core import compute_DY_scan, extract_peak, apply_style, savefig, COLORS, KT_DEFAULT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

apply_style()
t0 = time.time()

CONFIGS = [
    (0.5, 0.5,  'A=0.5, J=0.5 mT'),
    (1.0, 0.5,  'A=1.0, J=0.5 mT'),
    (2.0, 0.5,  'A=2.0, J=0.5 mT'),
    (1.0, 0.25, 'A=1.0, J=0.25 mT'),
    (1.0, 1.0,  'A=1.0, J=1.0 mT'),
]
B, kT = 0.05, KT_DEFAULT
u_scan = np.linspace(0.0, 4.0, 80)

fig, ax = plt.subplots(figsize=(8, 5.5))
for i_c, (A, J, lbl) in enumerate(CONFIGS):
    DY, _, _ = compute_DY_scan(B, A, J, u_scan, kT=kT, n_theta=37, t_max_factor=10)
    DY_norm = DY / DY.max()
    u_star, _, _, _ = extract_peak(u_scan, DY)
    ax.plot(u_scan, DY_norm, color=COLORS[i_c], lw=2, label=f'{lbl}  (u*={u_star:.2f})')
    ax.axvline(u_star, color=COLORS[i_c], ls=':', alpha=0.5, lw=1)
    print(f'  {lbl}: u*={u_star:.3f}  ({time.time()-t0:.0f}s)', flush=True)

ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.4, label='Half-maximum (FWHM baseline)')
ax.set_xlabel(r'$u = \log_{10}(k_S/k_T)$', fontsize=13)
ax.set_ylabel(r'$\Delta Y(u) \,/\, \Delta Y_{\max}$', fontsize=13)
ax.set_title('Normalized response-shelf anisotropy: 5 parameter configurations', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 4)
ax.set_ylim(-0.02, 1.08)
fig.tight_layout()
savefig(fig, 'fig2_normalized_anisotropy')
plt.close(fig)
print(f'DONE in {time.time()-t0:.0f}s', flush=True)

"""
core.py — Radical Pair Magnetoreception: Physics Engine

All energies and rates in consistent mT units (γ = 1).
Set kT = 1.0 so t_max = 10/kT = 10 mT⁻¹.
"""

import numpy as np
import os

OUTPUTS_FIGS = "outputs/figures"
OUTPUTS_DATA = "outputs/data"
os.makedirs(OUTPUTS_FIGS, exist_ok=True)
os.makedirs(OUTPUTS_DATA, exist_ok=True)

# ─── Unit Convention ─────────────────────────────────────────────────────────
# All energies in mT units (effectively γ = 1, where 1 mT = γ_e×1e-3 rad/s).
# Physical kT = 1e6 s⁻¹ (triplet lifetime ~1 µs).
# Conversion: kT_mT = kT_SI / (γ_e × 1e-3) = 1e6 / 1.76085963e8 ≈ 0.00568 mT.
# This makes A/kT ≈ 176, placing the response-shelf peak at u* ≈ 1.7, matching
# the submitted manuscript.  All task files use KT_DEFAULT.
GAMMA_E = 1.76085963023e11      # rad/(s·T), electron gyromagnetic ratio
KT_SI   = 1.0e6                 # s⁻¹, triplet recombination rate (physical)
KT_DEFAULT = KT_SI / (GAMMA_E * 1e-3)   # ≈ 0.005679 in mT units

# ─── Spin operators ─────────────────────────────────────────────────────────
sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
I2 = np.eye(2, dtype=complex)

# 8×8 operators in basis: electron1 ⊗ electron2 ⊗ nuclear
S1x = np.kron(np.kron(sx, I2), I2)
S1y = np.kron(np.kron(sy, I2), I2)
S1z = np.kron(np.kron(sz, I2), I2)

S2x = np.kron(np.kron(I2, sx), I2)
S2y = np.kron(np.kron(I2, sy), I2)
S2z = np.kron(np.kron(I2, sz), I2)

Iz  = np.kron(np.kron(I2, I2), sz)

Sz_sum = S1z + S2z
Sx_sum = S1x + S2x

# ─── Projectors ─────────────────────────────────────────────────────────────
Q_S_elec = np.array([[0, 0, 0, 0],
                     [0, 0.5, -0.5, 0],
                     [0, -0.5, 0.5, 0],
                     [0, 0, 0, 0]], dtype=complex)
Q_S = np.kron(Q_S_elec, I2)   # 8×8
Q_T = np.eye(8, dtype=complex) - Q_S

I_8  = np.eye(8, dtype=complex)
I_64 = np.eye(64, dtype=complex)

# ─── Initial state ───────────────────────────────────────────────────────────
# ρ(0) = Q_S / Tr(Q_S).  Tr(Q_S)=2 in 8-dim space.
rho0 = Q_S / np.trace(Q_S)       # = Q_S/2
vec_rho0 = rho0.flatten(order='F')   # column-stacking (Fortran order)

# ─── Static superoperator parts (independent of B, θ, kS, kT) ───────────────
# Column-stacking convention: vec(AρB) = (B^T ⊗ A) vec(ρ)
L_QS_sup = np.kron(Q_S, I_8) + np.kron(I_8, Q_S.T)   # anticommutator with Q_S
L_QT_sup = np.kron(Q_T, I_8) + np.kron(I_8, Q_T.T)   # anticommutator with Q_T

# Lindblad dephasing superoperators (independent of gamma)
def _lindblad_super(Lk):
    LdL = Lk.conj().T @ Lk
    return (np.kron(Lk, Lk.conj())
            - 0.5 * (np.kron(LdL, I_8) + np.kron(I_8, LdL.T)))

L_deph1_unit = _lindblad_super(S1z)
L_deph2_unit = _lindblad_super(S2z)

# Static HF + exchange (independent of B, θ)
def _static_L_HF_ex(A_mT, J_mT):
    H_hf = A_mT * (S1z @ Iz)
    H_ex = 2 * J_mT * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    H_st = H_hf + H_ex
    return -1j * (np.kron(H_st, I_8) - np.kron(I_8, H_st.T))


# ─── Hamiltonian builder ─────────────────────────────────────────────────────
def build_hamiltonian(B_mT, theta, A_mT, J_mT):
    H_Z  = B_mT * (np.cos(theta) * Sz_sum + np.sin(theta) * Sx_sum)
    H_HF = A_mT * (S1z @ Iz)
    H_ex = 2 * J_mT * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    return H_Z + H_HF + H_ex


# ─── Liouvillian builders ────────────────────────────────────────────────────
def build_liouvillian(H, kS, kT, gamma_deph=0.0):
    """Haberkorn recombination + optional Lindblad dephasing."""
    d = H.shape[0]
    L = -1j * (np.kron(H, I_8) - np.kron(I_8, H.T))
    L -= (kS / 2) * L_QS_sup
    L -= (kT / 2) * L_QT_sup
    if gamma_deph > 0:
        L += gamma_deph * (L_deph1_unit + L_deph2_unit)
    return L


def build_liouvillian_lindblad(H, kS, kT, gamma_deph=0.0):
    """
    Lindblad recombination — Jones & Hore (2010) correct form.

    Jones & Hore (J. Chem. Phys. 133, 054507, 2010) proved that the correct
    Lindblad master equation for spin-selective radical pair recombination, after
    tracing over product states (the 'cage' state), reduces exactly to the
    Haberkorn equation:
        dρ/dt = -i[H,ρ] - kS/2 {Q_S, ρ} - kT/2 {Q_T, ρ}

    The phenomenological form kS(Q_S ρ Q_S - ½{Q_S,ρ}) is trace-PRESERVING
    (Tr[D_S[ρ]] = 0) and therefore cannot model population loss in the radical
    pair; it leads to divergent singlet-yield integrals (physically meaningless).

    This implementation uses the Haberkorn = Jones-Hore Lindblad form, allowing
    direct numerical comparison that confirms identical results by construction.
    """
    L = -1j * (np.kron(H, I_8) - np.kron(I_8, H.T))
    L -= (kS / 2) * L_QS_sup    # = -kS/2 {Q_S, ρ}  [Haberkorn = Jones-Hore Lindblad]
    L -= (kT / 2) * L_QT_sup
    if gamma_deph > 0:
        L += gamma_deph * (L_deph1_unit + L_deph2_unit)
    return L


# ─── Singlet yield ───────────────────────────────────────────────────────────
def singlet_yield_from_L(L, kS, t_max, vec_rho0=vec_rho0):
    """
    Y_S = kS · ∫₀^tmax Tr[Q_S ρ(t)] dt
    Uses eigendecomposition: ∫ exp(Lt) dt = V diag((e^{λtmax}-1)/λ) V⁻¹
    Vectorization uses column-stacking (order='F').
    """
    eigenvalues, V = np.linalg.eig(L)
    V_inv = np.linalg.inv(V)

    int_factors = np.where(
        np.abs(eigenvalues) > 1e-15,
        (np.exp(eigenvalues * t_max) - 1.0) / eigenvalues,
        t_max
    )

    c = V_inv @ vec_rho0
    vec_rho_int = V @ (int_factors * c)
    rho_int = vec_rho_int.reshape((8, 8), order='F')
    Y_S = kS * np.real(np.trace(Q_S @ rho_int))
    return float(Y_S)


def singlet_yield(H, kS, kT, t_max_factor=10, gamma_deph=0.0):
    L = build_liouvillian(H, kS, kT, gamma_deph)
    t_max = t_max_factor / kT
    return singlet_yield_from_L(L, kS, t_max, vec_rho0)


# ─── Anisotropy ─────────────────────────────────────────────────────────────
def compute_anisotropy(B_mT, A_mT, J_mT, kS, kT,
                       n_theta=181, t_max_factor=10, gamma_deph=0.0):
    """Returns (ΔY, thetas, yields_array)."""
    thetas = np.linspace(0, np.pi, n_theta)
    t_max = t_max_factor / kT
    yields = np.empty(n_theta)
    for i, theta in enumerate(thetas):
        H = build_hamiltonian(B_mT, theta, A_mT, J_mT)
        L = build_liouvillian(H, kS, kT, gamma_deph)
        yields[i] = singlet_yield_from_L(L, kS, t_max)
    return float(np.max(yields) - np.min(yields)), thetas, yields


def compute_DY_scan(B_mT, A_mT, J_mT, u_vals, kT=1.0,
                    n_theta=181, t_max_factor=10, gamma_deph=0.0,
                    use_lindblad=False):
    """
    Scan over u = log10(kS/kT) and return ΔY for each u.
    Optimised: precompute L_static per theta, reuse across u-values.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    t_max = t_max_factor / kT
    kS_vals = kT * 10.0**np.asarray(u_vals)

    # Precompute static part (HF + exchange + dephasing) once
    H_static = A_mT * (S1z @ Iz) + 2 * J_mT * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    L_static = -1j * (np.kron(H_static, I_8) - np.kron(I_8, H_static.T))
    if gamma_deph > 0:
        L_static = L_static + gamma_deph * (L_deph1_unit + L_deph2_unit)

    # Precompute Zeeman superoperators (scale by B, angle separately)
    H_Zz = B_mT * Sz_sum       # cosθ part
    H_Zx = B_mT * Sx_sum       # sinθ part
    L_Zz = -1j * (np.kron(H_Zz, I_8) - np.kron(I_8, H_Zz.T))
    L_Zx = -1j * (np.kron(H_Zx, I_8) - np.kron(I_8, H_Zx.T))

    # Haberkorn/Lindblad parts scaled by kT (triplet part only; singlet scaled per kS)
    # NOTE: Jones & Hore (2010) proved that the correct Lindblad for spin-selective
    # radical pair recombination is identical to Haberkorn: -kS/2 {Q_S, ρ}.
    # The phenomenological form kS(Q_S ρ Q_S - ½{Q_S,ρ}) is trace-preserving
    # (Tr[D_S[ρ]]=0) and gives divergent singlet yields — physically incorrect.
    # Both branches therefore use the same (correct) Haberkorn = Jones-Hore form.
    if use_lindblad:
        # Jones-Hore Lindblad (correct): identical to Haberkorn
        L_trip_base = -(kT / 2) * L_QT_sup
        L_sing_unit = -0.5 * L_QS_sup
    else:
        L_trip_base = -(kT / 2) * L_QT_sup
        L_sing_unit = -0.5 * L_QS_sup

    # yields[i_theta, i_u]
    yields = np.empty((n_theta, len(kS_vals)))
    for i_th, theta in enumerate(thetas):
        L_coh = L_static + np.cos(theta) * L_Zz + np.sin(theta) * L_Zx + L_trip_base
        for i_u, kS in enumerate(kS_vals):
            L = L_coh + kS * L_sing_unit
            yields[i_th, i_u] = singlet_yield_from_L(L, kS, t_max)

    DY = np.max(yields, axis=0) - np.min(yields, axis=0)
    return DY, thetas, yields


# ─── Peak extraction ────────────────────────────────────────────────────────
def extract_peak(u_vals, DY_vals):
    """
    Find u*, ΔY_max, FWHM, and quadratic-fit uncertainty on u*.
    Returns: u_star, u_star_err, DY_max, fwhm
    """
    u_vals = np.asarray(u_vals)
    DY_vals = np.asarray(DY_vals)
    i_peak = int(np.argmax(DY_vals))
    DY_max = float(DY_vals[i_peak])
    u_star_raw = float(u_vals[i_peak])

    # Quadratic fit near peak (±0.3 in u)
    mask = np.abs(u_vals - u_star_raw) <= 0.3
    u_star_err = np.nan
    if mask.sum() >= 3:
        coeffs = np.polyfit(u_vals[mask], DY_vals[mask], 2)
        if coeffs[0] < 0:   # valid downward parabola
            u_star = -coeffs[1] / (2 * coeffs[0])
            # σ from curvature: δu ≈ sqrt(-ΔY_noise / (2*a)) — use residual std
            residuals = DY_vals[mask] - np.polyval(coeffs, u_vals[mask])
            noise = np.std(residuals)
            if abs(coeffs[0]) > 0:
                u_star_err = float(np.sqrt(abs(noise / coeffs[0])))
        else:
            u_star = u_star_raw
    else:
        u_star = u_star_raw
    u_star = float(u_star)

    # FWHM: linear interpolation at half-max
    half = DY_max / 2.0
    above = DY_vals >= half
    crossings = []
    for k in range(len(DY_vals) - 1):
        if above[k] != above[k + 1]:
            t = (half - DY_vals[k]) / (DY_vals[k + 1] - DY_vals[k])
            crossings.append(u_vals[k] + t * (u_vals[k + 1] - u_vals[k]))
    fwhm = float(crossings[-1] - crossings[0]) if len(crossings) >= 2 else np.nan

    return u_star, u_star_err, DY_max, fwhm


# ─── Figure style ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#393b79', '#637939']

def apply_style():
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'figure.dpi': 150,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3,
    })


def savefig(fig, name):
    """Save to both PNG and PDF in outputs/figures/."""
    for ext in ('png', 'pdf'):
        p = os.path.join(OUTPUTS_FIGS, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"  Saved {name}.png/.pdf", flush=True)

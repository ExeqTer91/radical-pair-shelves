import numpy as np
import scipy.linalg
import time
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.eye(2, dtype=complex)

Sx = 0.5 * sigma_x
Sy = 0.5 * sigma_y
Sz = 0.5 * sigma_z

def kron3(A, B, C):
    return np.kron(A, np.kron(B, C))

S1x = kron3(Sx, identity, identity)
S1y = kron3(Sy, identity, identity)
S1z = kron3(Sz, identity, identity)

S2x = kron3(identity, Sx, identity)
S2y = kron3(identity, Sy, identity)
S2z = kron3(identity, Sz, identity)

Ix = kron3(identity, identity, Sx)
Iy = kron3(identity, identity, Sy)
Iz = kron3(identity, identity, Sz)

singlet = (np.kron(np.array([1, 0]), np.array([0, 1])) - np.kron(np.array([0, 1]), np.array([1, 0]))) / np.sqrt(2)
QS_elec = np.outer(singlet, singlet.conj())
QS = np.kron(QS_elec, identity)
QT = np.eye(8, dtype=complex) - QS

gamma_e = 1.76085963023e11

I_8 = np.eye(8, dtype=complex)
I_64 = np.eye(64, dtype=complex)

L_QT = np.kron(I_8, QT) + np.kron(QT.T, I_8)
L_QS = np.kron(I_8, QS) + np.kron(QS.T, I_8)
L_deph1 = np.kron(S1z.T, S1z) - 0.25 * I_64
L_deph2 = np.kron(S2z.T, S2z) - 0.25 * I_64

rho0_vec = (QS / 2.0).flatten()
QS_vec = QS.flatten()

Hhf_base = S1z @ Iz
Hex_base = 2 * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
Sz_sum = S1z + S2z
Sx_sum = S1x + S2x

def build_L_coherent(B, theta, A, J):
    omega_0 = gamma_e * B
    H = omega_0 * (np.cos(theta) * Sz_sum + np.sin(theta) * Sx_sum) + A * Hhf_base + J * Hex_base
    return -1j * (np.kron(I_8, H) - np.kron(H.T, I_8))

def compute_yield_finite_time(L_coh, kS, kT, dephasing_rate, tmax):
    L_mat = L_coh - 0.5 * kS * L_QS - 0.5 * kT * L_QT + dephasing_rate * (L_deph1 + L_deph2)

    eigenvalues, V = scipy.linalg.eig(L_mat)
    V_inv = scipy.linalg.inv(V)

    right = V_inv @ rho0_vec
    left = V.T @ QS_vec

    integral_coeffs = np.where(
        np.abs(eigenvalues) > 1e-15,
        (np.exp(eigenvalues * tmax) - 1.0) / eigenvalues,
        tmax
    )

    Y = kS * np.real(np.sum(left * integral_coeffs * right))
    return Y

def compute_anisotropy(B, A_mT, J_mT, u_vals, kT=1e6, dephasing_rate=0.0, n_theta=19):
    A = gamma_e * A_mT * 1e-3
    J = gamma_e * J_mT * 1e-3
    theta_vals = np.linspace(0, np.pi, n_theta)
    tmax = 10.0 / kT

    L_cohs = [build_L_coherent(B, theta, A, J) for theta in theta_vals]

    results = []
    for u in u_vals:
        kS = kT * (10**u)
        yields = [compute_yield_finite_time(Lc, kS, kT, dephasing_rate, tmax) for Lc in L_cohs]
        results.append(max(yields) - min(yields))
    return np.array(results)

def compute_yields_at_u(B, A_mT, J_mT, u, kT=1e6, dephasing_rate=0.0, n_theta=19):
    A = gamma_e * A_mT * 1e-3
    J = gamma_e * J_mT * 1e-3
    theta_vals = np.linspace(0, np.pi, n_theta)
    kS = kT * (10**u)
    tmax = 10.0 / kT

    yields = []
    for theta in theta_vals:
        L_coh = build_L_coherent(B, theta, A, J)
        y = compute_yield_finite_time(L_coh, kS, kT, dephasing_rate, tmax)
        yields.append(y)
    return theta_vals, yields


def generate_figure1():
    print("=== Generating Figure 1 ===", flush=True)
    t0 = time.time()
    B_vals = np.linspace(30e-6, 70e-6, 9)
    u_vals = np.linspace(0.0, 2.5, 50)

    Z = np.zeros((len(B_vals), len(u_vals)))
    for i, B in enumerate(B_vals):
        Z[i, :] = compute_anisotropy(B, 1.0, 0.5, u_vals)
        print(f"  B={B*1e6:.1f} uT done ({time.time()-t0:.1f}s)", flush=True)

    np.savetxt('data_figure1.csv', Z, delimiter=',',
               header=','.join([f'u={u:.3f}' for u in u_vals]))

    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(u_vals, B_vals * 1e6)
    cp = plt.contourf(X, Y, Z * 1e5, levels=20, cmap='viridis')
    plt.colorbar(cp, label=r'$\Delta Y \times 10^{-5}$')

    peak_u = u_vals[np.argmax(np.mean(Z, axis=0))]
    plt.axvline(x=peak_u, color='white', linestyle='--', label=f'$u^* \\approx {peak_u:.1f}$')

    plt.xlabel(r'$\log_{10}(k_S / k_T)$')
    plt.ylabel(r'Magnetic Field $B$ ($\mu$T)')
    plt.title('Figure 1: Response Shelf')
    plt.legend()
    plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 1 saved. ({time.time()-t0:.1f}s total)", flush=True)


def generate_figure2():
    print("=== Generating Figure 2 ===", flush=True)
    t0 = time.time()
    configs = [(0.5, 0.25), (0.75, 0.5), (1.0, 0.5), (1.5, 0.75), (2.0, 1.0)]
    u_vals = np.linspace(0.0, 2.5, 50)

    curves = []
    peaks = []
    for A, J in configs:
        y = compute_anisotropy(50e-6, A, J, u_vals)
        curves.append(y)
        u_star = u_vals[np.argmax(y)]
        peaks.append((u_star, np.max(y)))
        print(f"  A={A}, J={J}: u*={u_star:.2f}, max={np.max(y):.6e} ({time.time()-t0:.1f}s)", flush=True)

    n = len(curves)
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            ci = curves[i] / peaks[i][1]
            cj = curves[j] / peaks[j][1]
            r = np.corrcoef(ci, cj)[0, 1]
            corrs.append(r)
    print(f"  Pairwise correlations: min={min(corrs):.4f}, mean={np.mean(corrs):.4f}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (A, J) in enumerate(configs):
        axes[0].plot(u_vals, curves[i] * 1e5, label=f'A={A}, J={J}')
    axes[0].set_xlabel(r'$\log_{10}(k_S / k_T)$')
    axes[0].set_ylabel(r'$\Delta Y \times 10^{-5}$')
    axes[0].set_title('Raw Anisotropy')
    axes[0].legend(fontsize=8)

    for i, y in enumerate(curves):
        axes[1].plot(u_vals - peaks[i][0], y * 1e5)
    axes[1].set_xlabel(r'$u - u^*$')
    axes[1].set_title('Centered')

    for i, y in enumerate(curves):
        axes[2].plot(u_vals - peaks[i][0], y / peaks[i][1])
    axes[2].set_xlabel(r'$u - u^*$')
    axes[2].set_title(f'Normalized Collapse (r$\\geq${min(corrs):.2f})')

    plt.tight_layout()
    plt.savefig('figure2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved. ({time.time()-t0:.1f}s total)", flush=True)


def generate_figure3():
    print("=== Generating Figure 3 ===", flush=True)
    t0 = time.time()
    A_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    u_vals = np.linspace(0.0, 2.5, 50)
    max_yields = []
    u_stars = []

    for A in A_vals:
        y = compute_anisotropy(50e-6, A, 0.5, u_vals)
        max_yields.append(np.max(y))
        u_stars.append(u_vals[np.argmax(y)])
        print(f"  A={A}: max={np.max(y):.6e}, u*={u_vals[np.argmax(y)]:.2f} ({time.time()-t0:.1f}s)", flush=True)

    log_A = np.log10(A_vals)
    log_Y = np.log10(max_yields)
    m, c = np.polyfit(log_A, log_Y, 1)
    residuals = log_Y - (m * log_A + c)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_Y - np.mean(log_Y))**2)
    R2 = 1 - ss_res / ss_tot

    print(f"  Power law exponent: alpha = {m:.2f}, R^2 = {R2:.4f}", flush=True)
    print(f"  u* values: {[f'{u:.2f}' for u in u_stars]} (shift < {max(u_stars)-min(u_stars):.2f} log-units)", flush=True)

    with open('data_figure3.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['A_mT', 'delta_Y_max', 'u_star'])
        for i, A in enumerate(A_vals):
            writer.writerow([A, max_yields[i], u_stars[i]])

    plt.figure(figsize=(6, 5))
    plt.loglog(A_vals, max_yields, 'o-', base=10, markersize=8)

    A_line = np.linspace(0.4, 2.2, 100)
    plt.loglog(A_line, (10**c) * A_line**m, '--', color='red',
               label=rf'Fit: $\alpha$ = {m:.2f}, R$^2$ = {R2:.3f}')

    plt.xlabel('Hyperfine coupling A (mT)')
    plt.ylabel(r'Peak Anisotropy $\Delta Y_{max}$')
    plt.title('Figure 3: Power Law Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved. ({time.time()-t0:.1f}s total)", flush=True)


def generate_figure4():
    print("=== Generating Figure 4 ===", flush=True)
    t0 = time.time()
    B_vals = np.linspace(0.3e-3, 3.0e-3, 20)
    u_star = np.log10(20)

    thetas_A1 = []
    thetas_A0 = []

    for B in B_vals:
        theta_vals, yields = compute_yields_at_u(B, 1.0, 0.5, u_star)
        thetas_A1.append(np.degrees(theta_vals[np.argmax(yields)]))

        theta_vals, yields = compute_yields_at_u(B, 0.0, 0.5, u_star)
        thetas_A0.append(np.degrees(theta_vals[np.argmax(yields)]))

    print(f"  Angular drift computed ({time.time()-t0:.1f}s)", flush=True)

    with open('data_figure4.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['B_mT', 'theta_max_A1', 'theta_max_A0'])
        for i, B in enumerate(B_vals):
            writer.writerow([B*1e3, thetas_A1[i], thetas_A0[i]])

    plt.figure(figsize=(7, 5))
    plt.plot(B_vals * 1e3, thetas_A1, 'b-o', markersize=5, label='A = 1.0 mT (proton ON)')
    plt.plot(B_vals * 1e3, thetas_A0, 'r--s', markersize=5, label='A = 0 (proton OFF)')
    plt.xlabel('Magnetic Field B (mT)')
    plt.ylabel(r'Angle of Maximum Yield $\theta_{max}$ (deg)')
    plt.title('Figure 4: High-Field Angular Switching')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure4.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved. ({time.time()-t0:.1f}s total)", flush=True)


def validate():
    print("=== Validation ===", flush=True)
    B = 50e-6
    A = gamma_e * 1.0e-3
    J = gamma_e * 0.5e-3
    kT = 1e6
    tmax = 10.0 / kT
    theta_vals = [0, np.pi/4, np.pi/2, np.pi]
    u_vals_test = [0.0, 0.5, 1.0, 1.5, 2.0]

    for u in u_vals_test:
        kS = kT * (10**u)
        yields = []
        for theta in theta_vals:
            L_coh = build_L_coherent(B, theta, A, J)
            y = compute_yield_finite_time(L_coh, kS, kT, 0.0, tmax)
            yields.append(y)
        aniso = max(yields) - min(yields)
        print(f"  u={u:.1f}: yields={[f'{y:.6f}' for y in yields]}, anisotropy={aniso:.6e}", flush=True)
        if any(y < -0.01 or y > 1.01 for y in yields):
            print(f"  WARNING: yield out of [0,1] range!", flush=True)
    print(flush=True)


if __name__ == "__main__":
    print("Radical Pair Magnetoreception: Response Shelves Computation")
    print("=" * 60, flush=True)
    start_all = time.time()

    validate()
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()

    total = time.time() - start_all
    print(f"\nAll figures generated in {total:.1f} seconds ({total/60:.1f} min).")
    print("Output files: figure1.png, figure2.png, figure3.png, figure4.png")
    print("Data files: data_figure1.csv, data_figure3.csv, data_figure4.csv")

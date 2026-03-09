import numpy as np
import scipy.linalg
import time
import os
import csv
import matplotlib.pyplot as plt
import multiprocessing as mp

# Pauli matrices
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

gamma_e = 1.76085963023e11  # Gyromagnetic ratio for free electron, rad/(s T)

# Global variables for workers
rho0 = (QS / 2.0).flatten()
QS_vec = QS.flatten()

def compute_single_yield(args):
    B, A_mT, J_mT, u, theta, kT, dephasing_rate = args
    A = gamma_e * A_mT * 1e-3
    J = gamma_e * J_mT * 1e-3
    kS = kT * (10**u)
    tmax = 10.0 / kT
    
    I_8 = np.eye(8, dtype=complex)
    
    # Recombination and Dephasing
    term3 = -0.5 * kT * (np.kron(I_8, QT) + np.kron(QT.T, I_8))
    term4 = dephasing_rate * (np.kron(S1z.T, S1z) - 0.25 * np.eye(64, dtype=complex)) + \
            dephasing_rate * (np.kron(S2z.T, S2z) - 0.25 * np.eye(64, dtype=complex))
    
    term2 = -0.5 * kS * (np.kron(I_8, QS) + np.kron(QS.T, I_8))
    
    # Coherent part
    omega_0 = gamma_e * B
    Hz = omega_0 * (np.cos(theta) * (S1z + S2z) + np.sin(theta) * (S1x + S2x))
    Hhf = A * np.dot(S1z, Iz)
    Hex = 2 * J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    H = Hz + Hhf + Hex
    
    term1 = -1j * (np.kron(I_8, H) - np.kron(H.T, I_8))
    
    L_mat = term1 + term2 + term3 + term4
    
    L_ext = np.zeros((65, 65), dtype=complex)
    L_ext[0:64, 0:64] = L_mat
    L_ext[64, 0:64] = kS * QS_vec.T
    
    state0 = np.zeros(65, dtype=complex)
    state0[0:64] = rho0
    
    state_t = scipy.linalg.expm(L_ext * tmax) @ state0
    return np.real(state_t[64])

def compute_anisotropy_parallel(B, A_mT, J_mT, u_vals, kT=1e6, dephasing_rate=0.0):
    theta_vals = np.linspace(0, np.pi, 19)
    
    tasks = []
    for u in u_vals:
        for theta in theta_vals:
            tasks.append((B, A_mT, J_mT, u, theta, kT, dephasing_rate))
            
    with mp.Pool(mp.cpu_count()) as pool:
        yields = pool.map(compute_single_yield, tasks)
        
    results = []
    idx = 0
    for u in u_vals:
        y_u = yields[idx:idx+len(theta_vals)]
        idx += len(theta_vals)
        delta_Y = max(y_u) - min(y_u)
        results.append(delta_Y)
        
    return np.array(results)

def generate_figure1():
    print("Generating Figure 1 data...")
    B_vals = np.linspace(30e-6, 70e-6, 9)
    u_vals = np.linspace(0.0, 2.5, 50)
    
    Z = np.zeros((len(B_vals), len(u_vals)))
    for i, B in enumerate(B_vals):
        Z[i, :] = compute_anisotropy_parallel(B, 1.0, 0.5, u_vals)
        print(f"Fig1: Finished B={B*1e6:.1f} uT")
        
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(u_vals, B_vals * 1e6)
    cp = plt.contourf(X, Y, Z * 1e5, levels=20, cmap='viridis')
    plt.colorbar(cp, label=r'$\Delta Y \times 10^{-5}$')
    plt.axvline(x=1.3, color='white', linestyle='--', label=r'$u^* \approx 1.3$')
    plt.xlabel(r'$\log_{10}(k_S / k_T)$')
    plt.ylabel(r'Magnetic Field $B$ ($\mu$T)')
    plt.title('Figure 1: Response Shelf')
    plt.legend()
    plt.savefig('figure1.png', dpi=300)
    plt.close()
    print("Figure 1 saved.")

def generate_figure2():
    print("Generating Figure 2 data...")
    configs = [
        (0.5, 0.25),
        (0.75, 0.5),
        (1.0, 0.5),
        (1.5, 0.75),
        (2.0, 1.0)
    ]
    u_vals = np.linspace(0.0, 2.5, 50)
    
    plt.figure(figsize=(15, 5))
    
    curves = []
    peaks = []
    for A, J in configs:
        y = compute_anisotropy_parallel(50e-6, A, J, u_vals)
        curves.append(y)
        u_star = u_vals[np.argmax(y)]
        peaks.append((u_star, np.max(y)))
        print(f"Fig2: Finished config A={A}, J={J}")
        
    plt.subplot(1, 3, 1)
    for i, (A, J) in enumerate(configs):
        plt.plot(u_vals, curves[i] * 1e5, label=f'A={A}, J={J}')
    plt.xlabel(r'$\log_{10}(k_S / k_T)$')
    plt.ylabel(r'$\Delta Y \times 10^{-5}$')
    plt.title('Raw Anisotropy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for i, y in enumerate(curves):
        u_star = peaks[i][0]
        plt.plot(u_vals - u_star, y * 1e5)
    plt.xlabel(r'$u - u^*$')
    plt.title('Centered')
    
    plt.subplot(1, 3, 3)
    for i, y in enumerate(curves):
        u_star = peaks[i][0]
        max_y = peaks[i][1]
        plt.plot(u_vals - u_star, y / max_y)
    plt.xlabel(r'$u - u^*$')
    plt.title('Normalized Collapse')
    
    plt.tight_layout()
    plt.savefig('figure2.png', dpi=300)
    plt.close()
    print("Figure 2 saved.")

def generate_figure3():
    print("Generating Figure 3 data...")
    A_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    u_vals = np.linspace(0.0, 2.5, 50)
    max_yields = []
    
    for A in A_vals:
        y = compute_anisotropy_parallel(50e-6, A, 0.5, u_vals)
        max_yields.append(np.max(y))
        print(f"Fig3: Finished config A={A}")
        
    plt.figure(figsize=(6, 5))
    plt.loglog(A_vals, max_yields, 'o-', base=10)
    
    log_A = np.log10(A_vals)
    log_Y = np.log10(max_yields)
    m, c = np.polyfit(log_A, log_Y, 1)
    
    A_line = np.linspace(0.4, 2.2, 100)
    plt.loglog(A_line, (10**c) * A_line**m, '--', label=f'Fit: $\\alpha$ = {m:.2f}')
    
    plt.xlabel('Hyperfine coupling A (mT)')
    plt.ylabel(r'Peak Anisotropy $\Delta Y_{max}$')
    plt.title('Figure 3: Power Law Scaling')
    plt.legend()
    plt.savefig('figure3.png', dpi=300)
    plt.close()
    print("Figure 3 saved.")

def compute_single_drift(args):
    B, A_mT, J_mT, u, theta, kT = args
    A = gamma_e * A_mT * 1e-3
    J = gamma_e * J_mT * 1e-3
    kS = kT * (10**u)
    tmax = 10.0 / kT
    
    I_8 = np.eye(8, dtype=complex)
    term3 = -0.5 * kT * (np.kron(I_8, QT) + np.kron(QT.T, I_8))
    term2 = -0.5 * kS * (np.kron(I_8, QS) + np.kron(QS.T, I_8))
    
    omega_0 = gamma_e * B
    Hz = omega_0 * (np.cos(theta) * (S1z + S2z) + np.sin(theta) * (S1x + S2x))
    Hhf = A * np.dot(S1z, Iz)
    Hex = 2 * J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    H = Hz + Hhf + Hex
    
    term1 = -1j * (np.kron(I_8, H) - np.kron(H.T, I_8))
    L_mat = term1 + term2 + term3
    
    L_ext = np.zeros((65, 65), dtype=complex)
    L_ext[0:64, 0:64] = L_mat
    L_ext[64, 0:64] = kS * QS_vec.T
    
    state0 = np.zeros(65, dtype=complex)
    state0[0:64] = rho0
    
    state_t = scipy.linalg.expm(L_ext * tmax) @ state0
    return np.real(state_t[64])

def compute_angular_drift_parallel(B_vals, A_mT, J_mT, kS_kT_ratio=20, kT=1e6):
    u = np.log10(kS_kT_ratio)
    theta_vals = np.linspace(0, np.pi, 19)
    
    tasks = []
    for B in B_vals:
        for theta in theta_vals:
            tasks.append((B, A_mT, J_mT, u, theta, kT))
            
    with mp.Pool(mp.cpu_count()) as pool:
        yields = pool.map(compute_single_drift, tasks)
        
    max_thetas = []
    idx = 0
    for B in B_vals:
        y_B = yields[idx:idx+len(theta_vals)]
        idx += len(theta_vals)
        max_idx = np.argmax(y_B)
        max_thetas.append(np.degrees(theta_vals[max_idx]))
        
    return max_thetas

def generate_figure4():
    print("Generating Figure 4 data...")
    B_vals = np.linspace(0.3e-3, 3.0e-3, 20)
    
    thetas_A1 = compute_angular_drift_parallel(B_vals, 1.0, 0.5)
    print("Fig4: Finished A=1.0")
    thetas_A0 = compute_angular_drift_parallel(B_vals, 0.0, 0.5)
    print("Fig4: Finished A=0.0")
    
    plt.figure(figsize=(7, 5))
    plt.plot(B_vals * 1e3, thetas_A1, 'b-', label='A = 1.0 mT (proton ON)')
    plt.plot(B_vals * 1e3, thetas_A0, 'r--', label='A = 0 (proton OFF)')
    plt.xlabel('Magnetic Field B (mT)')
    plt.ylabel(r'Angle of Maximum Yield $\theta_{max}$ (deg)')
    plt.title('Figure 4: High-Field Angular Switching')
    plt.legend()
    plt.savefig('figure4.png', dpi=300)
    plt.close()
    print("Figure 4 saved.")

if __name__ == "__main__":
    start_all = time.time()
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    print(f"All figures generated in {time.time() - start_all:.2f} seconds.")

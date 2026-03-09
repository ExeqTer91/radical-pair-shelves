# Radical Pair Magnetoreception: Noise-Assisted Response Shelves

This repository contains the Python implementation of the numerical experiments described in the paper **"Noise-Assisted Response Shelves and Angular Drift in Radical Pair Magnetoreception"** (Andrei Ursachi).

The code simulates a minimal radical pair model with an axial hyperfine coupling on one electron, exploring the self-organization of magnetic-field sensitivity into discrete response shelves.

## Features

- **Radical Pair Hamiltonian Construction**: Includes Zeeman interaction, axial hyperfine coupling (S₁z·Iz), and isotropic exchange interaction (2J·S₁·S₂).
- **Liouvillian Dynamics**: Models coherent evolution, Haberkorn recombination (singlet/triplet), and optional environmental dephasing using the Lindblad pure-dephasing channel.
- **Efficient Integration**: Uses eigendecomposition of the 64×64 Liouvillian superoperator for analytic time integration of the singlet yield Y_S = k_S ∫₀^{t_max} Tr(Q_S ρ(t)) dt.
- **Figure Generation**: Automatically computes and generates the key figures from the paper:
  - `figure1.png`: The main response shelf across recombination asymmetry u = log₁₀(k_S/k_T) and magnetic field B ∈ [30, 70] μT.
  - `figure2.png`: Curve collapse analysis demonstrating approximate ratio-controlled locking across different (A, J) parameter combinations.
  - `figure3.png`: Power law scaling of peak anisotropy ΔY_max with hyperfine coupling strength A (fixed J = 0.5 mT).
  - `figure4.png`: High-field angular switching showing the Zeeman-hyperfine interference transition (θ_max vs B).

## Model

- **Hilbert space**: 8-dimensional (2 electron spins × 1 nuclear spin-½)
- **Spins**: Two electrons (S₁, S₂) and one nucleus (I, spin-½ proton on radical 1)
- **Initial state**: Singlet-projected state ρ(0) = |S⟩⟨S| ⊗ I₂/2
- **Hamiltonian**: H = ω₀(cosθ·S_z^tot + sinθ·S_x^tot) + A·S₁z·Iz + 2J·S₁·S₂
- **Recombination**: Haberkorn (−½k_S{Q_S,ρ} − ½k_T{Q_T,ρ})
- **Dephasing**: Optional Lindblad channel on each electron (γ·D[S_iz])
- **Integration time**: t_max = 10/k_T

## Parameters

| Parameter | Value |
|-----------|-------|
| k_T | 10⁶ s⁻¹ |
| u = log₁₀(k_S/k_T) | 0 to 2.5 (50 points) |
| θ | 0 to π (19 points) |
| B (Figure 1) | 30–70 μT (9 points) |
| A (Figure 3) | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 mT |
| J (Figure 3) | 0.5 mT (fixed) |
| B (Figure 4) | 0.3–3.0 mT (20 points) |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ with NumPy, SciPy, and Matplotlib.

## Usage

```bash
python main.py
```

Runtime is approximately 18 minutes on a single core. The computation involves eigendecomposition of 64×64 complex matrices across a parameter grid of ~9000 points.

## Outputs

After running the script, the following files are generated:
- `figure1.png` through `figure4.png`: Publication-quality figures
- `data_figure1.csv`, `data_figure3.csv`, `data_figure4.csv`: Raw numerical data

## Key Results

- **Response shelf** (Figure 1): Anisotropy ΔY peaks near u* ≈ 1.7 across all B-field values, forming a shelf-like plateau in the (u, B) landscape.
- **Curve collapse** (Figure 2): Normalized anisotropy curves show partial collapse across different (A, J) combinations (minimum pairwise r ≈ 0.42).
- **Power law** (Figure 3): Peak anisotropy scales as ΔY_max ~ A^α with α ≈ 0.35 (R² = 0.92).
- **Angular switching** (Figure 4): With hyperfine coupling, θ_max transitions from 0° (low B) to 90° (high B ≳ 1.8 mT). Without hyperfine, θ_max shows chaotic dependence on B.

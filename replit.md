# Radical Pair Magnetoreception: Revision Computations

## Project Overview

Numerical simulation suite for the Stage 4 peer-review revision of:
**"Noise-Assisted Response Shelves and Angular Drift in Radical Pair Magnetoreception"**
Manuscript 1826192 | Frontiers in Physics

Implements all 7 reviewer-requested computational tasks, producing publication-quality
figures and CSV data tables for the revision letter and supplementary materials.

## Architecture

```
core.py                 Physics engine (shared by all tasks)
run_all.py              Sequential task runner with --tasks filter
main.py                 Entry point → delegates to run_all.py
gen_fig2.py             Standalone Figure 2 generator (normalized anisotropy)
task1_angular.py        Task 1: Finer angular sampling (Fig 4)
task2_convergence.py    Task 2: Convergence tests (Table S2)
task3_rate_invariance.py Task 3: Rate invariance / biological regime (Fig S2)
task4_crosssections.py  Task 4: Unnormalized cross-sections + heatmap (Fig 1)
task5_scaling.py        Task 5: Power-law robustness / Michaelis-Menten (Fig 3)
task6_dephasing.py      Task 6: Dephasing scan (Fig 5, promoted from S1)
task7_hab_vs_lind.py    Task 7: Haberkorn vs Lindblad comparison (Table S1)
outputs/figures/        30 PDF + PNG figures (all tasks)
outputs/data/           11 CSV / TXT data tables
outputs/publication_tiff/ 11 × 300 DPI TIFF files (Frontiers upload-ready)
outputs/everything_revision.zip  Single ZIP: TIFFs + PDFs + PNGs + data + source
```

## Physics Model

- **Hilbert space**: 8-dimensional (2 electron spins × 1 nuclear spin-½)
- **Liouvillian**: 64×64 complex superoperator, column-major (Fortran) vectorization
- **Time integration**: Eigendecomposition analytic integral — `Y_S = kS · Re[Tr(Q_S · V · diag((e^{λt}-1)/λ) · V⁻¹ · vec(ρ₀))]`
- **Hamiltonian**: `H = A·S1z⊗Iz + 2J·(S1·S2) + B·(γ_e·S1z + γ_e·S2z)·cos/sinθ`
- **Hyperfine**: Axial only (A⊥ = 0); isotropic gives zero anisotropy by symmetry
- **Recombination**: Haberkorn = Jones-Hore Lindblad (proven equivalent, Task 7)

## Critical Unit Convention

```
KT_DEFAULT = 1e6 s⁻¹ / (γ_e × 1e-3 T/mT) ≈ 0.005679 mT
```
This places the shelf at u* ≈ 1.682 and ΔY_max = 2.09 × 10⁻³, matching the
submitted manuscript. Using kT = 1.0 mT (wrong) gives u* ≈ −0.75.

## Task Results Summary

| Task | Description | Key Result | Time |
|------|-------------|-----------|------|
| 1 | Finer angular sampling | Angular switching confirmed at 181θ pts | 160s |
| 2 | Convergence tests | n_θ=9, t_max=10/kT, u* stable at 1.682 | 1363s |
| 3 | Rate invariance | CV=59.2%; stable regime at kT ≪ A | 672s |
| 4 | Unnormalized cross-sections | u*=1.62–1.74, FWHM≈2.0 (field-invariant) | 4824s |
| 5 | Power-law robustness | Michaelis-Menten wins (ΔAIC=11.2, α≈0.002) | 2657s |
| 6 | Dephasing scan | 75× ΔY enhancement at γ=0.5 mT | 384s |
| 7 | Haberkorn vs Lindblad | 0.00% difference (Jones & Hore 2010) | 489s |

**Total: 182.9 min | 30 figures | 11 data files**

## Supplementary Material Mapping

| Supplementary item | Content | Source task |
|--------------------|---------|-------------|
| Table S1 | Haberkorn vs Lindblad, 5 configs | Task 7 |
| Table S2 | Grid convergence (t_max, n_θ, n_u, n_B) | Task 2 |
| Figure 5 | Dephasing-enhanced compass (promoted from S1) | Task 6 |
| Figure S2 | Rate invariance / biological regime | Task 3 |

## Publication Figure Mapping (Frontiers upload)

| TIFF file | Manuscript figure | Source |
|-----------|------------------|--------|
| Figure1.tiff | Fig 1 (heatmap panel) | Task 4 |
| Figure1b.tiff | Fig 1 (cross-sections panel) | Task 4 |
| Figure2.tiff | Fig 2 (normalized anisotropy, 5 configs) | gen_fig2.py |
| Figure3.tiff | Fig 3 (amplitude saturation) | Task 5 |
| Figure4.tiff | Fig 4 (angular switching) | Task 1 |
| Figure4b/c.tiff | Fig 4 panels B, C | Task 1 |
| Figure5.tiff | Fig 5 (dephasing sensitivity) | Task 6 |
| FigureS2.tiff | Fig S2 (rate invariance) | Task 3 |

## Key Physics Findings (Revision)

1. **Convergence**: Fully converged at n_θ=9, t_max=10/kT across all grid parameters
2. **Haberkorn = Lindblad**: 0.00% difference — Jones & Hore (2010) confirmed numerically
3. **FWHM field-invariant**: ≈ 2.0 decades in log₁₀(kS/kT) across B=30–70 μT
4. **Power law overturned**: α≈0.002 (not 0.35); Michaelis-Menten saturating model preferred (ΔAIC=11.2, R²=0.942)
5. **Noise-assisted enhancement**: γ=0.5 mT dephasing gives 75× ΔY enhancement; shelf shifts to u*≈0.13
6. **Anisotropic tensor**: A⊥=0.5 shifts u* from 1.682→1.875 and reduces ΔY by 4× (vs submitted axial-only model)

## Workflow

- **"Start application"** runs `python3 main.py`
- To re-run specific tasks: edit `sys.argv` in main.py, e.g. `--tasks=6,7`
- To re-run all tasks: set `sys.argv = ["run_all.py"]`

## GitHub Repository

- https://github.com/ExeqTer91/radical-pair-shelves/tree/main/revision
- All 41 output files + 10 source files uploaded via GitHub REST API
- Token: `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable
- Upload method: Python `urllib.request` (curl fails for large files)

## Dependencies

- Python 3.11 (via Nix)
- numpy, scipy, matplotlib, Pillow (PIL)

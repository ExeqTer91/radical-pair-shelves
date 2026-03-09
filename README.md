# Radical Pair Magnetoreception: Noise-Assisted Response Shelves

This repository contains the Python implementation of the numerical experiments described in the paper **"Noise-Assisted Response Shelves and Angular Drift in Radical Pair Magnetoreception"**.

The code simulates a minimal radical pair model with an axial hyperfine coupling on one electron, exploring the self-organization of magnetic-field sensitivity into discrete response shelves.

## Features

- **Radical Pair Hamiltonian Construction**: Includes Zeeman interaction, axial Hyperfine coupling, and Exchange interaction.
- **Liouvillian Dynamics**: Models coherent evolution, Haberkorn recombination (singlet/triplet), and environmental dephasing using the Lindblad pure-dephasing channel.
- **Parallelized Integration**: Uses `scipy.linalg.expm` and `multiprocessing` to efficiently compute the time evolution of the 64-dimensional density matrix over large parameter grids.
- **Figure Generation**: Automatically computes and generates the key figures from the paper:
  - `figure1.png`: The main response shelf across recombination asymmetry and magnetic fields.
  - `figure2.png`: Collapse of normalized response curves demonstrating ratio-controlled locking.
  - `figure3.png`: Power law scaling of shelf amplitude with hyperfine coupling strength.
  - `figure4.png`: High-field angular drift and Zeeman-hyperfine interference transition.

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

*(Note: Requires Python 3.8+)*

## Usage

To run the experiments and generate the figures, simply execute:

```bash
python main.py
```

*Note: The computation performs high-resolution parameter scans (e.g., 50 points in recombination asymmetry, 19 points in angular orientation, multiple magnetic field strengths). The script leverages multiprocessing to speed up the density matrix exponentiation, but it may still take a few minutes to complete depending on your CPU.*

## Outputs

After running the script, the following files will be generated in the working directory:
- `figure1.png`
- `figure2.png`
- `figure3.png`
- `figure4.png`

## Model Details

- **Spins**: Two electrons (S1, S2) and one nucleus (I, spin-1/2, representing a proton).
- **Initial State**: Singlet-projected thermal state.
- **Integration**: Computed via exact matrix exponentiation of the vectorized Liouvillian superoperator.

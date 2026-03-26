"""
run_all.py — Execute all 7 revision computation tasks sequentially.

Usage: python run_all.py [--tasks 1,2,3,4,5,6,7]
All outputs saved to outputs/figures/ and outputs/data/
"""

import sys
import time
import os
import numpy as np

def _verify_checks():
    """Quick sanity checks on core physics before full run."""
    from core import (Q_S, rho0, vec_rho0, build_hamiltonian,
                      build_liouvillian, singlet_yield_from_L, singlet_yield)
    import numpy as np

    print("─" * 55, flush=True)
    print("Pre-run sanity checks:", flush=True)

    # Tr(ρ₀) = 1
    tr = np.real(np.trace(rho0))
    status = "OK" if abs(tr - 1.0) < 1e-10 else "FAIL"
    print(f"  Tr(ρ₀) = {tr:.10f}  [{status}]", flush=True)

    # Q_S trace = 2 in 8-dim
    tr_QS = np.real(np.trace(Q_S))
    status = "OK" if abs(tr_QS - 2.0) < 1e-10 else "FAIL"
    print(f"  Tr(Q_S) = {tr_QS:.1f}  [{status}]", flush=True)

    # Liouvillian eigenvalues have Re ≤ 0
    H = build_hamiltonian(0.05, 0.5, 1.0, 0.5)
    L = build_liouvillian(H, kS=10**1.7, kT=1.0)
    eigvals = np.linalg.eigvals(L)
    max_re = np.max(np.real(eigvals))
    status = "OK" if max_re < 1e-10 else f"WARN (max Re={max_re:.2e})"
    print(f"  max Re(λ) = {max_re:.2e}  [{status}]", flush=True)

    # Y_S in [0,1]
    Y = singlet_yield(H, kS=10**1.7, kT=1.0)
    status = "OK" if 0 <= Y <= 1 else f"FAIL (Y={Y:.4f})"
    print(f"  Y_S sample = {Y:.6f}  [{status}]", flush=True)
    print("─" * 55, flush=True)


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/data",    exist_ok=True)

    # Parse task selection
    selected = None
    for arg in sys.argv[1:]:
        if arg.startswith("--tasks"):
            parts = arg.split("=") if "=" in arg else [arg, sys.argv[sys.argv.index(arg)+1]]
            nums  = parts[-1].split(",")
            selected = set(int(n.strip()) for n in nums if n.strip().isdigit())

    print("=" * 55, flush=True)
    print("Radical Pair Magnetoreception — Revision Computations", flush=True)
    print("Manuscript 1826192  |  Frontiers in Physics", flush=True)
    print("=" * 55, flush=True)

    _verify_checks()

    t_start = time.time()

    tasks = [
        (1, "Finer Angular Sampling (Fig 4)",       "task1_angular",       "run"),
        (2, "Convergence Tests (Table S2)",          "task2_convergence",   "run"),
        (3, "Rate Invariance Test",                  "task3_rate_invariance","run"),
        (4, "Unnormalized Cross-Sections (Fig 1)",   "task4_crosssections", "run"),
        (5, "Power-Law Robustness (Fig 3)",          "task5_scaling",       "run"),
        (6, "Dephasing Scan (Fig S1)",               "task6_dephasing",     "run"),
        (7, "Haberkorn vs Lindblad (Table S1)",      "task7_hab_vs_lind",   "run"),
    ]

    completed = []
    failed    = []

    for task_num, task_desc, module_name, func_name in tasks:
        if selected is not None and task_num not in selected:
            print(f"\n[SKIP] Task {task_num}: {task_desc}", flush=True)
            continue

        print(f"\n{'='*55}", flush=True)
        print(f"[Task {task_num}/7] {task_desc}", flush=True)
        print(f"{'='*55}", flush=True)
        t0 = time.time()
        try:
            import importlib
            mod  = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            func()
            elapsed = time.time() - t0
            completed.append((task_num, task_desc, elapsed))
            print(f"  ✓ Task {task_num} DONE in {elapsed:.1f}s", flush=True)
        except Exception as e:
            import traceback
            print(f"  ✗ Task {task_num} FAILED: {e}", flush=True)
            traceback.print_exc()
            failed.append((task_num, task_desc, str(e)))

    # ── Final summary ──────────────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{'='*55}", flush=True)
    print(f"COMPLETED {len(completed)}/{len(completed)+len(failed)} tasks in {total/60:.1f} min",
          flush=True)
    for t_num, desc, elapsed in completed:
        print(f"  ✓ Task {t_num}: {desc}  ({elapsed:.0f}s)", flush=True)
    for t_num, desc, err in failed:
        print(f"  ✗ Task {t_num}: {desc}  ERROR: {err}", flush=True)

    print(f"\nFigures → outputs/figures/", flush=True)
    print(f"Data    → outputs/data/", flush=True)
    figs = sorted(os.listdir("outputs/figures"))
    data = sorted(os.listdir("outputs/data"))
    print(f"\nFigures ({len(figs)}):", flush=True)
    for f in figs:
        print(f"  {f}", flush=True)
    print(f"\nData ({len(data)}):", flush=True)
    for d in data:
        print(f"  {d}", flush=True)


if __name__ == "__main__":
    main()

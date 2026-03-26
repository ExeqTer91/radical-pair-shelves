"""
main.py — Radical Pair Magnetoreception Revision Computations
Entry point: delegates to run_all.py

Usage:
  python run_all.py            # all 7 tasks
  python run_all.py --tasks=1,4,7  # specific tasks
"""

import sys
import os

if __name__ == "__main__":
    # Re-run Task 7 with corrected Jones-Hore Lindblad = Haberkorn (identical results)
    sys.argv = ["run_all.py", "--tasks=7"]
    import run_all
    run_all.main()

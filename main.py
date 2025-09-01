#!/usr/bin/env python3
# main.py — Orchestrates Case Study 3 Phases 2, 3, and 4/5

import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PHASE2 = PROJECT_ROOT / "phase2.py"
PHASE3 = PROJECT_ROOT / "phase3.py"
PHASE45 = PROJECT_ROOT / "phase4_5.py"
DATASET = PROJECT_ROOT / "dataset" / "MSFT_enriched_phase2.csv"
OUT45 = PROJECT_ROOT / "out_phase4_5"

def run_script(path: Path, globals_ns=None):
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), globals_ns if globals_ns is not None else {})

def banner(msg: str):
    print(f"\n=== {msg} ===")

def ensure_dataset():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATASET}\n"
            "Run Phase 2 first or adjust DATASET path in main.py."
        )

def run_phase2(skip: bool):
    if skip: return
    banner("Phase 2: Data Understanding & Feature Engineering")
    t0 = time.time()
    run_script(PHASE2, globals())
    print(f"Phase 2 done in {time.time()-t0:.1f}s")

def run_phase3(skip: bool):
    if skip: return
    banner("Phase 3: Visualization")
    t0 = time.time()
    run_script(PHASE3, globals())
    print(f"Phase 3 done in {time.time()-t0:.1f}s")

def run_phase45(skip: bool, cutoff: str):
    if skip: return
    ensure_dataset()
    banner("Phase 4/5: Modeling & Evaluation")
    # phase4_5.py already uses defaults: data=./dataset/MSFT_enriched_phase2.csv, out=./out_phase4_5
    # But allow override via function call if present.
    t0 = time.time()
    ns = {}
    run_script(PHASE45, ns)
    if "run" in ns:
        ns["run"](str(DATASET), str(OUT45), cutoff)
    print(f"Phase 4/5 done in {time.time()-t0:.1f}s")
    print(f"Outputs -> {OUT45}")

def main():
    ap = argparse.ArgumentParser(description="Run Case Study 3 pipeline")
    ap.add_argument("--skip2", action="store_true", help="Skip Phase 2")
    ap.add_argument("--skip3", action="store_true", help="Skip Phase 3")
    ap.add_argument("--skip45", action="store_true", help="Skip Phase 4/5")
    ap.add_argument("--cutoff", default="2017-01-01", help="Train/Test split date for Phase 4/5")
    args = ap.parse_args()

    try:
        run_phase2(args.skip2)
        run_phase3(args.skip3)
        run_phase45(args.skip45, args.cutoff)
        banner("All requested phases completed")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
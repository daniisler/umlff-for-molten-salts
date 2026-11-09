#!/usr/bin/env python3
"""Shear viscosity of one melt via Green-Kubo (separate from the clean-5 run).

Viscosity needs a long, well-sampled NVT trajectory and is noisy, so it is kept
out of the main property sweep. Point it at a dedicated long NVT run.

Example:
  uv run python scripts/viscosity.py results/md_NaCl_visc_1100K 1100
"""

import argparse
import json
import os

from moltensaltcalc import MoltenSaltAnalyzer


def main():
    """Load the NVT trajectory and report the Green-Kubo shear viscosity."""
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="results/md_<label>/ with an nvt.traj inside")
    ap.add_argument("temperature", type=float, help="run temperature in K")
    ap.add_argument("--tmax-fs", type=int, default=20000, help="max correlation time (fs)")
    ap.add_argument("--eq-fraction", type=float, default=0.2)
    args = ap.parse_args()

    nvt = os.path.join(args.run_dir, "nvt.traj")
    if not os.path.exists(nvt):
        raise SystemExit(f"no nvt.traj in {args.run_dir}")

    ana = MoltenSaltAnalyzer(traj_files_nvt=nvt, temperatures_nvt=[args.temperature])
    eta, _acf = ana.compute_viscosity(T=args.temperature, eq_fraction=args.eq_fraction, tmax_fs=args.tmax_fs)
    eta_val = float(eta) if not hasattr(eta, "__len__") else float(eta[-1])
    eta_mpa_s = eta_val * 1e3  # Pa s -> mPa s (= cP)
    print(f"viscosity (Green-Kubo) = {eta_val:.4e} Pa s  =  {eta_mpa_s:.3f} mPa s (cP)")

    path = os.path.join(args.run_dir, "viscosity.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"T_K": args.temperature, "eta_Pa_s": eta_val, "eta_mPa_s": eta_mpa_s, "tmax_fs": args.tmax_fs},
            fh,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

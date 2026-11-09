#!/usr/bin/env python3
"""Compute the five liquid properties for one salt from its temperature sweep.

Reads the NPT + NVT trajectories that md_moltensaltcalc.py wrote at several
temperatures (results/md_<prefix>_<T>K/{npt,nvt}.traj) and computes:

  per temperature : density, RDF first-peaks (structure), self-diffusion, Cv
  across temperatures : thermal expansion alpha_V, heat capacity Cp

Bulk modulus is intentionally excluded (it is a 0 K solid property).
Viscosity is computed separately (scripts/viscosity.py), as it needs much
longer trajectories.

Example:
  uv run python scripts/analyze_properties.py NaCl 1100 1200 1300
"""

import argparse
import json
import os

import numpy as np
from moltensaltcalc import MoltenSaltAnalyzer

RMIN = 1.8  # A: below this is the r->0 normalisation spike, not real structure


def first_peaks(rdf: dict) -> dict:
    """First real g(r) peak position (A) for each pair, artefact region excluded."""
    peaks = {}
    for pair, (dists, g, _std) in rdf.items():
        d = np.asarray(dists)
        gg = np.asarray(g)
        mask = d >= RMIN
        if not mask.any():
            continue
        idx = int(np.where(mask)[0][np.argmax(gg[mask])])
        label = f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair)
        peaks[label] = round(float(d[idx]), 2)
    return peaks


def main():  # pylint: disable=too-many-locals,too-many-statements
    """Build the analyzer from the sweep trajectories and compute all five properties."""
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="salt label prefix, e.g. NaCl (dirs results/md_<prefix>_<T>K/)")
    ap.add_argument("temperatures", type=float, nargs="+", help="sweep temperatures in K (>=2 for alpha_V / Cp)")
    ap.add_argument("--rmax", type=float, default=6.0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    temps = sorted(args.temperatures)
    npt = [f"results/md_{args.prefix}_{int(t)}K/npt.traj" for t in temps]
    nvt = [f"results/md_{args.prefix}_{int(t)}K/nvt.traj" for t in temps]
    missing = [p for p in npt + nvt if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing trajectories: {missing}")

    ana = MoltenSaltAnalyzer(
        traj_files_npt=npt,
        temperatures_npt=temps,
        traj_files_nvt=nvt,
        temperatures_nvt=temps,
    )

    out = {"salt": args.prefix, "temperatures_K": temps, "per_T": {}}
    for t in temps:
        rec = {"density_g_cm3": round(float(ana.compute_eq_density(T=t, eq_fraction=0.5)), 4)}
        try:
            rec["cv_J_g_K"] = round(float(ana.compute_heat_capacity_cv(T=t)), 4)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            rec["cv_J_g_K"] = None
            print(f"  [{t} K] Cv failed: {exc}")
        try:
            diff = ana.compute_diffusion_coefficient(T=t)  # Angstrom^2 / fs
            rec["diffusion_A2_fs"] = {k: float(v) for k, v in diff.items()}
            rec["diffusion_cm2_s"] = {k: float(v) * 0.1 for k, v in diff.items()}  # 1 A^2/fs = 0.1 cm^2/s
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            rec["diffusion_A2_fs"] = None
            print(f"  [{t} K] diffusion failed: {exc}")
        try:
            rec["rdf_first_peak_A"] = first_peaks(ana.compute_rdf(T=t, rmax=args.rmax, nbins=120))
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            rec["rdf_first_peak_A"] = None
            print(f"  [{t} K] RDF failed: {exc}")
        out["per_T"][int(t)] = rec

    # ---- across-temperature properties ----
    if len(temps) >= 2:
        try:
            texp = ana.compute_thermal_expansion()
            out["thermal_expansion_1_K"] = float(texp["thermal_expansion"])
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            print(f"  thermal expansion failed: {exc}")
        try:
            mid = temps[len(temps) // 2]
            out["cp_J_g_K"] = round(float(ana.compute_heat_capacity_cp(T=mid)), 4)
            out["cp_central_T_K"] = mid
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            print(f"  Cp failed: {exc}")
    else:
        print("  (only one temperature given -> alpha_V and Cp need >= 2; skipped)")

    path = os.path.join(args.outdir, f"{args.prefix}_properties.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

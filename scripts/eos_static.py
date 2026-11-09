#!/usr/bin/env python3
"""0 K equation-of-state (static E-V) for a fitted MoltenSaltCalc model.

Takes an equilibrated snapshot, scales the box isotropically over a range of
volumes, (optionally) relaxes atomic positions at each FIXED cell, fits a
Birch-Murnaghan EOS, and reports the 0 K equilibrium volume V0, density, and
bulk modulus B0.

Static single-point sweep (no dynamics) -> cheap on a laptop even at 250 atoms.
It isolates the EOS *shape* from thermal effects: compare the 0 K density here
against the NPT density(T) to tell a ground-state volume error apart from a
thermal-expansion error.

Calculator is built exactly as scripts/md_moltensaltcalc.py does.

Example:
  uv run python scripts/eos_static.py results/2body_CsZrCl_zbl/model.json \
    results/md_ZrCl4_pure_710K_big/npt.traj --points 15 --range 0.90 1.10 --relax
"""

import argparse
import contextlib
import json
import os
import sys

import numpy as np
from ase.eos import EquationOfState
from ase.io import read
from ase.optimize import LBFGS
from moltensaltcalc import MoltenSaltSimulator  # same import path as the MD driver


def main():  # pylint: disable=too-many-locals,too-many-statements
    """Static 0 K E-V sweep: fit Birch-Murnaghan, report B0 and 0 K density."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model_json")
    ap.add_argument("snapshot", help="extxyz or .traj; last frame is used")
    ap.add_argument("--points", type=int, default=15)
    ap.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=(0.90, 1.10),
        metavar=("LO", "HI"),
        help="volume scale range (fraction of V_snapshot)",
    )
    ap.add_argument("--relax", action="store_true", help="relax atomic positions at each fixed cell (recommended)")
    ap.add_argument("--fmax", type=float, default=0.05, help="relax force tol (eV/A)")
    ap.add_argument("--relax-steps", type=int, default=60)
    ap.add_argument("--model-name", default="fit_uf3")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # --- build the calculator the same way md_moltensaltcalc.py does ---
    sim = MoltenSaltSimulator(
        model_name=args.model_name,
        model_parameters={"model_path": args.model_json},
        device="cpu",
        dispersion=None,  # DFT training data already has D3
    )
    calc = sim.calc

    atoms0 = read(args.snapshot, index=-1)
    atoms0.calc = calc
    n = len(atoms0)
    mass_sum = atoms0.get_masses().sum()  # amu
    cell0 = atoms0.get_cell().array.copy()
    v0_snap = atoms0.get_volume()

    outdir = args.outdir or os.path.join(os.path.dirname(args.snapshot) or ".", "eos_static")
    os.makedirs(outdir, exist_ok=True)

    scales = np.linspace(args.range[0], args.range[1], args.points)
    volumes, energies = [], []
    print(f"snapshot: {args.snapshot}  ({n} atoms, V={v0_snap:.1f} A^3)")
    print(f"model   : {args.model_json}  (elements: {list(getattr(calc, 'elements', []))})")
    print(
        f"sweep {args.points} volumes x{args.range[0]:.2f}-{args.range[1]:.2f}"
        f"{'  (relaxing positions)' if args.relax else '  (frozen positions)'}"
    )
    print(f"{'scale':>6} {'V (A^3)':>10} {'E (eV)':>12}")
    for s in scales:
        a = atoms0.copy()
        a.calc = calc
        a.set_cell(cell0 * s ** (1.0 / 3.0), scale_atoms=True)  # isotropic volume x s
        if args.relax:
            dyn = LBFGS(a, logfile=None)  # positions only; cell held fixed = EOS needs this
            try:
                dyn.run(fmax=args.fmax, steps=args.relax_steps)
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                print(f"  (relax warn at s={s:.3f}: {e})", file=sys.stderr)
        V = a.get_volume()
        e_pot = a.get_potential_energy()
        volumes.append(V)
        energies.append(e_pot)
        print(f"{s:6.3f} {V:10.1f} {e_pot:12.3f}")

    volumes = np.array(volumes)
    energies = np.array(energies)
    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    try:
        v0, e0, bulk = eos.fit()  # B in eV/A^3
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"EOS fit failed: {e}", file=sys.stderr)
        np.savetxt(
            os.path.join(outdir, "ev.csv"), np.c_[volumes, energies], delimiter=",", header="V_A3,E_eV", comments=""
        )
        sys.exit(1)

    b_gpa = bulk * 160.21766208  # eV/A^3 -> GPa
    rho0 = mass_sum * 1.66053906660 / v0  # amu/A^3 -> g/cm^3
    print("\n=== EOS (Birch-Murnaghan, 0 K) ===")
    print(f"  V0  = {v0:.1f} A^3   ({v0 / n:.2f} A^3/atom)")
    print(f"  rho0 (0 K) = {rho0:.3f} g/cm^3")
    print(f"  B0  = {b_gpa:.2f} GPa")
    inrange = volumes.min() <= v0 <= volumes.max()
    print(f"  V0 {'inside' if inrange else 'OUTSIDE'} the sampled window -> {'ok' if inrange else 'WIDEN --range'}")

    np.savetxt(os.path.join(outdir, "ev.csv"), np.c_[volumes, energies], delimiter=",", header="V_A3,E_eV", comments="")
    with open(os.path.join(outdir, "eos.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "V0_A3": float(v0),
                "E0_eV": float(e0),
                "B0_GPa": float(b_gpa),
                "rho0_gcc": float(rho0),
                "natoms": n,
                "relaxed": bool(args.relax),
                "snapshot": args.snapshot,
                "model": args.model_json,
            },
            f,
            indent=2,
        )
    with contextlib.suppress(Exception):
        eos.plot(os.path.join(outdir, "eos.png"))
    print(f"\nwrote {outdir}/eos.json + ev.csv + eos.png")


if __name__ == "__main__":
    main()

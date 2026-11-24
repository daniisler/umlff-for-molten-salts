"""Per-element 2-body force-error breakdown (locked operating point, full 595 test).

Fits one 2-body model at the locked point (n=200, seed 42, cutoff2=8, res2=20,
kappa=0.3, ridge2=1e-4, curv2=1e-6), then uses the UF3 ASE calculator to get
per-atom predicted forces -- which come back in atom order, so each atom's error
is attributed to its own element -- and reports the force RMSE separately for
Cl / Cs / Zr. Shows WHERE the pair model fails (expected: concentrated on Zr).

The pooled 'overall' line is a cross-check: it must reproduce ~266.4 meV/A.

Run from the repo root in the project venv:
    uv run python scripts/per_element_force.py
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
from ase.io import read
from fit_uf3 import RunConfig, build_basis, featurize, fit_model, reservoir_sample, set_seeds
from uf3.forcefield.calculator import UFCalculator

DATA = Path(__file__).resolve().parents[1] / "data"
ELEMENTS = ["Cl", "Cs", "Zr"]
N_TRAIN = 200
SEED = 42


def make_cfg() -> RunConfig:
    return RunConfig(
        data_dir=DATA,
        train_file="sub_CsZrCl_train.extxyz",
        test_file="sub_CsZrCl_test.extxyz",
        mode="2body",
        n_train=N_TRAIN,
        n_test=0,
        elements=ELEMENTS,
        cutoff2=8.0,
        res2=20,
        cutoff3=5.0,
        res3=6,
        weight=0.3,
        ridge2=1e-4,
        ridge3=1e-4,
        curv2=1e-6,
        curv3=1e-6,
        seed=SEED,
        run_name="pe",
        results_dir=DATA,
        max_features=60000,
        force=False,
        probe_only=False,
        save_predictions=False,
        n_jobs=1,
    )


def main() -> None:
    cfg = make_cfg()
    rng = set_seeds(SEED)
    train = reservoir_sample(DATA / "sub_CsZrCl_train.extxyz", N_TRAIN, rng, "sampling train")
    _, basis = build_basis(cfg, ELEMENTS)
    feats_train = featurize(train, basis, "train", n_jobs=1, progress="none")
    model = fit_model(cfg, basis, feats_train)
    calc = UFCalculator(model)

    test = read(str(DATA / "sub_CsZrCl_test.extxyz"), index=":")
    sq = collections.defaultdict(list)  # element -> squared force-component errors
    pooled = []
    for atoms in test:
        f_ref = np.asarray(atoms.arrays["force"])
        probe = atoms.copy()
        probe.calc = calc
        f_pred = probe.get_forces()
        d = f_pred - f_ref  # (N, 3), aligned to atom order
        for i, sym in enumerate(atoms.get_chemical_symbols()):
            sq[sym].extend((d[i] ** 2).tolist())
        pooled.extend((d.ravel() ** 2).tolist())

    overall = np.sqrt(np.mean(pooled)) * 1000.0
    print(f"\n[cross-check] pooled force RMSE = {overall:.2f} meV/A  (should be ~266.4)\n")
    print(f"{'element':>8} {'# force components':>18} {'force RMSE (meV/A)':>20}")
    for e in ELEMENTS:
        arr = np.asarray(sq[e])
        print(f"{e:>8} {len(arr):>18} {np.sqrt(arr.mean()) * 1000:>20.1f}")


if __name__ == "__main__":
    main()

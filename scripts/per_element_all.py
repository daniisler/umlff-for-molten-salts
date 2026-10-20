"""Per-element 2-body force-error breakdown across the FULL chemistry.

Fits ONE 2-body model at the locked operating point (cutoff2=8, res2=20,
kappa=0.3, ridge2=1e-4, curv2=1e-6) on the full Training set, then uses the UF3
ASE calculator to get per-atom predicted forces on the test set. Each atom's
force error is attributed to its own element, so we get a force RMSE per element
in a single fit -- directly ranking which cations break the pair model.

Two columns per element:
  * force RMSE (meV/A)          -- absolute error (scale + bias)
  * RMSE / ref_RMS  (%)         -- error normalized by that element's own force
                                   scale, so a high value = genuinely mis-modelled,
                                   not just large forces. THIS is 'problematic'.

Run from the repo root in the project venv (2-body => laptop-friendly):
    uv run python scripts/per_element_all.py                 # defaults
    uv run python scripts/per_element_all.py 2000 500        # n_train n_test_frames

If RAM strains (MacBook Air), lower n_train (first arg). Full chemistry has ~78
pairs, so memory grows with n_train; 1000-1500 is a safe diagnostic.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from fit_uf3 import RunConfig, build_basis, featurize, fit_model, reservoir_sample, set_seeds
from uf3.forcefield.calculator import UFCalculator

DATA = Path(__file__).resolve().parents[1] / "data"
TRAIN_FILE = "Training.extxyz"
TEST_FILE = "Validation.extxyz"

# Cl first (anion), then cations grouped by nominal charge for readability.
ELEMENTS = ["Cl", "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Zr"]
VALENCE = {"Cl": -1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Zr": 4}

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEED = 1


def force_ref(atoms):
    """Return a frame's reference forces, handling the 'force'/'forces' array keys."""
    for k in ("force", "forces"):
        if k in atoms.arrays:
            return np.asarray(atoms.arrays[k])
    return np.asarray(atoms.get_forces())  # fallback if a calculator is attached


def make_cfg() -> RunConfig:
    """Build the locked 2-body RunConfig for the full-chemistry per-element fit."""
    return RunConfig(
        data_dir=DATA,
        train_file=TRAIN_FILE,
        test_file=TEST_FILE,
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
        run_name="pe_all",
        results_dir=DATA,
        max_features=60000,
        force=False,
        probe_only=False,
        save_predictions=False,
        n_jobs=8,
    )


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Fit the full-chemistry 2-body model and report the per-element force breakdown."""
    cfg = make_cfg()
    rng = set_seeds(SEED)
    print(f"[per-element] fitting 2-body on {TRAIN_FILE}  n_train={N_TRAIN}  elements={len(ELEMENTS)}")
    train = reservoir_sample(DATA / TRAIN_FILE, N_TRAIN, rng, "sampling train")
    _, basis = build_basis(cfg, ELEMENTS)
    feats_train = featurize(train, basis, "train", n_jobs=cfg.n_jobs, progress="bar")
    model = fit_model(cfg, basis, feats_train)
    calc = UFCalculator(model)

    print(f"[per-element] scoring per element on first {N_TEST} frames of {TEST_FILE}")
    test = read(str(DATA / TEST_FILE), index=f":{N_TEST}")
    sq = collections.defaultdict(list)  # element -> squared force-component ERRORS
    refsq = collections.defaultdict(list)  # element -> squared reference force COMPONENTS
    pooled = []
    for atoms in test:
        f_ref = force_ref(atoms)
        probe = atoms.copy()
        probe.calc = calc
        d = probe.get_forces() - f_ref  # (N,3), aligned to atom order
        for i, sym in enumerate(atoms.get_chemical_symbols()):
            sq[sym].extend((d[i] ** 2).tolist())
            refsq[sym].extend((f_ref[i] ** 2).tolist())
        pooled.extend((d.ravel() ** 2).tolist())

    overall = np.sqrt(np.mean(pooled)) * 1000.0
    print(f"\n[cross-check] pooled force RMSE = {overall:.1f} meV/A\n")
    print(f"{'elem':>5} {'val':>4} {'#comp':>9} {'RMSE(meV/A)':>12} {'ref_RMS':>9} {'RMSE/ref %':>11}")
    rows = []
    for e in ELEMENTS:
        if not sq[e]:
            continue
        rmse = np.sqrt(np.mean(sq[e])) * 1000.0
        ref = np.sqrt(np.mean(refsq[e])) * 1000.0
        rel = 100.0 * rmse / ref if ref > 0 else float("nan")
        rows.append((e, VALENCE[e], len(sq[e]), rmse, ref, rel))
    # print ranked by absolute RMSE (worst first)
    for e, v, n, rmse, ref, rel in sorted(rows, key=lambda r: -r[3]):
        print(f"{e:>5} {v:>+4} {n:>9} {rmse:>12.1f} {ref:>9.0f} {rel:>11.1f}")
    print("\n(sorted worst-first by absolute force RMSE; RMSE/ref % = charge-normalized difficulty)")


if __name__ == "__main__":
    main()

"""Per-element 2-body force-error breakdown across the FULL chemistry -- BATCHED fit.

Same output as per_element_all.py, but the fit uses the gram-accumulation (batched)
solver, so peak RAM is ~one frame-batch regardless of n_train. This lets you run a
LARGE n (e.g. 2000-5000) on a laptop without the non-batched design-matrix blowing up.

Run from the repo root in the project venv:
    uv run python scripts/per_element_all_batched.py                    # defaults: n=2000, 400 test, fb=300
    uv run python scripts/per_element_all_batched.py 3000 600 300       # n_train n_test_frames frame_batch

If RAM ever strains, LOWER --frame-batch (3rd arg), not n_train -- memory tracks the
batch size, time tracks n_train. 2-body features ~1,800 at full chemistry, so fb=300
is ~1.6 GB/batch.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from fit_uf3 import RunConfig, build_basis, reservoir_sample, set_seeds
from fit_uf3_batched import fit_model_batched
from uf3.forcefield.calculator import UFCalculator

DATA = Path(__file__).resolve().parents[1] / "data"
TRAIN_FILE = "Training.extxyz"
TEST_FILE = "Validation.extxyz"
ELEMENTS = ["Cl", "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Zr"]
VALENCE = {"Cl": -1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Zr": 4}

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 400
FRAME_BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 300
SEED = 1


def force_ref(atoms):
    """Return a frame's reference forces, handling the 'force'/'forces' array keys."""
    for k in ("force", "forces"):
        if k in atoms.arrays:
            return np.asarray(atoms.arrays[k])
    return np.asarray(atoms.get_forces())


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
        run_name="pe_all_batched",
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
    print(
        f"[per-element/batched] 2-body on {TRAIN_FILE}  n_train={N_TRAIN}  frame_batch={FRAME_BATCH}  elements={len(ELEMENTS)}"
    )
    train = reservoir_sample(DATA / TRAIN_FILE, N_TRAIN, rng, "sampling train")
    _, basis = build_basis(cfg, ELEMENTS)
    model, n_batches = fit_model_batched(cfg, basis, train, FRAME_BATCH, cfg.n_jobs)
    print(
        f"[per-element/batched] fit done over {n_batches} batches; scoring per element on first {N_TEST} frames of {TEST_FILE}"
    )
    calc = UFCalculator(model)

    test_index = ":" if N_TEST <= 0 else f":{N_TEST}"
    test = read(str(DATA / TEST_FILE), index=test_index)
    print(f"[per-element/batched] test frames loaded: {len(test)}")
    sq = collections.defaultdict(list)
    refsq = collections.defaultdict(list)
    pooled = []
    for atoms in test:
        f_ref = force_ref(atoms)
        probe = atoms.copy()
        probe.calc = calc
        d = probe.get_forces() - f_ref
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
    for e, v, n, rmse, ref, rel in sorted(rows, key=lambda r: -r[3]):
        print(f"{e:>5} {v:>+4} {n:>9} {rmse:>12.1f} {ref:>9.0f} {rel:>11.1f}")
    print("\n(sorted worst-first by absolute force RMSE; RMSE/ref % = charge-normalized difficulty)")
    # also dump a CSV so cluster runs are easy to collect
    import csv as _csv

    out = Path(DATA).parents[0] / "results" / f"per_element_n{N_TRAIN}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(
            ["element", "valence", "n_components", "force_rmse_meV_per_ang", "ref_rms_meV_per_ang", "rmse_over_ref_pct"]
        )
        for e, v, n, rmse, ref, rel in sorted(rows, key=lambda r: -r[3]):
            w.writerow([e, v, n, round(rmse, 2), round(ref, 2), round(rel, 2)])
        w.writerow(["POOLED", "", len(pooled), round(overall, 2), "", ""])
    print(f"[per-element/batched] wrote {out}")


if __name__ == "__main__":
    main()

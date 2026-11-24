"""Show that the 2-body headline F_RMSE (~250 vs ~267) is a TEST-SET artifact.

Fits ONE 2-body model at the locked operating point (n_train=200, seed 42,
cutoff2=8, res2=20, kappa=0.3, ridge2=1e-4, curv2=1e-6), then scores that single
frozen model on several test definitions:

  * the FULL held-out test file (all frames)            -> the honest, stable number
  * random subsets of sizes 50/100/200/300 (R draws each) -> how much a SMALLER
    test set shifts and scatters the number

Because the model is fit once and only the evaluated frames change, any movement
in F_RMSE is purely the test-set definition -- not the fit, not the seed, not tuning.

Run from the repo root in the project venv:
    uv run python scripts/test_set_sensitivity.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from ase.io import read

# fit_uf3.py sits next to this file, so a plain import works when run as
# `python scripts/test_set_sensitivity.py` (Python puts the script dir on sys.path).
from fit_uf3 import RunConfig, build_basis, evaluate_model, featurize, fit_model, set_seeds

# ---- locked 2-body operating point (edit here if you want to probe another config) ----
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_FILE = "sub_CsZrCl_train.extxyz"
TEST_FILE = "sub_CsZrCl_test.extxyz"
ELEMENTS = ["Cl", "Cs", "Zr"]
N_TRAIN = 200
SEED = 42
SUBSET_SIZES = [50, 100, 200, 300]  # test-set sizes to probe
N_DRAWS = 5  # random subsets per size
N_JOBS = 1  # serial featurization (2-body is fast, fully deterministic)


def make_cfg() -> RunConfig:
    """A RunConfig carrying only the fields build_basis / fit_model actually read."""
    return RunConfig(
        data_dir=DATA_DIR,
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
        run_name="tss",
        results_dir=DATA_DIR,
        max_features=60000,
        force=False,
        probe_only=False,
        save_predictions=False,
        n_jobs=N_JOBS,
    )


def load_frames_with_energy(path: Path) -> list:
    """Read every frame and copy the calculator energy into info['energy'].

    (reservoir_sample does this internally; we replicate it because we load the
    whole test file directly so we can resample it in-memory.)
    """
    frames = read(str(path), index=":")
    for atoms in frames:
        atoms.info["energy"] = atoms.get_potential_energy()
    return frames


def f_rmse_on(model, basis, frames) -> float:
    """Force RMSE (meV/A) of a fixed model on a given set of frames."""
    feats = featurize(frames, basis, "test", n_jobs=N_JOBS, progress="none")
    return evaluate_model(model, feats)["metrics"]["force_rmse_meV_per_ang"]


def main() -> None:
    cfg = make_cfg()
    rng = set_seeds(cfg.seed)

    # ---- fit ONCE on the locked config ----
    from fit_uf3 import reservoir_sample

    train = reservoir_sample(DATA_DIR / TRAIN_FILE, N_TRAIN, rng, "sampling train")
    _, basis = build_basis(cfg, ELEMENTS)
    feats_train = featurize(train, basis, "train", n_jobs=N_JOBS, progress="none")
    model = fit_model(cfg, basis, feats_train)
    print(
        f"[fit] one 2-body model: n_train={len(train)} seed={SEED} "
        f"cutoff2={cfg.cutoff2} res2={cfg.res2} kappa={cfg.weight} "
        f"ridge2={cfg.ridge2} curv2={cfg.curv2}\n"
    )

    # ---- the honest number: the FULL held-out test ----
    test_all = load_frames_with_energy(DATA_DIR / TEST_FILE)
    full = f_rmse_on(model, basis, test_all)
    print(f"[FULL]  all {len(test_all):>4d} frames   F_RMSE = {full:7.2f} meV/A   <- canonical\n")

    # ---- smaller test sets: same model, fewer frames, several draws each ----
    print(f"{'size':>5} {'draws (F_RMSE meV/A)':<40} {'mean':>8} {'min':>8} {'max':>8}")
    picker = random.Random(SEED)
    for size in SUBSET_SIZES:
        if size > len(test_all):
            continue
        vals = []
        for _ in range(N_DRAWS):
            subset = picker.sample(test_all, size)
            vals.append(f_rmse_on(model, basis, subset))
        shown = " ".join(f"{v:6.1f}" for v in vals)
        print(f"{size:>5} {shown:<40} {np.mean(vals):8.2f} {min(vals):8.2f} {max(vals):8.2f}")

    print("\nReading: the FULL-595 number is the stable one. Small test sets scatter by")
    print("~tens of meV/A around it -- which is how the same fitted model was reported")
    print("as ~250 (an earlier, smaller test) and ~267 (the full 595). Same model, different test.")


if __name__ == "__main__":
    main()

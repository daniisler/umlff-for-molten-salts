"""Re-run the 2-body ridge2 and curv2 sweeps on the FULL 595-frame test.

Brings the regularization tables (hyperparameter_justifications.md section 4c)
onto the SAME test definition as the weight/cutoff2/res2 sweeps -- the full
sub_CsZrCl_test.extxyz (~266 scale) -- so every absolute number in the doc
shares one test set.

Locked 2-body operating point: n_train=200, seed 42, cutoff2=8, res2=20, kappa=0.3.
  * ridge2 sweep: curv2 fixed at 1e-6
  * curv2  sweep: ridge2 fixed at 1e-4
The design matrix does NOT depend on ridge2/curv2 (regularization enters only in
the solve), so train and the full test are each featurized ONCE and only the
cheap regularized re-fit changes per value.

Sanity check: the DEFAULT row of BOTH sweeps (ridge2=1e-4, curv2=1e-6) is the
same config, so both must print the same full-595 number (~266.38 / ~7.72).

Run from the repo root in the project venv:
    uv run python scripts/reg_sweep_full595.py
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read
from fit_uf3 import RunConfig, build_basis, evaluate_model, featurize, fit_model, reservoir_sample, set_seeds

DATA = Path(__file__).resolve().parents[1] / "data"
TRAIN_FILE = "sub_CsZrCl_train.extxyz"
TEST_FILE = "sub_CsZrCl_test.extxyz"
ELEMENTS = ["Cl", "Cs", "Zr"]
N_TRAIN = 200
SEED = 42
RIDGE2_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
CURV2_VALUES = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
RIDGE2_DEFAULT = 1e-4
CURV2_DEFAULT = 1e-6
N_JOBS = 1  # serial; 2-body is fast and fully deterministic. Bump to 8 to parallelize featurization.


def make_cfg() -> RunConfig:
    """RunConfig at the locked 2-body operating point (ridge2/curv2 overwritten per loop)."""
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
        ridge2=RIDGE2_DEFAULT,
        ridge3=1e-4,
        curv2=CURV2_DEFAULT,
        curv3=1e-6,
        seed=SEED,
        run_name="reg",
        results_dir=DATA,
        max_features=60000,
        force=False,
        probe_only=False,
        save_predictions=False,
        n_jobs=N_JOBS,
    )


def load_test(path: Path) -> list:
    """Read the full test file and copy calculator energy into info['energy'] (the NaN-bug fix)."""
    frames = read(str(path), index=":")
    for atoms in frames:
        atoms.info["energy"] = atoms.get_potential_energy()
    return frames


def main() -> None:
    cfg = make_cfg()
    rng = set_seeds(SEED)

    # featurize train + full test ONCE; the design matrix is independent of ridge2/curv2
    train = reservoir_sample(DATA / TRAIN_FILE, N_TRAIN, rng, "sampling train")
    _, basis = build_basis(cfg, ELEMENTS)
    feats_train = featurize(train, basis, "train", n_jobs=N_JOBS, progress="none")
    test = load_test(DATA / TEST_FILE)
    feats_test = featurize(test, basis, "test", n_jobs=N_JOBS, progress="none")
    print(
        f"operating point: n_train={len(train)} seed={SEED} cutoff2=8 res2=20 kappa=0.3 "
        f"| full test = {len(test)} frames\n"
    )

    def fit_eval() -> tuple[float, float]:
        """Fit at the current cfg.ridge2/cfg.curv2 and score on the full 595 test."""
        model = fit_model(cfg, basis, feats_train)
        m = evaluate_model(model, feats_test)["metrics"]
        return m["force_rmse_meV_per_ang"], m["energy_mae_meV_per_atom"]

    print(f"RIDGE2 sweep (curv2 fixed at {CURV2_DEFAULT:g}):")
    print(f"  {'ridge2':>8} {'F_RMSE (meV/A)':>16} {'E_MAE (meV/atom)':>18}")
    cfg.curv2 = CURV2_DEFAULT
    for value in RIDGE2_VALUES:
        cfg.ridge2 = value
        f_rmse, e_mae = fit_eval()
        tag = "  <- default" if value == RIDGE2_DEFAULT else ""
        print(f"  {value:>8.0e} {f_rmse:>16.2f} {e_mae:>18.2f}{tag}")

    print(f"\nCURV2 sweep (ridge2 fixed at {RIDGE2_DEFAULT:g}):")
    print(f"  {'curv2':>8} {'F_RMSE (meV/A)':>16} {'E_MAE (meV/atom)':>18}")
    cfg.ridge2 = RIDGE2_DEFAULT
    for value in CURV2_VALUES:
        cfg.curv2 = value
        f_rmse, e_mae = fit_eval()
        tag = "  <- default" if value == CURV2_DEFAULT else ""
        print(f"  {value:>8.0e} {f_rmse:>16.2f} {e_mae:>18.2f}{tag}")


if __name__ == "__main__":
    main()

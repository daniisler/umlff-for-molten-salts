"""Fit and evaluate a UF3 force field on the SuperSalt molten-salt data.

The model is a many-body expansion of the total energy, truncated at the
two-body (``2body``) or two+three-body (``3body``) level. Each term is a sum of
cubic B-splines, and because the energy is linear in the spline coefficients the
whole model is fit by a single regularized linear least-squares solve.

All paths are resolved relative to the repository root, so the script runs
unchanged on a laptop or on the cluster. Progress on the slow featurization step
is shown with a tqdm bar, and every source of randomness is seeded for
reproducibility.

Typical use::

    python scripts/fit_uf3.py --mode 2body --n-train 4000 --n-test 500
    python scripts/fit_uf3.py --mode 3body --n-train 2000 --n-test 300 --force

Run ``python scripts/fit_uf3.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase.io import iread
from tqdm import tqdm
from uf3.data.composition import ChemicalSystem
from uf3.data.io import DataCoordinator
from uf3.regression.least_squares import WeightedLinearModel, dataframe_to_tuples
from uf3.representation.bspline import BSplineBasis
from uf3.representation.process import BasisFeaturizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class RunConfig:  # pylint: disable=too-many-instance-attributes
    """All parameters that define a single UF3 fit-and-evaluate run."""

    data_dir: Path
    train_file: str
    test_file: str
    mode: str
    n_train: int
    n_test: int
    elements: list[str] | None
    cutoff2: float
    res2: int
    cutoff3: float
    res3: int
    weight: float
    ridge2: float
    ridge3: float
    curv2: float
    curv3: float
    seed: int
    run_name: str
    results_dir: Path
    max_features: int
    force: bool
    probe_only: bool
    save_predictions: bool
    n_jobs: int

    @property
    def degree(self) -> int:
        """Body-order of the model: 2 for a pair model, 3 with the angular term."""
        return 3 if self.mode == "3body" else 2


def set_seeds(seed: int) -> np.random.Generator:
    """Seed every RNG used here and return a seeded NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def count_frames(path: Path) -> int:
    """Count structures in an extended-xyz file (one ``Lattice=`` line each)."""
    total = 0
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "Lattice=" in line:
                total += 1
    return total


def reservoir_sample(path: Path, n_keep: int, rng: np.random.Generator, desc: str) -> list:
    """Uniformly sample ``n_keep`` frames in one streaming pass (Algorithm R).

    Streaming keeps memory flat regardless of file size, and the seeded ``rng``
    makes the selection reproducible. The energy stored in each frame's attached
    calculator is copied into ``info['energy']`` because UF3 reads it from there.
    """
    total = count_frames(path)
    reservoir: list = []
    for index, atoms in enumerate(tqdm(iread(str(path)), total=total, desc=desc, unit="frame")):
        if index < n_keep:
            reservoir.append(atoms)
        else:
            swap = int(rng.integers(0, index + 1))
            if swap < n_keep:
                reservoir[swap] = atoms
    for atoms in reservoir:
        atoms.info["energy"] = atoms.get_potential_energy()
    return reservoir


def build_basis(cfg: RunConfig, elements: list[str]) -> tuple[ChemicalSystem, BSplineBasis]:
    """Build the B-spline basis (pairs, and triangles when ``mode == 3body``)."""
    chem = ChemicalSystem(element_list=elements, degree=cfg.degree)
    pairs = chem.interactions_map[2]
    r_min = dict.fromkeys(pairs, 1.0)
    r_max = dict.fromkeys(pairs, cfg.cutoff2)
    resolution = dict.fromkeys(pairs, cfg.res2)
    if cfg.degree >= 3:
        trios = chem.interactions_map[3]
        r_min.update({trio: [1.0, 1.0, 1.0] for trio in trios})
        # Two bonded edges reach cutoff3; the third edge can reach 2*cutoff3.
        r_max.update({trio: [cfg.cutoff3, cfg.cutoff3, 2 * cfg.cutoff3] for trio in trios})
        resolution.update({trio: [cfg.res3, cfg.res3, cfg.res3] for trio in trios})
    basis = BSplineBasis(chem, r_min_map=r_min, r_max_map=r_max, resolution_map=resolution)
    return chem, basis


def default_n_jobs() -> int:
    """Number of usable CPU cores (respects a SLURM/cgroup allocation on Linux)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


def featurize(
    frames: list,
    basis: BSplineBasis,
    prefix: str,
    n_jobs: int = 1,
    progress: str = "bar",
):
    """Turn structures into the B-spline design matrix (the slow step).

    Featurization is the bottleneck; with ``n_jobs >= 2`` it is parallelized
    across CPU cores via ``evaluate_parallel`` (identical output to serial).
    """
    coordinator = DataCoordinator(energy_key="energy", force_key="force")
    coordinator.dataframe_from_lists(frames, prefix=prefix)
    df_data = coordinator.consolidate()
    featurizer = BasisFeaturizer(basis)
    if n_jobs >= 2:
        with ProcessPoolExecutor(max_workers=n_jobs) as client:
            return featurizer.evaluate_parallel(
                df_data,
                client=client,
                energy_key="energy",
                n_jobs=n_jobs,
                shuffle=False,
                progress=progress,
            )
    return featurizer.evaluate(df_data, energy_key="energy", progress=progress)


def natoms_per_energy(feats) -> np.ndarray:
    """Atoms per structure, aligned with the energy rows of the design matrix."""
    structures = feats.index.get_level_values(0)
    is_energy_row = feats.index.get_level_values(-1) == "energy"
    energy_structures = list(structures[is_energy_row])
    force_rows = Counter(structures[~is_energy_row])
    return np.array([force_rows[name] // 3 for name in energy_structures])


def fit_model(cfg: RunConfig, basis: BSplineBasis, feats) -> WeightedLinearModel:
    """Solve the regularized linear least-squares problem for the coefficients."""
    ridge_map = {1: 1e-6, 2: cfg.ridge2}
    curvature_map = {2: cfg.curv2}
    if cfg.degree >= 3:
        ridge_map[3] = cfg.ridge3
        curvature_map[3] = cfg.curv3
    regularizer = basis.get_regularization_matrix(ridge_map=ridge_map, curvature_map=curvature_map)
    model = WeightedLinearModel(basis, regularizer=regularizer)
    x_energy, y_energy, x_force, y_force = dataframe_to_tuples(feats)
    model.fit(x_energy, y_energy, x_force, y_force, weight=cfg.weight)
    return model


def evaluate_model(model: WeightedLinearModel, feats) -> dict:
    """Predict on featurized test data and return metrics plus raw arrays."""
    x_energy, y_energy, x_force, y_force = dataframe_to_tuples(feats)
    n_atoms = natoms_per_energy(feats)
    e_pred = model.predict(x_energy)
    f_pred = model.predict(x_force)
    e_mae = float(np.mean(np.abs((e_pred - y_energy) / n_atoms)) * 1000.0)
    e_rmse = float(np.sqrt(np.mean(((e_pred - y_energy) / n_atoms) ** 2)) * 1000.0)
    f_rmse = float(np.sqrt(np.mean((f_pred - y_force) ** 2)) * 1000.0)
    f_mae = float(np.mean(np.abs(f_pred - y_force)) * 1000.0)
    metrics = {
        "energy_mae_meV_per_atom": e_mae,
        "energy_rmse_meV_per_atom": e_rmse,
        "force_rmse_meV_per_ang": f_rmse,
        "force_mae_meV_per_ang": f_mae,
    }
    arrays = {
        "e_pred": e_pred,
        "e_ref": y_energy,
        "n_atoms": n_atoms,
        "f_pred": f_pred,
        "f_ref": y_force,
    }
    return {"metrics": metrics, "arrays": arrays}


def probe_feature_count(basis: BSplineBasis, frame) -> tuple[int, float]:
    """Featurize a single frame to report the feature count and gram-matrix RAM."""
    probe = featurize([frame], basis, "probe", progress=None)
    n_features = dataframe_to_tuples(probe)[0].shape[1]
    gram_gb = (n_features**2 * 8) / 1e9
    return n_features, gram_gb


def save_outputs(cfg: RunConfig, model: WeightedLinearModel, summary: dict) -> Path:
    """Write metrics.json, the fitted model, and (optionally) predictions."""
    out_dir = cfg.results_dir / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary["report"], handle, indent=2)
    model.to_json(str(out_dir / "model.json"))
    if cfg.save_predictions:
        np.savez_compressed(out_dir / "predictions.npz", **summary["arrays"])
    return out_dir


def run(cfg: RunConfig) -> None:  # pylint: disable=too-many-locals
    """Execute one full run: load, (probe), fit, evaluate, and save."""
    started = time.time()
    rng = set_seeds(cfg.seed)
    print(f"[UF3] mode={cfg.mode} seed={cfg.seed} n_jobs={cfg.n_jobs} host={socket.gethostname()}")

    train_path = cfg.data_dir / cfg.train_file
    test_path = cfg.data_dir / cfg.test_file
    train = reservoir_sample(train_path, cfg.n_train, rng, "sampling train")
    elements = cfg.elements or sorted({sym for atoms in train for sym in atoms.get_chemical_symbols()})
    chem, basis = build_basis(cfg, elements)
    print(
        f"[UF3] elements={len(elements)} pairs={len(chem.interactions_map[2])} "
        f"trios={len(chem.interactions_map[3]) if cfg.degree >= 3 else 0} train={len(train)}"
    )

    n_features, gram_gb = probe_feature_count(basis, train[0])
    rows = len(train) * (1 + 3 * len(train[0]))
    design_gb = rows * n_features * 8 / 1e9
    print(
        f"[UF3] features={n_features:,}  estimated peak RAM: design matrix ~{design_gb:.1f} GB + gram ~{gram_gb:.1f} GB"
    )
    if cfg.probe_only:
        return
    if cfg.degree >= 3 and n_features > cfg.max_features and not cfg.force:
        raise SystemExit(
            f"[UF3] STOP: {n_features:,} features exceed --max-features ({cfg.max_features:,}). "
            "Lower --cutoff3/--res3, restrict --elements, or pass --force."
        )

    feats_train = featurize(train, basis, "train", n_jobs=cfg.n_jobs)
    model = fit_model(cfg, basis, feats_train)
    finite = bool(np.all(np.isfinite(model.coefficients)))
    print(f"[UF3] fit done, coefficients finite={finite}")

    test = reservoir_sample(test_path, cfg.n_test, rng, "sampling test")
    feats_test = featurize(test, basis, "test", n_jobs=cfg.n_jobs)
    evaluation = evaluate_model(model, feats_test)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "elements": elements,
        "n_features": int(n_features),
        "n_pairs": len(chem.interactions_map[2]),
        "n_trios": len(chem.interactions_map[3]) if cfg.degree >= 3 else 0,
        "n_train": len(train),
        "n_test": len(test),
        "metrics": evaluation["metrics"],
        "wall_time_s": round(time.time() - started, 1),
        "coefficients_finite": finite,
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "host": socket.gethostname(),
        },
    }
    out_dir = save_outputs(cfg, model, {"report": report, "arrays": evaluation["arrays"]})
    metrics = evaluation["metrics"]
    print(
        f"[UF3] RESULT  E_MAE {metrics['energy_mae_meV_per_atom']:.2f} meV/atom   "
        f"F_RMSE {metrics['force_rmse_meV_per_ang']:.2f} meV/A   ({report['wall_time_s']:.0f} s)"
    )
    print(f"[UF3] outputs written to {out_dir}")


def parse_args() -> RunConfig:  # pylint: disable=too-many-locals
    """Parse the command line into a :class:`RunConfig`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-file", default="Training.extxyz")
    parser.add_argument("--test-file", default="Validation.extxyz")
    parser.add_argument("--mode", choices=["2body", "3body"], default="2body")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--elements", nargs="*", default=None)
    parser.add_argument("--cutoff2", type=float, default=6.0)
    parser.add_argument("--res2", type=int, default=15)
    parser.add_argument("--cutoff3", type=float, default=4.0)
    parser.add_argument("--res3", type=int, default=5)
    parser.add_argument("--weight", type=float, default=0.3)
    parser.add_argument("--ridge2", type=float, default=1e-4)
    parser.add_argument("--ridge3", type=float, default=1e-4)
    parser.add_argument("--curv2", type=float, default=1e-6)
    parser.add_argument("--curv3", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="parallel featurization cores (0 = all allocated)",
    )
    args = parser.parse_args()

    run_name = args.run_name or f"{args.mode}_n{args.n_train}_seed{args.seed}"
    return RunConfig(
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        mode=args.mode,
        n_train=args.n_train,
        n_test=args.n_test,
        elements=args.elements,
        cutoff2=args.cutoff2,
        res2=args.res2,
        cutoff3=args.cutoff3,
        res3=args.res3,
        weight=args.weight,
        ridge2=args.ridge2,
        ridge3=args.ridge3,
        curv2=args.curv2,
        curv3=args.curv3,
        seed=args.seed,
        run_name=run_name,
        results_dir=args.results_dir,
        max_features=args.max_features,
        force=args.force,
        probe_only=args.probe_only,
        save_predictions=not args.no_predictions,
        n_jobs=args.n_jobs if args.n_jobs else default_n_jobs(),
    )


def main() -> None:
    """Entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()

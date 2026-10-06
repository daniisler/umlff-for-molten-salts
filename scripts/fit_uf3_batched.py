"""Memory-bounded UF3 fit: accumulate the normal equations over frame batches.

This is the "split-RAM" companion to ``fit_uf3.py``. It produces the *same fit*
(identical coefficients, to floating point) but never holds the full design
matrix in memory.

Why it works
------------
The least-squares solve only needs the gram ``X^T W X`` (features x features) and
the ordinate ``X^T W y`` (features long). Both are *sums over frames*, so we can
featurize the training frames in batches of ``--frame-batch``, add each batch's
``X^T X`` / ``X^T y`` to a running total, discard the batch, and solve once at the
end. Peak RAM is one batch's design matrix plus the gram (~features^2, a few tens
of MB), so it is *independent of the number of frames* -- which lets the full
12-element 2-body fit run on a normal node instead of needing hundreds of GB.

It reuses ``fit_uf3.py`` for sampling, basis building, featurization, evaluation
and output, and reuses uf3's own ``combine_weighted_gram`` / ``fit_with_gram`` for
the weighting + solve, so the only thing done differently here is *where* the gram
is summed (over frame batches instead of over rows of one in-memory matrix).

Usage::

    python scripts/fit_uf3_batched.py --mode 2body --n-train 20000 --n-test 2000 \
        --train-file Training.extxyz --test-file Validation.extxyz --frame-batch 500

Note: only the TRAINING featurization is batched (that is the RAM blow-up). The
held-out test set is featurized in one pass, so keep ``--n-test`` modest
(a few thousand), as with ``fit_uf3.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import socket
import time
from dataclasses import asdict
from pathlib import Path

import fit_uf3 as F  # same scripts/ dir: reuse the shared pipeline
import numpy as np
from tqdm import tqdm
from uf3.regression.least_squares import (
    VarianceRecorder,
    WeightedLinearModel,
    batched_moore_penrose,
    dataframe_to_tuples,
    freeze_columns,
)

# CLI script: run/fit/parse functions are intentionally linear top-to-bottom
# pylint: disable=too-many-locals,too-many-statements


def fit_model_batched(cfg: F.RunConfig, basis, train_frames: list, frame_batch: int, n_jobs: int):
    """Solve the regularized least-squares fit by accumulating the gram over frame batches.

    Identical math to ``fit_uf3.fit_model`` + ``WeightedLinearModel.fit``; the gram and
    ordinate are summed batch-by-batch so the full design matrix never exists at once.
    Returns (model, n_batches).
    """
    ridge_map = {1: 1e-6, 2: cfg.ridge2}
    curvature_map = {2: cfg.curv2}
    if cfg.degree >= 3:
        ridge_map[3] = cfg.ridge3
        curvature_map[3] = cfg.curv3
    regularizer = basis.get_regularization_matrix(ridge_map=ridge_map, curvature_map=curvature_map)
    model = WeightedLinearModel(basis, regularizer=regularizer)

    gram_e = gram_f = ord_e = ord_f = None
    rec_e, rec_f = VarianceRecorder(), VarianceRecorder()  # global mean/std of energies & forces
    n_batches = 0
    for start in tqdm(range(0, len(train_frames), frame_batch), desc="fit (gram over batches)", unit="batch"):
        chunk = train_frames[start : start + frame_batch]
        feats = F.featurize(chunk, basis, f"batch{start}", n_jobs=n_jobs, progress=None)
        x_e, y_e, x_f, y_f = dataframe_to_tuples(feats)  # NO n_elements -> extensive energies (matches fit_uf3)
        # drop masked/frozen columns exactly as WeightedLinearModel.fit() does, so the gram
        # is built in the same reduced feature space the solve + un-masking expect
        x_e, y_e = freeze_columns(x_e, y_e, model.mask, model.frozen_c, model.col_idx)
        x_f, y_f = freeze_columns(x_f, y_f, model.mask, model.frozen_c, model.col_idx)
        ge, oe = batched_moore_penrose(x_e, y_e)  # X^T X, X^T y for this batch's energy rows
        gf, of = batched_moore_penrose(x_f, y_f)  # ... and force rows
        if gram_e is None:
            gram_e, ord_e, gram_f, ord_f = ge, oe, gf, of
        else:
            gram_e += ge
            ord_e += oe
            gram_f += gf
            ord_f += of
        rec_e.update(y_e)  # pooled std/count over all energies, streamed
        rec_f.update(y_f)  # ... and all force components
        n_batches += 1
        del feats, x_e, y_e, x_f, y_f, ge, oe, gf, of

    # same per-count / per-sigma normalization uf3's fit() applies, using GLOBAL counts+std
    try:
        energy_weight = 1 / rec_e.n / float(rec_e.std)
        force_weight = 1 / rec_f.n / float(rec_f.std)
    except (ZeroDivisionError, FloatingPointError):
        energy_weight = 1.0
        force_weight = 1 / rec_f.n
    gram, ordinate = model.combine_weighted_gram(gram_e, gram_f, ord_e, ord_f, energy_weight, force_weight, cfg.weight)
    model.fit_with_gram(gram, ordinate)  # adds R^T R regularizer, then LU solve
    return model, n_batches


def run(cfg: F.RunConfig, frame_batch: int) -> None:
    """Batched fit-and-evaluate; same stages as fit_uf3.run, but step 5 accumulates the gram."""
    started = time.time()
    rng = F.set_seeds(cfg.seed)
    print(
        f"[UF3-batched] mode={cfg.mode} seed={cfg.seed} frame_batch={frame_batch} "
        f"n_jobs={cfg.n_jobs} host={socket.gethostname()}"
    )

    train_path = cfg.data_dir / cfg.train_file
    test_path = cfg.data_dir / cfg.test_file
    train = F.reservoir_sample(train_path, cfg.n_train, rng, "sampling train")
    elements = cfg.elements or sorted({sym for atoms in train for sym in atoms.get_chemical_symbols()})
    chem, basis = F.build_basis(cfg, elements)
    print(
        f"[UF3-batched] elements={len(elements)} pairs={len(chem.interactions_map[2])} "
        f"trios={len(chem.interactions_map[3]) if cfg.degree >= 3 else 0} train={len(train)}"
    )

    n_features, gram_gb = F.probe_feature_count(basis, train[0])
    batch_rows = min(frame_batch, len(train)) * (1 + 3 * len(train[0]))
    batch_design_gb = batch_rows * n_features * 8 / 1e9
    print(
        f"[UF3-batched] features={n_features:,}  peak RAM ~ one batch "
        f"(~{batch_design_gb:.2f} GB) + gram (~{gram_gb:.3f} GB), independent of n_train"
    )
    if cfg.probe_only:
        return
    if cfg.degree >= 3 and n_features > cfg.max_features and not cfg.force:
        raise SystemExit(
            f"[UF3-batched] STOP: {n_features:,} features exceed --max-features ({cfg.max_features:,}). "
            "Lower --cutoff3/--res3, restrict --elements, or pass --force."
        )

    model, n_batches = fit_model_batched(cfg, basis, train, frame_batch, cfg.n_jobs)
    finite = bool(np.all(np.isfinite(model.coefficients)))
    print(f"[UF3-batched] fit done over {n_batches} batches, coefficients finite={finite}")

    test = F.reservoir_sample(test_path, cfg.n_test, rng, "sampling test")
    feats_test = F.featurize(test, basis, "test", n_jobs=cfg.n_jobs)
    evaluation = F.evaluate_model(model, feats_test)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "fit_method": "batched_gram_accumulation",
        "frame_batch": frame_batch,
        "n_batches": n_batches,
        "elements": elements,
        "n_features": int(n_features),
        "n_pairs": len(chem.interactions_map[2]),
        "n_trios": len(chem.interactions_map[3]) if cfg.degree >= 3 else 0,
        "n_train": len(train),
        "n_test": len(test),
        "metrics": evaluation["metrics"],
        "wall_time_s": round(time.time() - started, 1),
        "coefficients_finite": finite,
        "env": {"python": platform.python_version(), "numpy": np.__version__, "host": socket.gethostname()},
    }
    out_dir = F.save_outputs(cfg, model, {"report": report, "arrays": evaluation["arrays"]})
    m = evaluation["metrics"]
    print(
        f"[UF3-batched] RESULT  E_MAE {m['energy_mae_meV_per_atom']:.2f} meV/atom   "
        f"F_RMSE {m['force_rmse_meV_per_ang']:.2f} meV/A   ({report['wall_time_s']:.0f} s)"
    )
    print(f"[UF3-batched] outputs written to {out_dir}")


def parse_args() -> tuple[F.RunConfig, int]:
    """Same CLI as fit_uf3.py plus --frame-batch; returns (RunConfig, frame_batch)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=F.DEFAULT_DATA_DIR, help="folder containing the .extxyz data files")
    p.add_argument("--train-file", default="Training.extxyz", help="training structures file inside --data-dir")
    p.add_argument("--test-file", default="Validation.extxyz", help="evaluation structures file inside --data-dir")
    p.add_argument(
        "--mode", choices=["2body", "3body"], default="2body", help="2body = pair model; 3body adds the three-body term"
    )
    p.add_argument("--n-train", type=int, default=2000, help="number of training frames to sample")
    p.add_argument("--n-test", type=int, default=500, help="number of evaluation frames to sample")
    p.add_argument("--elements", nargs="*", default=None, help="species to include (default: auto-detect from frames)")
    p.add_argument("--cutoff2", type=float, default=8.0, help="two-body cutoff radius in Angstrom")
    p.add_argument("--res2", type=int, default=20, help="two-body spline resolution (knot intervals)")
    # 3-body defaults: cutoff3=5 (force knee from the cutoff sweep), res3=6 (working value, not yet swept)
    p.add_argument("--cutoff3", type=float, default=5.0, help="three-body cutoff radius in Angstrom")
    p.add_argument("--res3", type=int, default=6, help="three-body spline resolution per triplet edge")
    p.add_argument(
        "--weight",
        type=float,
        default=0.3,
        help="energy weight kappa in [0,1]; higher favors energy, lower favors forces",
    )
    p.add_argument("--ridge2", type=float, default=1e-4, help="two-body ridge (L2) regularization strength")
    p.add_argument("--ridge3", type=float, default=1e-4, help="three-body ridge (L2) regularization strength")
    p.add_argument("--curv2", type=float, default=1e-6, help="two-body curvature (smoothness) penalty strength")
    p.add_argument("--curv3", type=float, default=1e-6, help="three-body curvature (smoothness) penalty strength")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducible frame sampling")
    p.add_argument(
        "--frame-batch",
        type=int,
        default=500,
        help="training frames featurized per gram-accumulation batch (the RAM lever)",
    )
    p.add_argument("--run-name", default=None, help="output subfolder name (default: auto)")
    p.add_argument(
        "--results-dir", type=Path, default=F.DEFAULT_RESULTS_DIR, help="parent folder for run output subfolders"
    )
    p.add_argument("--max-features", type=int, default=60000, help="abort 3-body runs above this feature count")
    p.add_argument("--force", action="store_true", help="override the --max-features guard and run anyway")
    p.add_argument("--probe-only", action="store_true", help="report feature count + RAM, then stop (no fit)")
    p.add_argument("--no-predictions", action="store_true", help="skip writing predictions.npz (keep metrics+model)")
    p.add_argument("--n-jobs", type=int, default=0, help="parallel featurization cores (0 = all allocated)")
    a = p.parse_args()
    if not 0.0 <= a.weight <= 1.0:
        p.error(
            "--weight is uf3's energy weight kappa and must be in [0, 1] (higher favors energy, lower favors forces)"
        )
    if a.frame_batch < 1:
        p.error("--frame-batch must be >= 1")

    elems = f"{''.join(a.elements)}_" if a.elements else ""
    cutoff = a.cutoff3 if a.mode == "3body" else a.cutoff2
    tag = hashlib.md5(repr(vars(a)).encode()).hexdigest()[:6]  # unique run-name suffix per config
    run_name = a.run_name or (
        f"{a.mode}_{elems}n{a.n_train}_c{cutoff:g}_w{a.weight:g}_seed{a.seed}_fb{a.frame_batch}_{tag}"
    )
    cfg = F.RunConfig(
        data_dir=a.data_dir,
        train_file=a.train_file,
        test_file=a.test_file,
        mode=a.mode,
        n_train=a.n_train,
        n_test=a.n_test,
        elements=a.elements,
        cutoff2=a.cutoff2,
        res2=a.res2,
        cutoff3=a.cutoff3,
        res3=a.res3,
        weight=a.weight,
        ridge2=a.ridge2,
        ridge3=a.ridge3,
        curv2=a.curv2,
        curv3=a.curv3,
        seed=a.seed,
        run_name=run_name,
        results_dir=a.results_dir,
        max_features=a.max_features,
        force=a.force,
        probe_only=a.probe_only,
        save_predictions=not a.no_predictions,
        n_jobs=a.n_jobs if a.n_jobs else F.default_n_jobs(),
    )
    return cfg, a.frame_batch


def main() -> None:
    """Entry point."""
    cfg, frame_batch = parse_args()
    run(cfg, frame_batch)


if __name__ == "__main__":
    main()

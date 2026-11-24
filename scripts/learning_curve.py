"""Learning-curve driver: sweep training-set size, reusing ONE fixed test set.

Why this exists: running ``fit_uf3.py`` once per n re-featurizes the held-out test
set every time — on the 3-body (~seconds per frame) that is the dominant cost and
it is pure waste, since the test set never changes. This driver featurizes the test
set ONCE, then for each n in ``--n-list`` reservoir-samples the training frames,
fits with the batched gram accumulation (``fit_uf3_batched``, so RAM stays flat and
it works for 2-body or 3-body subsystems), evaluates on the cached test features,
and appends a row to a CSV. Test is a single fixed sample held constant across all
n (the correct learning-curve design); each n is an independent seeded train draw.

Example — 3-body learning curve on Cs-Zr-Cl::

    python scripts/learning_curve.py --mode 3body \
        --n-list 60,120,250,500,1000,1500 --n-test 300 \
        --train-file sub_CsZrCl_train.extxyz --test-file sub_CsZrCl_test.extxyz \
        --elements Cl Cs Zr --cutoff3 5.0 --res3 6 --frame-batch 300 \
        --out results/lc_3body_CsZrCl.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import fit_uf3 as F
import fit_uf3_batched as FB
import numpy as np


def main() -> None:
    """Featurize the test set once, sweep n, and write a learning-curve CSV."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-list", required=True, help="comma-separated training sizes, e.g. 60,120,250,500,1000")
    p.add_argument("--data-dir", type=Path, default=F.DEFAULT_DATA_DIR)
    p.add_argument("--train-file", default="Training.extxyz")
    p.add_argument("--test-file", default="Validation.extxyz")
    p.add_argument("--mode", choices=["2body", "3body"], default="2body")
    p.add_argument("--n-test", type=int, default=400, help="held-out test frames (sampled once, fixed for all n)")
    p.add_argument("--elements", nargs="*", default=None, help="species (default: auto-detect from the test frames)")
    p.add_argument("--cutoff2", type=float, default=8.0)
    p.add_argument("--res2", type=int, default=20)
    p.add_argument("--cutoff3", type=float, default=10.0)
    p.add_argument("--res3", type=int, default=11)
    p.add_argument("--weight", type=float, default=0.3)
    p.add_argument("--ridge2", type=float, default=1e-4)
    p.add_argument("--ridge3", type=float, default=1e-4)
    p.add_argument("--curv2", type=float, default=1e-6)
    p.add_argument("--curv3", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-batch", type=int, default=500)
    p.add_argument("--results-dir", type=Path, default=F.DEFAULT_RESULTS_DIR)
    p.add_argument("--max-features", type=int, default=60000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-jobs", type=int, default=0)
    p.add_argument("--out", default=None, help="output CSV path (default: results/lc_<mode>.csv)")
    a = p.parse_args()

    n_list = [int(x) for x in a.n_list.split(",") if x.strip()]
    n_jobs = a.n_jobs if a.n_jobs else F.default_n_jobs()

    def make_cfg(n_train: int) -> F.RunConfig:
        return F.RunConfig(
            data_dir=a.data_dir,
            train_file=a.train_file,
            test_file=a.test_file,
            mode=a.mode,
            n_train=n_train,
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
            run_name="lc",
            results_dir=a.results_dir,
            max_features=a.max_features,
            force=a.force,
            probe_only=False,
            save_predictions=False,
            n_jobs=n_jobs,
        )

    # --- fixed test set: sample + featurize ONCE, reuse for every n ---
    test_path = a.data_dir / a.test_file
    test_frames = F.reservoir_sample(test_path, a.n_test, np.random.default_rng(a.seed), "sampling test")
    elements = a.elements or sorted({s for atoms in test_frames for s in atoms.get_chemical_symbols()})
    _, basis = F.build_basis(make_cfg(n_list[0]), elements)
    print(
        f"[LC] mode={a.mode} elements={elements} fixed test={len(test_frames)} frames; featurizing once...", flush=True
    )
    t0 = time.time()
    feats_test = F.featurize(test_frames, basis, "test", n_jobs=n_jobs, progress="bar")
    print(f"[LC] test featurized in {time.time() - t0:.0f}s", flush=True)

    out = Path(a.out) if a.out else a.results_dir / f"lc_{a.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    train_path = a.data_dir / a.train_file
    for n in n_list:
        ts = time.time()
        cfg = make_cfg(n)
        train = F.reservoir_sample(train_path, n, np.random.default_rng(a.seed), f"train n={n}")
        model, nb = FB.fit_model_batched(cfg, basis, train, a.frame_batch, n_jobs)
        m = F.evaluate_model(model, feats_test)["metrics"]
        row = {
            "n_train": len(train),
            "force_rmse_meV_per_ang": round(m["force_rmse_meV_per_ang"], 2),
            "energy_mae_meV_per_atom": round(m["energy_mae_meV_per_atom"], 2),
            "n_batches": nb,
            "seconds": round(time.time() - ts, 1),
        }
        rows.append(row)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(
            f"[LC] n={n:5d}  F_RMSE {row['force_rmse_meV_per_ang']:7.2f}  "
            f"E_MAE {row['energy_mae_meV_per_atom']:6.2f}  ({row['seconds']:.0f}s)",
            flush=True,
        )

    print(f"[LC] done — wrote {out}", flush=True)


if __name__ == "__main__":
    main()

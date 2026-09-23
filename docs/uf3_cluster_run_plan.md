# UF3 cluster run plan

Author: Marios Poulis · for review by Daniel Isler
Script: `scripts/fit_uf3.py` · Branch: `marios/uf3-reproduction`

This document says exactly what to run, with which commands, why each
configuration was chosen, and what I expect to get back. It is written so the
jobs can be launched without further clarification.

---

## 0. TL;DR — this is a CPU + RAM job, not a GPU job

**UF3 has no GPU code path.** The `uf3` library does featurization (building the
B-spline design matrix) and one regularized linear least-squares solve, both on
CPU. There is nothing for a CUDA device to do. So on the L4 / A2 nodes the GPU
will sit idle — that is expected and fine; we are using those nodes for their
**CPU cores and RAM**, not the accelerator. If GPU utilization is a hard
scheduling requirement, UF3 is simply the wrong kind of workload (the MACE /
SuperSalt side is the GPU job). See §5.2 for the full answer.

## 1. Environment (reproducible)

- Python: **3.12.14** (pinned in `.python-version` as `3.12`).
- Dependencies + lock: `pyproject.toml` (adds `tqdm`) and `uv.lock` are committed.
  On the cluster: `uv sync` reproduces the exact environment, then `uv run ...`.
- Seeds: every run seeds `random`, `numpy`, and the sampling RNG from `--seed`
  (default 42). Same seed + same config → identical numbers (verified: seed 42
  reproduced `F_RMSE 271.08` exactly twice; seed 123 gave `297.60`).
- Data: unpack `umlff_data_archive.zip` so `data/` sits next to the repo
  (git-ignores `data/`). Files: `Training.extxyz` (68 400 frames),
  `Validation.extxyz` (5 820), `Testing_2.extxyz` (3 294), `Tessting_1.extxyz`,
  and three single-salt subsystems `sub_CaMgCl / sub_CsZrCl / sub_KRbCl`.

Outputs per run land in `results/<run_name>/`:
`metrics.json` (config + errors + timing + env), `model.json` (the fitted
coefficients + basis, via UF3's own serializer), `predictions.npz`
(`e_pred, e_ref, n_atoms, f_pred, f_ref` for parity/error analysis).

---

## 2. What to run (job matrix)

All commands are `uv run python scripts/fit_uf3.py ...` from the repo root. They
are independent — send them as separate jobs / a SLURM array (see
`scripts/run_slurm.sh`). Wall-times are rough, on a single many-core CPU node.

### Track A — 2-body learning curve (the "is the floor structural?" test)

| id | command (flags) | ~RAM | ~time |
|----|-----------------|------|-------|
| A1 | `--mode 2body --n-train 1000  --n-test 1000 --seed 42` | <4 GB | ~10 min |
| A2 | `--mode 2body --n-train 4000  --n-test 1000 --seed 42` | <8 GB | ~30 min |
| A3 | `--mode 2body --n-train 16000 --n-test 1000 --seed 42` | ~16 GB | ~2 h |
| A4 | `--mode 2body --n-train 64000 --n-test 1000 --seed 42` | ~32 GB | ~6 h |
| A2b/A3b | repeat A2, A3 with `--seed 7` and `--seed 123` | — | — |

Default `--train-file Training.extxyz --test-file Validation.extxyz`, so nothing
else is needed. Repeats give error bars on the learning curve.

### Track B — full 12-element 3-body (feasibility + attempt)

| id | command (flags) | note |
|----|-----------------|------|
| B0 | `--mode 3body --probe-only --n-train 50` | instant: prints feature count + gram-matrix RAM estimate, does NOT fit |
| B1 | `--mode 3body --n-train 2000 --n-test 500 --cutoff3 3.5 --res3 4 --force` | needs a **high-memory** node (see B0's estimate; likely ≥256 GB) |
| B2 | `--mode 3body --n-train 2000 --n-test 500 --cutoff3 3.0 --res3 3 --force` | smaller fallback if B1 OOMs |

Run **B0 first** — it tells us the RAM before we commit a big node. The default
`--max-features` guard (60 000) will refuse a hopeless configuration unless
`--force` is passed, precisely so we don't OOM a node by accident.

### Track C — 3-body on single-salt subsystems, at higher fidelity than the laptop allowed (the physics result)

For each subsystem, a 2-body baseline and a 2+3-body run at matched `--n-train`,
so the *difference* isolates the angular term.

| id | command (flags) | cation |
|----|-----------------|--------|
| C1a | `--mode 2body --n-train 800 --n-test 200 --train-file sub_CaMgCl.extxyz --test-file sub_CaMgCl.extxyz --elements Ca Cl Mg` | Mg (directional) |
| C1b | `--mode 3body --n-train 800 --n-test 200 --train-file sub_CaMgCl.extxyz --test-file sub_CaMgCl.extxyz --elements Ca Cl Mg --cutoff3 5.0 --res3 6` | Mg |
| C2a | `--mode 2body ... --train-file sub_CsZrCl.extxyz --test-file sub_CsZrCl.extxyz --elements Cl Cs Zr` | Zr (directional) |
| C2b | `--mode 3body ... --train-file sub_CsZrCl.extxyz --test-file sub_CsZrCl.extxyz --elements Cl Cs Zr --cutoff3 5.0 --res3 6` | Zr |
| C3a | `--mode 2body ... --train-file sub_KRbCl.extxyz --test-file sub_KRbCl.extxyz --elements Cl K Rb` | K/Rb (spherical control) |
| C3b | `--mode 3body ... --train-file sub_KRbCl.extxyz --test-file sub_KRbCl.extxyz --elements Cl K Rb --cutoff3 5.0 --res3 6` | K/Rb |

> Note: for the subsystems, train and test are drawn from the same MD pool
> (disjoint by the seeded reservoir sample), so treat the absolute numbers as
> proof-of-concept; the **2-body-vs-3-body contrast** is the robust quantity.

---

## 3. Why this configuration (answers to the specific questions)

### 5.1 How much was evaluated beforehand
The whole pipeline is already validated end-to-end at small scale — on my
laptop, on Colab, and in a cloud sandbox — and reproduced independently:
- 2-body, all 12 elements: **78 pair types → 1 416 coefficients**; force error
  sits at a **~280–300 meV/Å floor** and does not fall as data is added
  (~60-frame runs). A 40-frame cluster smoke test here gave `F_RMSE 271` /
  `E_MAE 25.6`, consistent.
- 3-body subsystems at `cutoff3=4.0, res3=5`: force error dropped sharply for
  directional cations (Mg ≈ 282→185, Zr ≈ 208→164) and barely for the spherical
  control (K/Rb ≈ 107→101).
- Full 12-element 3-body was measured to be **~111 000 features** — intractable
  on a laptop (gram matrix ~100 GB). That number is the whole reason for Track B.

So these are **scale-ups of an already-working pipeline**, not exploration.

### 5.2 Why a GPU / why not the laptop
It does **not** need a GPU — there is no GPU code path in UF3 (§0). What the
laptop cannot provide, and the cluster can, is:
- **RAM.** The linear solve forms a gram matrix of size `n_features²`. For the
  full 3-body model `n_features ≈ 1e5`, i.e. `1e5² × 8 bytes ≈ 80–100 GB` — far
  beyond a 16 GB laptop. This is the binding constraint for Track B.
- **CPU cores + wall-time.** Featurization costs seconds per frame (≈ 8 frames/s
  for 2-body, ≈ 0.04 frames/s for full 3-body on my machine). Thousands of
  frames = many hours; a many-core node with fast BLAS turns that around.

The GPUs on the assigned nodes will be idle; we are borrowing the node for CPU
and memory. (Future work could port the solve to a GPU linear-algebra backend,
but that is a code change, not this run.)

### 5.3 Why these training fractions
- Track A sweeps `n_train = 1k → 64k` precisely to see whether the force error
  **plateaus**. A flat curve is the signature of a model at its structural limit
  (high bias), which is the scientific claim we want evidence for. 64k ≈ the full
  training set is the endpoint that closes the argument.
- Track C uses `n_train = 800` per subsystem: enough to constrain a 3-body model
  whose coefficient count grows like `res³` per triplet, without overfitting the
  few-thousand-frame subsystem pools. Matched between the 2- and 3-body runs so
  the comparison is fair.
- Track B uses `n_train = 2000`: the fit there is **memory-limited, not
  data-limited**, so a moderate frame count is enough to populate the (reduced)
  feature space; adding frames wouldn't change whether it fits in RAM.

### 5.4 Why this UF3 configuration (coefficients, pairs/triangles)
- **2-body:** 12 elements → `C(12,2)+12 = 78` pair types. Each pair is a cubic
  B-spline on `r ∈ [1.0, 6.0] Å` at **resolution 15** (~18 basis functions),
  giving `78×18 + 12 one-body ≈ 1 416` coefficients. `r_min=1.0` captures the
  steep repulsive wall (which dominates forces); `r_max=6.0` is the interaction
  cutoff that makes cost scale linearly in atom count; resolution 15 is enough
  knots to shape the well without over-fitting (curvature regularization keeps it
  smooth where data is sparse).
- **3-body:** each triplet term is a tensor-product B-spline over the three edge
  lengths, with edge cutoffs `[c, c, 2c]` — two bonded edges to `cutoff3`, the
  third (neighbour–neighbour) edge to `2·cutoff3`, since two sides ≤ c can sum to
  2c. Full 12-element enumeration gives 936 triplet types; that is why Track B
  must *reduce* `cutoff3`/`res3` and why Track C restricts to 3-element
  subsystems (18 triplet types, tractable).
- **Regularization:** ridge (pull coefficients toward zero) + curvature (penalize
  jagged splines), values `ridge2=1e-4, curv2=1e-6` (and `ridge3=1e-4,
  curv3=1e-6`). **Fit weight 0.3** prioritizes forces over energies (forces are
  the weaker, more important metric here). All are exposed as flags.

### 5.6 Which outputs I want back
Files, yes — per run: `metrics.json`, `model.json`, `predictions.npz` (see §1).
The `results/` tree zipped is all I need back; `predictions.npz` carries the
per-structure/per-atom arrays for the plots below.

### 5.7 Analysis I will run on them
- Track A → **learning curve**: `force_rmse` vs `n_train` (log-x), with seed
  repeats as error bars.
- All tracks → **parity plots** from `predictions.npz`: predicted vs DFT forces
  and energies (tightness + systematic bias).
- Track C → **grouped bar chart**: 2-body vs 2+3-body force error for Mg, Zr, and
  the K/Rb control.
- Track B → a small **feasibility table**: feature count, peak RAM, whether it
  fit, and the resulting error if it did.

### 5.8 What insight I expect
- Track A: confirm the 2-body error is **structural (a floor), not
  data-limited** — the flat learning curve — which is the justification for
  adding physics (3-body / electrostatics) rather than more data.
- Track B: establish **how large a full 3-body model actually fits** on the
  biggest node, and what accuracy that buys over the 2-body floor.
- Track C: quantify the angular term's value and show it **tracks cation
  directionality** (big for Mg/Zr, ~0 for spherical K/Rb) — the physical story
  that a pair potential is blind to coordination geometry.

### 5.9 What I expect for the numbers (explicit guesses)

| case | 2-body F_RMSE (meV/Å) | 2+3-body F_RMSE (meV/Å) | E_MAE (meV/atom) |
|------|----------------------|-------------------------|------------------|
| Track A (all n_train) | ~260–300, **roughly flat** | — | ~10–20 |
| Track B full 3-body (if it fits, reduced) | (~280 baseline) | ~180–230 | ~8–12 |
| C Mg (CaMgCl) | ~270–290 | **~170–190 (−30–40%)** | ~5–10 |
| C Zr (CsZrCl) | ~200–220 | **~155–175 (−20–25%)** | ~5–10 |
| C K/Rb (control) | ~100–110 | ~95–105 (**~−5%**) | ~3–6 |

Headline prediction: the 3-body improvement **scales with cation charge
density/directionality** — large for Mg and Zr, negligible for the spherical
control — while Track A stays flat, together making the case that the next gain
comes from many-body physics, not more data.

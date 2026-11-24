# Cluster campaign — complete the UF3-vs-SuperSalt comparison

This package runs everything the Week 10 "Toward the complete comparison" slide lists, so
every cell of the density / Cp / αᵥ / structure / diffusion tables is filled **directly**
(no interpolation) across all 11 pure salts **and** the 5 SuperSalt blends.

## What it produces

| Phase | Script | Fills |
|-------|--------|-------|
| **1. Fit** 14 models | `01_fit.slurm` ← `fit_manifest.txt` | the models phase 2 needs |
| **2. UF3 MD** 20 jobs | `02_md_uf3.slurm` ← `md_manifest.txt` | UF3 **3-body** pure (11) + **blends** 2-body & 3-body (9) |
| **3. SuperSalt MD** 16 jobs | `03_md_supersalt.slurm` ← `../md_jobs_supersalt.txt` | the **SuperSalt** column for all 11 pure salts + 5 blends, at our exact T |

The UF3 **2-body pure** campaign is already done (Week 10 data) and is **not** repeated.
Outputs: `results/<salt>_properties.json` (UF3) and `results/supersalt_<name>_properties.json`.

## The models fitted in phase 1 (none committed yet)
- **`2body_full_zbl`** — one full-chemistry 2-body + ZBL covering every blend (Big included).
- **`3body_NaKSrCl`, `3body_NaMgCl`, `3body_NaCaZnCl`, `3body_NaKZnCl`** — per-blend subsystem
  3-body + ZBL (the full 12-element 3-body gram matrix is ~100 GB → intractable; Big gets
  **no** 3-body and is 2-body + SuperSalt only).
- **9 pure single-chloride 3-body + ZBL** (Li K Rb Cs Ca Sr Ba Zn Zr; Na, Mg already in `results/`).

All at the locked operating points (2-body cutoff2=8 res2=20 ridge2=1e-4 κ=0.3; 3-body
cutoff3=5 res3=10 ridge3=1e-5 κ=0.3; ZBL r_in=4 r_out=5, matching the MD driver's wall).

## One code change shipped with this package
`scripts/fit_uf3_batched.py` gained a **`--zbl`** path. The MD driver **always** adds the
ZBL wall at inference, so the full 2-body model **must** be fit on the ZBL residual — the
batched fitter previously could not do that (and was stale against the current `RunConfig`).

## Run it (SLURM)
From the repo root on the login node, after editing the `#SBATCH --partition/--account`
and `module load` / `uv` lines in the three `.slurm` files:
```bash
# one-time: locked env + data
uv sync
# (data/Training.extxyz must be present for the fits — unzip the data archive)

bash scripts/cluster/submit_all.sh
```
`submit_all.sh` submits all three arrays in dependency order (UF3 MD waits for the fits;
SuperSalt runs in parallel) and prints the job IDs. Watch with `squeue -u $USER` and
`tail -f logs/*.log`.

Phases can also be submitted by hand:
```bash
sbatch --array=1-14 scripts/cluster/01_fit.slurm
sbatch --array=1-20 --dependency=afterok:<FITJOBID> scripts/cluster/02_md_uf3.slurm
sbatch --array=1-16 scripts/cluster/03_md_supersalt.slurm
```

## Run it (no SLURM — plain bash, e.g. one big node or locally)
```bash
# 1. fit (sequential; split the manifest across terminals/nodes to parallelise)
while IFS= read -r l; do [ "${l#\#}" = "$l" ] && bash -c "$l"; done < scripts/cluster/fit_manifest.txt
# 2. UF3 MD  (1 fs and 0.5 fs batches)
JOBS_FILE=scripts/md_jobs_3body.txt bash scripts/run_md_jobs.sh 1 8
DT_FS=0.5 JOBS_FILE=scripts/md_jobs_3body.txt bash scripts/run_md_jobs.sh 9 11
JOBS_FILE=scripts/md_jobs_blends.txt bash scripts/run_md_jobs.sh 1 1      # NaKSr 2b
DT_FS=0.5 JOBS_FILE=scripts/md_jobs_blends.txt bash scripts/run_md_jobs.sh 2 5
JOBS_FILE=scripts/md_jobs_blends.txt bash scripts/run_md_jobs.sh 6 6      # NaKSr 3b
DT_FS=0.5 JOBS_FILE=scripts/md_jobs_blends.txt bash scripts/run_md_jobs.sh 7 9
# 3. SuperSalt reference
bash scripts/run_supersalt_jobs.sh            # all 16
```

## Compute notes
- **UF3 MD** uses the fast backends (~0.14 s/step at ~100 atoms) → each salt's 3-T sweep is
  a few hours on 16 cores. 24 h walltime is comfortable.
- **SuperSalt MD** is the long pole: MACE on CPU is ~1000× slower per step, so systems are
  kept small (`ATOMS=150`, `NPT=15000`, `NVT=8000`). Expect several hours to ~a day per
  salt; 48 h walltime is set. Drop `ATOMS`/`NPT`/`NVT` if the queue is tight — density just
  gets noisier, not wrong.
- UF3 has **no GPU path**; request CPU nodes only (no `--gres=gpu`).

## Caveats
- **"Big"** cation fractions are assumed **equimolar** (the archive gives no structure
  file) — confirm against Shen et al. Nat. Commun. 2025 before trusting the Big density.
- The UF3 MD array uses `afterok`, so one failed fit holds the whole MD array; re-run the
  failed fit and `scontrol release` (notes printed by `submit_all.sh`).

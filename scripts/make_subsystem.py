"""Build a subsystem extxyz file from the full SuperSalt training set.

Streams the source file (memory-safe) and keeps only the frames whose elements
are a *subset* of the requested list, writing them to
``data/sub_<Elements>.extxyz``. Use it to create new or larger subsystems for the
3-body study (e.g. add Zn, Sr, Ba, or use 4-5 elements for a bigger test) without
editing any code.

Examples::

    # a new 3-element subsystem
    python scripts/make_subsystem.py --elements Cl Ca Zn

    # a larger 4-element subsystem (more frames qualify, richer 3-body test)
    python scripts/make_subsystem.py --elements Cl Cs Zr Sr

Then fit it with the same element list::

    python scripts/fit_uf3.py --mode 3body --train-file sub_CaClZn.extxyz \
        --test-file sub_CaClZn.extxyz --elements Ca Cl Zn --cutoff3 5.0 --res3 6

Note: more elements means more triplet types, so the 3-body feature count grows
quickly. Check it first with ``fit_uf3.py --mode 3body --probe-only`` on the new file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import iread, write
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def count_frames(path: Path) -> int:
    """Count structures in an extended-xyz file (one ``Lattice=`` line each)."""
    total = 0
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "Lattice=" in line:
                total += 1
    return total


def main() -> None:
    """Filter the source file to the chosen subsystem and write it out."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--elements",
        nargs="+",
        required=True,
        help="elements that define the subsystem",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source", default="Training.extxyz", help="source file inside --data-dir")
    parser.add_argument("--out", default=None, help="output name (default: sub_<Elements>.extxyz)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after keeping this many frames",
    )
    args = parser.parse_args()

    allowed = set(args.elements)
    source = args.data_dir / args.source
    tag = "".join(sorted(args.elements))
    out_path = args.data_dir / (args.out or f"sub_{tag}.extxyz")

    total = count_frames(source)
    kept = []
    for atoms in tqdm(iread(str(source)), total=total, desc="scanning", unit="frame"):
        if set(atoms.get_chemical_symbols()) <= allowed:
            kept.append(atoms)
            if args.max_frames and len(kept) >= args.max_frames:
                break

    if not kept:
        raise SystemExit(f"[make_subsystem] no frames are pure subsets of {sorted(allowed)} — nothing written.")
    # The attached calculator carries energy/forces; do NOT touch atoms.info before writing.
    write(str(out_path), kept)
    print(f"[make_subsystem] kept {len(kept)} / {total} frames -> {out_path}")


if __name__ == "__main__":
    main()

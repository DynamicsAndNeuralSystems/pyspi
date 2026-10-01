"""Thin CLI for pyspi: compute all SPIs on a saved dataset.

    python -m pyspi compute \
        --data ts.npy \
        --config benchmarked_p90 \
        --output table.npz \
        --n-jobs 4 \
        --checkpoint-dir results/

If ``--config`` is omitted, the bundled ``full`` config is used. If
``--output`` is omitted, the result table is written next to the data file as
``<data-stem>.spi.npz``. The format follows the extension: ``.npz``
(round-trips via ``pyspi.load_table``) or ``.csv`` (human-readable, one-way).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .calculator import Calculator, bundled_configs


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".csv":
        return np.genfromtxt(path, delimiter=",")
    if path.suffix == ".txt":
        return np.genfromtxt(path)
    raise ValueError(f"Unsupported data extension: {path.suffix} (use .npy, .csv, or .txt)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pyspi", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("compute", help="Compute SPIs on a saved dataset.")
    cp.add_argument("--data", type=Path, required=True,
                    help="Path to time series array (.npy/.csv/.txt). Shape (processes, observations).")
    cp.add_argument("--config", default="full",
                    help="Bundled config name or path to your own YAML (default: full). "
                         "Bundled: " + ", ".join(bundled_configs()) + ".")
    cp.add_argument("--output", type=Path, default=None,
                    help="Where to write results; format follows the extension "
                         "(.npz, .csv). Default: <data>.spi.npz.")
    cp.add_argument("--n-jobs", type=int, default=1,
                    help="Worker process count. 1 = serial (default).")
    cp.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Directory for per-SPI .npy checkpoints. Enables resume.")
    cp.add_argument("--no-resume", action="store_true",
                    help="Ignore existing checkpoints; recompute every SPI.")
    cp.add_argument("--mp-context", choices=["spawn", "fork", "forkserver"], default=None,
                    help="Multiprocessing start method. Default: fork on Linux "
                         "(measured ~2x faster end to end), spawn elsewhere "
                         "-- fork is unsafe on macOS and absent on Windows.")
    cp.add_argument("--no-zscore", action="store_true",
                    help="Skip z-scoring each time series before computing.")
    cp.add_argument("--quiet", action="store_true",
                    help="Suppress INFO logging; show warnings/errors only.")
    cp.add_argument("--allow-partial", action="store_true",
                    help="Exit 0 even if some SPIs failed. The default is to "
                         "exit 1 whenever calc.errors is non-empty: a results "
                         "table with failed columns is one an automated "
                         "pipeline must not ingest silently, and a NaN column "
                         "is indistinguishable from a legitimately undefined "
                         "statistic once the process has exited. The failed "
                         "identifiers are printed on stderr and stored in the "
                         "NPZ either way. A run in which *nothing* succeeded "
                         "exits 1 regardless of this flag.")

    args = parser.parse_args(argv)

    arr = _load_array(args.data)
    if arr.ndim != 2:
        raise SystemExit(f"Data must be 2D (processes x observations); got shape {arr.shape}")

    calc = Calculator(
        dataset=arr,
        config=args.config,
        zscore=not args.no_zscore,
        verbose=not args.quiet,
    )
    calc.compute(
        n_jobs=args.n_jobs,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        mp_context=args.mp_context,
    )

    out = args.output or args.data.with_suffix(".spi.npz")
    calc.save(out)
    print(f"Wrote results table -> {out}")

    # Report failures on the way out, unconditionally. `--quiet` suppresses the
    # computation summary, which used to be the only place a failed SPI was
    # mentioned -- so `pyspi compute --quiet` printed "Wrote results table" and
    # exited 0 over a table that could be entirely NaN.
    n_failed = len(calc.errors)
    if n_failed:
        print(f"{n_failed} of {calc.n_spis} SPI(s) failed: "
              f"{', '.join(sorted(calc.errors))}", file=sys.stderr)

    values = np.stack([calc.table[k].to_numpy(dtype=float) for k in calc.spis])
    off_diagonal = ~np.eye(calc.dataset.n_processes, dtype=bool)
    n_empty = int(sum(not np.isfinite(v[off_diagonal]).any() for v in values))
    if n_empty:
        print(f"{n_empty} of {calc.n_spis} SPI(s) produced no finite value.",
              file=sys.stderr)

    if n_empty == calc.n_spis:
        print("Every SPI is empty; the results table carries no information.",
              file=sys.stderr)
        return 1
    if n_failed and not args.allow_partial:
        print("Exiting 1 because SPIs failed; pass --allow-partial to accept "
              "a partial table.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

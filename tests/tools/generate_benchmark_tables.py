"""Regenerate the baseline SPI tables used by ``tests/test_baseline_drift.py``.

For each frozen test fixture in ``tests/data/fixtures/`` this runs the full
Calculator (all 317 SPIs)
once and stores the resulting MxM matrix per SPI in a single compressed
``.npz`` file under ``tests/data/baselines/``.

Why a single pass rather than repeated trials
---------------------------------------------
The previous generator ran ten "trials" and stored mean/std, but reseeded numpy
to the same value inside the loop, so all ten draws were identical and every
std matrix was zero by construction. Fixing that by varying the seed would
produce a baseline that *no* individual run reproduces, which defeats the
purpose of the drift test: we want an exact oracle so genuine sub-percent
regressions in deterministic SPIs are visible. The benchmark datasets are
frozen fixtures and the overwhelming majority of SPIs are deterministic given
the data, so a single seeded pass is both honest and strictly more useful. The
handful of estimator-based SPIs that consume the global RNG are pinned by
``--seed`` (default 42) and are compared under a looser tolerance by the test.

Usage
-----
    python tests/tools/generate_benchmark_tables.py                # all three
    python tests/tools/generate_benchmark_tables.py -d cml_M5_T100
    python tests/tools/generate_benchmark_tables.py --out /tmp/baselines
"""
import argparse
import json
import os
import platform
import time
from importlib import metadata

import numpy as np

from pyspi.calculator import Calculator
from pyspi.data import Data

# Datasets that the drift suite tracks. These are test fixtures, not shipped
# data: they live under tests/data/fixtures/ and are built by
# tests/tools/generate_fixtures.py. Their names encode (M processes, T obs).
DATASETS = ("var1_M3_T100", "cml_M5_T100", "kuramoto_M7_T100")

# <repo>/tests/tools/this_file.py -> <repo>/tests/data/{baselines,fixtures}
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(_HERE), "data", "baselines")
FIXTURE_DIR = os.path.join(os.path.dirname(_HERE), "data", "fixtures")


def load_fixture(dataset_name):
    """Load a frozen test fixture; stored (observations, processes) -> 'sp'."""
    return Data(data=os.path.join(FIXTURE_DIR, f"{dataset_name}.npy"),
                dim_order="sp", name=dataset_name)

# Reserved npz keys for provenance; the loader ignores anything dunder-wrapped.
META_PREFIX = "__"

# SPIs that legitimately cannot be estimated on a given fixture, with the
# statistical reason. Everything else must produce a finite value, and a run
# with any other failure is not frozen. Imported by tests/test_baseline_drift.py
# so the test suite and the generator cannot drift apart on what is expected.
#
# Group delay regresses coherence phase on at least three *independent*
# frequencies, one multitaper bandwidth (six bins) apart, so a pair needs a
# significant coherence run of at least 13 of the 51 bins at T=100. The longest
# run on any pair is 12 bins on var1_M3_T100 and 10 on cml_M5_T100, in every
# band.
_GD_SHORT_RUNS = ("no significant coherence run spans three independent "
                  "frequencies (13 bins) at T=100")
KNOWN_UNESTIMABLE = {
    dataset: {
        f"gd_multitaper_delay_fs-1_{band}": _GD_SHORT_RUNS
        for band in ("fmin-0_fmax-0-5", "fmin-0_fmax-0-25",
                     "fmin-0-25_fmax-0-5")
    }
    for dataset in ("var1_M3_T100", "cml_M5_T100")
}


def build_tables(dataset_name, config="full", seed=42):
    """Compute every SPI on ``dataset_name`` and return ``{spi_key: MxM array}``."""
    np.random.seed(seed)
    calc = Calculator(dataset=load_fixture(dataset_name), config=config)
    calc.compute()
    expected = KNOWN_UNESTIMABLE.get(dataset_name, {})
    unexpected = {k: v for k, v in calc.errors.items() if k not in expected}
    if unexpected:
        # Refusing rather than freezing. A baseline written from a run with
        # failed SPIs records the failure as the expected answer, and the drift
        # suite then agrees with it forever -- which is how three all-NaN
        # `gd_*` columns stayed in the baselines unnoticed.
        raise RuntimeError(
            f"[{dataset_name}] {len(unexpected)} SPI(s) raised; refusing to "
            f"freeze a baseline over them:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in sorted(unexpected.items()))
        )
    tables = {spi: calc.table[spi].to_numpy() for spi in calc.spis}
    off_diagonal = ~np.eye(calc.dataset.n_processes, dtype=bool)
    empty = sorted(k for k, v in tables.items()
                   if k not in expected
                   and not np.isfinite(np.asarray(v, dtype=float)[off_diagonal]).any())
    if empty:
        raise RuntimeError(
            f"[{dataset_name}] {len(empty)} SPI(s) produced no finite value; "
            f"an empty column cannot serve as an oracle:\n  " + "\n  ".join(empty)
        )
    return tables


PROVENANCE_PACKAGES = ("numpy", "scipy", "scikit-learn", "statsmodels", "pyEDM",
                       "spectral-connectivity")


def provenance():
    """Environment the baselines were frozen in, as a JSON-serialisable dict.

    Stored in each archive so a drift failure can be attributed to a dependency
    or platform change rather than to the code.
    """
    from pyspi import _parallel

    versions = {}
    for pkg in PROVENANCE_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "computation_version": _parallel.COMPUTATION_VERSION,
    }


def write_npz(tables, path, dataset_name, config, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(tables)
    payload[META_PREFIX + "dataset" + META_PREFIX] = np.array(dataset_name)
    payload[META_PREFIX + "config" + META_PREFIX] = np.array(config)
    payload[META_PREFIX + "seed" + META_PREFIX] = np.array(seed)
    payload[META_PREFIX + "provenance" + META_PREFIX] = np.array(
        json.dumps(provenance(), sort_keys=True))
    np.savez_compressed(path, **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-d", "--dataset", choices=DATASETS + ("all",), default="all",
        help="Bundled dataset to regenerate (default: all).",
    )
    parser.add_argument(
        "-o", "--out", default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "-c", "--config", default="full",
        help="Calculator config name or path (default: full).",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Global numpy seed set once before compute (default: 42).",
    )
    args = parser.parse_args()

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in names:
        t0 = time.time()
        tables = build_tables(name, config=args.config, seed=args.seed)
        path = os.path.join(args.out, f"{name}.npz")
        write_npz(tables, path, name, args.config, args.seed)
        size_kb = os.path.getsize(path) / 1024
        print(f"[{name}] {len(tables)} SPIs -> {path} "
              f"({size_kb:.0f} KB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

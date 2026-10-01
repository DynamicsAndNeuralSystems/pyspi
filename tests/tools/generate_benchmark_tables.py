"""Regenerate the baseline SPI tables used by ``tests/test_baseline_drift.py``.

For each frozen test fixture in ``tests/data/fixtures/`` this runs the full
Calculator (all 322 SPIs)
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
import os
import time

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
# `sgc_parametric_*_order-None`: nitime's automatic AR order search walks the
# lag up to `max_order` and raises if the information criterion never turns
# over. On kuramoto_M7_T100 -- 100 observations, a smooth oscillatory process
# -- BIC improves all the way to lag 49, which at that length is over-fitting
# rather than a genuinely high order. The fixed-order variants (order-1,
# order-20) are estimated normally on the same fixture, and the automatic
# variant is estimated normally on var1_M3_T100 and cml_M5_T100, so this is a
# property of the record, not of the estimator.
KNOWN_UNESTIMABLE = {
    "kuramoto_M7_T100": {
        key: "automatic AR order selection does not converge at T=100"
        for key in (
            "sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-None",
            "sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-None",
            "sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-None",
            "sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-None",
            "sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-25_order-None",
            "sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-None",
        )
    },
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


def write_npz(tables, path, dataset_name, config, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(tables)
    payload[META_PREFIX + "dataset" + META_PREFIX] = np.array(dataset_name)
    payload[META_PREFIX + "config" + META_PREFIX] = np.array(config)
    payload[META_PREFIX + "seed" + META_PREFIX] = np.array(seed)
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

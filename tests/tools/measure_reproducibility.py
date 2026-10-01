"""Measure which SPIs are bit-reproducible under the drift suite's protocol.

``tests/test_baseline_drift.py`` compares a fresh computation against a frozen
baseline. How tight that comparison may be is a property of each SPI, not of
the module it lives in: a deterministic estimator must reproduce its baseline
to float round-off, while one consuming unpinned randomness cannot. Guessing
which is which is how a blanket 1e-2 band ended up covering ~50 deterministic
SPIs, wide enough to hide a real regression in any of them.

This computes the full config twice per fixture under exactly the suite's
protocol -- ``np.random.seed(SEED)``, then a fresh ``Calculator`` -- and reports
the largest disagreement between the two passes for every SPI. Anything
non-zero belongs in ``LOOSE_SPIS`` in the drift suite, with its mechanism named.

    python tests/tools/measure_reproducibility.py
    python tests/tools/measure_reproducibility.py -d var1_M3_T100 --json out.json

Two full passes per fixture: budget ~15 minutes for all three.
"""
import argparse
import json
import os

import numpy as np

from pyspi.calculator import Calculator
from pyspi.data import Data

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(os.path.dirname(_HERE), "data", "fixtures")

# Must match tests/test_baseline_drift.py.
DATASETS = ("var1_M3_T100", "cml_M5_T100", "kuramoto_M7_T100")
SEED = 42


def _one_pass(dataset_name, config, seed):
    np.random.seed(seed)
    calc = Calculator(
        dataset=Data(data=os.path.join(FIXTURE_DIR, f"{dataset_name}.npy"),
                     dim_order="sp", name=dataset_name),
        config=config,
    )
    calc.compute()
    return ({s: calc.table[s].to_numpy() for s in calc.spis},
            {s: calc.spis[s].__module__.split(".")[-1] for s in calc.spis})


def measure(datasets, config="full", seed=SEED):
    """spi -> {module, abs, rel}, maximised over datasets."""
    out = {}
    for dataset_name in datasets:
        first, modules = _one_pass(dataset_name, config, seed)
        second, _ = _one_pass(dataset_name, config, seed)
        for key, a in first.items():
            b = second[key]
            # Compared as masks first. Restricting to entries finite in both
            # and reporting the numeric difference there cannot see an SPI
            # whose NaN *pattern* moved between the two runs -- it would report
            # a difference of 0 for a column that had gone from finite to NaN.
            rec = out.setdefault(key, {"module": modules[key], "abs": 0.0,
                                       "rel": 0.0})
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                rec["abs"] = rec["rel"] = float("inf")
                continue
            finite = np.isfinite(a) & np.isfinite(b)
            abs_diff = np.abs(a[finite] - b[finite])
            worst_abs = float(abs_diff.max()) if abs_diff.size else 0.0
            nonzero = np.abs(a[finite]) > 0
            worst_rel = (float((abs_diff[nonzero] / np.abs(a[finite][nonzero])).max())
                         if nonzero.any() else 0.0)
            rec["abs"] = max(rec["abs"], worst_abs)
            rec["rel"] = max(rec["rel"], worst_rel)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--config", default="full")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", default=None, help="Also write the raw table here.")
    args = ap.parse_args(argv)

    results = measure(args.datasets, config=args.config, seed=args.seed)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, sort_keys=True)

    drifting = {k: v for k, v in results.items() if v["abs"] > 0.0}
    print(f"\n{len(results) - len(drifting)}/{len(results)} SPIs reproduce bit-exactly "
          f"on {', '.join(args.datasets)}")
    if not drifting:
        print("LOOSE_SPIS in tests/test_baseline_drift.py should stay empty.")
        return 0
    print("\nNot bit-reproducible -- add to LOOSE_SPIS with the mechanism named:")
    for key, rec in sorted(drifting.items(), key=lambda kv: -kv[1]["rel"]):
        print(f"  {rec['module']:10s} {key:58s} abs={rec['abs']:.3e} rel={rec['rel']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

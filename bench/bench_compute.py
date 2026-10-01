#!/usr/bin/env python
"""Benchmark Calculator.compute() across an (M, T, n_jobs) grid.

Reproducible, non-notebook timing suite. Each grid cell is run ``--repeats``
times on freshly generated synthetic data and written to its **own** JSON
file named ``<label>_M<M>_T<T>_n<n_jobs>.json`` — one cell per file, with
M, T, n_jobs recorded at the top level (not just in the filename). Each
file is self-contained: environment metadata (pyspi git sha, dependency
versions + fingerprint, platform) lives in every cell file. ``--resume``
skips cells whose JSON already exists with ``repeats >= --repeats``.

Usage:
    python -m bench.bench_compute --m 8,16 --t 200,800 --n-jobs 1,4 --config fast
    python -m bench.bench_compute --preset amortized --config full
    python -m bench.bench_compute --preset parallel --array-index $PBS_ARRAY_INDEX

Presets (each fixes an M/T/n_jobs grid; --config still applies):
    headline   (M=10,T=500), (M=20,T=1000), n_jobs=1.
    scaling    M={4,8,16,32} x T={200,400,800,1600}, n_jobs=1.
    parallel   M=16, T=800, n_jobs={1,2,4,8,16}.
    amortized  M={8,16}, T=800, n_jobs=1.

Two-axis parallelism: this script benchmarks INNER parallelism
(Calculator.compute(n_jobs=)). OUTER parallelism (many datasets at once)
belongs to the job scheduler — e.g. a PBS array over --array-index.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as im
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import psutil

from pyspi._parallel import COMPUTATION_VERSION
from pyspi.calculator import Calculator, bundled_configs, resolve_config

CATEGORY_PREFIX = ".statistics."  # python module suffix becomes the category field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "results" / "cells"

PRESETS = {
    "headline": {"points": [(10, 500), (20, 1000)], "n_jobs": [1]},
    "scaling": {"M": [4, 8, 16, 32], "T": [200, 400, 800, 1600], "n_jobs": [1]},
    "parallel": {"M": [16], "T": [800], "n_jobs": [1, 2, 4, 8, 16]},
    "amortized": {"M": [8, 16], "T": [800], "n_jobs": [1]},
}

TRACKED_DEPS = (
    "pyspi", "numpy", "scipy", "pandas", "scikit-learn", "statsmodels",
    "mne", "mne-connectivity", "spectral-connectivity", "nitime",
    "hyppo", "tslearn", "dtaidistance", "pyEDM",
    "h5py", "pyyaml", "tqdm",
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=_parse_int_list, default=[8],
                   help="Comma-separated process counts (ignored if --preset is set).")
    p.add_argument("--t", type=_parse_int_list, default=[200],
                   help="Comma-separated observation counts (ignored if --preset is set).")
    p.add_argument("--n-jobs", dest="n_jobs", type=_parse_int_list, default=[1],
                   help="Comma-separated worker counts (ignored if --preset is set).")
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="Predefined M/T/n_jobs grid; overrides --m/--t/--n-jobs.")
    p.add_argument("--config", default="fabfour",
                   help=f"Bundled config name ({'/'.join(bundled_configs())}) "
                        "or a path to your own YAML.")
    p.add_argument("--mp-context", choices=["spawn", "fork", "forkserver"], default="spawn",
                   help="Multiprocessing start method for n_jobs>1 (default: spawn).")
    p.add_argument("--repeats", type=int, default=2, help="Repeats per cell (default: 2).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (default: 0).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for per-cell result JSONs (default: bench/results/cells/).")
    p.add_argument("--label", default=None,
                   help="Filename prefix; default = stem of --config.")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells whose JSON already exists with repeats >= --repeats.")
    p.add_argument("--array-index", type=int, default=None,
                   help="Run only the Nth (1-indexed) cell of the resolved grid. For PBS arrays.")
    return p.parse_args(argv)


def make_calculator(config: str, dataset: np.ndarray) -> Calculator:
    return Calculator(dataset=dataset, config=config, zscore=False, verbose=False)


def spi_metadata(calc: Calculator) -> dict[str, dict]:
    """Return {identifier: {"category": <basic|distance|...>, "labels": [...]}}.

    ``category`` is the python module suffix (``.statistics.basic`` -> ``basic``);
    ``labels`` is the SPI's merged label list (class labels + per-config overrides),
    which includes the ``Mxx`` size-applicability tags alongside stat-type tags.
    """
    out: dict[str, dict] = {}
    for ident, spi in calc._spis.items():
        mod = type(spi).__module__
        cat = mod.split(CATEGORY_PREFIX, 1)[1] if CATEGORY_PREFIX in mod else mod
        labels = list(getattr(spi, "labels", []) or [])
        out[ident] = {"category": cat, "labels": labels}
    return out


_PROC = psutil.Process()


def rss_mb() -> float:
    """Process current RSS in MB (NOT the high-watermark — that monotonically
    accumulates across cells in the same process and gives misleading deltas)."""
    return _PROC.memory_info().rss / (1024.0 * 1024.0)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def dep_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_DEPS:
        try:
            out[name] = im.version(name)
        except im.PackageNotFoundError:
            continue
    return out


def build_environment() -> dict:
    versions = dep_versions()
    payload = ";".join(f"{k}=={v}" for k, v in sorted(versions.items()))
    return {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "pyspi_git_sha": git_sha(),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "dep_versions": versions,
        "dep_fingerprint": "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16],
        "env": {k: os.environ.get(k, "") for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "PYSPI_N_JOBS")},
    }


def cell_seed(base_seed: int, M: int, T: int) -> int:
    """RNG seed for a cell's *data* -- a pure function of (base seed, M, T).

    Deliberately not a function of ``n_jobs``. The whole point of the n_jobs
    sweep is to time the same problem at different worker counts, and seeding on
    n_jobs handed each column of that sweep a different dataset, so a scaling
    curve compared runs on data that were never the same.

    Also not a function of position. It was ``args.seed + i`` with ``i`` the
    index in the *selected* cell list, and under ``--array-index k`` that list
    has one element -- so every array task used ``seed + 1`` while a sequential
    run gave cell k ``seed + k``. Hashing the cell instead survives adding a
    point to the grid, which a positional seed does not.
    """
    digest = hashlib.blake2b(f"{base_seed}|{M}|{T}".encode(),
                             digest_size=8).digest()
    return int.from_bytes(digest, "little")


def cell_identity(base_seed, M, T, n_jobs, config_path, mp_context,
                  environment) -> str:
    """What a stored cell must match for ``--resume`` to reuse it.

    Resume previously checked only that the file existed, had at least
    ``repeats`` repeats and carried no ``"error"`` -- so a cell measured under a
    different config, seed, multiprocessing context or dependency set was
    silently reused, and the grid mixed measurements that were never comparable.

    ``n_jobs`` is part of the identity (it is what the cell measures) but not of
    the data seed. ``repeats`` is deliberately *absent*: reuse is already
    allowed whenever the stored run has at least as many repeats as requested,
    so folding the requested count in here would discard a perfectly good
    10-repeat cell the moment someone asked for 5. ``COMPUTATION_VERSION``
    covers the estimator implementations, so a cell measured before an
    output-changing change is not reused after it.
    """
    payload = json.dumps({
        "cell": [M, T, n_jobs],
        "config": Path(config_path).read_text(),
        "seed": cell_seed(base_seed, M, T),
        "mp_context": mp_context,
        "computation": COMPUTATION_VERSION,
        "pyspi_git_sha": environment["pyspi_git_sha"],
        "python": environment["python_version"],
        "platform": environment["platform"],
        "deps": environment["dep_fingerprint"],
    }, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def summarise(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=0)), 6),
        "values": [round(float(v), 6) for v in arr],
    }


def resolve_grid(args) -> list[tuple[int, int, int]]:
    if args.preset is not None:
        spec = PRESETS[args.preset]
        n_jobs = spec["n_jobs"]
        if "points" in spec:
            mt = list(spec["points"])
        else:
            mt = [(m, t) for m in spec["M"] for t in spec["T"]]
    else:
        mt = [(m, t) for m in args.m for t in args.t]
        n_jobs = args.n_jobs
    return [(m, t, nj) for (m, t) in mt for nj in n_jobs]


def run_cell(M, T, n_jobs, config, mp_context, repeats, seed) -> dict:
    """Run one (M, T, n_jobs) cell ``repeats`` times. Returns a self-contained entry dict."""
    rss_before = rss_mb()
    rng = np.random.default_rng(seed)
    totals: list[float] = []
    per_spi: dict[str, list[float]] = {}
    failed_ids: set[str] = set()
    n_spis = 0
    error = None
    meta: dict[str, dict] = {}

    for _ in range(repeats):
        arr = rng.standard_normal((M, T)).astype(np.float64)
        try:
            calc = make_calculator(config, arr)
            if not meta:
                meta = spi_metadata(calc)
            t0 = time.perf_counter()
            calc.compute(n_jobs=n_jobs, mp_context=mp_context, progress=False)
            totals.append(time.perf_counter() - t0)
            n_spis = len(calc.spis)
            for k, v in calc.timings.items():
                per_spi.setdefault(k, []).append(float(v))
            tbl = calc.table
            for s in calc.spis:
                if bool(np.all(np.isnan(np.asarray(tbl[s])[~np.eye(M, dtype=bool)]))):
                    failed_ids.add(s)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            break

    spi_seconds = {}
    for k, v in per_spi.items():
        entry_v = summarise(v)
        if k in meta:
            entry_v["category"] = meta[k]["category"]
            entry_v["labels"] = meta[k]["labels"]
        spi_seconds[k] = entry_v

    entry = {
        "M": M, "T": T, "n_jobs": n_jobs, "repeats": len(totals),
        "cell_wall_seconds": summarise(totals) if totals else None,
        "n_spis": n_spis,
        "n_spis_failed": len(failed_ids),
        "failed_spis": sorted(failed_ids),
        "rss_mb_end": round(rss_mb(), 1),
        "rss_mb_delta": round(rss_mb() - rss_before, 1),
        "spi_seconds": spi_seconds,
    }
    if error is not None:
        entry["error"] = error
    return entry


def cell_filename(label: str, M: int, T: int, n_jobs: int) -> str:
    return f"{label}_M{M}_T{T}_n{n_jobs}.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = resolve_config(args.config)
    cfg_label = args.config

    cells = resolve_grid(args)
    if args.array_index is not None:
        if not 1 <= args.array_index <= len(cells):
            raise SystemExit(
                f"--array-index {args.array_index} out of range [1, {len(cells)}].")
        cells = [cells[args.array_index - 1]]

    output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(cfg_label).stem

    env = build_environment()
    print(f"[bench] config={cfg_label}  cells={len(cells)}  repeats={args.repeats}  "
          f"mp={args.mp_context}", file=sys.stderr)
    print(f"[bench] output_dir={output_dir}  label={label}", file=sys.stderr)

    t_total = time.perf_counter()
    for i, (M, T, n_jobs) in enumerate(cells, 1):
        path = output_dir / cell_filename(label, M, T, n_jobs)
        identity = cell_identity(args.seed, M, T, n_jobs, config,
                                 args.mp_context, env)
        if args.resume and path.exists():
            try:
                existing = json.loads(path.read_text())
                reusable = (existing.get("repeats", 0) >= args.repeats
                            and "error" not in existing
                            and existing.get("cell_identity") == identity)
                if reusable:
                    print(f"[bench] [{i}/{len(cells)}] M={M} T={T} n_jobs={n_jobs}"
                          f" — skipped (resume: {path.name})", file=sys.stderr)
                    continue
                if existing.get("cell_identity") != identity:
                    print(f"[bench] [{i}/{len(cells)}] {path.name} was measured "
                          f"under different conditions; recomputing.",
                          file=sys.stderr)
            except Exception:
                pass

        print(f"[bench] [{i}/{len(cells)}] M={M} T={T} n_jobs={n_jobs} x{args.repeats}"
              f" -> {path.name}", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        seed = cell_seed(args.seed, M, T)
        entry = run_cell(M, T, n_jobs, config, args.mp_context, args.repeats,
                         seed)
        wall = time.perf_counter() - t0
        # Self-contained per-cell file: include run metadata + environment.
        entry["config"] = cfg_label
        entry["mp_context"] = args.mp_context
        # Both: the base is what was asked for, the effective seed is what the
        # data was actually generated from and is the one a rerun must match.
        entry["seed_base"] = args.seed
        entry["seed"] = seed
        entry["cell_identity"] = identity
        entry["environment"] = env

        if "error" in entry:
            print(f"[bench]   ERROR after {wall:.1f}s: {entry['error']}", file=sys.stderr)
        else:
            cw = entry["cell_wall_seconds"]
            print(f"[bench]   {wall:.1f}s wall (cell mean {cw['mean']:.2f}s "
                  f"+/- {cw['std']:.2f}s, {entry['n_spis']} SPIs, "
                  f"{entry['n_spis_failed']} failed, "
                  f"RSS {entry['rss_mb_end']:.0f} MB (+{entry['rss_mb_delta']:+.0f}))",
                  file=sys.stderr)
        _atomic_write_json(path, entry)

    print(f"[bench] done in {time.perf_counter() - t_total:.1f}s -> {output_dir}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Baseline drift detector for the full SPI set.

For each (dataset, SPI) pair this recomputes the SPI on a frozen dataset and
compares it element-wise against a stored baseline matrix.

What is ENFORCED (hard assertion failure):
  * the baseline SPI set and the current Calculator's SPI set are identical,
    so a newly-broken or renamed SPI cannot disappear by having no baseline;
  * the shape of each matrix;
  * the NaN *pattern*. A SPI going from finite to all-NaN (or back) is a
    categorical regression, not drift — this fork changed NaN-on-failure
    semantics, so that is precisely the signal worth failing on.

What is also ENFORCED:
  * numerical drift in the finite entries. Exceedances are routed to a
    session-end summary table via ``spi_warning_logger`` (see conftest.py) and
    hard-fail this test.

Baselines live in ``tests/data/baselines/<dataset>.npz`` (one MxM array per
SPI identifier) and are regenerated from the *current* fork by
``tests/tools/generate_benchmark_tables.py``. They are a forward-looking
change detector, not an independent oracle: the fork deliberately rewrote the
information-theoretic estimators, so upstream pyspi 2.0.1 values are the wrong
reference for exactly the code that most needs one. Independent correctness
lives in analytic and structural contract tests, including
``test_infotheory_analytic.py`` and ``test_structural_traits.py``.

Frozen fixtures live in ``tests/data/fixtures/`` (not in ``pyspi/data/``: they
are test inputs, not shipped demo data) and are built by
``tests/tools/generate_fixtures.py``. Three generating processes at three widths
— VAR(1) at M=3, coupled map lattice at M=5, Kuramoto at M=7, all T=100 — so the
SPI set is exercised across a range of M.
"""
import os

import sys

import numpy as np
import pytest

from pyspi.calculator import Calculator
from pyspi.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
# Single source of truth for the documented per-fixture exceptions, shared with
# the generator so the two cannot disagree about what is expected to fail.
from generate_benchmark_tables import KNOWN_UNESTIMABLE  # noqa: E402

# Whole-file marker: this suite takes ~3.5 minutes. Skipped by default; run with
#   pytest -m slow tests/test_baseline_drift.py
# or
#   pytest -m '' tests/
pytestmark = pytest.mark.slow

DATASETS = ("var1_M3_T100", "cml_M5_T100", "kuramoto_M7_T100")

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASELINE_DIR = os.path.join(_DATA_DIR, "baselines")
FIXTURE_DIR = os.path.join(_DATA_DIR, "fixtures")

# Seed used when the baselines were generated; must match the generator's
# default so the RNG-consuming SPIs land in the same place.
SEED = 42

# Drift thresholds. A value is OK if EITHER the absolute or the relative test
# passes (the absolute one protects near-zero references).
#
# TIGHT is the default: the benchmark datasets are frozen and every SPI
# measured so far is a deterministic function of them, so re-running the same
# code on the same machine reproduces the baseline to float round-off. 1e-9
# sits well above that (~1e-12) while still catching genuine sub-percent
# regressions that the old blanket RTOL=1e-2 would have hidden.
TIGHT = (1e-12, 1e-9)   # (atol, rtol)
# LOOSE is for SPIs that are genuinely not a pure function of the data under
# this suite's protocol (seed the global RNG, then compute).
LOOSE = (1e-6, 1e-2)

# Per-SPI, not per-module. The previous version applied LOOSE to every SPI in
# the `causal` and `misc` modules on the assumption that the causal-discovery
# optimisers, GP
# restarts and randomised independence tests made them irreproducible. Measured,
# that is false: `tests/tools/measure_reproducibility.py` computes the full
# config twice per fixture and, on all three fixtures, **322 of 322 SPIs
# reproduce bit-exactly** -- max |difference| identically 0, including every
# `anm`/`cds`/`reci`/`ccm`, every `coint_*`, `gpfit_*` (GaussianProcessRegressor
# defaults to n_restarts_optimizer=0, so there are no random restarts),
# `lmfit_*` (random_state pinned to 42) and `ids` (consumes the global RNG,
# which the suite seeds). A module-wide 1e-2 band over 62 SPIs, ~50 of them
# deterministic, is slack that hides deterministic regressions -- exactly the
# failure mode this suite exists to catch.
#
# So the map is empty, and that is a measurement, not an assumption. Re-measure
# with:
#     python tests/tools/measure_reproducibility.py
# and add an entry here -- keyed by SPI identifier, valued (atol, rtol) -- for
# anything that comes back non-zero, with the mechanism named. The tier stays
# defined because a genuinely stochastic SPI (an unpinned permutation test, a
# GPU-backed estimator) is a plausible future addition, and it needs a home
# that is not "the whole module it happens to live in".
LOOSE_SPIS = {}          # e.g. {"some_stochastic_spi": LOOSE}


def _baseline_path(dataset_name):
    return os.path.join(BASELINE_DIR, f"{dataset_name}.npz")


def _load_fixture(dataset_name):
    """Load a frozen fixture; stored (observations, processes) -> 'sp'.

    Must stay in step with ``tests/tools/generate_benchmark_tables.load_fixture``,
    or the baselines and the test would be reading different data.
    """
    return Data(data=os.path.join(FIXTURE_DIR, f"{dataset_name}.npy"),
                dim_order="sp", name=dataset_name)


def _baseline_keys(dataset_name):
    """SPI identifiers stored in a baseline archive, or None if unreadable.

    Reads only the zip central directory, so this is cheap enough to run at
    collection time. Returning None instead of raising is deliberate: a missing
    or corrupt baseline must never break collection of unrelated tests.
    """
    try:
        with np.load(_baseline_path(dataset_name)) as archive:
            return sorted(k for k in archive.files if not k.startswith("__"))
    except Exception:
        return None


def pytest_generate_tests(metafunc):
    """Parametrise over (dataset, SPI) using only the baseline archives.

    Nothing here constructs a Calculator or decompresses a matrix; the actual
    tables come from session-scoped fixtures, so importing this module costs
    nothing.
    """
    if "spi_key" not in metafunc.fixturenames:
        return
    params = []
    for dataset_name in DATASETS:
        keys = _baseline_keys(dataset_name)
        if keys is None:
            params.append(pytest.param(
                dataset_name, None,
                marks=pytest.mark.skip(
                    reason=f"missing/unreadable baseline {_baseline_path(dataset_name)}"
                ),
                id=f"{dataset_name}:<no-baseline>",
            ))
            continue
        params.extend(pytest.param(dataset_name, k, id=f"{dataset_name}:{k}")
                      for k in keys)
    metafunc.parametrize("dataset_name, spi_key", params)


@pytest.fixture(scope="session")
def baseline_tables():
    """dataset -> {spi_key: matrix}, loaded once per session on first use."""
    cache = {}

    def get(dataset_name):
        if dataset_name not in cache:
            with np.load(_baseline_path(dataset_name)) as archive:
                cache[dataset_name] = {
                    k: archive[k] for k in archive.files if not k.startswith("__")
                }
        return cache[dataset_name]

    return get


@pytest.fixture(scope="session")
def current_tables():
    """dataset -> (tables, spi_objects); one full Calculator run per dataset."""
    cache = {}

    def get(dataset_name):
        if dataset_name not in cache:
            np.random.seed(SEED)
            calc = Calculator(dataset=_load_fixture(dataset_name))
            calc.compute()
            cache[dataset_name] = (
                {spi: calc.table[spi].to_numpy() for spi in calc.spis},
                dict(calc.spis),
                dict(calc.errors),
            )
        return cache[dataset_name]

    return get


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_baseline_covers_every_spi(dataset_name, baseline_tables, current_tables):
    """The baseline and the current Calculator must expose the same SPI set.

    Without this, an SPI that is renamed or newly added is silently untested,
    which is how ~45-50 SPIs per dataset escaped the old suite.
    """
    baseline = baseline_tables(dataset_name)
    _, spis, _ = current_tables(dataset_name)
    missing_from_baseline = sorted(set(spis) - set(baseline))
    missing_from_current = sorted(set(baseline) - set(spis))
    assert not missing_from_baseline and not missing_from_current, (
        f"[{dataset_name}] SPI set mismatch. "
        f"No baseline for: {missing_from_baseline}. "
        f"Baseline-only: {missing_from_current}. "
        f"Regenerate with tests/tools/generate_benchmark_tables.py."
    )


def test_baseline_drift(dataset_name, spi_key, baseline_tables, current_tables,
                        spi_warning_logger):
    """Hard-fail on shape, NaN-pattern, or excessive numerical drift."""
    ref = baseline_tables(dataset_name)[spi_key]
    tables, spis, _ = current_tables(dataset_name)
    assert spi_key in tables, (
        f"[{dataset_name}] {spi_key}: present in baseline but not in the current "
        f"Calculator (see test_baseline_covers_every_spi)."
    )
    new = tables[spi_key]

    assert ref.shape == new.shape, (
        f"[{dataset_name}] {spi_key}: shape mismatch "
        f"baseline={ref.shape} new={new.shape}"
    )

    # --- Enforced: NaN pattern -------------------------------------------
    # Compared as masks, never coerced to 0. Folding NaN into 0 on both sides
    # (the old behaviour) turns "this SPI now fails everywhere" into a small
    # numeric drift entry, hiding the one regression class that matters most.
    ref_nan = ~np.isfinite(ref)
    new_nan = ~np.isfinite(new)
    if not np.array_equal(ref_nan, new_nan):
        gained = int(np.sum(new_nan & ~ref_nan))
        lost = int(np.sum(ref_nan & ~new_nan))
        pytest.fail(
            f"[{dataset_name}] {spi_key}: non-finite pattern changed "
            f"({gained} entries became NaN/inf, {lost} became finite). "
            f"baseline non-finite={int(ref_nan.sum())}/{ref.size}, "
            f"current non-finite={int(new_nan.sum())}/{new.size}."
        )

    # --- Enforced: a baseline with nothing in it is not an oracle ---------
    # `if not finite.any(): return` used to pass here, so an SPI that produced
    # an all-NaN column at freeze time was recorded as such and then agreed
    # with itself forever. All three `gd_*` SPIs sat in that state.
    off_diagonal = ~np.eye(ref.shape[0], dtype=bool)
    if spi_key in KNOWN_UNESTIMABLE.get(dataset_name, {}):
        pytest.skip(KNOWN_UNESTIMABLE[dataset_name][spi_key])
    assert np.isfinite(ref[off_diagonal]).any(), (
        f"[{dataset_name}] {spi_key}: the frozen baseline has no finite "
        f"off-diagonal value. An empty column cannot detect a regression; "
        f"either the SPI is broken or it does not belong in the config."
    )

    finite = ~ref_nan

    module_name = spis[spi_key].__module__.split(".")[-1]
    atol, rtol = LOOSE_SPIS.get(spi_key, TIGHT)

    abs_diff = np.zeros_like(ref, dtype=np.float64)
    abs_diff[finite] = np.abs(new[finite] - ref[finite])
    ok = ~finite | (abs_diff <= atol) | (abs_diff <= rtol * np.abs(np.nan_to_num(ref)))
    if np.all(ok):
        return

    max_abs = float(abs_diff[~ok].max())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(ref) > 0, abs_diff / np.abs(ref), np.nan)
    bad_rel = rel[~ok]
    bad_rel = bad_rel[np.isfinite(bad_rel)]
    max_rel = float(bad_rel.max()) if bad_rel.size else float("nan")

    num_interactions = new.size - new.shape[0]
    num_exceed = int(np.count_nonzero(~ok))
    if "undirected" in spis[spi_key].labels:
        num_exceed //= 2
        num_interactions //= 2

    spi_warning_logger(
        f"{dataset_name}:{spi_key}",
        module_name,
        max_abs,
        max_rel,
        num_exceed,
        num_interactions,
    )
    # Reported *and* failed. Logging alone made every tolerance in this file
    # decorative: a deterministic SPI could move by any amount and the suite
    # still exited 0, with the evidence in a summary banner nobody gates on.
    pytest.fail(
        f"[{dataset_name}] {spi_key}: {num_exceed} of {num_interactions} "
        f"interaction(s) exceed the drift tolerance "
        f"(atol={atol:g}, rtol={rtol:g}); max |delta|={max_abs:.4g}, "
        f"max relative={max_rel:.4g}. If the change is intended, say why in "
        f"CHANGELOG.md and regenerate with "
        f"tests/tools/generate_benchmark_tables.py."
    )


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_no_spi_raises_on_the_frozen_fixtures(dataset_name, current_tables):
    """A completed computation is not the same as a clean one.

    `Calculator.compute()` catches per-SPI exceptions and records them in
    `calc.errors`, so the suite could run the whole config to completion over a
    table with failed columns in it and report nothing. Nothing in the config
    is expected to fail on these fixtures; if something legitimately cannot be
    estimated on data this small, the exception belongs in an explicit
    allow-list here with the statistical reason, not in silence.
    """
    _, _, errors = current_tables(dataset_name)
    expected = KNOWN_UNESTIMABLE.get(dataset_name, {})
    unexpected = {k: v for k, v in errors.items() if k not in expected}
    assert not unexpected, (
        f"[{dataset_name}] {len(unexpected)} SPI(s) raised:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(unexpected.items()))
    )
    still_failing = sorted(set(expected) - set(errors))
    assert not still_failing, (
        f"[{dataset_name}] these are listed as unestimable but now succeed; "
        f"remove them from KNOWN_UNESTIMABLE:\n  " + "\n  ".join(still_failing)
    )


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_every_spi_produces_a_finite_value_on_the_frozen_fixtures(
        dataset_name, current_tables):
    """No shipped SPI may be an entirely empty column.

    Partial NaN is legitimate and common -- `gd_*` is defined only where the
    coherence is significant, `sgc_*` where the factorisation converges. An
    SPI with *no* finite off-diagonal value anywhere is not a measurement.
    """
    tables, _, _ = current_tables(dataset_name)
    expected = KNOWN_UNESTIMABLE.get(dataset_name, {})
    empty = []
    for key, matrix in tables.items():
        if key in expected:
            continue
        matrix = np.asarray(matrix, dtype=float)
        off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
        if not np.isfinite(matrix[off_diagonal]).any():
            empty.append(key)
    assert not empty, (
        f"[{dataset_name}] {len(empty)} SPI(s) produced no finite value:\n  "
        + "\n  ".join(sorted(empty))
    )

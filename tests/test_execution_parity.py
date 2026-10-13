"""Serial and parallel must agree on more than the numbers.

``test_parallel.py`` already pins numerical parity. What is unpinned is
*failure* parity: whether an exception, a warning, a wrong-shaped return, or a
non-finite value is reported the same way in both paths, and whether the
failure survives into a structured, inspectable place rather than only a
transient ``warnings.warn``.

These began as red tests and are now green: both paths route through
``_parallel.run_spi``, failures land in ``calc.errors``, worker-side warnings
are returned to the parent and re-emitted there, and ``calc.run_spec`` records
the resolved run.

The misbehaving SPIs live in ``tests/failing_spis.py``; ``tests/`` is put on
PYTHONPATH so spawned workers can import them too.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyspi.calculator import Calculator

CONFIG = str(Path(__file__).parent / "parity_failure_config.yaml")
TESTS_DIR = str(Path(__file__).parent)

MODES = [("serial", 1), ("parallel", 2)]


@pytest.fixture(autouse=True)
def _tests_on_path(monkeypatch):
    """Make tests/ importable here and in spawned workers."""
    monkeypatch.syspath_prepend(TESTS_DIR)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", TESTS_DIR + (os.pathsep + existing if existing else "")
    )


def _dataset():
    rng = np.random.default_rng(0)
    return rng.standard_normal((3, 120))


def _run(n_jobs):
    calc = Calculator(dataset=_dataset(), config=CONFIG, zscore=False, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute(n_jobs=n_jobs, mp_context="spawn", progress=False)
    return calc, [str(w.message) for w in caught]


# --------------------------------------------------------------------------
# Structured error reporting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,n_jobs", MODES)
def test_failures_are_recorded_structurally(mode, n_jobs):
    """A failed SPI must be inspectable after the run, not just warned about."""
    calc, _ = _run(n_jobs)

    errors = getattr(calc, "errors", None)
    assert errors is not None, "Calculator exposes no `errors` mapping."
    assert "always_raises" in errors, (
        f"[{mode}] the failing SPI is absent from calc.errors: {sorted(errors)}"
    )
    assert "deliberate test failure" in str(errors["always_raises"])


def test_serial_and_parallel_report_the_same_failures():
    serial, _ = _run(1)
    parallel, _ = _run(2)

    s_err = set(getattr(serial, "errors", {}) or {})
    p_err = set(getattr(parallel, "errors", {}) or {})
    assert s_err == p_err, (
        f"Failure sets diverge: serial-only={s_err - p_err}, parallel-only={p_err - s_err}"
    )


def test_warnings_survive_the_parallel_boundary():
    """A warning raised inside an SPI must reach the caller in both paths."""
    _, serial_warns = _run(1)
    _, parallel_warns = _run(2)

    assert any("deliberate test warning" in w for w in serial_warns), (
        "Precondition failed: serial path did not surface the SPI's warning."
    )
    assert any("deliberate test warning" in w for w in parallel_warns), (
        "Warning raised in a worker never reached the parent process."
    )


# --------------------------------------------------------------------------
# Output validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,n_jobs", MODES)
def test_wrong_shape_is_a_recorded_failure(mode, n_jobs):
    calc, _ = _run(n_jobs)
    errors = getattr(calc, "errors", {}) or {}
    assert "wrong_shape" in errors, (
        f"[{mode}] an SPI returning a (2,5) matrix for a 3-process dataset was "
        "not recorded as a failure."
    )


@pytest.mark.parametrize("mode,n_jobs", MODES)
def test_non_finite_output_is_flagged(mode, n_jobs):
    calc, _ = _run(n_jobs)
    errors = getattr(calc, "errors", {}) or {}
    assert "non_finite" in errors, (
        f"[{mode}] an SPI returning +inf was not flagged."
    )


# --------------------------------------------------------------------------
# Config snapshot
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,n_jobs", MODES)
def test_run_records_its_resolved_specification(mode, n_jobs):
    """One canonical, immutable description of what was actually run."""
    calc, _ = _run(n_jobs)
    spec = getattr(calc, "run_spec", None)
    assert spec is not None, "Calculator records no resolved run specification."
    for field in ("config", "zscore", "detrend", "n_processes", "spi_identifiers"):
        assert field in spec, f"[{mode}] run_spec is missing {field!r}."

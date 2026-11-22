"""SPIs that refuse their input as not appropriate: ``pyspi.base.NotAppropriate``.

An SPI or a pair whose input cannot support the estimate (too short, tied)
raises ``NotAppropriate``. The Calculator then stores NaN with
``Quality.NAN`` and keeps the reason: in ``calc.not_appropriate`` for the
whole SPI, in ``calc.pair_not_appropriate`` for single pairs. This is not a
failure, so neither is in ``calc.errors`` or ``calc.pair_errors``, where any
other exception goes, as ``Quality.ERROR``. These tests pin that in the
serial and parallel paths, through checkpoints and through ``.npz`` files.

The stub SPIs live in ``tests/failing_spis.py``; ``tests/`` is put on
PYTHONPATH so spawned workers can import them too.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyspi import _parallel
from pyspi.base import NotAppropriate, collect_pair_failures, fill_pairwise
from pyspi.calculator import Calculator, load_results
from pyspi.quality import Quality

CONFIG = str(Path(__file__).parent / "not_appropriate_config.yaml")
TESTS_DIR = str(Path(__file__).parent)
M = 3
OFF = ~np.eye(M, dtype=bool)
GOOD, ERROR, NAN = Quality.GOOD, Quality.ERROR, Quality.NAN

WHOLE = "needs more than 40 samples"
PAIRS = "the series are too short for this lag"
ONE_PAIR = "pair (0, 1) has tied values"


@pytest.fixture(autouse=True)
def _tests_on_path(monkeypatch):
    """Make tests/ importable here and in spawned workers."""
    monkeypatch.syspath_prepend(TESTS_DIR)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", TESTS_DIR + (os.pathsep + existing if existing else "")
    )


def _calc():
    data = np.random.default_rng(0).standard_normal((M, 40))
    return Calculator(dataset=data, config=CONFIG, zscore=False, verbose=False)


def _compute(calc, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False, **kwargs)
    return calc


def _state(calc):
    """Everything a run records about cells with no value, for comparing runs."""
    quality = {k: calc.quality[k].to_numpy().tolist() for k in calc.spis}
    return (quality, calc.errors, calc.pair_errors, calc.not_appropriate,
            calc.pair_not_appropriate)


@pytest.fixture(scope="module")
def serial():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(TESTS_DIR)
        return _compute(_calc())


def _q(calc, key):
    return calc.quality[key].to_numpy()


def test_not_appropriate_is_a_value_error():
    assert issubclass(NotAppropriate, ValueError)
    from pyspi.data import Data
    from failing_spis import NeverAppropriate
    data = Data(np.random.default_rng(1).standard_normal((2, 5)), dim_order="ps")
    with pytest.raises(ValueError, match="needs more than 5 samples"):
        NeverAppropriate().multivariate(data)


def test_a_whole_spi_not_appropriate_is_nan_with_its_reason(serial):
    q = _q(serial, "never_appropriate")
    assert (q == NAN).all()
    assert np.isnan(serial.table["never_appropriate"].to_numpy(dtype=float)).all()
    assert serial.not_appropriate["never_appropriate"] == WHOLE
    assert "never_appropriate" not in serial.errors
    assert "never_appropriate" not in serial.pair_not_appropriate


def test_every_pair_not_appropriate_gives_the_reason_not_the_generic_message(serial):
    assert (_q(serial, "no_pair_appropriate") == NAN).all()
    assert serial.not_appropriate["no_pair_appropriate"] == PAIRS
    assert "no_pair_appropriate" not in serial.errors


def test_one_pair_not_appropriate_is_nan_and_the_others_stay_good(serial):
    q = _q(serial, "one_pair_not_appropriate")
    expected = np.full((M, M), GOOD, dtype=np.uint8)
    np.fill_diagonal(expected, NAN)
    expected[0, 1] = NAN
    np.testing.assert_array_equal(q, expected)
    procs = list(serial.dataset.procnames)
    assert serial.pair_not_appropriate["one_pair_not_appropriate"] == [
        (procs[0], procs[1], ONE_PAIR)]
    for record in (serial.errors, serial.pair_errors, serial.not_appropriate):
        assert "one_pair_not_appropriate" not in record


def test_an_ordinary_exception_is_still_an_error(serial):
    q = _q(serial, "always_raises")
    assert (q[OFF] == ERROR).all()
    assert serial.errors["always_raises"].startswith("RuntimeError: ")
    assert "always_raises" not in serial.not_appropriate
    assert (_q(serial, "cov_EmpiricalCovariance")[OFF] == GOOD).all()


def test_a_real_failure_among_not_appropriate_pairs_is_the_spi_error(serial):
    """When every pair raises, an ordinary exception outranks the refusals."""
    assert serial.errors["mixed_pair_failures"].startswith("RuntimeError: ")
    assert (_q(serial, "mixed_pair_failures")[OFF] == ERROR).all()


def test_fill_pairwise_marks_not_appropriate_pairs_by_prefix():
    A = np.full((2, 2), np.nan)

    def compute(i, j):
        if (i, j) == (0, 1):
            raise NotAppropriate("too short")
        raise KeyError("bug")

    with collect_pair_failures() as failures, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(KeyError):
            fill_pairwise("x", A, [(0, 1), (1, 0)], compute)
    assert failures == []  # re-raised for the whole SPI, so nothing per pair

    with collect_pair_failures() as failures, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fill_pairwise("x", A, [(0, 1), (1, 0)],
                      lambda i, j: 1.0 if i else compute(i, j))
    assert failures == [(0, 1, "NotAppropriate: too short")]


def test_derive_quality_reads_a_not_appropriate_reason_as_nan():
    values = np.full((M, M), np.nan)
    np.testing.assert_array_equal(
        _parallel.derive_quality(values, "NotAppropriate: " + WHOLE)[OFF], NAN)
    np.testing.assert_array_equal(
        _parallel.derive_quality(values, "RuntimeError: boom")[OFF], ERROR)


def test_two_spawned_workers_agree_with_serial(serial):
    parallel = _compute(_calc(), n_jobs=2, mp_context="spawn")
    assert _state(parallel) == _state(serial)


def test_checkpoint_resume_keeps_codes_and_reasons(serial, tmp_path):
    _compute(_calc(), checkpoint_dir=tmp_path)
    resumed = _compute(_calc(), checkpoint_dir=tmp_path, retry_failed=False)
    assert _state(resumed) == _state(serial)


def test_checkpoint_without_quality_sidecars_derives_nan(serial, tmp_path):
    _compute(_calc(), checkpoint_dir=tmp_path)
    for path in tmp_path.glob("*.quality"):
        path.unlink()
    resumed = _compute(_calc(), checkpoint_dir=tmp_path, retry_failed=False)
    assert (_q(resumed, "never_appropriate") == NAN).all()
    assert resumed.not_appropriate["never_appropriate"] == WHOLE
    assert (_q(resumed, "always_raises")[OFF] == ERROR).all()


def test_resume_retries_failures_and_keeps_not_appropriate_spis(tmp_path):
    """A not-appropriate SPI has not failed, so its checkpoint is not recomputed."""
    _compute(_calc(), checkpoint_dir=tmp_path)
    resumed = _compute(_calc(), checkpoint_dir=tmp_path)
    recomputed = {k for k, seconds in resumed.timings.items() if seconds > 0}
    assert recomputed == set(resumed.errors)
    assert set(resumed.not_appropriate) == {"never_appropriate", "no_pair_appropriate"}


def test_a_not_appropriate_checkpoint_that_holds_values_is_recomputed(tmp_path):
    """The message and the values disagree, so the checkpoint is not trusted."""
    _compute(_calc(), checkpoint_dir=tmp_path)
    np.save(tmp_path / "never_appropriate.npy", np.ones((M, M)))
    (tmp_path / "never_appropriate.quality").unlink()
    resumed = _compute(_calc(), checkpoint_dir=tmp_path)
    assert resumed.timings["never_appropriate"] > 0
    assert (_q(resumed, "never_appropriate") == NAN).all()
    assert resumed.not_appropriate["never_appropriate"] == WHOLE


def test_load_table_keeps_the_not_appropriate_reasons(serial, tmp_path):
    from pyspi import load_table
    table = load_table(serial.save(tmp_path / "r.npz"))
    assert table.attrs["not_appropriate"] == serial.not_appropriate
    assert table.attrs["errors"] == serial.errors


def test_npz_round_trip_keeps_codes_and_reasons(serial, tmp_path):
    res = load_results(serial.save(tmp_path / "r.npz"))
    for key in serial.spis:
        np.testing.assert_array_equal(res.quality[key].to_numpy(), _q(serial, key))
    assert res.errors == serial.errors
    assert res.pair_errors == serial.pair_errors
    assert res.not_appropriate == serial.not_appropriate
    assert res.pair_not_appropriate == serial.pair_not_appropriate


def test_summary_counts_a_not_appropriate_spi_apart_from_the_failed(serial):
    summary = serial.summary()
    assert summary["not_appropriate"] == ["never_appropriate", "no_pair_appropriate"]
    assert summary["n_not_appropriate"] == 2
    assert summary["failed"] == sorted(serial.errors)
    assert summary["n_failed"] == len(serial.errors)
    assert not set(summary["failed"]) & set(summary["not_appropriate"])
    assert "2 not appropriate" in repr(serial)


def test_a_not_appropriate_spi_warns_that_it_is_not_appropriate():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(TESTS_DIR)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _calc().compute(progress=False)
    messages = [str(w.message) for w in record]
    assert any(m == f'SPI "never_appropriate" is not appropriate for this '
                    f'dataset: {WHOLE}' for m in messages)
    assert not any("Caught error" in m and "never_appropriate" in m for m in messages)

"""Per-cell quality codes: ``Calculator.quality`` and everything that stores it.

Each cell of ``calc.table`` has a ``uint8`` code in ``calc.quality``
(``pyspi.quality.Quality``, shared with pyhctsa): GOOD cells hold a finite
value, and every other cell is NaN with a code saying why. These tests pin the
codes for each way an SPI can misbehave, that the serial and parallel paths
agree on them, and that checkpoints and ``.npz`` files keep them -- including
files written before the codes existed.

The stub SPIs live in ``tests/failing_spis.py``; ``tests/`` is put on
PYTHONPATH so spawned workers can import them too.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyspi
from pyspi import _parallel
from pyspi.base import collect_pair_failures, fill_pairwise
from pyspi.calculator import Calculator, load_results, load_table
from pyspi.quality import Quality

CONFIG = str(Path(__file__).parent / "quality_config.yaml")
TESTS_DIR = str(Path(__file__).parent)
M = 3
OFF = ~np.eye(M, dtype=bool)

GOOD, ERROR, NAN = Quality.GOOD, Quality.ERROR, Quality.NAN


@pytest.fixture(autouse=True)
def _tests_on_path(monkeypatch):
    """Make tests/ importable here and in spawned workers."""
    monkeypatch.syspath_prepend(TESTS_DIR)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", TESTS_DIR + (os.pathsep + existing if existing else "")
    )


def _dataset():
    return np.random.default_rng(0).standard_normal((M, 80))


def _calc():
    return Calculator(dataset=_dataset(), config=CONFIG, zscore=False, verbose=False)


def _compute(calc, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute(progress=False, **kwargs)
    return [str(w.message) for w in caught]


@pytest.fixture(scope="module")
def serial():
    """One serial run, shared by the read-only tests below."""
    calc = _calc()
    msgs = _compute(calc)
    return calc, msgs


def _q(calc, key):
    return calc.quality[key].to_numpy()


def _v(calc, key):
    return calc.table[key].to_numpy(dtype=float)


def _expected(off_code, cells=()):
    """Diagonal NAN, off-diagonal ``off_code``, then ``cells`` overrides."""
    q = np.full((M, M), off_code, dtype=np.uint8)
    np.fill_diagonal(q, NAN)
    for (i, j), code in cells:
        q[i, j] = code
    return q


# --------------------------------------------------------------------------
# The code for each kind of result
# --------------------------------------------------------------------------

def test_quality_frame_matches_the_table_and_starts_not_computed():
    calc = _calc()
    assert calc.quality.shape == calc.table.shape
    assert calc.quality.index.equals(calc.table.index)
    assert calc.quality.columns.equals(calc.table.columns)
    assert set(calc.quality.dtypes) == {np.dtype(np.uint8)}
    assert (calc.quality.to_numpy() == Quality.NOT_COMPUTED).all()


def test_values_are_nan_exactly_where_the_code_is_not_good(serial):
    calc, _ = serial
    assert set(calc.quality.dtypes) == {np.dtype(np.uint8)}
    q = calc.quality.to_numpy()
    v = calc.table.to_numpy(dtype=float)
    assert np.array_equal(q == GOOD, np.isfinite(v))
    assert not np.isinf(v).any()


def test_a_raising_spi_is_error_off_the_diagonal(serial):
    calc, _ = serial
    assert np.array_equal(_q(calc, "always_raises"), _expected(ERROR))
    assert "deliberate test failure" in calc.errors["always_raises"]


def test_an_infinite_cell_keeps_the_finite_ones(serial):
    """Inf used to fail the whole SPI; now only its cells are lost."""
    calc, msgs = serial
    key = "one_infinite"
    assert np.array_equal(_q(calc, key), _expected(
        GOOD, [((0, 1), Quality.INF), ((1, 0), Quality.NEG_INF)]))
    expected = np.arange(M * M, dtype=float).reshape(M, M) + 1.0
    good = _q(calc, key) == GOOD
    np.testing.assert_array_equal(_v(calc, key)[good], expected[good])
    assert key not in calc.errors
    assert any(key in m and "1 Inf, 1 -Inf" in m for m in msgs), msgs


def test_complex_output_is_labelled_not_truncated(serial):
    """`np.array(S, dtype=float)` used to drop the imaginary part silently."""
    calc, msgs = serial
    key = "complex_output"
    assert np.array_equal(_q(calc, key), _expected(GOOD, [((0, 2), Quality.COMPLEX)]))
    assert np.isnan(_v(calc, key)[0, 2])
    assert _v(calc, key)[2, 0] == 7.0           # a real cell keeps its real part
    assert key not in calc.errors
    assert any(key in m and "1 complex" in m for m in msgs), msgs


def test_an_all_nan_spi_keeps_its_nan_codes_and_its_error(serial):
    """The all-NaN error entry stays: three group-delay SPIs once shipped
    all-NaN with no warning."""
    calc, _ = serial
    assert np.array_equal(_q(calc, "all_nan"), _expected(NAN))
    assert calc.errors["all_nan"] == _parallel.NO_FINITE_VALUES


def test_a_wrong_shape_is_error():
    from failing_spis import WrongShape
    res = _parallel.execute_spi(WrongShape(), pyspi.calculator.Data(_dataset()),
                                "wrong_shape", M)
    assert res.err.startswith("ValueError: SPI returned shape")
    assert np.array_equal(res.quality, _expected(ERROR))


def test_a_healthy_spi_is_good_off_the_diagonal(serial):
    calc, _ = serial
    assert np.array_equal(_q(calc, "cov_EmpiricalCovariance"), _expected(GOOD))


# --------------------------------------------------------------------------
# Pair failures inside fill_pairwise
# --------------------------------------------------------------------------

def test_a_failed_pair_is_error_and_its_message_is_kept(serial):
    calc, msgs = serial
    key = "one_pair_raises"
    assert np.array_equal(_q(calc, key), _expected(GOOD, [((0, 1), ERROR)]))
    assert key not in calc.errors
    procs = list(calc.dataset.procnames)
    assert calc.pair_errors[key] == [
        (procs[0], procs[1], "ValueError: pair (0, 1) is undefined")]
    # The existing warning is still raised.
    assert any("undefined and set to NaN" in m and key in m for m in msgs)


def test_an_undirected_failure_follows_the_mirror(serial):
    """Undirected copies the upper triangle over the lower one, so an upper
    failure is two ERROR cells and a lower one is overwritten by a good value."""
    calc, _ = serial
    upper, lower = "one_pair_raises_undirected_01", "one_pair_raises_undirected_10"
    assert np.array_equal(_q(calc, upper),
                          _expected(GOOD, [((0, 1), ERROR), ((1, 0), ERROR)]))
    assert [(s, t) for s, t, _ in calc.pair_errors[upper]] == [
        (calc.dataset.procnames[0], calc.dataset.procnames[1]),
        (calc.dataset.procnames[1], calc.dataset.procnames[0])]
    assert np.array_equal(_q(calc, lower), _expected(GOOD))
    assert lower not in calc.pair_errors
    assert _v(calc, lower)[1, 0] == 1.0


def test_fill_pairwise_reports_to_an_open_collector_only():
    def compute(i, j):
        if (i, j) == (0, 1):
            raise ValueError("nope")
        return 1.0

    pairs = [(0, 1), (0, 2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with collect_pair_failures() as failures:
            fill_pairwise("x", np.full((3, 3), np.nan), pairs, compute, mirror=True)
        assert failures == [(0, 1, "ValueError: nope"), (1, 0, "ValueError: nope")]
        # No collector open: nothing recorded, nothing raised.
        fill_pairwise("x", np.full((3, 3), np.nan), pairs, compute)
        with collect_pair_failures() as failures:
            with pytest.raises(ValueError):     # every pair fails: the SPI raises
                fill_pairwise("x", np.full((3, 3), np.nan), [(0, 1)], compute)
        assert failures == []


# --------------------------------------------------------------------------
# Execution paths and checkpoints
# --------------------------------------------------------------------------

def test_serial_and_parallel_give_identical_quality(serial):
    calc_s, _ = serial
    calc_p = _calc()
    _compute(calc_p, n_jobs=2, mp_context="spawn")
    assert calc_p.quality.equals(calc_s.quality)
    assert calc_p.pair_errors == calc_s.pair_errors
    assert calc_p.errors == calc_s.errors
    np.testing.assert_array_equal(calc_p.table.to_numpy(), calc_s.table.to_numpy())


def test_checkpoint_resume_restores_quality(serial, tmp_path):
    calc_s, _ = serial
    first = _calc()
    _compute(first, checkpoint_dir=tmp_path)
    assert first.quality.equals(calc_s.quality)

    resumed = _calc()
    # retry_failed=False, so the failed SPIs are resumed too, not recomputed.
    _compute(resumed, checkpoint_dir=tmp_path, retry_failed=False)
    assert all(t == 0.0 for t in resumed.timings.values()), resumed.timings
    assert resumed.quality.equals(calc_s.quality)
    assert resumed.pair_errors == calc_s.pair_errors
    assert resumed.errors == calc_s.errors


def test_checkpoint_without_quality_derives_it(tmp_path):
    """A checkpoint written before quality codes existed: values plus `.error`."""
    calc = _calc()
    _parallel.write_manifest(tmp_path, calc.run_digest, calc.run_spec)
    good = np.arange(9.0).reshape(3, 3)
    np.fill_diagonal(good, np.nan)
    good[0, 2] = np.nan                         # one undefined cell
    for key in calc.spis:
        np.save(tmp_path / f"{key}.npy", good)
    # A pre-quality failure: all NaN, with the message the SPI raised.
    np.save(tmp_path / "always_raises.npy", np.full((3, 3), np.nan))
    (tmp_path / "always_raises.error").write_text("RuntimeError: old failure")
    np.save(tmp_path / "all_nan.npy", np.full((3, 3), np.nan))
    (tmp_path / "all_nan.error").write_text(_parallel.NO_FINITE_VALUES)

    _compute(calc, checkpoint_dir=tmp_path, retry_failed=False)
    assert np.array_equal(_q(calc, "always_raises"), _expected(ERROR))
    assert np.array_equal(_q(calc, "all_nan"), _expected(NAN))
    assert np.array_equal(_q(calc, "one_pair_raises"),
                          _expected(GOOD, [((0, 2), NAN)]))
    assert calc.pair_errors == {}


def test_a_stale_quality_sidecar_is_not_trusted(tmp_path):
    """Codes that disagree with the stored values are re-derived from them."""
    calc = _calc()
    _compute(calc, checkpoint_dir=tmp_path)
    key = "cov_EmpiricalCovariance"
    with open(tmp_path / f"{key}.quality", "wb") as fh:
        np.save(fh, np.full((3, 3), ERROR, dtype=np.uint8))
    resumed = _calc()
    _compute(resumed, checkpoint_dir=tmp_path)
    assert np.array_equal(_q(resumed, key), _expected(GOOD))


# --------------------------------------------------------------------------
# .npz files
# --------------------------------------------------------------------------

def test_npz_round_trips_quality(serial, tmp_path):
    calc, _ = serial
    path = calc.save(tmp_path / "r.npz")
    with np.load(path) as f:
        assert f["quality"].dtype == np.uint8
        assert f["quality"].shape == (calc.n_spis, M, M)
        assert int(f["schema"]) == 2

    res = load_results(path)
    assert isinstance(res, pyspi.Results)
    assert res.table.equals(calc.table)
    assert res.quality.equals(calc.quality)
    assert res.quality.index.equals(res.table.index)
    assert res.quality.columns.equals(res.table.columns)
    assert res.errors == calc.errors
    assert res.pair_errors == calc.pair_errors
    assert res.run_spec["spi_identifiers"] == calc.run_spec["spi_identifiers"]


def test_load_table_keeps_quality_out_of_attrs(serial, tmp_path):
    """A DataFrame in `attrs` breaks `pd.concat` and is deep-copied by every
    slice, so `load_table` keeps only its small metadata there."""
    calc, _ = serial
    path = calc.save(tmp_path / "r.npz")
    t1, t2 = load_table(path), load_table(path)
    assert set(t1.attrs) == {"schema", "run_spec", "errors", "run_digest"}
    assert t1.equals(calc.table)
    pd.concat([t1, t2])


def _write_schema1(path, calc):
    """What `save()` wrote before quality codes existed (schema 1)."""
    keys = list(calc.spis)
    np.savez_compressed(
        path,
        values=np.stack([calc.table[k].to_numpy(dtype=float) for k in keys]),
        spis=np.array(keys, dtype="U"),
        processes=np.array(calc.dataset.procnames, dtype="U"),
        schema=np.array(1),
        run_spec=np.array(json.dumps(calc.run_spec, default=str), dtype="U"),
        run_digest=np.array(calc.run_digest, dtype="U"),
        errors=np.array(json.dumps(calc.errors), dtype="U"),
    )


def test_an_old_schema_file_loads_with_derived_quality(serial, tmp_path):
    calc, _ = serial
    path = tmp_path / "old.npz"
    _write_schema1(path, calc)

    table = load_table(path)
    assert table.equals(calc.table) and table.attrs["schema"] == 1

    res = load_results(path)
    assert res.table.equals(calc.table)
    assert set(res.quality.dtypes) == {np.dtype(np.uint8)}
    q = lambda k: res.quality[k].to_numpy()
    assert np.array_equal(q("always_raises"), _expected(ERROR))
    assert np.array_equal(q("all_nan"), _expected(NAN))
    assert np.array_equal(q("cov_EmpiricalCovariance"), _expected(GOOD))
    # What a schema-1 file cannot say: which NaN cells were failed pairs, and
    # what the Inf or complex cells were.
    assert np.array_equal(q("one_pair_raises"), _expected(GOOD, [((0, 1), NAN)]))
    assert np.array_equal(q("one_infinite"),
                          _expected(GOOD, [((0, 1), NAN), ((1, 0), NAN)]))
    assert res.pair_errors is None
    assert res.errors == calc.errors


def test_a_schema0_file_with_only_values_loads(tmp_path):
    path = tmp_path / "bare.npz"
    v = np.array([[[np.nan, 1.0], [np.inf, np.nan]]])
    np.savez_compressed(path, values=v, spis=np.array(["s"], dtype="U"),
                        processes=np.array(["a", "b"], dtype="U"))
    res = load_results(path)
    assert res.quality["s"].to_numpy().tolist() == [[NAN, GOOD], [Quality.INF, NAN]]
    assert res.errors is None and res.run_spec is None and res.pair_errors is None


# --------------------------------------------------------------------------
# Reporting and the older entry points
# --------------------------------------------------------------------------

def test_inspect_calc_results_reports_by_label(serial, capsys):
    from pyspi.utils import inspect_calc_results
    calc, _ = serial
    inspect_calc_results(calc)
    out = capsys.readouterr().out
    assert "SPI Computation Results Summary" in out
    assert "always_raises (6 error)" in out
    assert "all_nan (6 NaN)" in out
    assert "one_infinite (1 Inf, 1 -Inf)" in out
    assert "complex_output (1 complex)" in out
    assert "one_pair_raises (1 error)" in out


# --------------------------------------------------------------------------
# Review fixes: the shared module's edge cases, Undirected, checkpoints, M=1
# --------------------------------------------------------------------------

def test_shared_module_edge_cases():
    from pyspi.quality import classify, quality_from_values
    nanj = complex(np.nan, np.nan)
    # An undefined complex value (0/0) is NaN, not a bug.
    assert classify(nanj)[1] == NAN
    assert classify(np.complex128(nanj))[1] == NAN
    assert classify(1 + 2j)[1] == Quality.COMPLEX
    x, q = quality_from_values(np.array([[nanj, 1 + 0j], [1 + 1j, np.inf + 0j]]))
    assert q.tolist() == [[NAN, GOOD], [Quality.COMPLEX, Quality.INF]]
    assert np.isnan(x[0, 0]) and x[0, 1] == 1.0
    # A Python int beyond the float range is +/-Inf, not an OverflowError.
    assert classify(10 ** 400)[1] == Quality.INF
    assert classify(-10 ** 400)[1] == Quality.NEG_INF
    # A one-element object array holding None is EMPTY.
    assert classify(np.array([None], dtype=object))[1] == Quality.EMPTY


def test_a_complex_nan_cell_from_an_spi_is_nan():
    from failing_spis import ComplexOutput

    class ComplexNaN(ComplexOutput):
        def multivariate(self, data):
            A = super().multivariate(data)
            A[1, 2] = complex(np.nan, np.nan)
            return A

    res = _parallel.execute_spi(ComplexNaN(), pyspi.calculator.Data(_dataset()),
                                "complex_nan", M)
    assert np.array_equal(res.quality, _expected(
        GOOD, [((0, 2), Quality.COMPLEX), ((1, 2), NAN)]))


def _undirected(values):
    """An Undirected stub whose bivariate returns ``values[(i, j)]``, raising
    when that is an exception and returning 1.0 for unlisted pairs."""
    from pyspi.base import Undirected, Unsigned, parse_bivariate

    class Stub(Undirected, Unsigned):
        identifier = "stub_undirected"
        labels = ["test"]

        @parse_bivariate
        def bivariate(self, data, i=None, j=None):
            v = values.get((i, j), 1.0)
            if isinstance(v, Exception):
                raise v
            return v
    return Stub()


def _execute(spi):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _parallel.execute_spi(spi, pyspi.calculator.Data(_dataset()), "stub", M)
    return res, [str(w.message) for w in caught] + res.warns


def test_an_undirected_lower_failure_is_discarded_with_its_value():
    """(0, 1) returns NaN and (1, 0) raises: the lower cell is overwritten by
    the upper one, so its failure is discarded too, and the codes stay
    symmetric (both NAN), with no pair error and no warning."""
    res, msgs = _execute(_undirected({(0, 1): np.nan, (1, 0): ValueError("lower")}))
    assert np.array_equal(res.quality, _expected(GOOD, [((0, 1), NAN), ((1, 0), NAN)]))
    assert np.array_equal(res.quality, res.quality.T)
    assert res.pair_errors == []
    assert not any("undefined and set to NaN" in m for m in msgs), msgs


def test_an_undirected_upper_failure_is_mirrored_and_counted_once():
    res, msgs = _execute(_undirected({(0, 1): ValueError("upper"),
                                      (1, 0): ValueError("lower")}))
    assert np.array_equal(res.quality, _expected(GOOD, [((0, 1), ERROR), ((1, 0), ERROR)]))
    assert res.pair_errors == [(0, 1, "ValueError: upper"), (1, 0, "ValueError: upper")]
    warned = [m for m in msgs if "undefined and set to NaN" in m]
    assert len(warned) == 1 and "1 of 3 pairs" in warned[0], warned


def test_an_undirected_spi_whose_kept_pairs_all_fail_raises():
    """Only the lower triangle succeeds: none of it is kept, so the SPI fails
    with the estimator's message."""
    res, _ = _execute(_undirected({(i, j): ValueError("kept pair")
                                   for i in range(M) for j in range(M) if i < j}))
    assert res.err == "ValueError: kept pair"
    assert np.array_equal(res.quality, _expected(ERROR))


def test_an_interrupted_checkpoint_write_is_not_resumed(tmp_path, monkeypatch):
    """The values file is the commit point: an interruption after the sidecars
    must not leave the previous run's values beside them."""
    key = "k"
    old = np.ones((M, M))
    np.fill_diagonal(old, np.nan)
    _parallel.write_checkpoint(tmp_path, key, old, None, _expected(GOOD), [])
    done, _ = _parallel.load_checkpoints(tmp_path, [key], M, retry_failed=False)
    assert key in done

    real = _parallel._atomic_npy_write

    def interrupted(path, arr):
        if path.name == f"{key}.npy":
            raise KeyboardInterrupt
        real(path, arr)

    monkeypatch.setattr(_parallel, "_atomic_npy_write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _parallel.write_checkpoint(tmp_path, key, np.full((M, M), np.nan),
                                   "RuntimeError: new failure", _expected(ERROR), [])
    done, remaining = _parallel.load_checkpoints(tmp_path, [key], M, retry_failed=False)
    assert remaining == [key] and not done


def test_a_checkpoint_with_an_inf_cell_resumes_it_as_inf(tmp_path):
    """Inf is a per-cell code, so a checkpoint holding one is not a failure."""
    arr = np.ones((M, M))
    np.fill_diagonal(arr, np.nan)
    arr[0, 1], arr[1, 0] = np.inf, -np.inf
    np.save(tmp_path / "k.npy", arr)
    done, remaining = _parallel.load_checkpoints(tmp_path, ["k"], M)
    assert remaining == []
    res = done["k"]
    assert res.err is None
    assert np.array_equal(res.quality, _expected(
        GOOD, [((0, 1), Quality.INF), ((1, 0), Quality.NEG_INF)]))
    assert np.isnan(res.values[0, 1]) and res.values[0, 2] == 1.0


def test_inspect_calc_results_with_one_process(capsys):
    from pyspi.utils import inspect_calc_results
    calc = Calculator(dataset=np.random.default_rng(1).standard_normal((1, 80)),
                      config=CONFIG, zscore=False, verbose=False)
    _compute(calc)
    inspect_calc_results(calc)
    out = capsys.readouterr().out
    assert "no off-diagonal values" in out

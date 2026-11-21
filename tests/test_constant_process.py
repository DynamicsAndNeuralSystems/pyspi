"""A constant process is refused by ``Data``, by name and index, whatever the preprocessing.

No pairwise statistic is defined for a series without variation. Z-scoring
would turn it into NaN and detrending an exactly linear series would leave
only rounding noise, so the check runs on the input, before either step.
"""
import numpy as np
import pytest

from pyspi import Calculator
from pyspi import __main__ as cli
from pyspi.data import Data

PREPROCESSING = [
    {"zscore": z, "detrend": d} for z in (True, False) for d in (True, False)
]


def _series(M=4, T=120, seed=0):
    return np.random.default_rng(seed).standard_normal((M, T))


@pytest.mark.parametrize("prep", PREPROCESSING, ids=lambda p: f"z{p['zscore']:d}-d{p['detrend']:d}")
def test_one_constant_process_is_named(prep):
    X = _series()
    X[2] = 3.0
    with pytest.raises(ValueError, match=r"Constant process\(es\) 'c' \(index 2\)") as exc:
        Data(X, dim_order="ps", procnames=list("abcd"), **prep)
    assert "non-finite" not in str(exc.value)
    assert "no pairwise statistic is defined" in str(exc.value)


@pytest.mark.parametrize("prep", PREPROCESSING, ids=lambda p: f"z{p['zscore']:d}-d{p['detrend']:d}")
def test_several_constant_processes_are_all_named(prep):
    X = _series(M=5)
    X[1] = 0.0
    X[4] = -7.5
    with pytest.raises(ValueError) as exc:
        Data(X, dim_order="ps", procnames=["v", "w", "x", "y", "z"], **prep)
    message = str(exc.value)
    assert "'w' (index 1), 'z' (index 4)" in message
    assert "'x'" not in message


def test_default_names_are_used_without_procnames():
    X = _series()
    X[3] = 1.0
    with pytest.raises(ValueError, match=r"'proc-3' \(index 3\)"):
        Data(X, dim_order="ps")


def test_dataframe_labels_name_the_constant_process():
    pd = pytest.importorskip("pandas")
    X = _series().T
    X[:, 1] = 2.0
    frame = pd.DataFrame(X, columns=["Fz", "Cz", "Pz", "Oz"])
    with pytest.raises(ValueError, match=r"'Cz' \(index 1\)"):
        Data(frame, dim_order="sp")


@pytest.mark.parametrize("zscore", [True, False])
def test_exactly_linear_process_is_constant_once_detrended(zscore):
    X = _series()
    X[0] = 5.0 + 0.3 * np.arange(X.shape[1])
    with pytest.raises(ValueError, match=r"'proc-0' \(index 0\).*linear trend"):
        Data(X, dim_order="ps", detrend=True, zscore=zscore)
    # Without detrending a ramp varies, and is accepted.
    Data(X, dim_order="ps", detrend=False, zscore=zscore)


@pytest.mark.parametrize("offset", [0.0, 1.0, -1e6, 1e9])
@pytest.mark.parametrize("slope", [1e-3, 1.0, 1e3])
def test_linear_series_of_any_offset_and_slope_is_caught_when_detrended(offset, slope):
    X = _series(M=3, T=5000)
    X[1] = offset + slope * np.arange(X.shape[1])
    with pytest.raises(ValueError, match=r"'proc-1' \(index 1\)"):
        Data(X, dim_order="ps", detrend=True)


@pytest.mark.parametrize("prep", PREPROCESSING, ids=lambda p: f"z{p['zscore']:d}-d{p['detrend']:d}")
def test_tiny_variation_around_zero_is_accepted(prep):
    X = 1e-12 * _series()
    data = Data(X, dim_order="ps", **prep)
    assert np.isfinite(data.to_numpy()).all()


@pytest.mark.parametrize("prep", PREPROCESSING, ids=lambda p: f"z{p['zscore']:d}-d{p['detrend']:d}")
def test_small_variation_on_a_large_offset_is_accepted(prep):
    X = 1e6 + 1e-3 * _series()
    data = Data(X, dim_order="ps", **prep)
    assert np.isfinite(data.to_numpy()).all()
    if prep["zscore"]:
        assert np.allclose(data.to_numpy().std(axis=1, ddof=1), 1.0)


def test_constant_in_one_replication_is_refused():
    # Preprocessing acts on each replication separately, so one constant
    # replication is enough to make the process undefined there.
    X = np.random.default_rng(1).standard_normal((3, 80, 2))
    X[1, :, 1] = 4.0
    with pytest.raises(ValueError, match=r"'proc-1' \(index 1\).*at least one replication"):
        Data(X, dim_order="psr")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("zscore", [True, False])
def test_non_finite_input_keeps_its_own_error(bad, zscore):
    X = _series()
    X[0, 10] = bad
    X[2] = 3.0  # a constant process too: the non-finite value is reported first
    with pytest.raises(ValueError, match=r"non-finite values \(NaN/inf\) in processes: \[0\]") as exc:
        Data(X, dim_order="ps", zscore=zscore)
    assert "Constant" not in str(exc.value)


def test_single_observation_keeps_its_error():
    with pytest.raises(ValueError, match="non-finite"):
        Data(np.ones((3, 1)), dim_order="ps")
    Data(np.ones((3, 1)), dim_order="ps", zscore=False)


def test_add_process_refuses_a_constant_process_by_its_new_name():
    data = Data(_series(M=2), dim_order="ps", procnames=["u", "v"])
    with pytest.raises(ValueError, match=r"'proc-2' \(index 2\)"):
        data.add_process(np.full(120, 0.5))
    assert data.n_processes == 2 and data.procnames == ["u", "v"]
    data.add_process(_series(M=1, seed=3)[0])
    assert data.procnames == ["u", "v", "proc-2"]


def test_add_process_to_empty_data_refuses_a_constant_process():
    with pytest.raises(ValueError, match=r"'proc-0' \(index 0\)"):
        Data().add_process(np.zeros(50))


def test_calculator_refuses_a_constant_process():
    X = _series()
    X[1] = 2.0
    with pytest.raises(ValueError, match=r"'proc-1' \(index 1\)"):
        Calculator(dataset=X, config="fabfour", zscore=False)
    calc = Calculator(dataset=_series(), config="fabfour")
    with pytest.raises(ValueError, match=r"'proc-1' \(index 1\)"):
        calc.load_dataset(X)


def test_cli_exits_with_the_message(tmp_path, capsys):
    X = _series()
    X[3] = 0.0
    path = tmp_path / "ts.npy"
    np.save(path, X)
    with pytest.raises(SystemExit) as exc:
        cli.main(["compute", "--data", str(path), "--config", "fabfour", "--quiet"])
    # A string code is printed on stderr and gives exit status 1.
    assert isinstance(exc.value.code, str)
    assert "'proc-3' (index 3)" in exc.value.code
    assert not (tmp_path / "ts.spi.npz").exists()

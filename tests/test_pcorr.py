"""Partial correlation ``pcorr_<estimator>`` (``basic.PartialCorrelation``).

``pcorr_ij = -P_ij / sqrt(P_ii P_jj)`` with ``P`` the precision matrix of the
cached scikit-learn covariance fit. Two independent references:

1. by hand, ``P = np.linalg.inv(np.cov(Z, ddof=0))`` (``EmpiricalCovariance``
   is the maximum-likelihood estimate, ``ddof=0``);
2. by regression: the Pearson correlation of the OLS residuals of ``x_i`` and
   ``x_j`` after regressing each on the other ``M - 2`` processes.
"""
import os
import warnings

import numpy as np
import pytest
import yaml
import sklearn.covariance

import pyspi
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.basic import Covariance, PartialCorrelation, Precision

ESTIMATORS = ["EmpiricalCovariance", "GraphicalLasso", "GraphicalLassoCV",
              "LedoitWolf", "MinCovDet", "OAS", "ShrunkCovariance"]


def _series(M=5, T=400, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((M, T))
    X[1] += 0.6 * X[0]
    X[2] -= 0.5 * X[1]
    X[3] += 0.4 * X[0] - 0.3 * X[2]
    return X


def _data(X):
    return Data(data=np.array(X, dtype=float), dim_order="ps")


def _pcorr_from_inverse(Z):
    P = np.linalg.inv(np.cov(Z, ddof=0))
    d = np.sqrt(np.diag(P))
    R = -P / np.outer(d, d)
    np.fill_diagonal(R, np.nan)
    return P, R


def _pcorr_from_residuals(Z, i, j):
    rest = [k for k in range(Z.shape[0]) if k not in (i, j)]
    A = np.column_stack([np.ones(Z.shape[1]), Z[rest].T])
    ri = Z[i] - A @ np.linalg.lstsq(A, Z[i], rcond=None)[0]
    rj = Z[j] - A @ np.linalg.lstsq(A, Z[j], rcond=None)[0]
    return np.corrcoef(ri, rj)[0, 1]


def test_matches_hand_inverted_empirical_covariance():
    data = _data(_series())
    Z = data.to_numpy(squeeze=True)
    P, ref = _pcorr_from_inverse(Z)

    spi = PartialCorrelation(estimator="EmpiricalCovariance")
    A = spi.multivariate(data)
    np.testing.assert_allclose(A, ref, rtol=1e-10, atol=1e-12, equal_nan=True)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[0]):
            if i != j:
                assert spi.bivariate(data, i=i, j=j) == pytest.approx(
                    ref[i, j], rel=1e-10, abs=1e-12)
    # The ddof=0 reference also reproduces the cached precision itself.
    prec = Precision(estimator="EmpiricalCovariance").multivariate(data)
    off = ~np.eye(Z.shape[0], dtype=bool)
    np.testing.assert_allclose(prec[off], P[off], rtol=1e-9)


def test_matches_ols_residual_correlation():
    data = _data(_series())
    Z = data.to_numpy(squeeze=True)
    spi = PartialCorrelation(estimator="EmpiricalCovariance")
    A = spi.multivariate(data)
    M = Z.shape[0]
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            ref = _pcorr_from_residuals(Z, i, j)
            assert abs(A[i, j] - ref) < 1e-10
            assert abs(spi.bivariate(data, i=i, j=j) - ref) < 1e-10


def test_chain_removes_the_indirect_link():
    """x0 -> x1 -> x2 (contemporaneous): corr(0, 2) ~ 1/sqrt(3), pcorr ~ 0."""
    rng = np.random.default_rng(5)
    T = 4000
    x0 = rng.standard_normal(T)
    x1 = x0 + rng.standard_normal(T)
    x2 = x1 + rng.standard_normal(T)
    data = _data(np.vstack([x0, x1, x2]))
    A = PartialCorrelation().multivariate(data)
    corr = Covariance().multivariate(data)  # z-scored, so cov is ~ corr
    assert abs(A[0, 2]) < 0.06
    assert corr[0, 2] > 0.5
    assert A[0, 1] > 0.5 and A[1, 2] > 0.5


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_structure_diagonal_symmetry_range(estimator):
    data = _data(_series())
    A = PartialCorrelation(estimator=estimator).multivariate(data)
    assert np.isnan(np.diag(A)).all()
    off = ~np.eye(A.shape[0], dtype=bool)
    assert np.isfinite(A[off]).all()
    np.testing.assert_allclose(A, A.T, rtol=0, atol=1e-12, equal_nan=True)
    assert (np.abs(A[off]) <= 1 + 1e-12).all()
    # Opposite sign to the precision off-diagonal, same zero pattern.
    P = Precision(estimator=estimator).multivariate(data)
    assert (np.sign(A[off]) == -np.sign(P[off])).all()


def test_squared_and_identifier():
    data = _data(_series())
    s = PartialCorrelation(estimator="LedoitWolf", squared=True)
    assert s.identifier == "pcorr-sq_LedoitWolf"
    assert not s.issigned()
    A = PartialCorrelation(estimator="LedoitWolf").multivariate(data)
    np.testing.assert_array_equal(s.multivariate(data), np.square(A))
    assert PartialCorrelation().identifier == "pcorr_EmpiricalCovariance"


def test_shares_the_cached_fit(monkeypatch):
    data = _data(_series())
    prec = Precision(estimator="EmpiricalCovariance")
    pcorr = PartialCorrelation(estimator="EmpiricalCovariance")
    assert type(pcorr)._cache_namespace == type(prec)._cache_namespace
    assert pcorr._cache_subkey == prec._cache_subkey

    prec.multivariate(data)
    assert list(data.covariance) == ["EmpiricalCovariance"]

    def _refit(*a, **k):
        raise AssertionError("pcorr refitted the covariance estimator")

    monkeypatch.setattr(sklearn.covariance.EmpiricalCovariance, "fit", _refit)
    A = pcorr.multivariate(data)
    assert np.isfinite(A[~np.eye(A.shape[0], dtype=bool)]).all()
    assert list(data.covariance) == ["EmpiricalCovariance"]


def test_does_not_hand_out_the_cache():
    data = _data(_series())
    A = PartialCorrelation().multivariate(data)
    A[:] = 7.0
    B = PartialCorrelation().multivariate(data)
    assert not np.any(B == 7.0)
    P = data.covariance["EmpiricalCovariance"].precision_
    assert np.isfinite(np.diag(P)).all()


def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.basic": {"PartialCorrelation": {
        "labels": ["undirected", "linear", "signed", "multivariate",
                   "contemporaneous", "M14"],
        "dependencies": None,
        "configs": [{"estimator": "EmpiricalCovariance"},
                    {"estimator": "LedoitWolf"}]}}}
    path = tmp_path / "pcorr.yaml"
    path.write_text(yaml.safe_dump(cfg))
    X = _series()
    calc = Calculator(dataset=_data(X), config=str(path))
    calc.compute()
    assert calc.errors == {}
    assert set(calc.spis) == {"pcorr_EmpiricalCovariance", "pcorr_LedoitWolf"}
    got = calc.table["pcorr_EmpiricalCovariance"].to_numpy()
    _, ref = _pcorr_from_inverse(_data(X).to_numpy(squeeze=True))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_full_and_fast_configs_construct():
    full = set(Calculator(config="full").spis)
    assert {f"pcorr_{e}" for e in ESTIMATORS} <= full
    assert not any(k.startswith("pcorr-sq") for k in full)
    fast = set(Calculator(config="fast").spis)
    assert {"pcorr_EmpiricalCovariance", "pcorr_LedoitWolf"} <= fast

    path = os.path.join(os.path.dirname(pyspi.__file__), "configs", "full.yaml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)[".statistics.basic"]
    assert ([c["estimator"] for c in cfg["PartialCorrelation"]["configs"]]
            == ESTIMATORS)
    assert cfg["PartialCorrelation"]["labels"] == cfg["Precision"]["labels"]


# Singular covariance: with no more samples than processes (T <= M) the
# centred empirical covariance has rank at most T - 1 < M, and so does the
# MinCovDet fit. scikit-learn then returns pinvh(covariance_) as precision_,
# which is not a precision matrix, so every precision-derived SPI is NaN.

PSEUDO_INVERSE_ESTIMATORS = ["EmpiricalCovariance", "MinCovDet"]
SHRINKAGE_ESTIMATORS = ["LedoitWolf", "OAS", "ShrunkCovariance"]


def _wide(M, T, seed=3):
    X = np.random.default_rng(seed).standard_normal((M, T))
    X[1] += 0.6 * X[0]
    return _data(X)


@pytest.mark.parametrize("M,T", [(12, 10), (20, 20)])
@pytest.mark.parametrize("estimator", PSEUDO_INVERSE_ESTIMATORS)
@pytest.mark.parametrize("cls", [Precision, PartialCorrelation])
@pytest.mark.parametrize("squared", [False, True])
def test_singular_covariance_gives_nan_and_one_warning(M, T, estimator, cls, squared):
    data = _wide(M, T)
    spi = cls(estimator=estimator, squared=squared)
    with pytest.warns(UserWarning) as caught:
        A = spi.multivariate(data)
    assert A.shape == (M, M)
    assert np.isnan(A).all()
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert estimator in messages[0]
    assert f"rank {T - 1} < M = {M}" in messages[0]
    assert spi.identifier in messages[0]


@pytest.mark.parametrize("M,T", [(12, 10), (20, 20)])
@pytest.mark.parametrize("estimator", PSEUDO_INVERSE_ESTIMATORS)
def test_singular_covariance_keeps_the_covariance(M, T, estimator):
    data = _wide(M, T)
    C = Covariance(estimator=estimator).multivariate(data)
    off = ~np.eye(M, dtype=bool)
    assert np.isfinite(C[off]).all()
    np.testing.assert_array_equal(
        C[off], data.covariance[estimator].covariance_[off])


@pytest.mark.parametrize("M,T", [(12, 10), (20, 20)])
@pytest.mark.parametrize("estimator", SHRINKAGE_ESTIMATORS)
def test_shrinkage_estimators_stay_finite_when_samples_are_few(M, T, estimator):
    data = _wide(M, T)
    off = ~np.eye(M, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A = PartialCorrelation(estimator=estimator).multivariate(data)
        P = Precision(estimator=estimator).multivariate(data)
    assert np.isfinite(A[off]).all() and np.isfinite(P[off]).all()
    assert (np.abs(A[off]) <= 1 + 1e-12).all()


@pytest.mark.parametrize("M,T", [(5, 400), (7, 100), (19, 20)])
@pytest.mark.parametrize("estimator", PSEUDO_INVERSE_ESTIMATORS + SHRINKAGE_ESTIMATORS)
def test_full_rank_fit_is_returned_unchanged(M, T, estimator):
    """At full rank the precision SPIs are exactly the fitted precision_."""
    data = _wide(M, T)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        P = Precision(estimator=estimator).multivariate(data)
        A = PartialCorrelation(estimator=estimator).multivariate(data)
    prec = data.covariance[estimator].precision_
    d = np.sqrt(np.diag(prec))
    ref_P = prec.copy()
    ref_A = -prec / np.outer(d, d)
    np.fill_diagonal(ref_P, np.nan)
    np.fill_diagonal(ref_A, np.nan)
    np.testing.assert_array_equal(P, ref_P)
    np.testing.assert_array_equal(A, ref_A)


def test_duplicated_process_makes_the_covariance_singular():
    X = _series()
    X[4] = X[0]
    data = _data(X)
    with pytest.warns(UserWarning, match=r"rank 4 < M = 5"):
        A = PartialCorrelation().multivariate(data)
    assert np.isnan(A).all()


def test_singular_covariance_in_the_calculator(tmp_path):
    from pyspi.quality import Quality

    cfg = {".statistics.basic": {
        name: {"labels": ["undirected", "linear", "signed", "multivariate",
                          "contemporaneous", "M14"],
               "dependencies": None,
               "configs": [{"estimator": "EmpiricalCovariance"},
                           {"estimator": "LedoitWolf"}]}
        for name in ("Covariance", "Precision", "PartialCorrelation")}}
    path = tmp_path / "singular.yaml"
    path.write_text(yaml.safe_dump(cfg))
    X = np.random.default_rng(3).standard_normal((12, 10))
    calc = Calculator(dataset=_data(X), config=str(path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute()
    messages = [str(w.message) for w in caught]
    off = ~np.eye(12, dtype=bool)
    for key in ("prec_EmpiricalCovariance", "pcorr_EmpiricalCovariance"):
        assert np.isnan(calc.table[key].to_numpy()).all()
        assert (calc.quality[key].to_numpy()[off] == Quality.NAN).all()
        assert any(key in m and "rank 9 < M = 12" in m for m in messages)
    for key in ("cov_EmpiricalCovariance", "cov_LedoitWolf",
                "prec_LedoitWolf", "pcorr_LedoitWolf"):
        assert np.isfinite(calc.table[key].to_numpy()[off]).all()
        assert (calc.quality[key].to_numpy()[off] == Quality.GOOD).all()
        assert key not in calc.errors

"""Independent references for every class in ``pyspi.statistics.basic``.

Each statistic is recomputed here from its textbook definition -- brute-force
pair counting for Kendall's tau-b, Pearson on average ranks for Spearman,
explicit lag sums for the cross-correlation profile, ``numpy`` for the
empirical covariance and its inverse -- on data with ties, a lagged pair and an
anticorrelated pair, and compared with what the SPI returns through
``multivariate`` (the path the Calculator takes) and ``bivariate``.
"""
import itertools

import numpy as np
import pytest
import yaml
from scipy import stats

from pyspi.data import Data
from pyspi.statistics import basic


def _series(T=120, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((4, T))
    X[1] = 0.6 * np.r_[np.zeros(2), X[0, :-2]] + 0.8 * X[1]   # 1 lags 0 by 2
    X[2] = -0.7 * X[0] + 0.5 * X[2]                          # anticorrelated
    X[3] = np.round(X[3] * 2) / 2                            # heavy ties
    return X


def _data(X, zscore=True):
    return Data(data=np.array(X, dtype=float), dim_order="ps", zscore=zscore)


def _pairs(M):
    return itertools.permutations(range(M), 2)


def _tau_b(a, b):
    """Kendall's tau-b by counting every pair (Kendall 1945)."""
    con = dis = tie_a = tie_b = 0
    for s, t in itertools.combinations(range(len(a)), 2):
        da, db = np.sign(a[s] - a[t]), np.sign(b[s] - b[t])
        if da == 0 and db == 0:
            continue
        if da == 0:
            tie_a += 1
        elif db == 0:
            tie_b += 1
        elif da == db:
            con += 1
        else:
            dis += 1
    return (con - dis) / np.sqrt((con + dis + tie_a) * (con + dis + tie_b))


def _rank_pearson(a, b):
    return np.corrcoef(stats.rankdata(a), stats.rankdata(b))[0, 1]


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


_CORR = {"pearson": _pearson, "spearman": _rank_pearson, "kendall": _tau_b}


# ---------------------------------------------------------------------------
# SpearmanR / KendallTau
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("squared", [False, True])
def test_spearman_is_pearson_on_average_ranks(squared):
    X = _series()
    Z = _data(X).to_numpy(squeeze=True)
    spi = basic.SpearmanR(squared=squared)
    got = spi.multivariate(_data(X))
    for i, j in _pairs(len(X)):
        ref = _rank_pearson(Z[i], Z[j]) ** (2 if squared else 1)
        assert got[i, j] == pytest.approx(ref, abs=1e-12)
        assert spi.bivariate(_data(X), i=i, j=j) == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("squared", [False, True])
def test_kendall_is_tau_b_with_ties(squared):
    """scipy's default variant, tau-b: ties are discounted in the denominator."""
    X = _series(T=60)
    Z = _data(X).to_numpy(squeeze=True)
    assert len(np.unique(Z[3])) < 20, "fixture must carry ties"
    spi = basic.KendallTau(squared=squared)
    got = spi.multivariate(_data(X))
    for i, j in _pairs(len(X)):
        ref = _tau_b(Z[i], Z[j]) ** (2 if squared else 1)
        assert got[i, j] == pytest.approx(ref, abs=1e-12)
        assert spi.bivariate(_data(X), i=i, j=j) == pytest.approx(ref, abs=1e-12)


# ---------------------------------------------------------------------------
# LaggedCorrelation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estimator", ["pearson", "spearman", "kendall"])
@pytest.mark.parametrize("tau", [1, 2, 20])
@pytest.mark.parametrize("squared", [False, True])
def test_lagged_correlation_matches_its_definition(estimator, tau, squared):
    """0.5 * (corr(x[tau:], y[:-tau]) + corr(y[tau:], x[:-tau])), as documented."""
    X = _series(T=60 if estimator == "kendall" else 120)
    Z = _data(X).to_numpy(squeeze=True)
    f = _CORR[estimator]
    spi = basic.LaggedCorrelation(estimator=estimator, tau=tau, squared=squared)
    got = spi.multivariate(_data(X))
    assert np.allclose(got, got.T, equal_nan=True, rtol=0, atol=1e-15)
    for i, j in _pairs(len(X)):
        x, y = Z[i], Z[j]
        ref = 0.5 * (f(x[tau:], y[:-tau]) + f(y[tau:], x[:-tau]))
        ref = ref ** 2 if squared else ref
        assert got[i, j] == pytest.approx(ref, abs=1e-12)
        assert spi.bivariate(_data(X), i=i, j=j) == pytest.approx(ref, abs=1e-12)


# ---------------------------------------------------------------------------
# CrossCorrelation
# ---------------------------------------------------------------------------

def _xcorr_profile(a, b):
    """r(l) = sum_t a[t+l] b[t] / (T sd_a sd_b), l in [-T//4, T//4], by hand."""
    T = len(a)
    a, b = a - a.mean(), b - b.mean()
    scale = T * np.sqrt(np.mean(a * a) * np.mean(b * b))
    L = T // 4
    return np.array([np.sum(a[l:] * b[:T - l]) if l >= 0
                     else np.sum(a[:T + l] * b[-l:])
                     for l in range(-L, L + 1)]) / scale


def _full_yaml_configs(cls):
    import os
    import pyspi
    path = os.path.join(os.path.dirname(pyspi.__file__), "configs", "full.yaml")
    with open(path) as fh:
        full = yaml.safe_load(fh)
    return full[".statistics.basic"][cls]["configs"]


@pytest.mark.parametrize("zscore", [True, False])
@pytest.mark.parametrize(
    "params", [{k: v for k, v in c.items() if k != "labels"}
               for c in _full_yaml_configs("CrossCorrelation")]
    + [{"statistic": "max", "sigonly": False}],
    ids=str)
def test_cross_correlation_matches_direct_lag_sums(params, zscore):
    X = _series()
    T = X.shape[1]
    Z = _data(X, zscore).to_numpy(squeeze=True)
    spi = basic.CrossCorrelation(**params)
    got = spi.multivariate(_data(X, zscore))
    for i, j in _pairs(len(X)):
        r = _xcorr_profile(Z[i], Z[j])
        if spi._sigonly:
            r = r[np.abs(r) > 1.96 / np.sqrt(T)]
        if r.size == 0:
            ref = 0.0
        else:
            r = r ** 2 if spi._squared else r
            ref = r.max() if spi._statistic == "max" else r.mean()
        assert got[i, j] == pytest.approx(ref, abs=1e-12), (i, j)


# ---------------------------------------------------------------------------
# Covariance / Precision
# ---------------------------------------------------------------------------

ESTIMATORS = sorted({c["estimator"] for c in _full_yaml_configs("Covariance")})


def test_empirical_covariance_and_precision_match_numpy():
    X = _series()
    d = _data(X)
    Z = d.to_numpy(squeeze=True)
    emp = np.cov(Z, ddof=0)
    off = ~np.eye(len(X), dtype=bool)
    cov = basic.Covariance().multivariate(d)
    prec = basic.Precision().multivariate(d)
    assert np.allclose(cov[off], emp[off], rtol=0, atol=1e-14)
    assert np.allclose(prec[off], np.linalg.inv(emp)[off], rtol=0, atol=1e-12)
    assert np.allclose(basic.Precision(squared=True).multivariate(d)[off],
                       np.linalg.inv(emp)[off] ** 2, rtol=0, atol=1e-12)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_covariance_estimator_is_the_named_one_and_is_deterministic(estimator):
    """The class the config names, at its scikit-learn defaults except a fixed
    seed (MinCovDet, EllipticEnvelope) and a tight tolerance (GraphicalLasso,
    GraphicalLassoCV); the same value whatever the global NumPy seed."""
    import warnings
    import sklearn.covariance as skcov

    X = _series()
    np.random.seed(1)
    d1 = _data(X)
    c1 = basic.Covariance(estimator=estimator).multivariate(d1)
    p1 = basic.Precision(estimator=estimator).multivariate(d1)
    assert type(d1.covariance[estimator]) is getattr(skcov, estimator)
    np.random.seed(2)
    d2 = _data(X)
    c2 = basic.Covariance(estimator=estimator).multivariate(d2)
    p2 = basic.Precision(estimator=estimator).multivariate(d2)
    assert np.array_equal(c1, c2, equal_nan=True)
    assert np.array_equal(p1, p2, equal_nan=True)
    # Undirected: both matrices are symmetric.
    assert np.allclose(c1, c1.T, equal_nan=True, rtol=0, atol=1e-14)
    assert np.allclose(p1, p1.T, equal_nan=True, rtol=0, atol=1e-12)

    kwargs = {}
    if estimator in ("MinCovDet", "EllipticEnvelope"):
        kwargs["random_state"] = 42
    if estimator.startswith("GraphicalLasso"):
        kwargs.update(tol=1e-8, enet_tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = getattr(skcov, estimator)(**kwargs).fit(
            d1.to_numpy(squeeze=True).T)
    off = ~np.eye(len(X), dtype=bool)
    assert np.array_equal(c1[off], ref.covariance_[off])
    assert np.array_equal(p1[off], ref.precision_[off])


@pytest.mark.parametrize("cls", [basic.Covariance, basic.Precision])
@pytest.mark.parametrize("squared", [False, True])
def test_covariance_output_does_not_alias_the_cached_fit(cls, squared):
    """The matrix handed back must be the caller's, not the cache's.

    ``multivariate`` returned ``fit.covariance_`` (or ``precision_``) itself,
    after writing NaN into the fitted estimator's diagonal. Anything the caller
    then did to the result -- thresholding, taking absolute values -- was
    written into the cache, and every later SPI read from the same Data
    (including the squared variant) silently returned the edited numbers.
    """
    X = _series()
    d = _data(X)
    plain = cls(squared=squared)
    first = plain.multivariate(d)
    expected = first.copy()
    first[0, 1] = 999.0
    assert plain.multivariate(d)[0, 1] == expected[0, 1]
    assert cls(squared=not squared).multivariate(d)[0, 1] != 999.0 ** 2
    fit = d.covariance["EmpiricalCovariance"]
    assert np.isfinite(np.diag(getattr(fit, cls(squared=squared)._kind + "_"))).all()


@pytest.mark.xfail(strict=True, reason=(
    "sklearn's EllipticEnvelope.fit is MinCovDet.fit plus a decision "
    "threshold (offset_), so with the shared fixed seed its covariance_ and "
    "precision_ are MinCovDet's, bit for bit: cov_/cov-sq_/prec_/prec-sq_"
    "EllipticEnvelope duplicate the MinCovDet SPIs on every input. Not "
    "salvageable as a distinct estimate; drop the four from the configs."))
def test_every_bundled_covariance_estimator_gives_a_distinct_matrix():
    X = _series()
    d = _data(X)
    mats = {e: basic.Covariance(estimator=e).multivariate(d) for e in ESTIMATORS}
    for a, b in itertools.combinations(ESTIMATORS, 2):
        assert not np.array_equal(mats[a], mats[b], equal_nan=True), (a, b)

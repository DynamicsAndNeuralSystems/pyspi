"""A pair of linearly dependent series is not appropriate, pair by pair.

Parametric spectral GC fits a VAR whose Yule-Walker equations and innovation
covariance it inverts, the Wilson-derived SPIs (nonparametric spectral GC,
directed coherence, ...) factorise the pair's spectral matrix from its
zero-lag covariance, and Johansen's test inverts two residual covariances.
For a series and a scaled and shifted, or nearly exact, copy of it those
matrices are singular, or singular to the precision they are computed with,
so that pair is NaN with ``Quality.NAN`` and its reason in
``calc.pair_not_appropriate``, and every other pair keeps the value it has
without the copy present. The backend's random start for Wilson's iteration
is never reached, so the values do not depend on NumPy's global random state.
"""
import re
import warnings

import numpy as np
import pytest
import yaml

from pyspi.base import NotAppropriate
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.quality import Quality
from pyspi.statistics.misc import Cointegration
from pyspi.statistics.spectral import (
    DirectDirectedTransferFunction,
    DirectedCoherence,
    DirectedTransferFunction,
    GeneralizedPartialDirectedCoherence,
    PartialDirectedCoherence,
    SpectralGrangerCausality,
)
from pyspi.statistics.var import _SINGULAR_RCOND
from pyspi.utils import (SINGULAR_RCOND, singular_covariance_error,
                         unit_diagonal_ratio)

# One config of each affected class, as bundled in `full`.
CONFIG = {
    ".statistics.spectral": {
        "SpectralGrangerCausality": {"labels": ["directed"], "configs": [
            {"method": "parametric", "fmin": 0, "fmax": 0.5, "order": 1},
            {"method": "parametric", "fmin": 0, "fmax": 0.25, "order": None},
            {"method": "nonparametric", "fmin": 0, "fmax": 0.5},
        ]},
        "DirectedCoherence": {"labels": ["directed"], "configs": [
            {"fmin": 0, "fmax": 0.5},
        ]},
    },
    ".statistics.misc": {
        "Cointegration": {"labels": ["undirected"], "configs": [
            {"method": "johansen", "statistic": "max_eig_stat",
             "det_order": 0, "k_ar_diff": 1},
            {"method": "johansen", "statistic": "trace_stat",
             "det_order": 1, "k_ar_diff": 10},
        ]},
    },
}


def _ar1(seed, M=3, T=500):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((M, T + 100))
    x = np.zeros((M, T + 100))
    for t in range(1, T + 100):
        x[:, t] = 0.5 * x[:, t - 1] + e[:, t]
    return x[:, 100:]


def _with_copy(kind):
    """Four processes: 1 depends on 0 as `kind` says; 2 and 3 are independent."""
    x = _ar1(0, M=3)
    if kind == "exact":
        copy = x[0].copy()
    elif kind == "affine":
        copy = -2.5 * x[0] + 3.0
    else:  # a copy plus noise of relative size 1e-9
        copy = x[0] + 1e-9 * _ar1(1, M=1)[0]
    return np.vstack([x[0], copy, x[1], x[2]])


def _calculator(X, tmp_path):
    config = tmp_path / "dependent.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    calc = Calculator(dataset=Data(X, dim_order="ps"), config=str(config),
                      verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    return calc


@pytest.mark.parametrize("kind", ["exact", "affine", "near"])
def test_only_the_dependent_pair_is_not_appropriate(kind, tmp_path):
    calc = _calculator(_with_copy(kind), tmp_path)
    procs = list(calc.dataset.procnames)
    assert not calc.errors and not calc.pair_errors
    assert len(calc.spis) == 6
    off = ~np.eye(4, dtype=bool)
    for key in calc.spis:
        q = calc.quality[key].to_numpy()
        assert q[0, 1] == q[1, 0] == Quality.NAN, (key, q)
        good = off.copy()
        good[0, 1] = good[1, 0] = False
        assert (q[good] == Quality.GOOD).all(), (key, q)
        reasons = {(a, b): msg for a, b, msg in calc.pair_not_appropriate[key]}
        assert set(reasons) == {(procs[0], procs[1]), (procs[1], procs[0])}
        for msg in reasons.values():
            assert "are linearly dependent" in msg and re.search(r"(is|number) -?(\d|inf)", msg)


def test_the_other_pairs_keep_the_values_they_have_without_the_copy(tmp_path):
    """The same values up to rounding, and the same NaN pattern."""
    X = _with_copy("affine")
    with_copy = _calculator(X, tmp_path)
    without = _calculator(X[[0, 2, 3]], tmp_path)
    keep = [0, 2, 3]
    for key in with_copy.spis:
        got = with_copy.table[key].to_numpy()[np.ix_(keep, keep)]
        want = without.table[key].to_numpy()
        # Not one computation: nonparametric GC takes one multitaper
        # estimate of all the processes, whose BLAS calls can round
        # differently for three and four of them.
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0, err_msg=key)


def test_a_strongly_correlated_pair_keeps_its_value(tmp_path):
    """r = 0.99 is far from the 1e-12 criterion: every pair is computed."""
    x = _ar1(2, M=3)
    rng = np.random.default_rng(3)
    x[1] = 0.99 * x[0] + np.sqrt(1 - 0.99 ** 2) * x[0].std() * rng.standard_normal(x.shape[1])
    assert np.corrcoef(x[0], x[1])[0, 1] > 0.98
    calc = _calculator(x, tmp_path)
    assert not calc.errors and not calc.pair_errors
    assert not calc.not_appropriate and not calc.pair_not_appropriate
    off = ~np.eye(3, dtype=bool)
    for key in calc.spis:
        assert (calc.quality[key].to_numpy()[off] == Quality.GOOD).all(), key


@pytest.mark.parametrize("make", [
    lambda: SpectralGrangerCausality(method="nonparametric", fmin=0, fmax=0.5),
    lambda: DirectedCoherence(fmin=0, fmax=0.5),
], ids=["sgc_nonparametric", "dcoh"])
def test_values_do_not_depend_on_the_global_random_state(make):
    """spectral_connectivity draws a random Wilson start from the global stream
    when its Cholesky start fails, as it would for this nearly identical pair;
    the pair is refused first, so neither the values nor the caller's stream
    change."""
    X = _with_copy("near")
    results = []
    for seed in (1, 2, 3):
        np.random.seed(seed)
        before = np.random.get_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(make().multivariate(Data(X, dim_order="ps")))
        after = np.random.get_state()
        assert before[0] == after[0] and np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    for A in results[1:]:
        np.testing.assert_array_equal(A, results[0])
    assert np.isnan(results[0][0, 1]) and np.isnan(results[0][1, 0])
    off = ~np.eye(4, dtype=bool)
    off[0, 1] = off[1, 0] = False
    assert np.isfinite(results[0][off]).all()


def test_every_pair_dependent_makes_the_spi_not_appropriate():
    x = _ar1(4, M=1)[0]
    X = np.vstack([x, 2 * x + 1, -x])
    for spi in (SpectralGrangerCausality(method="parametric", order=1),
                SpectralGrangerCausality(),
                DirectedCoherence()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotAppropriate, match="linearly dependent"):
                spi.multivariate(Data(X, dim_order="ps"))


def _near_copy(s, T=500):
    """Process 1 is process 0 plus noise of relative size ``s``; 2 is independent."""
    x = _ar1(5, M=3, T=T)
    return np.vstack([x[0], x[0] + s * x[1], x[2]])


def _lagged_copy(T=500):
    x = _ar1(6, M=2, T=T + 1)
    return np.vstack([x[0, 1:], x[0, :-1], x[1, 1:]])


@pytest.mark.parametrize("order", [None, 1, 5, 20])
@pytest.mark.parametrize("s", [1e-6, 2e-6, 3e-6, 1e-5])
def test_parametric_sgc_refuses_a_near_copy_whose_fit_is_rounding(s, order):
    """Just above the zero-lag tolerance the fitted innovation covariance is
    indefinite or below the precision of the Yule-Walker system: refused,
    never a negative or unexplained value."""
    spi = SpectralGrangerCausality(method="parametric", order=order)
    data = Data(_near_copy(s), dim_order="ps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = spi.multivariate(data)
    assert np.isnan(A[0, 1]) and np.isnan(A[1, 0])
    assert np.isfinite(A[[0, 2, 1, 2], [2, 0, 2, 1]]).all()
    with pytest.raises(NotAppropriate, match="linearly dependent"):
        from pyspi.statistics.spectral import _fit_bivariate_var
        _fit_bivariate_var(data.to_numpy(squeeze=True)[:2], order)


def test_parametric_sgc_keeps_a_near_copy_it_can_resolve(tmp_path):
    """At s = 1e-2 every order gives a finite, non-negative value."""
    data = Data(_near_copy(1e-2), dim_order="ps")
    for order in (None, 1, 5, 20):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = SpectralGrangerCausality(method="parametric",
                                         order=order).multivariate(data)
        off = ~np.eye(3, dtype=bool)
        assert np.isfinite(A[off]).all() and (A[off] >= 0).all(), order


JOHANSEN = [dict(det_order=d, k_ar_diff=k) for d in (0, 1) for k in (1, 10)]


@pytest.mark.parametrize("kwargs", JOHANSEN, ids=str)
@pytest.mark.parametrize("s", [1e-6, 3e-6])
def test_johansen_refuses_a_near_copy_in_both_orientations(kwargs, s):
    data = Data(_near_copy(s), dim_order="ps")
    for statistic in ("max_eig_stat", "trace_stat"):
        spi = Cointegration(method="johansen", statistic=statistic, **kwargs)
        for i, j in ((0, 1), (1, 0)):
            with pytest.raises(NotAppropriate, match="linearly dependent"):
                spi.bivariate(data, i=i, j=j)
        assert spi.bivariate(data, i=0, j=2) == spi.bivariate(data, i=2, j=0) >= 0


@pytest.mark.parametrize("kwargs", [dict(det_order=0, k_ar_diff=1),
                                    dict(det_order=0, k_ar_diff=10),
                                    dict(det_order=1, k_ar_diff=10)], ids=str)
def test_johansen_refuses_a_delayed_copy(kwargs, tmp_path):
    """x1(t) = x0(t - 1): the lagged levels are a lagged difference."""
    data = Data(_lagged_copy(), dim_order="ps")
    spi = Cointegration(method="johansen", statistic="max_eig_stat", **kwargs)
    for i, j in ((0, 1), (1, 0)):
        with pytest.raises(NotAppropriate, match="linearly dependent"):
            spi.bivariate(data, i=i, j=j)


def test_a_johansen_refusal_is_cached_for_both_orientations():
    data = Data(_with_copy("exact"), dim_order="ps")
    spi = Cointegration(method="johansen", statistic="trace_stat", det_order=0)
    with pytest.raises(NotAppropriate, match="residual covariance"):
        spi.bivariate(data, i=0, j=1)
    assert "refused" in data.coint[spi.key][(1, 0)]
    assert np.isfinite(spi.bivariate(data, i=0, j=2))


@pytest.mark.parametrize("cls", [PartialDirectedCoherence,
                                 GeneralizedPartialDirectedCoherence,
                                 DirectedTransferFunction,
                                 DirectDirectedTransferFunction],
                         ids=lambda c: c.__name__)
def test_every_wilson_derived_spi_refuses_the_pair(cls):
    data = Data(_with_copy("affine"), dim_order="ps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = cls().multivariate(data)
    assert np.isnan(A[0, 1]) and np.isnan(A[1, 0])
    off = ~np.eye(4, dtype=bool)
    off[0, 1] = off[1, 0] = False
    assert np.isfinite(A[off]).all()
    with pytest.raises(NotAppropriate, match="linearly dependent"):
        cls().bivariate(data, i=1, j=0)


def test_two_workers_record_the_same_refusals(tmp_path):
    X = _with_copy("affine")
    config = tmp_path / "dependent.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    calcs = []
    for n_jobs in (1, 2):
        calc = Calculator(dataset=Data(X, dim_order="ps"), config=str(config),
                          verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calc.compute(n_jobs=n_jobs, progress=False)
        calcs.append(calc)
    one, two = calcs
    assert not two.errors and not two.pair_errors
    assert one.pair_not_appropriate == two.pair_not_appropriate
    for key in one.spis:
        np.testing.assert_array_equal(one.table[key].to_numpy(),
                                      two.table[key].to_numpy(), err_msg=key)


def test_every_nan_cell_of_parametric_sgc_has_a_reason(tmp_path):
    config = tmp_path / "sgc.yaml"
    config.write_text(yaml.safe_dump({".statistics.spectral": {
        "SpectralGrangerCausality": {"labels": ["directed"], "configs": [
            {"method": "parametric", "order": o} for o in (None, 1, 5, 20)]}}}))
    for s in (1e-12, 1e-9, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        calc = Calculator(dataset=Data(_near_copy(s), dim_order="ps"),
                          config=str(config), verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calc.compute(progress=False)
        assert not calc.errors and not calc.pair_errors
        procs = list(calc.dataset.procnames)
        for key in calc.spis:
            v = calc.table[key].to_numpy()
            told = {(a, b) for a, b, _ in calc.pair_not_appropriate.get(key, [])}
            for a, b in zip(*np.nonzero(~np.eye(3, dtype=bool))):
                if np.isnan(v[a, b]):
                    assert (procs[a], procs[b]) in told, (s, key, a, b)
                else:
                    assert v[a, b] >= 0, (s, key, a, b)


@pytest.mark.parametrize("one_minus_r, refused", [
    (0.0, True), (1e-13, True), (1.9e-12, True), (2.1e-12, False),
    (1e-10, False), (0.01, False),
])
def test_the_floor_is_on_the_unit_diagonal_covariance(one_minus_r, refused):
    """(1 - |r|) / (1 + |r|) <= 1e-12, whatever the scales of the two series."""
    r = 1.0 - one_minus_r
    for scale in (1.0, 1e-6, 1e6):
        cov = np.array([[1.0, r * scale], [r * scale, scale ** 2]])
        err = singular_covariance_error("X", 0, 1, cov, "Y", of="Z")
        assert (err is not None) == refused, (one_minus_r, scale)
        if refused:
            assert isinstance(err, NotAppropriate)
    assert SINGULAR_RCOND == _SINGULAR_RCOND == 1e-12


def test_the_threshold_rises_with_the_condition_of_the_computation():
    cov = np.array([[1.0, 1 - 1e-9], [1 - 1e-9, 1.0]])   # ratio 5e-10
    assert singular_covariance_error("X", 0, 1, cov, "Y", of="Z") is None
    err = singular_covariance_error("X", 0, 1, cov, "Y", of="Z", condition=1e8)
    assert err is not None and "condition number 1e+08" in str(err)


def test_an_indefinite_or_nonpositive_covariance_is_refused_and_named():
    indefinite = np.array([[1.0, 1.1], [1.1, 1.0]])
    assert unit_diagonal_ratio(indefinite) < 0
    assert "-0.048" in str(singular_covariance_error("X", 0, 1, indefinite,
                                                     "Y", of="Z"))
    err = singular_covariance_error("X", 0, 1, np.array([[1.0, 0.0], [0.0, -1.0]]),
                                    "Y", of="Z")
    assert "a variance that is not positive" in str(err)
    assert singular_covariance_error("X", 0, 1, np.full((2, 2), np.nan),
                                     "Y", of="Z") is None
    with pytest.raises(TypeError):
        unit_diagonal_ratio(np.eye(2, dtype=complex))


def test_bic_search_keeps_the_orders_below_the_first_it_cannot_fit():
    """x and x + 2.4e-4 w: order 2 cannot be fitted to working precision,
    so the search selects among orders 0 and 1, picks 1, and gives the
    value the fixed order 1 gives; order 5 cannot be fitted at all."""
    x = _ar1(1, M=1, T=501)[0]
    w = _ar1(3, M=1, T=501)[0]
    data = Data(np.vstack([x[:500], x[:500] + 2.4e-4 * w[:500]]), dim_order="ps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected = SpectralGrangerCausality(method="parametric").multivariate(data)
        one = SpectralGrangerCausality(method="parametric", order=1).multivariate(data)
        with pytest.raises(NotAppropriate, match="linearly dependent"):
            SpectralGrangerCausality(method="parametric", order=5).multivariate(data)
    np.testing.assert_array_equal(selected, one)
    assert np.isfinite(selected[0, 1]) and selected[0, 1] > 0


def test_bic_search_refuses_when_no_order_with_lags_can_be_fitted():
    data = Data(_near_copy(1e-5), dim_order="ps")
    from pyspi.statistics.spectral import _fit_bivariate_var
    with pytest.raises(NotAppropriate, match=r"order 1|VAR\(1\)"):
        _fit_bivariate_var(data.to_numpy(squeeze=True)[:2], None)


@pytest.mark.parametrize("kwargs", JOHANSEN + [dict(det_order=-1, k_ar_diff=1)],
                         ids=str)
def test_johansen_values_are_statsmodels(kwargs):
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    data = Data(_near_copy(1e-2), dim_order="ps")
    z = data.to_numpy(squeeze=True)
    want = coint_johansen(z[[0, 2]].T, **kwargs)
    for statistic in ("max_eig_stat", "trace_stat"):
        got = Cointegration(method="johansen", statistic=statistic,
                            **kwargs).bivariate(data, i=0, j=2)
        assert got == getattr(want, statistic)[0]


def test_johansen_accepts_eigenvalues_at_rounding_level_below_zero(monkeypatch):
    from pyspi.statistics import misc
    real = misc.coint_johansen

    def shifted(*args, **kwargs):
        result = real(*args, **kwargs)
        result._eig = np.array([result._eig[0], -1e-17])
        return result

    monkeypatch.setattr(misc, "coint_johansen", shifted)
    data = Data(_near_copy(1e-2), dim_order="ps")
    value = Cointegration(method="johansen", det_order=0).bivariate(data, i=0, j=2)
    assert np.isfinite(value)

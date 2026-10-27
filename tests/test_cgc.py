"""Conditional Granger causality, ``cgc_*`` (``pyspi/statistics/var.py``).

Checked against statsmodels' VAR fit and restricted fits written out by hand,
on a chain and a common driver, for BIC order selection, and for its scale
relative to the bivariate ``gc_gaussian_*``.
"""
import numpy as np
import pytest
import yaml

from pyspi import _parallel
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.infotheory import TransferEntropy
from pyspi.statistics.var import (
    ConditionalGrangerCausality as CGC,
    fit_var,
    max_feasible_order,
    select_order_bic,
)

GC_ID = "gc_gaussian_k-1_kt-1_l-1_lt-1"
BUNDLED = {"cgc_p-1": dict(order=1), "cgc_p-2": dict(order=2),
           "cgc_p-bic": dict(order="bic"),
           "cgc_p-bic_coef": dict(order="bic", statistic="coef")}


def _simulate(coefs, T, seed, burn=500):
    """VAR(len(coefs)) with unit-variance Gaussian innovations; coefs[k-1][j, i]."""
    coefs = [np.asarray(A, dtype=float) for A in coefs]
    M = coefs[0].shape[0]
    rng = np.random.default_rng(seed)
    x = np.zeros((M, T + burn))
    for t in range(len(coefs), T + burn):
        x[:, t] = sum(A @ x[:, t - k] for k, A in enumerate(coefs, start=1))
        x[:, t] += rng.standard_normal(M)
    return x[:, burn:]


def _data(x):
    return Data(data=np.asarray(x), dim_order="ps")


def _gaussian_gc():
    spi = TransferEntropy(estimator="gaussian", k_history=1, l_history=1)
    assert spi.identifier == GC_ID
    return spi


def _offdiag(M):
    return ~np.eye(M, dtype=bool)


def _reference_cgc(Z, p):
    """Conditional GC by brute force: one least-squares fit per model.

    Regressors are assembled per model, process by process (a different column
    order from the module's lag-major design), and every restricted model is
    refitted with ``np.linalg.lstsq``.
    """
    M, T = Z.shape

    def regressors(exclude):
        cols = [np.ones(T - p)]
        for m in range(M):
            if m != exclude:
                cols += [Z[m, p - k:T - k] for k in range(1, p + 1)]
        return np.column_stack(cols)

    def residual_variance(y, X):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
        return r @ r / len(y)

    F = np.full((M, M), np.nan)
    for j in range(M):
        y = Z[j, p:]
        full = residual_variance(y, regressors(None))
        for i in range(M):
            if i != j:
                F[i, j] = np.log(residual_variance(y, regressors(i)) / full)
    return F


# A random stable VAR(2) on four processes, for the reference comparisons.
_A1 = np.array([[0.4, 0.2, 0.0, 0.0], [0.0, 0.3, -0.3, 0.0],
                [0.2, 0.0, 0.3, 0.1], [0.0, 0.0, 0.25, 0.2]])
_A2 = np.array([[-0.2, 0.0, 0.0, 0.1], [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.2, 0.0], [0.0, 0.15, 0.0, 0.0]])


# --------------------------------------------------------------------------
# 1. Independent references
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", [1, 2, 3])
def test_fit_matches_statsmodels_var(p):
    """Coefficients, intercept and residual covariance of statsmodels'
    ``VAR(...).fit(p, trend="c")``; ours divides by the T - p fitted samples,
    which is statsmodels' ``sigma_u_mle`` (its ``sigma_u`` divides by the
    residual degrees of freedom)."""
    from statsmodels.tsa.vector_ar.var_model import VAR

    Z = _data(_simulate([_A1, _A2], 300, seed=1)).to_numpy(squeeze=True)
    res = VAR(Z.T).fit(p, trend="c")
    fit = fit_var(Z, p)

    assert fit["coef"].shape == (p, 4, 4)
    np.testing.assert_allclose(fit["coef"], res.coefs, rtol=0, atol=1e-8)
    np.testing.assert_allclose(fit["intercept"], res.intercept, rtol=0, atol=1e-8)
    np.testing.assert_allclose(fit["sigma"], res.sigma_u_mle, rtol=0, atol=1e-8)
    assert fit["n_samples"] == res.nobs == 300 - p
    np.testing.assert_allclose(fit["rss"], np.diag(res.sigma_u_mle) * res.nobs,
                               rtol=1e-10)


@pytest.mark.parametrize("p", [1, 2])
def test_coef_statistic_is_the_net_lagged_coefficient(p):
    """``coef[i, j] = sum_k A_k[j, i]`` from statsmodels, through both paths."""
    from statsmodels.tsa.vector_ar.var_model import VAR

    data = _data(_simulate([_A1, _A2], 300, seed=2))
    Z = data.to_numpy(squeeze=True)
    want = VAR(Z.T).fit(p, trend="c").coefs.sum(axis=0).T   # [j, i] -> [i, j]

    spi = CGC(order=p, statistic="coef")
    got = spi.multivariate(data)
    off = _offdiag(4)
    np.testing.assert_allclose(got[off], want[off], rtol=0, atol=1e-8)
    assert np.isnan(np.diag(got)).all()
    for i in range(4):
        for j in range(4):
            if i != j:
                assert spi.bivariate(data, i=i, j=j) == pytest.approx(want[i, j], abs=1e-8)


@pytest.mark.parametrize("order", [1, 2, 3, "bic"])
def test_F_matches_restricted_fits_written_out_by_hand(order):
    """Multivariate and bivariate against one lstsq refit per restricted model."""
    data = _data(_simulate([_A1, _A2], 400, seed=3))
    Z = data.to_numpy(squeeze=True)
    spi = CGC(order=order)
    got = spi.multivariate(data)
    p = data.var[spi._cache_subkey]["order"]
    if order == "bic":
        assert p == select_order_bic(Z)[0]
    want = _reference_cgc(Z, p)

    off = _offdiag(4)
    np.testing.assert_allclose(got[off], want[off], rtol=1e-8, atol=1e-12)
    assert np.isnan(np.diag(got)).all()
    for i in range(4):
        for j in range(4):
            if i != j:
                assert spi.bivariate(data, i=i, j=j) == pytest.approx(
                    want[i, j], rel=1e-8, abs=1e-12)


def test_bivariate_on_a_fresh_dataset_matches_multivariate():
    """``bivariate`` builds the same cached fit when it is called first."""
    x = _simulate([_A1, _A2], 300, seed=4)
    whole = CGC(order=2).multivariate(_data(x))
    fresh = _data(x)
    assert CGC(order=2).bivariate(fresh, i=2, j=1) == whole[2, 1]
    assert list(fresh.var) == [(2, None)]


def test_collinear_processes_take_the_refit_path():
    """A duplicated process makes the design rank-deficient: restricted fits
    are refitted, the duplicates give F = 0, other pairs match the reference,
    and BIC (needing ln det of a singular covariance) raises."""
    x = _simulate([_A1], 300, seed=5)
    x = np.vstack([x, x[0]])                 # process 4 duplicates process 0
    data = _data(x)
    Z = data.to_numpy(squeeze=True)
    fit = fit_var(Z, 1)
    assert not fit["_full_rank"]

    got = CGC(order=1).multivariate(data)
    want = _reference_cgc(Z, 1)
    off = _offdiag(5)
    np.testing.assert_allclose(got[off], want[off], rtol=1e-7, atol=1e-10)
    assert got[0, 1] == pytest.approx(0, abs=1e-10)   # 4 stands in for 0
    assert got[4, 1] == pytest.approx(0, abs=1e-10)

    with pytest.raises(ValueError, match="singular"):
        CGC(order="bic").multivariate(_data(x))


# --------------------------------------------------------------------------
# 2-3. What conditioning buys: an indirect link and a common driver
# --------------------------------------------------------------------------

def test_chain_conditional_gc_drops_the_indirect_link():
    """x0 -> x1 -> x2. The bivariate Gaussian GC sees 0 -> 2 through x1;
    conditioning on x1's past removes it."""
    A = [[0.5, 0.0, 0.0],
         [0.8, 0.5, 0.0],
         [0.0, 0.8, 0.5]]
    data = _data(_simulate([A], 2000, seed=0))
    F = CGC(order=1).multivariate(data)
    bivariate = 2 * _gaussian_gc().multivariate(data)   # on Geweke's scale

    assert F[0, 1] > 0.3 and F[1, 2] > 0.3
    assert F[0, 2] < 0.02
    for i, j in [(1, 0), (2, 1), (2, 0)]:
        assert F[i, j] < 0.02, (i, j)
    # The contrast is the point of the SPI.
    assert bivariate[0, 2] > 0.04
    assert bivariate[0, 2] > 10 * F[0, 2]

    # BIC finds the VAR(1) and so the same matrix.
    assert np.array_equal(CGC(order="bic").multivariate(data), F, equal_nan=True)

    # The signed net coefficient: two positive direct links, no direct 0 -> 2.
    coef = CGC(order="bic", statistic="coef").multivariate(data)
    assert coef[0, 1] > 0.3 and coef[1, 2] > 0.3
    assert abs(coef[0, 2]) < 0.05


def test_coef_carries_the_sign_and_F_does_not():
    """An inhibitory link: coef is negative, F is positive either way."""
    A = [[0.5, 0.0], [-0.6, 0.4]]
    data = _data(_simulate([A], 1000, seed=6))
    coef = CGC(order=1, statistic="coef").multivariate(data)
    F = CGC(order=1).multivariate(data)
    assert coef[0, 1] < -0.3 and abs(coef[1, 0]) < 0.1
    assert F[0, 1] > 0.1 and F[1, 0] < 0.02


@pytest.mark.parametrize("lag", [1, 2])
def test_common_driver_gives_no_link_between_the_driven(lag):
    """x0 -> x1 (lag 1) and x0 -> x2 (lag ``lag``), nothing between x1 and x2.
    With the second delay longer, x1's past anticipates x2 and the bivariate
    GC reports 1 -> 2 strongly; conditioning on x0's past removes it."""
    A1 = np.array([[0.5, 0.0, 0.0], [0.8, 0.5, 0.0], [0.0, 0.0, 0.5]])
    A2 = np.zeros((3, 3))
    (A1 if lag == 1 else A2)[2, 0] = 0.8
    data = _data(_simulate([A1, A2], 2000, seed=0))
    F = CGC(order=2).multivariate(data)

    assert F[0, 1] > 0.3 and F[0, 2] > 0.3
    assert F[1, 2] < 0.02 and F[2, 1] < 0.02
    if lag == 2:
        bivariate = 2 * _gaussian_gc().multivariate(data)
        assert bivariate[1, 2] > 0.1
        assert bivariate[1, 2] > 10 * F[1, 2]


# --------------------------------------------------------------------------
# 4. Order selection
# --------------------------------------------------------------------------

_B1 = np.array([[0.4, 0.0, 0.0], [0.3, 0.3, 0.0], [0.0, 0.3, 0.2]])
_B2 = np.array([[-0.4, 0.0, 0.0], [0.0, -0.3, 0.3], [0.3, 0.0, -0.3]])


@pytest.mark.parametrize("coefs,p_true", [([_B1, _B2], 2), ([_B1], 1)],
                         ids=["VAR2", "VAR1"])
def test_bic_selects_the_true_order_and_matches_statsmodels(coefs, p_true):
    """BIC = ln det Sigma + (M^2 p + M) ln N / N on a common sample, which is
    statsmodels' ``select_order`` criterion value for value."""
    from statsmodels.tsa.vector_ar.var_model import VAR

    data = _data(_simulate(coefs, 2000, seed=7))
    Z = data.to_numpy(squeeze=True)
    p, bic = select_order_bic(Z, max_order=10)
    assert p == p_true

    sm = VAR(Z.T).select_order(10, trend="c")
    assert sm.bic == p_true
    np.testing.assert_allclose(bic, np.asarray(sm.ics["bic"])[1:], rtol=0, atol=1e-10)

    # The SPI uses that order; it is cached, not in the identifier.
    spi = CGC(order="bic")
    spi.multivariate(data)
    assert spi.identifier == "cgc_p-bic"
    assert data.var[("bic", 10)]["order"] == p_true
    np.testing.assert_array_equal(data.var[("bic", 10)]["bic"], bic)


def test_bic_search_is_capped_by_feasibility():
    """M=3, T=30: the cap is (29 // 4) - 1 = 6, so max_order=10 searches 1..6."""
    Z = _data(_simulate([_B1], 30, seed=8)).to_numpy(squeeze=True)
    assert max_feasible_order(3, 30) == 6
    _, bic = select_order_bic(Z, max_order=10)
    assert len(bic) == 6
    _, bic = select_order_bic(Z, max_order=4)
    assert len(bic) == 4


# --------------------------------------------------------------------------
# 5. Scale relative to the bivariate Gaussian GC
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_two_processes_give_twice_the_gaussian_gc(seed):
    """With M = 2 there is nothing else to condition on, so cgc_p-1 and the
    Gaussian TE with k = l = 1 fit the same model; the TE reports half of
    ln(var_r / var_f). The tolerance covers the Gaussian estimator's 1e-8
    relative ridge (``GAUSSIAN_RIDGE``)."""
    A = [[0.5, 0.0], [0.6, 0.4]]
    data = _data(_simulate([A], 2000, seed=seed))
    F = CGC(order=1).multivariate(data)
    gc = _gaussian_gc().multivariate(data)
    off = _offdiag(2)
    np.testing.assert_allclose(F[off], 2 * gc[off], rtol=1e-6, atol=1e-9)
    assert F[0, 1] > 0.3 and F[1, 0] < 0.01


# --------------------------------------------------------------------------
# 6. Infeasible orders and argument validation
# --------------------------------------------------------------------------

def test_infeasible_order_raises_a_clear_error():
    data = _data(np.random.default_rng(9).standard_normal((5, 20)))
    assert max_feasible_order(5, 20) == 2
    with pytest.raises(ValueError, match=r"p=3 is infeasible.*p <= .* = 2"):
        CGC(order=3).multivariate(data)
    with pytest.raises(ValueError, match="infeasible"):
        CGC(order=3).bivariate(data, i=0, j=1)
    assert CGC(order=2).multivariate(data).shape == (5, 5)

    tiny = _data(np.random.default_rng(9).standard_normal((10, 20)))
    assert max_feasible_order(10, 20) == 0
    with pytest.raises(ValueError, match="No feasible VAR order"):
        CGC(order="bic").multivariate(tiny)


def test_calculator_records_an_infeasible_order(tmp_path):
    cfg = {".statistics.var": {"ConditionalGrangerCausality": {
        "labels": ["directed"],
        "configs": [{"order": 3}, {"order": "bic"}]}}}
    path = tmp_path / "cgc.yaml"
    path.write_text(yaml.safe_dump(cfg))
    calc = Calculator(dataset=_data(np.random.default_rng(9).standard_normal((5, 20))),
                      config=str(path))
    with pytest.warns(UserWarning, match='"cgc_p-3"'):
        calc.compute()
    assert set(calc.errors) == {"cgc_p-3"}
    assert "infeasible" in calc.errors["cgc_p-3"]
    assert np.isnan(calc.table["cgc_p-3"].to_numpy(dtype=float)).all()
    assert np.isfinite(calc.table["cgc_p-bic"].to_numpy(dtype=float)[_offdiag(5)]).all()


@pytest.mark.parametrize("kwargs,exc,match", [
    (dict(order="aic"), ValueError, "'bic'"),
    (dict(order=0), ValueError, ">= 1"),
    (dict(order=1.5), TypeError, "integer"),
    (dict(order=True), TypeError, "integer"),
    (dict(order=2, max_order=5), ValueError, "only used by order='bic'"),
    (dict(order="bic", max_order=0), ValueError, ">= 1"),
    (dict(statistic="mean"), ValueError, "'F' or 'coef'"),
])
def test_constructor_rejects_bad_arguments(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        CGC(**kwargs)


def test_identifiers_and_labels():
    assert {CGC(**kw).identifier for kw in BUNDLED.values()} == set(BUNDLED)
    assert CGC(order="bic", max_order=10).identifier == "cgc_p-bic"
    assert CGC(order="bic", max_order=5).identifier == "cgc_p-bic_pmax-5"
    assert CGC(order=3, statistic="coef").identifier == "cgc_p-3_coef"

    F, coef = CGC(), CGC(statistic="coef")
    assert not F.issigned() and coef.issigned()
    assert {"directed", "linear", "multivariate", "unsigned"} <= set(F.labels)
    assert "signed" in coef.labels and "unsigned" not in coef.labels
    assert "unsigned" in CGC.labels                  # the class list is not mutated


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_F_and_coef_share_one_fit_per_order():
    data = _data(_simulate([_A1, _A2], 300, seed=10))
    CGC(order="bic").multivariate(data)
    entry = data.var[("bic", 10)]
    CGC(order="bic", statistic="coef").multivariate(data)
    CGC(order="bic").bivariate(data, i=0, j=1)
    assert list(data.var) == [("bic", 10)] and data.var[("bic", 10)] is entry
    CGC(order=1).multivariate(data)
    CGC(order="bic", max_order=3).multivariate(data)
    assert set(data.var) == {("bic", 10), (1, None), ("bic", 3)}

    # One scheduling bucket per shared fit.
    bucket = _parallel.cache_bucket
    assert bucket(CGC(order="bic")) == bucket(CGC(order="bic", statistic="coef"))
    assert bucket(CGC(order="bic")) != bucket(CGC(order=1))
    assert "var" in Data._CACHE_ATTRS


def test_results_are_not_the_cache_and_the_cache_follows_the_data():
    x = _simulate([_A1], 300, seed=11)
    data = _data(x)
    spi = CGC(order=1, statistic="coef")
    first = spi.multivariate(data)
    first[0, 1] = 1e6
    assert spi.multivariate(data)[0, 1] != 1e6

    y = _simulate([_A1], 300, seed=12)
    data.set_data(y, dim_order="ps")
    np.testing.assert_array_equal(spi.multivariate(data),
                                  spi.multivariate(_data(y)))


# --------------------------------------------------------------------------
# 7. Calculator round trip and bundled configs
# --------------------------------------------------------------------------

def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.var": {"ConditionalGrangerCausality": {
        "labels": ["directed", "linear", "unsigned", "multivariate"],
        "configs": list(BUNDLED.values())}}}
    path = tmp_path / "cgc.yaml"
    path.write_text(yaml.safe_dump(cfg))
    x = _simulate([_A1, _A2], 500, seed=13)
    calc = Calculator(dataset=_data(x), config=str(path))
    calc.compute()

    assert calc.errors == {}
    assert set(calc.spis) == set(BUNDLED)
    off = _offdiag(4)
    for ident, kwargs in BUNDLED.items():
        got = calc.table[ident].to_numpy(dtype=float)
        want = CGC(**kwargs).multivariate(_data(x))
        np.testing.assert_allclose(got[off], want[off], rtol=1e-12, atol=1e-14)
    assert calc.spis["cgc_p-bic_coef"].labels.count("signed") == 1


def test_bundled_configs_construct():
    full = Calculator(config="full").spis
    assert set(BUNDLED) <= set(full)
    assert "signed" in full["cgc_p-bic_coef"].labels
    assert "unsigned" in full["cgc_p-bic"].labels
    fast = Calculator(config="fast").spis
    assert "cgc_p-bic" in fast
    assert not {"cgc_p-1", "cgc_p-2", "cgc_p-bic_coef"} & set(fast)

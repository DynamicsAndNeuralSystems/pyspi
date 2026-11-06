"""Chatterjee's xi (`xi_tau-<tau>`, `pyspi.statistics.basic.ChatterjeeXi`).

xi(X, Y) = 1 - n sum_k |r_{k+1} - r_k| / (2 sum_k l_k (n - l_k)), with the
pairs sorted by X, r_k = #{j : Y_j <= Y_(k)} and l_k = #{j : Y_j >= Y_(k)}
(Chatterjee 2021, JASA 116:2009). A[i, j] = xi(X = x_i, Y = x_j): row = source.
"""

import numpy as np
import pytest
import yaml
from scipy import stats

from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.basic import ChatterjeeXi


def _data(arr, zscore=False):
    return Data(data=np.asarray(arr, dtype=float), dim_order="ps", zscore=zscore)


def _xi_bruteforce(x, y):
    """O(n^2) xi from the definition, with ties in X broken as in the SPI."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    perm = np.random.default_rng(0).permutation(n)
    order = perm[np.argsort(x[perm], kind="stable")]
    ys = y[order]
    r = np.array([np.sum(y <= v) for v in ys])
    l = np.array([np.sum(y >= v) for v in ys])
    num = n * np.sum(np.abs(np.diff(r)))
    den = 2 * np.sum(l * (n - l))
    return 1 - num / den


def _offdiag(a):
    return a[~np.eye(a.shape[0], dtype=bool)]


# --------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------

def test_matches_scipy_chatterjeexi_both_orientations():
    chatterjeexi = getattr(stats, "chatterjeexi", None)
    if chatterjeexi is None:
        pytest.skip("scipy.stats.chatterjeexi needs SciPy >= 1.15")
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 300))
    z[1] += np.sin(2 * z[0])
    z[3] = z[2] ** 2 + 0.1 * rng.standard_normal(300)
    spi = ChatterjeeXi()
    data = _data(z)
    A = spi.multivariate(data)
    for i in range(4):
        for j in range(4):
            if i == j:
                assert np.isnan(A[i, j])
                continue
            ref = chatterjeexi(z[i], z[j]).statistic
            assert A[i, j] == pytest.approx(ref, abs=1e-12)
            assert spi.bivariate(data, i=i, j=j) == pytest.approx(ref, abs=1e-12)
            assert spi.bivariate(data, i=i, j=j) == A[i, j]  # bit-identical


def test_tied_target_matches_bruteforce():
    """Integer-valued data: heavy ties in Y (and in X) use the general formula."""
    rng = np.random.default_rng(2)
    z = rng.integers(0, 6, size=(3, 120)).astype(float)
    z[1] = (z[0] + rng.integers(0, 2, size=120)) % 6
    spi = ChatterjeeXi()
    data = _data(z)
    A = spi.multivariate(data)
    for i in range(3):
        for j in range(3):
            if i != j:
                ref = _xi_bruteforce(z[i], z[j])
                assert A[i, j] == pytest.approx(ref, abs=1e-12)
                assert spi.bivariate(data, i=i, j=j) == pytest.approx(ref, abs=1e-12)


def test_tied_target_with_tiefree_source_matches_scipy():
    """Ties in Y only: no tie-breaking involved, so scipy is an exact reference."""
    chatterjeexi = getattr(stats, "chatterjeexi", None)
    if chatterjeexi is None:
        pytest.skip("scipy.stats.chatterjeexi needs SciPy >= 1.15")
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    y = np.round(2 * x + rng.standard_normal(200))
    A = ChatterjeeXi().multivariate(_data([x, y]))
    assert A[0, 1] == pytest.approx(chatterjeexi(x, y).statistic, abs=1e-12)


def test_identity_gives_n_minus_2_over_n_plus_1():
    """y = x: sorted by x the ranks of y are 1, 2, ..., n, so
    sum_k |r_{k+1} - r_k| = n - 1, and without ties
    xi = 1 - 3 (n - 1) / (n^2 - 1) = 1 - 3 / (n + 1) = (n - 2) / (n + 1)."""
    n = 57
    x = np.random.default_rng(4).standard_normal(n)
    A = ChatterjeeXi().multivariate(_data([x, x.copy()]))
    assert A[0, 1] == pytest.approx((n - 2) / (n + 1), abs=1e-14)
    assert A[1, 0] == pytest.approx((n - 2) / (n + 1), abs=1e-14)


# --------------------------------------------------------------------------
# Known-truth sanity checks
# --------------------------------------------------------------------------

def test_functional_dependence_is_directed_where_rank_correlations_are_blind():
    """y = x^2 + small noise: y is a function of x but x is not one of y."""
    rng = np.random.default_rng(5)
    T = 1000
    x = rng.standard_normal(T)
    y = x ** 2 + 0.05 * rng.standard_normal(T)
    A = ChatterjeeXi().multivariate(_data([x, y], zscore=True))
    assert A[0, 1] > 0.7
    assert A[1, 0] < 0.3
    assert A[0, 1] - A[1, 0] > 0.4
    assert abs(stats.spearmanr(x, y).correlation) < 0.1
    assert abs(stats.kendalltau(x, y).correlation) < 0.1


def test_independent_series_give_xi_near_zero():
    rng = np.random.default_rng(6)
    z = rng.standard_normal((5, 2000))
    A = ChatterjeeXi().multivariate(_data(z, zscore=True))
    # Under independence sqrt(n) xi -> N(0, 2/5) (it can be slightly negative).
    assert np.all(np.abs(_offdiag(A)) < 5 * np.sqrt(0.4 / 2000))


def test_lag_one_dependence_is_seen_in_the_right_direction():
    """x_1[t] = cos(3 x_0[t-1]) + noise: xi_tau-1(0 -> 1) large, (1 -> 0) small."""
    rng = np.random.default_rng(7)
    T = 1000
    x0 = rng.standard_normal(T)
    x1 = np.empty(T)
    x1[0] = rng.standard_normal()
    x1[1:] = np.cos(3 * x0[:-1]) + 0.05 * rng.standard_normal(T - 1)
    data = _data([x0, x1], zscore=True)
    A1 = ChatterjeeXi(tau=1).multivariate(data)
    assert A1[0, 1] > 0.6
    assert A1[1, 0] < 0.1
    # The contemporaneous xi sees nothing: the dependence is purely lagged.
    A0 = ChatterjeeXi(tau=0).multivariate(data)
    assert abs(A0[0, 1]) < 0.1
    # And the lagged bivariate path agrees with the definition on aligned slices.
    assert A1[0, 1] == pytest.approx(
        _xi_bruteforce(data.to_numpy(squeeze=True)[0, :-1],
                       data.to_numpy(squeeze=True)[1, 1:]), abs=1e-12)
    assert ChatterjeeXi(tau=1).bivariate(data, i=0, j=1) == A1[0, 1]
    assert ChatterjeeXi(tau=1).bivariate(data, i=1, j=0) == A1[1, 0]


def test_ties_in_source_are_broken_deterministically():
    rng = np.random.default_rng(8)
    z = rng.integers(0, 3, size=(4, 400)).astype(float)
    z[1] = z[0] * 2 + rng.standard_normal(400)
    A = ChatterjeeXi().multivariate(_data(z))
    B = ChatterjeeXi().multivariate(_data(z.copy()))
    np.testing.assert_array_equal(A, B)
    data = _data(z)
    for i, j in [(0, 1), (2, 3), (3, 0)]:
        assert ChatterjeeXi().bivariate(data, i=i, j=j) == A[i, j]


def test_constant_target_is_undefined():
    rng = np.random.default_rng(9)
    z = np.vstack([rng.standard_normal(50), np.ones(50), rng.standard_normal(50)])
    spi = ChatterjeeXi()
    data = _data(z)
    A = spi.multivariate(data)
    assert np.isnan(A[0, 1]) and np.isnan(A[2, 1])
    assert np.isfinite(A[1, 0]) and np.isfinite(A[0, 2])
    assert np.isnan(spi.bivariate(data, i=0, j=1))


def test_identifier_and_labels():
    assert ChatterjeeXi().identifier == "xi_tau-0"
    assert ChatterjeeXi(tau=3).identifier == "xi_tau-3"
    assert "unordered" in ChatterjeeXi().labels
    assert "temporal" in ChatterjeeXi(tau=1).labels
    assert not ChatterjeeXi().issigned()
    with pytest.raises((TypeError, ValueError)):
        ChatterjeeXi(tau=-1)


# --------------------------------------------------------------------------
# Calculator round trip and bundled config
# --------------------------------------------------------------------------

def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.basic": {"ChatterjeeXi": {
        "labels": ["directed", "nonlinear", "unsigned", "bivariate", "M14"],
        "dependencies": None,
        "configs": [{"tau": 0}, {"tau": 1}]}}}
    path = tmp_path / "xi.yaml"
    path.write_text(yaml.safe_dump(cfg))
    rng = np.random.default_rng(10)
    z = rng.standard_normal((3, 200))
    calc = Calculator(dataset=Data(data=z, dim_order="ps"), config=str(path))
    calc.compute()
    assert calc.errors == {}
    assert set(calc.spis) == {"xi_tau-0", "xi_tau-1"}
    ref = ChatterjeeXi().multivariate(Data(data=z, dim_order="ps"))
    got = calc.table["xi_tau-0"].to_numpy()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12, equal_nan=True)


def test_full_config_constructs_xi_entries():
    spis = Calculator(config="full").spis
    for ident, tau in [("xi_tau-0", 0), ("xi_tau-1", 1)]:
        assert ident in spis
        assert isinstance(spis[ident], ChatterjeeXi)
        assert spis[ident]._tau == tau
    assert "contemporaneous" in spis["xi_tau-0"].labels
    assert "time-dependent" in spis["xi_tau-1"].labels
    assert "xi_tau-0" in Calculator(config="fast").spis

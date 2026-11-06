"""EdgeTimeSeries (``edge_*``): coupling nonstationarity from the edge time series.

``e_t = z_x(t) z_y(t)`` on ddof=1 z-scored series. Three statistics, each
referenced to a Gaussian iid pair with the same r:

- ``excessvar``: ``var(e) - c^2 [1 + r^2 - (2 + 4 r^2)/(T - 1)]``, ``c = (T-1)/T``;
- ``topshare``: ``sum_top e / sum |e|`` minus its Gaussian large-T value;
- ``acf1``: lag-1 autocorrelation of ``e`` minus the Isserlis reference
  ``(rho_x(1) rho_y(1) + r_xy(1) r_yx(1)) / (1 + r^2)``.
"""
import os
import tempfile

import numpy as np
import pytest
import yaml
from scipy import integrate, optimize, special

from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.misc import EdgeTimeSeries

STATS = ("excessvar", "topshare", "acf1")
IDS = {"excessvar": "edge_excessvar", "topshare": "edge_topshare_f-0-05",
       "acf1": "edge_acf1"}


# --------------------------------------------------------------------------
# Independent reference
# --------------------------------------------------------------------------

def _k0_null(r, frac):
    """Gaussian topshare null from the product-normal density.

    ``f(z) = exp(rho z / s) K0(|z| / s) / (pi sqrt(s))``, ``s = 1 - rho^2``,
    integrated directly (the SPI conditions on x instead).
    """
    rho = abs(r)
    s = 1 - rho**2
    f = lambda z: (np.exp((rho * z - abs(z)) / s) * special.k0e(abs(z) / s)
                   / (np.pi * np.sqrt(s)))
    tail = lambda q: integrate.quad(lambda z: f(z) + f(-z), q, np.inf, limit=200)[0]
    q = optimize.brentq(lambda q: tail(q) - frac, 1e-2, 100, xtol=1e-13)
    num = integrate.quad(lambda z: z * (f(z) - f(-z)), q, np.inf, limit=200)[0]
    den = 2 / np.pi * (np.sqrt(s) + rho * np.arcsin(rho))
    return np.sign(r) * num / den


def _reference(x, y, statistic, frac=0.05):
    T = len(x)
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    e = zx * zy
    c = (T - 1) / T
    m = e.mean()
    r = e.sum() / (T - 1)
    assert r == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)
    if statistic == "excessvar":
        return np.var(e) - c**2 * (1 + r**2 - (2 + 4 * r**2) / (T - 1))
    if statistic == "topshare":
        k = max(1, round(frac * T))
        top = np.argsort(-np.abs(e))[:k]
        return e[top].sum() / np.abs(e).sum() - _k0_null(r, k / T)
    if statistic == "acf1":
        d = e - m
        rho_e = np.sum(d[:-1] * d[1:]) / np.sum(d**2)
        rx = np.sum(zx[:-1] * zx[1:]) / (T - 1)
        ry = np.sum(zy[:-1] * zy[1:]) / (T - 1)
        rxy = np.sum(zx[:-1] * zy[1:]) / (T - 1)
        ryx = np.sum(zy[:-1] * zx[1:]) / (T - 1)
        return rho_e - (rx * ry + rxy * ryx) / (1 + r**2)
    raise ValueError(statistic)


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(7)
    M, T = 4, 203
    X = np.zeros((M, T))
    eps = rng.standard_normal((M, T))
    eps[1] += 0.6 * eps[0]
    for t in range(1, T):
        X[:, t] = 0.5 * X[:, t - 1] + eps[:, t]
        X[2, t] += 0.4 * X[0, t - 1]
    X[3] = X[3] ** 3 + 0.5 * X[1]  # a heavy-tailed, coupled process
    return X


@pytest.mark.parametrize("statistic", STATS)
def test_matches_hand_reference(fixture, statistic):
    X = fixture
    spi = EdgeTimeSeries(statistic)
    assert spi.identifier == IDS[statistic]
    A = spi.multivariate(Data(X, dim_order="ps"))
    data = Data(X, dim_order="ps")
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i == j:
                continue
            ref = _reference(X[i], X[j], statistic)
            assert A[i, j] == pytest.approx(ref, abs=1e-8), (i, j)
            assert spi.bivariate(data, i=i, j=j) == pytest.approx(ref, abs=1e-8)
    # z-scored internally (ddof=1), so un-z-scored Data gives the same values.
    raw = EdgeTimeSeries(statistic).multivariate(
        Data(3.0 * X + 2.0, dim_order="ps", zscore=False))
    np.testing.assert_allclose(raw, A, atol=1e-10, equal_nan=True)


def test_topshare_null_is_the_product_normal_closed_form():
    from pyspi.statistics.misc import _edge_gauss_topshare
    for r in (0.0, 0.02, 0.3, 0.5, 0.8, 0.97, -0.6):
        assert _edge_gauss_topshare(r, 0.05) == pytest.approx(_k0_null(r, 0.05), abs=1e-7)
    # r = 1: e = x^2, share = P(chi2_3 > q) with P(chi2_1 > q) = frac.
    q = special.chdtri(1, 0.05)
    assert _edge_gauss_topshare(1.0, 0.05) == pytest.approx(special.chdtrc(3, q), abs=1e-7)


# --------------------------------------------------------------------------
# Gaussian null: each statistic is ~0 in expectation
# --------------------------------------------------------------------------

def _pairs(X):
    """Compute every statistic on stacked pairs (rows 2k, 2k+1)."""
    data = Data(X, dim_order="ps")
    out = {}
    for s in STATS:
        A = EdgeTimeSeries(s).multivariate(data)
        out[s] = A[np.arange(0, len(X), 2), np.arange(1, len(X), 2)]
    return out


def _within(v, n_se=3.5):
    se = v.std(ddof=1) / np.sqrt(len(v))
    return abs(v.mean()) < n_se * se, (v.mean(), se)


def test_gaussian_iid_null():
    rng = np.random.default_rng(2026)
    n, T, r = 200, 1000, 0.5
    x = rng.standard_normal((n, T))
    y = r * x + np.sqrt(1 - r**2) * rng.standard_normal((n, T))
    X = np.empty((2 * n, T))
    X[0::2], X[1::2] = x, y
    out = _pairs(X)
    for s in ("excessvar", "acf1"):
        ok, info = _within(out[s])
        assert ok, (s, info)
    # topshare's reference is the large-T limit, so allow a small finite-T bias.
    assert abs(out["topshare"].mean()) < 0.004


def test_autocorrelated_gaussian_var_acf1_null():
    """For an AR(1) Gaussian VAR the edge series is persistent for linear
    reasons alone; the Isserlis reference removes it."""
    rng = np.random.default_rng(11)
    n, T, burn = 200, 1000, 200
    A = np.array([[0.6, 0.3], [0.0, 0.5]])
    eps = rng.standard_normal((n, 2, T + burn))
    eps[:, 1] += 0.5 * eps[:, 0]
    Z = np.zeros((n, 2, T + burn))
    for t in range(1, T + burn):
        Z[:, :, t] = Z[:, :, t - 1] @ A.T + eps[:, :, t]
    X = Z[:, :, burn:].reshape(2 * n, T)
    out = _pairs(X)
    ok, info = _within(out["acf1"])
    assert ok, info
    # premise: the raw lag-1 autocorrelation of e is far from 0
    zx = (X[0] - X[0].mean()) / X[0].std(ddof=1)
    zy = (X[1] - X[1].mean()) / X[1].std(ddof=1)
    e = zx * zy - np.mean(zx * zy)
    assert np.sum(e[:-1] * e[1:]) / np.sum(e**2) > 0.15


# --------------------------------------------------------------------------
# Sign checks on known coupling
# --------------------------------------------------------------------------

def _stack(pairs):
    X = np.empty((2 * len(pairs), len(pairs[0][0])))
    for k, (x, y) in enumerate(pairs):
        X[2 * k], X[2 * k + 1] = x, y
    return X


def _matched_gaussian(rng, n, T, r):
    return [(x, r * x + np.sqrt(1 - r**2) * rng.standard_normal(T))
            for x in rng.standard_normal((n, T))]


def test_bursty_coupling():
    """y = x in a random 5% of frames, independent noise elsewhere: e is
    heavy-tailed (excessvar ~ +0.1) but the coupled frames are of ordinary
    amplitude (e = x^2), so topshare stays ~0. Coupling carried
    by shared high-amplitude events moves both. A Gaussian pair matched to the
    same overall r gives ~0 for both."""
    rng = np.random.default_rng(5)
    n, T = 100, 1000

    switched = []
    for _ in range(n):
        x, y = rng.standard_normal(T), rng.standard_normal(T)
        on = rng.random(T) < 0.05
        y[on] = x[on]
        switched.append((x, y))
    events = []
    for _ in range(n):
        x, y = rng.standard_normal(T), rng.standard_normal(T)
        on = rng.random(T) < 0.05
        u = 2.0 * rng.standard_normal(T)
        x[on] += u[on]
        y[on] += u[on]
        events.append((x, y))

    for pairs, expect_top in ((switched, False), (events, True)):
        r = np.mean([np.corrcoef(x, y)[0, 1] for x, y in pairs])
        out = _pairs(_stack(pairs))
        gauss = _pairs(_stack(_matched_gaussian(rng, n, T, r)))
        v = out["excessvar"]
        assert v.mean() > 0.05 and v.mean() > 8 * v.std(ddof=1) / np.sqrt(n)
        assert _within(gauss["excessvar"])[0]
        assert _within(gauss["topshare"], 4)[0]
        t = out["topshare"]
        if expect_top:
            assert t.mean() > 0.05 and t.mean() > 8 * t.std(ddof=1) / np.sqrt(n)
            assert out["excessvar"].mean() > 0.5
        else:
            assert abs(t.mean()) < 0.01


def test_persistent_coupling_episodes_give_positive_acf1():
    """Coupling switched on in blocks of 10 frames persists: acf1 > 0."""
    rng = np.random.default_rng(9)
    n, T = 100, 1000
    pairs = []
    for _ in range(n):
        x, y = rng.standard_normal(T), rng.standard_normal(T)
        on = np.zeros(T, bool)
        for s in rng.choice(T - 10, 5, replace=False):
            on[s:s + 10] = True
        y[on] = x[on]
        pairs.append((x, y))
    a = _pairs(_stack(pairs))["acf1"]
    assert a.mean() > 0.02 and a.mean() > 8 * a.std(ddof=1) / np.sqrt(n)


def test_steady_coupling_is_negative():
    """y = 0.8 x + 0.6 z with bounded (uniform) x and z: sub-Gaussian e, so
    excessvar < 0 (its expectation is cum(x, x, y, y) = 0.64 * (-1.2) = -0.77).
    With Gaussian x the same bounded noise gives ~0: the statistic is the joint
    cumulant, which only the shared component's kurtosis enters."""
    rng = np.random.default_rng(3)
    n, T = 50, 1000
    half = np.sqrt(3)
    bounded = [(x, 0.8 * x + 0.6 * rng.uniform(-half, half, T))
               for x in rng.uniform(-half, half, (n, T))]
    v = _pairs(_stack(bounded))["excessvar"]
    assert v.mean() == pytest.approx(-0.768, abs=0.05)
    assert (v < 0).all()
    gauss_x = [(x, 0.8 * x + 0.6 * rng.uniform(-half, half, T))
               for x in rng.standard_normal((n, T))]
    assert _within(_pairs(_stack(gauss_x))["excessvar"])[0]


# --------------------------------------------------------------------------
# Structure, construction, Calculator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("statistic", STATS)
def test_symmetric_nan_diagonal_and_sign_flip(fixture, statistic):
    X = fixture
    A = EdgeTimeSeries(statistic).multivariate(Data(X, dim_order="ps"))
    assert np.isnan(np.diag(A)).all()
    off = ~np.eye(len(X), dtype=bool)
    assert np.isfinite(A[off]).all()
    np.testing.assert_array_equal(A, A.T)
    # y -> -y: excessvar and acf1 are even, topshare is odd (like r).
    Xf = X.copy()
    Xf[1] = -Xf[1]
    B = EdgeTimeSeries(statistic).multivariate(Data(Xf, dim_order="ps"))
    sign = -1 if statistic == "topshare" else 1
    assert B[0, 1] == pytest.approx(sign * A[0, 1], abs=1e-10)
    assert B[2, 3] == pytest.approx(A[2, 3], abs=1e-12)


def test_constructor():
    assert EdgeTimeSeries().identifier == "edge_excessvar"
    assert EdgeTimeSeries("topshare", frac=0.1).identifier == "edge_topshare_f-0-1"
    with pytest.raises(ValueError):
        EdgeTimeSeries("mean")
    with pytest.raises(ValueError):
        EdgeTimeSeries("acf1", frac=0.05)
    with pytest.raises(ValueError):
        EdgeTimeSeries("topshare", frac=1.0)
    spi = EdgeTimeSeries("acf1")
    assert spi.issigned() and "undirected" in spi.labels


def test_calculator_round_trip(fixture):
    cfg = {".statistics.misc": {"EdgeTimeSeries": {
        "labels": ["undirected", "nonlinear", "signed", "bivariate",
                   "time-dependent"],
        "dependencies": None,
        "configs": [{"statistic": "excessvar"},
                    {"statistic": "topshare", "frac": 0.05},
                    {"statistic": "acf1"}]}}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        calc = Calculator(dataset=Data(fixture, dim_order="ps"), config=f.name)
        calc.compute()
    finally:
        os.unlink(f.name)
    assert calc.errors == {}
    assert set(calc.spis) == set(IDS.values())
    for s, ident in IDS.items():
        ref = EdgeTimeSeries(s).multivariate(Data(fixture, dim_order="ps"))
        np.testing.assert_allclose(calc.table[ident].to_numpy(dtype=float), ref,
                                   atol=1e-12, equal_nan=True)


def test_bundled_configs():
    full = set(Calculator(config="full").spis)
    assert set(IDS.values()) <= full
    fast = set(Calculator(config="fast").spis)
    assert "edge_excessvar" in fast
    assert not {"edge_topshare_f-0-05", "edge_acf1"} & fast

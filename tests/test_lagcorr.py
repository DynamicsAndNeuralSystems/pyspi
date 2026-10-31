"""``LagResolvedCorrelation`` (``lagcorr_*``): directed reading of the xcorr profile.

The statistics are recomputed here from their definitions with explicit lag
sums -- ``r_{i->j}(tau) = sum_t x_i[t] x_j[t + tau] / (T s_i s_j)`` on the
demeaned series, and the first lag at which a biased sample ACF falls below 1/e
for the adaptive lags -- and compared with ``multivariate`` (the Calculator's
path) and ``bivariate``. The orientation of
the lag axis is checked on a pure delay and the sign on a negatively coupled
pair.
"""
import itertools
import os
import warnings

import numpy as np
import pytest
import yaml
from scipy import signal

from pyspi.data import Data
from pyspi.statistics import basic
from pyspi.statistics.basic import CrossCorrelation, LagResolvedCorrelation


def _data(X, zscore=True):
    return Data(data=np.array(X, dtype=float), dim_order="ps", zscore=zscore)


def _ar1(rng, T, phi):
    e = rng.standard_normal(T)
    x = np.zeros(T)
    x[0] = e[0]
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _series(T=160, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((4, T))
    X[1, 2:] += 0.8 * X[0, :-2]                       # 0 leads 1 by 2
    X[2, 1:] = -0.6 * X[1, :-1] + 0.7 * X[2, 1:]      # 1 leads 2 by 1, negative
    X[3] = _ar1(rng, T, 0.7) + 3.0                    # persistent, offset
    return X


def _full_yaml_configs():
    import pyspi
    path = os.path.join(os.path.dirname(pyspi.__file__), "configs", "full.yaml")
    with open(path) as fh:
        full = yaml.safe_load(fh)
    return full[".statistics.basic"]["LagResolvedCorrelation"]["configs"]


FULL_CONFIGS = [dict(c) for c in _full_yaml_configs()]
EXPECTED_IDS = [
    "lagcorr_lag_L-None",
    "lagcorr_peak_L-None",
    "lagcorr_asym_L-None",
    "lagcorr_onesided_tau-1_L-None",
    "lagcorr_onesided_tau-1e_L-None",
    "lagcorr_onesidedpeak_L-None",
    "lagcorr_lag_L-10",
    "lagcorr_peak_L-10",
    "lagcorr_asym_L-10",
]
# The bundled set plus variants that exercise a short window and a longer tau.
PARAMS = FULL_CONFIGS + [
    {"statistic": "onesided", "tau": 3},
    {"statistic": "onesided", "tau": "1e", "max_lag": 4},
    {"statistic": "onesidedpeak", "max_lag": 3},
    {"statistic": "lag", "max_lag": 3},
    {"statistic": "asym", "max_lag": 1},
]


# ---------------------------------------------------------------------------
# Independent reference
# ---------------------------------------------------------------------------

def _r_lead(a, b, tau):
    """corr(a[t], b[t + tau]) by an explicit lag sum, biased (divide by T)."""
    T = len(a)
    a = a - a.mean()
    b = b - b.mean()
    scale = T * np.sqrt(np.mean(a * a) * np.mean(b * b))
    if tau >= 0:
        return np.sum(a[:T - tau] * b[tau:]) / scale
    return np.sum(a[-tau:] * b[:T + tau]) / scale


def _length_1e(x):
    """First lag in 1..T//4 where the biased sample ACF is below 1/e, else inf."""
    T = len(x)
    x = x - x.mean()
    c0 = np.sum(x * x)
    for k in range(1, T // 4 + 1):
        if np.sum(x[:-k] * x[k:]) / c0 < np.exp(-1):
            return k
    return np.inf


def _adaptive_lag(l_i, l_j, T, L, statistic):
    """The pair's ``onesided`` 1e lag or ``onesidedpeak`` window, in 1..cap."""
    cap = max(1, min(T // 10, L))
    v = np.sqrt(l_i * l_j) if statistic == "onesided" else l_i + l_j
    if not np.isfinite(v):
        return cap
    return int(min(max(np.floor(v + 0.5), 1), cap))


def _reference(Z, statistic, max_lag=None, tau=None):
    M, T = Z.shape
    L = T // 4 if max_lag is None else min(max_lag, T // 4)
    lags = range(-L, L + 1)
    A = np.full((M, M), np.nan)
    for i, j in itertools.permutations(range(M), 2):
        r = {l: _r_lead(Z[i], Z[j], l) for l in lags}
        if statistic in ("lag", "peak"):
            best = max(lags, key=lambda l: abs(r[l]))
            A[i, j] = best if statistic == "lag" else r[best]
        elif statistic == "onesided":
            t = (_adaptive_lag(_length_1e(Z[i]), _length_1e(Z[j]), T, L, statistic)
                 if tau == "1e" else (1 if tau is None else tau))
            A[i, j] = r[t]
        elif statistic == "onesidedpeak":
            w = _adaptive_lag(_length_1e(Z[i]), _length_1e(Z[j]), T, L, statistic)
            A[i, j] = max(abs(r[k]) for k in range(1, w + 1))
        elif statistic == "asym":
            fwd = sum(r[l] ** 2 for l in range(1, L + 1))
            bwd = sum(r[-l] ** 2 for l in range(1, L + 1))
            A[i, j] = (fwd - bwd) / (fwd + bwd)
    return A


@pytest.mark.parametrize("zscore", [True, False])
@pytest.mark.parametrize("params", PARAMS, ids=str)
def test_matches_explicit_lag_sums(params, zscore):
    X = _series()
    Z = _data(X, zscore).to_numpy(squeeze=True)
    spi = LagResolvedCorrelation(**params)
    ref = _reference(Z, **params)

    mv = spi.multivariate(_data(X, zscore))
    data = _data(X, zscore)
    bv = np.full_like(mv, np.nan)
    for i, j in itertools.permutations(range(len(X)), 2):
        bv[i, j] = spi.bivariate(data, i=i, j=j)

    off = ~np.eye(len(X), dtype=bool)
    assert np.isnan(mv[~off]).all()
    if params["statistic"] == "lag":
        assert np.array_equal(mv[off], ref[off])
        assert np.array_equal(bv[off], ref[off])
    else:
        np.testing.assert_allclose(mv[off], ref[off], rtol=0, atol=1e-12)
        np.testing.assert_allclose(bv[off], ref[off], rtol=0, atol=1e-12)
    # The two paths share one per-pair computation, so they agree exactly.
    assert np.array_equal(mv[off], bv[off])


def test_reference_is_not_vacuous():
    """The fixture's pairs give distinct, non-trivial answers per statistic."""
    Z = _data(_series()).to_numpy(squeeze=True)
    lag = _reference(Z, "lag", max_lag=10)
    assert lag[0, 1] == 2 and lag[1, 2] == 1
    assert _reference(Z, "peak")[1, 2] < -0.3
    one = _reference(Z, "onesided", tau=1)
    # Process 3 is AR(1) with phi = 0.7: its ACF falls below 1/e after lag 1,
    # so its pairs get a lag and a window longer than 1.
    assert _length_1e(Z[3]) > 1 and _length_1e(Z[0]) == 1
    assert _reference(Z, "onesided", tau="1e")[3, 1] != one[3, 1]
    assert _reference(Z, "onesidedpeak")[3, 1] != abs(one[3, 1])


# ---------------------------------------------------------------------------
# Orientation, sign and structure on known couplings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", ["source_first", "source_last"])
def test_pure_delay_orientation(order):
    """x_j[t] = x_i[t - 3]: i leads j by 3, so (i, j) is +3 and (j, i) is -3.

    Both index orders, so the lower-index-first canonical orientation inside
    the class is exercised either way round.
    """
    rng = np.random.default_rng(0)
    T = 400
    a = rng.standard_normal(T + 3)
    x_i, x_j = a[3:], a[:-3]
    assert np.array_equal(x_j[3:], x_i[:-3])
    noise = rng.standard_normal(T)
    if order == "source_first":
        X, i, j = np.vstack([x_i, x_j, noise]), 0, 1
    else:
        X, i, j = np.vstack([noise, x_j, x_i]), 2, 1

    for max_lag in (None, 10):
        data = _data(X)
        lag = LagResolvedCorrelation("lag", max_lag=max_lag)
        A = lag.multivariate(data)
        assert A[i, j] == 3 and A[j, i] == -3
        assert lag.bivariate(data, i=i, j=j) == 3
        assert lag.bivariate(data, i=j, j=i) == -3

        peak = LagResolvedCorrelation("peak", max_lag=max_lag).multivariate(data)
        assert peak[i, j] == peak[j, i] > 0.98

        asym = LagResolvedCorrelation("asym", max_lag=max_lag).multivariate(data)
        assert asym[i, j] > 0.5 and asym[j, i] == -asym[i, j]

    onesided = LagResolvedCorrelation("onesided", tau=3).multivariate(_data(X))
    assert onesided[i, j] > 0.98 and abs(onesided[j, i]) < 0.15

    # And it is the cached profile read at index T//4 - 3, not T//4 + 3.
    data = _data(X)
    CrossCorrelation(sigonly=False).bivariate(data, i=i, j=j)
    P = data.xcorr[(i, j)]
    c = T // 4
    assert np.argmax(P) == c - 3


def test_negative_coupling_keeps_its_sign():
    """x_j[t] = -0.9 x_i[t - 2] + noise: peak < 0 at lag +2, asym > 0.

    ``xcorr_max_sig-True`` (the largest signed value) is positive on this pair.
    """
    rng = np.random.default_rng(0)
    T = 500
    x_i = rng.standard_normal(T)
    x_j = np.zeros(T)
    x_j[2:] = -0.9 * x_i[:-2]
    x_j += rng.standard_normal(T)
    X = np.vstack([x_i, x_j])

    for max_lag in (None, 10):
        data = _data(X)
        spi = lambda s: LagResolvedCorrelation(s, max_lag=max_lag)
        assert spi("lag").bivariate(data, i=0, j=1) == 2
        assert spi("lag").bivariate(data, i=1, j=0) == -2
        peak = spi("peak").bivariate(data, i=0, j=1)
        assert peak < -0.5
        assert spi("peak").bivariate(data, i=1, j=0) == peak
        assert spi("asym").bivariate(data, i=0, j=1) > 0.3
        assert spi("asym").bivariate(data, i=1, j=0) < -0.3

    data = _data(X)
    assert LagResolvedCorrelation("onesided", tau=2).bivariate(
        data, i=0, j=1) == pytest.approx(peak, abs=1e-15)
    assert CrossCorrelation(statistic="max", sigonly=True).bivariate(
        data, i=0, j=1) > 0


@pytest.mark.parametrize("max_lag", [None, 10])
def test_lag_and_asym_are_antisymmetric_and_peak_symmetric(max_lag):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, 300)).cumsum(axis=1) * 0.05 + rng.standard_normal((6, 300))
    X[2, 4:] += 0.5 * X[5, :-4]
    data = _data(X)
    off = ~np.eye(6, dtype=bool)
    for stat in ("lag", "asym"):
        A = LagResolvedCorrelation(stat, max_lag=max_lag).multivariate(data)
        assert np.isfinite(A[off]).all()
        assert np.array_equal(A, -A.T, equal_nan=True), stat
        assert not np.signbit(A[A == 0]).any(), "negative zero in " + stat
    A = LagResolvedCorrelation("lag", max_lag=max_lag).multivariate(data)
    assert np.array_equal(A, np.round(A), equal_nan=True)   # integer lags
    L = 300 // 4 if max_lag is None else max_lag
    assert np.abs(A[off]).max() <= L
    P = LagResolvedCorrelation("peak", max_lag=max_lag).multivariate(data)
    assert np.array_equal(P, P.T, equal_nan=True)
    asym = LagResolvedCorrelation("asym", max_lag=max_lag).multivariate(data)
    assert np.abs(asym[off]).max() <= 1


def test_labels_match_the_matrix_structure():
    traits = {"lag": "antisymmetric", "peak": "undirected",
              "onesided": "directed", "onesidedpeak": "directed",
              "asym": "antisymmetric"}
    for stat, trait in traits.items():
        spi = LagResolvedCorrelation(stat)
        sign = "unsigned" if stat == "onesidedpeak" else "signed"
        assert trait in spi.labels and sign in spi.labels and "linear" in spi.labels
        assert len({"antisymmetric", "undirected", "directed"} & set(spi.labels)) == 1
        assert len({"signed", "unsigned"} & set(spi.labels)) == 1
        assert spi.issigned() == (sign == "signed")
    # The class attribute is untouched by an unsigned instance.
    assert LagResolvedCorrelation("onesided").issigned()


def test_max_lag_limits_the_search():
    """A delay of 15 is found with the full window and not with max_lag=10."""
    rng = np.random.default_rng(4)
    T = 400
    a = rng.standard_normal(T + 15)
    X = np.vstack([a[15:], a[:-15] + 0.3 * rng.standard_normal(T)])
    data = _data(X)
    assert LagResolvedCorrelation("lag").bivariate(data, i=0, j=1) == 15
    short = LagResolvedCorrelation("lag", max_lag=10).bivariate(data, i=0, j=1)
    assert abs(short) <= 10 and short != 15
    # A max_lag beyond T // 4 is the full window.
    assert LagResolvedCorrelation("lag", max_lag=10_000).bivariate(
        data, i=0, j=1) == 15


# ---------------------------------------------------------------------------
# Adaptive lags: tau = "1e" and onesidedpeak
# ---------------------------------------------------------------------------

def _ar_pair(T=1200, seed=11):
    """A slow AR(1) source driving a faster target at lag 1."""
    rng = np.random.default_rng(seed)
    src = _ar1(rng, T, 0.9)
    dst = np.zeros(T)
    dst[1:] = 0.5 * src[:-1]
    dst += _ar1(rng, T, 0.3)
    return np.vstack([src, dst])


def _seeded(X, lengths):
    """Data whose cached 1/e lengths are ``lengths`` (exercises the pair rule)."""
    data = _data(X)
    data.acf_1e = dict(enumerate(lengths))
    return data


def test_1e_length_and_pair_lag_match_a_direct_reference():
    """First ACF < 1/e by explicit sums, then the rounded geometric mean."""
    X = _ar_pair()
    T = X.shape[1]
    Z = _data(X).to_numpy(squeeze=True)
    data = _data(X)
    l_src, l_dst = _length_1e(Z[0]), _length_1e(Z[1])
    assert basic._acf_1e_length(data, 0) == l_src
    assert basic._acf_1e_length(data, 1) == l_dst
    # phi = 0.9 crosses near lag 10; the target is faster.
    assert 5 < l_src < 20 and 1 <= l_dst < l_src

    tau = int(np.floor(np.sqrt(l_src * l_dst) + 0.5))
    assert 1 < tau < T // 10
    A = LagResolvedCorrelation("onesided", tau="1e").multivariate(data)
    assert A[0, 1] == pytest.approx(_r_lead(Z[0], Z[1], tau), abs=1e-12)
    assert A[1, 0] == pytest.approx(_r_lead(Z[1], Z[0], tau), abs=1e-12)
    np.testing.assert_array_equal(
        A, LagResolvedCorrelation("onesided", tau=tau).multivariate(_data(X)))


@pytest.mark.parametrize("lengths, max_lag, want", [
    ((1.0, 10.0), None, 3),     # sqrt(10) = 3.16 (the arithmetic mean is 5.5)
    ((3.0, 4.0), None, 3),      # sqrt(12) = 3.46 rounds down
    ((3.0, 5.0), None, 4),      # sqrt(15) = 3.87 rounds up
    ((2.0, 8.0), None, 4),      # exact
    ((40.0, 45.0), None, 20),   # clipped to T // 10
    ((3.0, 5.0), 3, 3),         # clipped to max_lag
    ((np.inf, 1.0), None, 20),  # no 1/e crossing: the cap, not sqrt(20 * 1)
    ((np.inf, np.inf), 7, 7),
])
def test_1e_pair_lag_rounding_and_clipping(lengths, max_lag, want):
    X = np.random.default_rng(0).standard_normal((2, 200))   # T // 10 = 20
    Z = _data(X).to_numpy(squeeze=True)
    spi = LagResolvedCorrelation("onesided", tau="1e", max_lag=max_lag)
    A = spi.multivariate(_seeded(X, lengths))
    assert A[0, 1] == pytest.approx(_r_lead(Z[0], Z[1], want), abs=1e-12)
    assert A[1, 0] == pytest.approx(_r_lead(Z[1], Z[0], want), abs=1e-12)
    assert spi.bivariate(_seeded(X, lengths), i=1, j=0) == A[1, 0]


def test_1e_length_without_a_crossing_is_unbounded():
    # A step: the ACF at lag 1 (= T // 4 for T = 7) is 0.56, above 1/e.
    step = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)
    assert _length_1e(step) == np.inf
    data = _data(np.vstack([step, step[::-1] + 0.1 * np.arange(7)]))
    assert basic._acf_1e_length(data, 0) == np.inf
    # A white series crosses at once.
    white = np.random.default_rng(5).standard_normal((1, 400))
    assert basic._acf_1e_length(_data(np.vstack([white, white])), 0) == 1


def test_1e_lag_is_shared_by_both_directions():
    """x = u + v/4, y = v + u/4 of an independent fast u and slow v.

    The pair is correlated at lag 0 with no directed coupling, and the two
    series have different 1/e lengths. One shared lag reads |r| about equally
    in both directions; each source's own lag would report x -> y.
    """
    rng = np.random.default_rng(0)
    T = 2000
    u, v = _ar1(rng, T, 0.2), _ar1(rng, T, 0.95)
    X = np.vstack([u + 0.25 * v, v + 0.25 * u])
    Z = _data(X).to_numpy(squeeze=True)
    l_x, l_y = _length_1e(Z[0]), _length_1e(Z[1])
    assert l_y > 3 * l_x
    tau = int(np.floor(np.sqrt(l_x * l_y) + 0.5))

    A = LagResolvedCorrelation("onesided", tau="1e").multivariate(_data(X))
    assert A[0, 1] == pytest.approx(_r_lead(Z[0], Z[1], tau), abs=1e-12)
    assert A[1, 0] == pytest.approx(_r_lead(Z[1], Z[0], tau), abs=1e-12)
    assert abs(abs(A[0, 1]) - abs(A[1, 0])) < 0.05
    per_source = abs(_r_lead(Z[0], Z[1], l_x)) - abs(_r_lead(Z[1], Z[0], l_y))
    assert per_source > 0.3


def test_1e_lengths_are_computed_once_per_series(monkeypatch):
    calls = []
    real_acf = basic.acf
    monkeypatch.setattr(basic, "acf", lambda x: calls.append(1) or real_acf(x))
    X = _series()
    data = _data(X)
    for spi in (LagResolvedCorrelation("onesided", tau="1e"),
                LagResolvedCorrelation("onesidedpeak"),
                LagResolvedCorrelation("onesidedpeak", max_lag=3)):
        spi.multivariate(data)
        spi.bivariate(data, i=2, j=0)
    assert len(calls) == len(X)
    assert set(data.acf_1e) == set(range(len(X)))


def test_onesidedpeak_matches_a_brute_force_window():
    X = _ar_pair()
    Z = _data(X).to_numpy(squeeze=True)
    w = int(_length_1e(Z[0]) + _length_1e(Z[1]))
    assert 1 < w < X.shape[1] // 10
    A = LagResolvedCorrelation("onesidedpeak").multivariate(_data(X))
    for i, j in ((0, 1), (1, 0)):
        want = max(abs(_r_lead(Z[i], Z[j], k)) for k in range(1, w + 1))
        assert A[i, j] == pytest.approx(want, abs=1e-12)
    assert A[0, 1] > A[1, 0] > 0


@pytest.mark.parametrize("delay, lengths, max_lag, found", [
    (2, None, None, True),          # white noise: w = 1 + 1 = 2
    (3, None, None, False),
    (3, (5.0, 5.0), None, True),    # w = 10
    (3, (5.0, 5.0), 2, False),      # w clipped to max_lag
    (3, (np.inf, 1.0), None, True), # w = T // 10 = 40
    (45, (np.inf, 1.0), None, False),
])
def test_onesidedpeak_window(delay, lengths, max_lag, found):
    """A pure delay is seen only when it lies inside the pair's window."""
    rng = np.random.default_rng(1)
    T = 400
    a = rng.standard_normal(T + delay)
    X = np.vstack([a[delay:], a[:-delay]])           # 0 leads 1 by `delay`
    data = _data(X) if lengths is None else _seeded(X, lengths)
    A = LagResolvedCorrelation("onesidedpeak", max_lag=max_lag).multivariate(data)
    assert (A[0, 1] > 0.85) == found
    assert A[1, 0] < 0.2
    assert (A[~np.eye(2, dtype=bool)] >= 0).all()


# ---------------------------------------------------------------------------
# Parameters, identifiers and failures
# ---------------------------------------------------------------------------

def test_identifiers():
    assert [LagResolvedCorrelation(**c).identifier for c in FULL_CONFIGS] == EXPECTED_IDS
    assert LagResolvedCorrelation("onesided").identifier == "lagcorr_onesided_tau-1_L-None"
    assert LagResolvedCorrelation("onesided", tau=3, max_lag=5).identifier == (
        "lagcorr_onesided_tau-3_L-5")
    assert LagResolvedCorrelation("onesided", tau="1e", max_lag=5).identifier == (
        "lagcorr_onesided_tau-1e_L-5")
    assert LagResolvedCorrelation("onesidedpeak").identifier == "lagcorr_onesidedpeak_L-None"
    assert LagResolvedCorrelation("onesidedpeak", max_lag=8).identifier == (
        "lagcorr_onesidedpeak_L-8")


@pytest.mark.parametrize("kwargs, err", [
    ({"statistic": "max"}, ValueError),
    ({"statistic": "lag", "tau": 1}, ValueError),
    ({"statistic": "onesided", "tau": 0}, ValueError),
    ({"statistic": "onesided", "tau": "pacf"}, TypeError),
    ({"statistic": "onesided", "tau": "acf"}, TypeError),
    ({"statistic": "onesidedpeak", "tau": 1}, ValueError),
    ({"statistic": "onesided", "tau": 1.5}, TypeError),
    ({"statistic": "lag", "max_lag": 0}, ValueError),
    ({"statistic": "lag", "max_lag": True}, TypeError),
])
def test_bad_parameters_raise(kwargs, err):
    with pytest.raises(err):
        LagResolvedCorrelation(**kwargs)


def test_tau_outside_the_window_raises():
    X = np.random.default_rng(0).standard_normal((2, 40))   # T // 4 = 10
    spi = LagResolvedCorrelation("onesided", tau=11)
    with pytest.raises(ValueError, match="outside the lag window"):
        spi.bivariate(_data(X), i=0, j=1)
    with pytest.raises(ValueError, match="outside the lag window"):
        spi.multivariate(_data(X))
    with pytest.raises(ValueError, match="outside the lag window"):
        LagResolvedCorrelation("onesided", tau=4, max_lag=3).bivariate(
            _data(X), i=0, j=1)


def test_too_short_for_any_lag_raises():
    X = np.random.default_rng(0).standard_normal((2, 3))     # T // 4 = 0
    with pytest.raises(ValueError, match="at least one lag"):
        LagResolvedCorrelation("lag").bivariate(_data(X), i=0, j=1)


def test_a_constant_process_voids_only_its_own_pairs():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 200))
    X[2] = 5.0
    for params in ({"statistic": "lag"}, {"statistic": "peak"}, {"statistic": "asym"},
                   {"statistic": "onesided"}, {"statistic": "onesided", "tau": "1e"},
                   {"statistic": "onesidedpeak"}):
        spi = LagResolvedCorrelation(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            A = spi.multivariate(_data(X, zscore=False))
        assert np.isnan(A[2]).all() and np.isnan(A[:, 2]).all()
        good = ~np.eye(4, dtype=bool)
        good[2] = good[:, 2] = False
        assert np.isfinite(A[good]).all()
        msgs = [str(w.message) for w in caught if "undefined and set to NaN" in str(w.message)]
        assert len(msgs) == 1 and spi.identifier in msgs[0]
        # 0/0 in the profile's normalisation (CrossCorrelation's own warning).
        with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="not finite"):
            spi.bivariate(_data(X, zscore=False), i=0, j=2)


def test_shares_the_xcorr_cache_and_scheduling_bucket():
    """Reads the profile CrossCorrelation cached, and fills it the same way."""
    from pyspi._parallel import cache_bucket

    assert cache_bucket(LagResolvedCorrelation("lag")) == cache_bucket(
        LagResolvedCorrelation("onesided", tau="1e", max_lag=10)) == cache_bucket(
        CrossCorrelation()) == ("xcorr",)

    X = _series()
    shared = _data(X)
    CrossCorrelation().multivariate(shared)
    before = {k: v for k, v in shared.xcorr.items()}
    for c in PARAMS:
        A = LagResolvedCorrelation(**c).multivariate(shared)
        B = LagResolvedCorrelation(**c).multivariate(_data(X))
        np.testing.assert_allclose(A, B, rtol=0, atol=1e-13)
    # No profile recomputed: the cache holds the very same arrays.
    assert all(shared.xcorr[k] is v for k, v in before.items())

    # Filled by this class first, the cache is what CrossCorrelation computes.
    mine = _data(X)
    LagResolvedCorrelation("peak").multivariate(mine)
    fresh = _data(X)
    for i, j in itertools.permutations(range(len(X)), 2):
        CrossCorrelation(sigonly=False).bivariate(fresh, i=i, j=j)
    assert set(mine.xcorr) == set(fresh.xcorr)


# ---------------------------------------------------------------------------
# CrossCorrelation values from the shared profile
# ---------------------------------------------------------------------------

def _reference_profile(data, i, j):
    """The xcorr profile computed inline, independently of `_xcorr_profile`."""
    T = data.n_observations
    x, y = data.to_numpy(squeeze=True)[[i, j]]
    x = x - x.mean()
    y = y - y.mean()
    scale = T * np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))
    r_full = signal.correlate(x, y, "full", method="fft") / scale
    lag_max = T // 4
    return r_full[T - 1 - lag_max: T + lag_max]


def _reference_xcorr(r_ij, T, statistic, squared, sigonly):
    if sigonly:
        above = np.abs(r_ij) > 1.96 / np.sqrt(T)
        if not above.any():
            return 0.0
        r_ij = r_ij[above]
    if squared:
        r_ij = r_ij ** 2
    return float(np.max(r_ij)) if statistic == "max" else float(np.mean(r_ij))


XCORR_PARAMS = [dict(statistic=s, squared=q, sigonly=g)
                for s in ("max", "mean") for q in (False, True) for g in (False, True)]


@pytest.mark.parametrize("zscore", [True, False])
@pytest.mark.parametrize("params", XCORR_PARAMS, ids=str)
def test_xcorr_values_match_inline_reference(params, zscore):
    """Profiles and xcorr_* values are bit-identical to an inline reference.

    `Directed.multivariate` visits (1, 0) before (0, 1), so the reference
    fills the (j, i) orientation of each pair first, as the cache does.
    """
    X = _series(T=97, seed=5)
    M, T = X.shape
    data = _data(X, zscore)
    spi = CrossCorrelation(**params)
    A = spi.multivariate(data)

    ref_profiles = {}
    ref = _data(X, zscore)
    for j in range(M):
        for i in range(M):
            if i == j:
                continue
            if (i, j) not in ref_profiles:
                ref_profiles[(i, j)] = _reference_profile(ref, i, j)
                ref_profiles[(j, i)] = ref_profiles[(i, j)][::-1]
    assert set(data.xcorr) == set(ref_profiles)
    for key, prof in ref_profiles.items():
        assert np.array_equal(data.xcorr[key], prof), key
    for i, j in itertools.permutations(range(M), 2):
        want = _reference_xcorr(ref_profiles[(i, j)], T, **params)
        assert abs(A[i, j] - want) <= 1e-15, (i, j)


def test_xcorr_values_match_reference_constants():
    """Reference values computed with CrossCorrelation at commit 3b2deb6.

    The tolerance allows for a different FFT build.
    """
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((3, 80))
    X[1, 2:] += 0.7 * X[0, :-2]
    X[2] = -0.6 * X[0] + 0.8 * X[2]
    expected = {
        "xcorr_max_sig-False": (0.5186610806879121, 0.16294961551037665, 0.2209728058503341),
        "xcorr_max_sig-True": (0.5186610806879121, -0.23302658723588135, 0.2209728058503341),
        "xcorr_mean_sig-False": (0.023741996320894965, -0.03828272284937119, -0.04378325616369184),
        "xcorr-sq_mean_sig-True": (0.2690093166203528, 0.133516059602504, 0.06277542070660734),
    }
    spis = [CrossCorrelation(statistic="max", sigonly=False), CrossCorrelation(statistic="max"),
            CrossCorrelation(statistic="mean", sigonly=False),
            CrossCorrelation(statistic="mean", squared=True)]
    for spi in spis:
        A = spi.multivariate(_data(X))
        got = (A[0, 1], A[0, 2], A[1, 2])
        np.testing.assert_allclose(got, expected[spi.identifier], rtol=0, atol=1e-13)
    data = _data(X)
    CrossCorrelation(sigonly=False).bivariate(data, i=0, j=1)
    np.testing.assert_allclose(
        data.xcorr[(0, 1)][[0, 18, 20, 40]],
        [0.00029826295061468423, 0.5186610806879121, 0.1088004042397863,
         -0.011497683493100878], rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# Calculator and bundled configs
# ---------------------------------------------------------------------------

def test_calculator_round_trip(tmp_path):
    """Only this class (and CrossCorrelation, which shares its cache)."""
    from pyspi.calculator import Calculator

    cfg = {".statistics.basic": {
        "CrossCorrelation": {"labels": ["undirected"], "configs": [{"statistic": "max"}]},
        "LagResolvedCorrelation": {
            "labels": ["directed", "linear", "signed", "bivariate",
                       "time-dependent", "M14"],
            "configs": FULL_CONFIGS},
    }}
    path = tmp_path / "lagcorr.yaml"
    path.write_text(yaml.safe_dump(cfg))
    X = _series()
    calc = Calculator(dataset=_data(X), config=str(path))
    calc.compute()
    assert calc.errors == {}
    ids = [k for k in calc.spis if k.startswith("lagcorr_")]
    assert ids == EXPECTED_IDS
    off = ~np.eye(len(X), dtype=bool)
    for c, ident in zip(FULL_CONFIGS, EXPECTED_IDS):
        got = calc.table[ident].to_numpy(dtype=float)
        want = LagResolvedCorrelation(**c).multivariate(_data(X))
        np.testing.assert_allclose(got[off], want[off], rtol=0, atol=1e-13)
        assert np.isfinite(got[off]).all()
    assert "antisymmetric" in calc.spis["lagcorr_lag_L-None"].labels
    assert "undirected" in calc.spis["lagcorr_peak_L-10"].labels
    assert "directed" in calc.spis["lagcorr_onesided_tau-1e_L-None"].labels
    peak1s = calc.spis["lagcorr_onesidedpeak_L-None"]
    assert "unsigned" in peak1s.labels and "signed" not in peak1s.labels
    assert not peak1s.issigned()


def test_full_config_lists_every_identifier():
    from pyspi.calculator import Calculator

    calc = Calculator(config="full")
    assert [k for k in calc.spis if k.startswith("lagcorr_")] == EXPECTED_IDS
    keys = list(calc.spis)
    xcorr_last = max(n for n, k in enumerate(keys) if k.startswith("xcorr"))
    assert keys[xcorr_last + 1] == EXPECTED_IDS[0]   # right after CrossCorrelation


def test_fast_config_lists_its_subset():
    from pyspi.calculator import load_spis_from_yaml, resolve_config

    spis = load_spis_from_yaml(resolve_config("fast"), quiet=True)
    assert [k for k in spis if k.startswith("lagcorr_")] == [
        "lagcorr_lag_L-None", "lagcorr_peak_L-None", "lagcorr_asym_L-None",
        "lagcorr_onesided_tau-1_L-None"]

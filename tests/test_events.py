"""Event synchronisation (``es_*``) and tail-dependence asymmetry (``tailasym_*``).

References are recomputed here from the definitions with plain loops, on event
trains small enough to count by hand, and checked against the SPIs through
both ``multivariate`` and ``bivariate``.
"""
import importlib

import numpy as np
import pytest
import yaml
from scipy import stats

from pyspi.calculator import Calculator, load_spis_from_yaml, resolve_config
from pyspi.data import Data
from pyspi.statistics.events import (
    EventSynchronisation,
    TailDependenceAsymmetry,
    _es_counts,
    _half_gaps,
)


# --------------------------------------------------------------------------
# Event synchronisation: brute-force reference
# --------------------------------------------------------------------------

def _es_reference(tx, ty, max_delay=None):
    """``(c(y|x), c(x|y))`` from the definition, by a double loop.

    ``c(y|x)``: events of y following an event of x within
    ``tau_ab = min(adjacent gaps of x_a and y_b) / 2`` (capped at
    ``max_delay``), plus 1/2 per equal-time pair. A coincidence whose x or y
    event also coincides in the opposite direction counts 1/2.
    """
    tx, ty = list(tx), list(ty)

    def tau(a, b):
        gaps = []
        for t, k in ((tx, a), (ty, b)):
            if k + 1 < len(t):
                gaps.append(t[k + 1] - t[k])
            if k > 0:
                gaps.append(t[k] - t[k - 1])
        value = min(gaps) / 2
        return value if max_delay is None else min(value, max_delay)

    follows, precedes, equal = [], [], 0
    for a, t_a in enumerate(tx):
        for b, t_b in enumerate(ty):
            d = t_b - t_a
            if d == 0:
                equal += 1
            elif 0 < d <= tau(a, b):
                follows.append((a, b))  # y_b follows x_a
            elif 0 < -d <= tau(a, b):
                precedes.append((a, b))  # x_a follows y_b

    def weight(pair, opposite):
        a, b = pair
        double = any(p[0] == a or p[1] == b for p in opposite)
        return 0.5 if double else 1.0

    c_yx = sum(weight(p, precedes) for p in follows) + 0.5 * equal
    c_xy = sum(weight(p, follows) for p in precedes) + 0.5 * equal
    return c_yx, c_xy


# Hand-counted. x gaps 10, 10, 20, 30 -> half gaps 5, 5, 5, 10, 15;
# y gaps 8, 16, 12, 42 -> half gaps 4, 4, 6, 6, 21.
#   x10 -> y12: d = 2 <= tau = min(5, 4) = 4                     follows, 1
#   x20 == y20: equal times                                       1/2 each
#   x30 -> y36: d = 6 >  tau = min(5, 6) = 5   (outside tau)      0
#   y48 -> x50: d = 2 <= tau = min(10, 6) = 6                     precedes, 1
#   x80 -> y90: d = 10 <= tau = min(15, 21) = 15                  follows, 1
# c(y|x) = 1 + 1 + 1/2 = 2.5, c(x|y) = 1 + 1/2 = 1.5, m_x = m_y = 5.
# With max_delay = 8 the x80 -> y90 pair (d = 10) drops out.
TX_A = [10, 20, 30, 50, 80]
TY_A = [12, 20, 36, 48, 90]
HAND_A = {None: (2.5, 1.5), 8: (1.5, 1.5)}

# Midpoint ties: x10 is exactly midway between y6 and y14 (tau = 4 to both),
# x30 between y26 and y34. All four coincidences are double, so each counts
# 1/2: c(y|x) = c(x|y) = 1, Q = 2 / sqrt(2 * 4). Counted at 1 each they would
# give Q = 4 / sqrt(8) = 1.41 > 1.
TX_B = [10, 30]
TY_B = [6, 14, 26, 34]
HAND_B = (1.0, 1.0)


def test_reference_reproduces_the_hand_counts():
    for max_delay, expected in HAND_A.items():
        assert _es_reference(TX_A, TY_A, max_delay) == expected
    assert _es_reference(TX_B, TY_B) == HAND_B


def _counts(tx, ty, max_delay=None):
    tx, ty = np.asarray(tx), np.asarray(ty)
    return _es_counts(tx, _half_gaps(tx), ty, _half_gaps(ty), max_delay)


def test_sparse_counts_match_the_reference_on_random_event_trains():
    """The searchsorted enumeration finds every coincidence the double loop does."""
    assert _counts(TX_A, TY_A) == HAND_A[None]
    assert _counts(TX_A, TY_A, 8) == HAND_A[8]
    assert _counts(TX_B, TY_B) == HAND_B
    rng = np.random.default_rng(0)
    n_double = 0
    for trial in range(300):
        T = int(rng.integers(20, 120))
        p = rng.uniform(0.05, 0.4)
        tx = np.flatnonzero(rng.random(T) < p)
        ty = np.flatnonzero(rng.random(T) < p)
        if trial % 3 == 0:  # a lagged, thinned copy: many coincidences
            ty = np.unique(np.concatenate([tx[rng.random(tx.size) < 0.8] + 2,
                                           np.flatnonzero(rng.random(T) < 0.05)]))
        if tx.size < 2 or ty.size < 2:
            continue
        max_delay = [None, 1, 3][trial % 3]
        ref = _es_reference(tx, ty, max_delay)
        assert _counts(tx, ty, max_delay) == ref, (trial, tx, ty, max_delay)
        # Swapping the roles swaps the two counts.
        assert _counts(ty, tx, max_delay) == ref[::-1]
        n_double += float(ref[0]) != int(ref[0])
    assert n_double > 10  # the half-weight paths were exercised


def _series_with_events(T, onsets, run_lengths, rng):
    """Baseline below 0 and runs of values above 1 starting at ``onsets``."""
    x = rng.uniform(-1.0, 0.0, T)
    for onset, length in zip(onsets, run_lengths):
        x[onset:onset + length] = 1.0 + rng.uniform(0.0, 1.0, length)
    return x


def _quantile_for(n_exceed, T):
    """np.quantile(x, 1 - n/T) lies strictly between the n-th and (n+1)-th
    largest values (linear interpolation), so exactly n samples exceed it."""
    return 1.0 - n_exceed / T


@pytest.mark.parametrize("max_delay", [None, 8])
def test_es_matches_the_hand_counts_on_series(max_delay):
    """Series built to exceed their quantile exactly at TX_A / TY_A.

    x has a run of two exceedances at 10 and y one at 90: each run is one event
    at its first sample, so both series have 6 exceedances and 5 events.
    """
    rng = np.random.default_rng(1)
    T = 100
    x = _series_with_events(T, TX_A, [2, 1, 1, 1, 1], rng)
    y = _series_with_events(T, TY_A, [1, 1, 1, 1, 2], rng)
    data = Data(np.vstack([x, y]), dim_order="ps")
    q = _quantile_for(6, T)
    c_yx, c_xy = HAND_A[max_delay]
    norm = np.sqrt(5 * 5)

    Q_spi = EventSynchronisation("Q", quantile=q, max_delay=max_delay)
    q_spi = EventSynchronisation("q", quantile=q, max_delay=max_delay)
    assert Q_spi.multivariate(data)[0, 1] == pytest.approx((c_yx + c_xy) / norm, abs=1e-12)
    assert Q_spi.bivariate(data, i=0, j=1) == pytest.approx((c_yx + c_xy) / norm, abs=1e-12)
    assert Q_spi.bivariate(data, i=1, j=0) == pytest.approx((c_yx + c_xy) / norm, abs=1e-12)
    A = q_spi.multivariate(data)
    assert A[0, 1] == pytest.approx((c_yx - c_xy) / norm, abs=1e-12)
    assert A[1, 0] == pytest.approx(-(c_yx - c_xy) / norm, abs=1e-12)
    assert q_spi.bivariate(data, i=0, j=1) == pytest.approx(A[0, 1], abs=1e-12)
    assert q_spi.bivariate(data, i=1, j=0) == pytest.approx(A[1, 0], abs=1e-12)
    if max_delay is None:
        assert (A[0, 1], Q_spi.multivariate(data)[0, 1]) == pytest.approx((0.2, 0.8))


def test_es_halves_midpoint_ties_so_Q_stays_at_most_one():
    rng = np.random.default_rng(2)
    T = 100
    x = _series_with_events(T, TX_B, [2, 2], rng)
    y = _series_with_events(T, TY_B, [1, 1, 1, 1], rng)
    data = Data(np.vstack([x, y]), dim_order="ps")
    q = _quantile_for(4, T)
    Q = EventSynchronisation("Q", quantile=q).bivariate(data, i=0, j=1)
    assert Q == pytest.approx(2 / np.sqrt(8))
    assert EventSynchronisation("q", quantile=q).bivariate(data, i=0, j=1) == 0.0


# --------------------------------------------------------------------------
# Event synchronisation: pyunicorn, if it happens to be installed
# --------------------------------------------------------------------------

def _pyunicorn_event_series():
    pytest.importorskip("pyunicorn")
    for name in ("pyunicorn.eventseries", "pyunicorn.timeseries"):
        try:
            cls = getattr(importlib.import_module(name), "EventSeries", None)
        except ImportError:
            continue
        if cls is not None and hasattr(cls, "event_synchronization"):
            return cls
    pytest.skip("this pyunicorn has no EventSeries.event_synchronization")


def _pad(binary, gap):
    """Sentinel events ``gap`` samples before and after the record.

    pyunicorn scores only the interior events of each train (the first and last
    set the gaps of their neighbours but are not counted, and the norm is
    ``(m_x - 2)(m_y - 2)``). With one far sentinel at each end, its interior
    events are ours and their tolerances are ours.
    """
    out = np.zeros(binary.size + 2 * gap, dtype=int)
    out[0] = out[-1] = 1
    out[gap:gap + binary.size] = binary
    return out


def test_es_agrees_with_pyunicorn():
    EventSeries = _pyunicorn_event_series()
    rng = np.random.default_rng(3)
    T, m = 300, 30
    for trial in range(20):
        # Random binary series with exactly m isolated events each, so that one
        # quantile turns the spike series below into exactly these events.
        trains = []
        for _ in range(2):
            starts = np.sort(rng.choice(np.arange(1, T // 2), m, replace=False)) * 2
            trains.append(starts)
        tx, ty = trains
        bx = np.zeros(T, dtype=int)
        bx[tx] = 1
        by = np.zeros(T, dtype=int)
        by[ty] = 1
        x = _series_with_events(T, tx, [1] * m, rng)
        y = _series_with_events(T, ty, [1] * m, rng)
        data = Data(np.vstack([x, y]), dim_order="ps")
        q = _quantile_for(m, T)
        for taumax, max_delay in ((np.inf, None), (3, 3)):
            es_xy, es_yx = EventSeries.event_synchronization(
                _pad(bx, 2 * T), _pad(by, 2 * T), taumax=taumax)
            # pyunicorn's es_XY counts x events *after* y events: c(x|y).
            Q = EventSynchronisation("Q", q, max_delay).bivariate(data, i=0, j=1)
            lead = EventSynchronisation("q", q, max_delay).bivariate(data, i=0, j=1)
            assert Q == pytest.approx(es_xy + es_yx, abs=1e-12)
            assert lead == pytest.approx(es_yx - es_xy, abs=1e-12)


# --------------------------------------------------------------------------
# Event synchronisation: direction on a delayed heavy-tailed pair
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def delayed_extremes():
    """x iid Student-t(3); y[t] = x[t-3] + 0.5 t(3) noise; z independent."""
    rng = np.random.default_rng(11)
    T, lag = 5000, 3
    x = rng.standard_t(3, T + lag)
    y = x[:-lag] + 0.5 * rng.standard_t(3, T)
    z = rng.standard_t(3, T)
    return Data(np.vstack([x[lag:], y, z]), dim_order="ps")


@pytest.mark.parametrize("quantile", [0.95, 0.9])
def test_es_detects_that_x_extremes_lead_y_extremes(delayed_extremes, quantile):
    Q = EventSynchronisation("Q", quantile).multivariate(delayed_extremes)
    q = EventSynchronisation("q", quantile).multivariate(delayed_extremes)
    assert q[0, 1] > 0.08
    assert q[1, 0] < -0.08
    # Q is not corrected for chance, so compare with the independent pairs.
    assert Q[0, 1] > 0.5
    assert Q[0, 1] > max(Q[0, 2], Q[1, 2]) + 0.1
    assert max(abs(q[0, 2]), abs(q[1, 2])) < 0.12


# --------------------------------------------------------------------------
# Tail-dependence asymmetry
# --------------------------------------------------------------------------

def _tail_reference(x, y, u):
    """Upper and lower empirical tail dependence by explicit counting."""
    T = len(x)
    k = int(np.floor((1 - u) * T + 1e-9))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    upper = sum(1 for a, b in zip(rx, ry) if a > T - k and b > T - k)
    lower = sum(1 for a, b in zip(rx, ry) if a <= k and b <= k)
    return upper / k, lower / k


def _clayton(T, theta, rng):
    """Clayton copula by conditional inversion: v = C^{-1}(w | u)."""
    u, w = rng.random(T), rng.random(T)
    v = ((w ** (-theta / (1 + theta)) - 1) * u ** (-theta) + 1) ** (-1 / theta)
    return u, v


@pytest.mark.parametrize("u", [0.95, 0.9, 0.8])
def test_tailasym_matches_the_reference(u):
    rng = np.random.default_rng(4)
    T = 237  # (1 - u) T is not an integer
    a, b = _clayton(T, 2.0, rng)
    Z = np.vstack([a, b, rng.standard_normal(T)])
    data = Data(Z, dim_order="ps")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        lam_u, lam_l = _tail_reference(Z[i], Z[j], u)
        for statistic, want in (("asymmetry", lam_u - lam_l),
                                ("upper", lam_u), ("lower", lam_l)):
            spi = TailDependenceAsymmetry(u, statistic)
            A = spi.multivariate(data)
            assert A[i, j] == pytest.approx(want, abs=1e-12)
            assert A[j, i] == pytest.approx(want, abs=1e-12)
            assert spi.bivariate(data, i=i, j=j) == pytest.approx(want, abs=1e-12)
            assert spi.bivariate(data, i=j, j=i) == pytest.approx(want, abs=1e-12)


def test_tailasym_sign_on_known_copulas():
    rng = np.random.default_rng(5)
    T = 5000
    u, v = _clayton(T, 3.0, rng)                      # lower-tail dependent
    a, c = rng.standard_normal((2, T))
    g = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], T).T
    Z = np.vstack([stats.norm.ppf(u), stats.norm.ppf(v),
                   a, np.maximum(a, c),               # upper-tail dependent
                   g[0], g[1]])                        # radially symmetric
    data = Data(Z, dim_order="ps")
    for quantile in (0.95, 0.9):
        A = TailDependenceAsymmetry(quantile).multivariate(data)
        assert A[0, 1] < -0.35
        assert A[2, 3] > 0.12
        assert abs(A[4, 5]) < 0.15
        # Spearman cannot tell the Clayton pair from the Gaussian one by sign.
        assert stats.spearmanr(Z[0], Z[1])[0] > 0 and stats.spearmanr(Z[4], Z[5])[0] > 0


def test_tailasym_reflection_negates_it_exactly():
    rng = np.random.default_rng(6)
    u, v = _clayton(1000, 2.0, rng)
    Z = np.vstack([u, v, rng.standard_normal(1000)])
    asym = TailDependenceAsymmetry(0.9)
    upper, lower = TailDependenceAsymmetry(0.9, "upper"), TailDependenceAsymmetry(0.9, "lower")
    fwd, rev = Data(Z, dim_order="ps"), Data(-Z, dim_order="ps")
    off = ~np.eye(3, dtype=bool)
    assert np.array_equal(asym.multivariate(rev)[off], -asym.multivariate(fwd)[off])
    assert np.array_equal(upper.multivariate(rev)[off], lower.multivariate(fwd)[off])


# --------------------------------------------------------------------------
# Structure, invariances and undefined pairs
# --------------------------------------------------------------------------

ALL_VARIANTS = [
    EventSynchronisation("Q"), EventSynchronisation("q"),
    EventSynchronisation("Q", 0.9, max_delay=2), EventSynchronisation("q", 0.9, max_delay=2),
    TailDependenceAsymmetry(0.95), TailDependenceAsymmetry(0.9),
    TailDependenceAsymmetry(0.9, "upper"), TailDependenceAsymmetry(0.9, "lower"),
]


@pytest.fixture(scope="module")
def heavy_data():
    rng = np.random.default_rng(7)
    Z = rng.standard_t(3, (5, 800))
    Z[1, 2:] += Z[0, :-2]
    Z[3] = np.maximum(Z[2], Z[3])
    return Z


@pytest.mark.parametrize("spi", ALL_VARIANTS, ids=lambda s: s.identifier)
def test_symmetry_and_antisymmetry(spi, heavy_data):
    A = spi.multivariate(Data(heavy_data, dim_order="ps"))
    off = ~np.eye(A.shape[0], dtype=bool)
    assert np.isfinite(A[off]).all()
    assert np.isnan(np.diag(A)).all()
    if "antisymmetric" in spi.labels:
        assert np.array_equal(A, -A.T, equal_nan=True)
        assert spi.issigned()
    else:
        assert "undirected" in spi.labels
        assert np.array_equal(A, A.T, equal_nan=True)
    if not spi.issigned():
        assert (A[off] >= 0).all() and (A[off] <= 1).all()
    else:
        assert (np.abs(A[off]) <= 1).all()


@pytest.mark.parametrize("spi", ALL_VARIANTS, ids=lambda s: s.identifier)
def test_invariant_to_increasing_transforms(spi, heavy_data):
    """Rank-based: exp() of one series and a cube of another change nothing."""
    Z = heavy_data
    W = Z.copy()
    W[0] = np.exp(Z[0] / 4)
    W[1] = Z[1] ** 3
    a = spi.multivariate(Data(Z, dim_order="ps", zscore=False))
    b = spi.multivariate(Data(W, dim_order="ps", zscore=False))
    assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("spi", ALL_VARIANTS, ids=lambda s: s.identifier)
def test_constant_series_gives_nan_for_its_pairs_only(spi, heavy_data):
    Z = heavy_data[:3].copy()
    Z[1] = 2.0
    A = spi.multivariate(Data(Z, dim_order="ps", zscore=False))
    assert np.isnan(A[1]).all() and np.isnan(A[:, 1]).all()
    assert np.isfinite(A[0, 2]) and np.isfinite(A[2, 0])
    assert np.isnan(spi.bivariate(Data(Z, dim_order="ps", zscore=False), i=0, j=1))


@pytest.mark.parametrize("statistic", ["Q", "q"])
def test_es_needs_two_events(statistic, heavy_data):
    Z = heavy_data[:3].copy()
    Z[1] = 0.0
    Z[1, 400] = 5.0  # the only exceedance of the 0.95 quantile: one event
    Z[2] = 0.0
    Z[2, 100:120] = 5.0  # one run of 20 exceedances: also one event
    data = Data(Z, dim_order="ps", zscore=False)
    spi = EventSynchronisation(statistic)
    A = spi.multivariate(data)
    assert np.isnan(A[0, 1]) and np.isnan(A[1, 2]) and np.isnan(A[0, 2])
    assert np.isnan(spi.bivariate(data, i=0, j=1))


def test_es_cache_is_keyed_on_quantile_only():
    """Alternating quantile and max_delay on one Data matches a fresh Data."""
    rng = np.random.default_rng(8)
    Z = rng.standard_t(3, (3, 400))
    makers = [lambda: EventSynchronisation("Q", 0.95),
              lambda: EventSynchronisation("q", 0.9),
              lambda: EventSynchronisation("Q", 0.95, max_delay=1),
              lambda: EventSynchronisation("Q", 0.9),
              lambda: EventSynchronisation("q", 0.95, max_delay=2)]
    shared = Data(Z, dim_order="ps")
    for _ in range(2):
        for make in makers:
            got = make().multivariate(shared)
            want = make().multivariate(Data(Z, dim_order="ps"))
            assert np.array_equal(got, want, equal_nan=True)
    assert set(shared.events) == {0.95, 0.9}
    assert EventSynchronisation("Q", 0.9)._cache_subkey == EventSynchronisation("q", 0.9, 3)._cache_subkey


def test_identifiers_and_parameter_validation():
    assert EventSynchronisation("Q").identifier == "es_Q_q-0-95"
    assert EventSynchronisation("q", 0.9).identifier == "es_q_q-0-9"
    assert EventSynchronisation("q", 0.9, max_delay=5).identifier == "es_q_q-0-9_maxdelay-5"
    assert TailDependenceAsymmetry().identifier == "tailasym_q-0-95"
    assert TailDependenceAsymmetry(0.9, "upper").identifier == "tailasym_upper_q-0-9"
    for bad in (dict(statistic="Qq"), dict(quantile=1.0), dict(quantile=0),
                dict(max_delay=0), dict(max_delay=2.5)):
        with pytest.raises((ValueError, TypeError)):
            EventSynchronisation(**bad)
    for bad in (dict(quantile=0.5), dict(quantile=1.2), dict(statistic="both")):
        with pytest.raises((ValueError, TypeError)):
            TailDependenceAsymmetry(**bad)


# --------------------------------------------------------------------------
# Calculator round trip and bundled configs
# --------------------------------------------------------------------------

BUNDLED = {"es_Q_q-0-95", "es_q_q-0-95", "es_Q_q-0-9", "es_q_q-0-9",
           "tailasym_q-0-95", "tailasym_q-0-9"}


def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.events": {
        "EventSynchronisation": {
            "labels": ["nonlinear", "bivariate", "time-dependent", "M14"],
            "configs": [{"statistic": "Q", "quantile": 0.95},
                        {"statistic": "q", "quantile": 0.95},
                        {"statistic": "q", "quantile": 0.9, "max_delay": 4}]},
        "TailDependenceAsymmetry": {
            "labels": ["nonlinear", "bivariate", "contemporaneous", "M14"],
            "configs": [{"quantile": 0.95}, {"quantile": 0.9, "statistic": "lower"}]},
    }}
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(cfg))
    rng = np.random.default_rng(9)
    Z = rng.standard_t(3, (4, 600))
    Z[1, 1:] += Z[0, :-1]
    calc = Calculator(dataset=Data(Z, dim_order="ps"), config=str(path))
    calc.compute()
    expected = {"es_Q_q-0-95", "es_q_q-0-95", "es_q_q-0-9_maxdelay-4",
                "tailasym_q-0-95", "tailasym_lower_q-0-9"}
    assert set(calc.spis) == expected
    assert calc.errors == {}
    for ident in expected:
        A = calc.table[ident].to_numpy(dtype=float)
        direct = calc.spis[ident].multivariate(Data(Z, dim_order="ps"))
        off = ~np.eye(4, dtype=bool)
        assert np.allclose(A[off], direct[off], equal_nan=True), ident
    # q survives the Calculator with its sign: it is not min-shifted.
    q = calc.table["es_q_q-0-95"].to_numpy(dtype=float)
    assert q[0, 1] > 0 > q[1, 0]


def test_bundled_configs_construct_the_new_spis():
    full = load_spis_from_yaml(resolve_config("full"), quiet=True)
    assert BUNDLED <= set(full)
    traits = {"directed", "undirected", "antisymmetric", "signed", "unsigned"}
    assert set(full["es_Q_q-0-95"].labels) & traits == {"undirected", "unsigned"}
    assert set(full["es_q_q-0-95"].labels) & traits == {"antisymmetric", "signed"}
    assert set(full["tailasym_q-0-95"].labels) & traits == {"undirected", "signed"}
    assert "time-dependent" in full["es_q_q-0-9"].labels
    assert "contemporaneous" in full["tailasym_q-0-9"].labels
    fast = load_spis_from_yaml(resolve_config("fast"), quiet=True)
    assert {"es_q_q-0-95", "tailasym_q-0-95"} <= set(fast)
    calc = Calculator(config="full")
    assert BUNDLED <= set(calc.spis)

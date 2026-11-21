"""NonlinearInterdependence (`nli_*`): S, H, N and L from delay embeddings.

Pinned against a brute-force O(N^2) reference written from the definitions in
Chicharro & Andrzejak (2009), eqs. (1)-(4), with the Theiler window applied to
every average over j as the paper's note 24 prescribes. Behaviour is checked
on identical, independent and unidirectionally coupled (Henon) processes, and
the Theiler window on slowly varying series.
"""
import warnings

import numpy as np
import pytest
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.causal import (
    NonlinearInterdependence, _nli_embed, _nli_neighbours,
)

STATS = ("S", "H", "N", "L")


# --------------------------------------------------------------------------
# Brute-force reference
# --------------------------------------------------------------------------

def _reference(x, y, m, tau, k, W):
    """S, H, N, L of X given Y from explicit distance matrices and argsort.

    Delay vectors x_n = (x_n, x_{n-tau}, ..., x_{n-(m-1)tau}); every average
    over j (neighbours, R_n, ranks) runs over |n - j| > W only.
    """
    T = len(x)
    N = T - (m - 1) * tau
    rows = np.arange(N) + (m - 1) * tau
    ex = np.column_stack([x[rows - l * tau] for l in range(m)])
    ey = np.column_stack([y[rows - l * tau] for l in range(m)])
    Dx = cdist(ex, ex, "sqeuclidean")
    Dy = cdist(ey, ey, "sqeuclidean")
    t = np.arange(N)
    out = dict.fromkeys(STATS, 0.0)
    for n in range(N):
        cand = t[np.abs(t - n) > W]
        r = cand[np.argsort(Dx[n, cand], kind="stable")[:k]]  # X's own k-NN
        s = cand[np.argsort(Dy[n, cand], kind="stable")[:k]]  # Y's k-NN
        Rk, Rk_cond, R = Dx[n, r].mean(), Dx[n, s].mean(), Dx[n, cand].mean()
        rank = dict(zip(cand, rankdata(Dx[n, cand])))  # 1 = nearest
        G_cond = np.mean([rank[j] for j in s])
        G, G_min = (len(cand) + 1) / 2, (k + 1) / 2
        out["S"] += Rk / Rk_cond / N
        out["H"] += np.log(R / Rk_cond) / N
        out["N"] += (R - Rk_cond) / R / N
        out["L"] += (G - G_cond) / (G - G_min) / N
    return out


def _fixture():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((3, 150))
    z[1, 1:] += 0.9 * np.sin(2 * z[0, :-1])  # a nonlinear 0 -> 1 coupling
    return Data(data=z, dim_order="ps")


@pytest.mark.parametrize("m, tau, k, theiler", [
    (5, 1, 5, None),   # the bundled setting, theiler = m * tau = 5
    (3, 2, 4, 0),      # only the point itself excluded: N_n = N - 1
    (2, 1, 3, 20),     # wide window: the ends keep more candidates
])
def test_matches_bruteforce_reference(m, tau, k, theiler):
    data = _fixture()
    z = data.to_numpy(squeeze=True)
    W = m * tau if theiler is None else theiler
    refs = {(i, j): _reference(z[i], z[j], m, tau, k, W)
            for i in range(3) for j in range(3) if i != j}
    for stat in STATS:
        spi = NonlinearInterdependence(stat, m=m, tau=tau, k=k, theiler=theiler)
        A = spi.multivariate(data)
        assert np.all(np.isnan(np.diag(A)))
        for (i, j), ref in refs.items():
            assert A[i, j] == pytest.approx(ref[stat], abs=1e-10), (stat, i, j)
            got = NonlinearInterdependence(
                stat, m=m, tau=tau, k=k, theiler=theiler).bivariate(
                    _fixture(), i=i, j=j)
            assert got == pytest.approx(ref[stat], abs=1e-10), (stat, i, j)


# --------------------------------------------------------------------------
# Identical and independent processes
# --------------------------------------------------------------------------

def test_identical_series_give_exactly_one_for_S_and_L():
    # Y = X: Y's neighbours are X's own, so R^k(X|Y) = R^k(X) and the mean
    # conditioned rank is the minimal (k+1)/2 -- exactly, whatever the
    # Theiler window, because the window applies to both searches alike.
    x = np.random.default_rng(3).standard_normal(500)
    data = Data(data=np.vstack([x, x]), dim_order="ps")
    for stat in ("S", "L"):
        A = NonlinearInterdependence(stat).multivariate(data)
        assert A[0, 1] == pytest.approx(1.0, abs=1e-12)
        assert A[1, 0] == pytest.approx(1.0, abs=1e-12)
    # N(X|X) = <1 - R^k(X)/R(X)>, high but below 1.
    assert 0.8 < NonlinearInterdependence("N").bivariate(data, i=0, j=1) < 1.0


def test_independent_series_are_near_their_null_values():
    # For independent processes S ~ (k/N)^(2/D) (about 0.12 here), N and L ~ 0.
    z = np.random.default_rng(100).standard_normal((4, 1000))
    data = Data(data=z, dim_order="ps")
    off = ~np.eye(4, dtype=bool)
    S = NonlinearInterdependence("S").multivariate(data)[off]
    N = NonlinearInterdependence("N").multivariate(data)[off]
    L = NonlinearInterdependence("L").multivariate(data)[off]
    assert np.all((S > 0.05) & (S < 0.15))
    assert np.max(np.abs(N)) < 0.06 and abs(N.mean()) < 0.02
    assert np.max(np.abs(L)) < 0.06 and abs(L.mean()) < 0.02


# --------------------------------------------------------------------------
# Unidirectionally coupled Henon maps (X drives Y)
# --------------------------------------------------------------------------

def _henon(C, T=2000, transient=1000, seed=0):
    """x[t+1] = 1.4 - x^2 + 0.3 x[t-1]; y[t+1] = 1.4 - (C x + (1-C) y) y + 0.3 y[t-1]."""
    rng = np.random.default_rng(seed)
    while True:  # redraw the (rare) initial conditions that escape
        x = np.zeros(T + transient)
        y = np.zeros(T + transient)
        x[:2], y[:2] = rng.uniform(0, 0.5, 2), rng.uniform(0, 0.5, 2)
        with np.errstate(all="ignore"):
            for t in range(1, T + transient - 1):
                x[t + 1] = 1.4 - x[t] ** 2 + 0.3 * x[t - 1]
                y[t + 1] = (1.4 - (C * x[t] + (1 - C) * y[t]) * y[t]
                            + 0.3 * y[t - 1])
        if np.all(np.isfinite(y)) and np.abs(y).max() < 10:
            return Data(data=np.vstack([x, y])[:, transient:], dim_order="ps")


def test_henon_coupling_is_detected_in_the_documented_direction():
    # A[i, j] = A(x_i | x_j): row 0 is the driver X, so A[0, 1] = A(X|Y) is
    # the driven-direction cell.
    L0 = NonlinearInterdependence("L").multivariate(_henon(0.0))
    assert abs(L0[0, 1]) < 0.05 and abs(L0[1, 0]) < 0.05
    for C in (0.2, 0.6):
        data = _henon(C)
        L = NonlinearInterdependence("L").multivariate(data)
        N = NonlinearInterdependence("N").multivariate(data)
        assert L[0, 1] > L0[0, 1] + 0.5, C
        assert L[0, 1] - L[1, 0] > 0.3, C
        assert N[0, 1] - N[1, 0] > 0.3, C
        if C == 0.2:
            # The dimension caveat of the docstring: S's asymmetry points the
            # wrong way here.
            S = NonlinearInterdependence("S").multivariate(data)
            assert S[0, 1] < S[1, 0]


# --------------------------------------------------------------------------
# Theiler window
# --------------------------------------------------------------------------

def _smooth(T, sigma, seed):
    e = np.random.default_rng(seed).standard_normal(T + 20 * sigma)
    return gaussian_filter1d(e, sigma)[10 * sigma:10 * sigma + T]


def test_theiler_window_excludes_temporal_neighbours():
    # Smoothed noise moves slowly, so without a window many nearest neighbours
    # of x_n are x_{n +- 1, 2, ...}: the same instants are then "neighbours" in
    # any slow process, and independent processes look interdependent.
    L_none, L_default = [], []
    for seed in range(3):
        z = np.vstack([_smooth(1000, 5, 10 + seed), _smooth(1000, 5, 20 + seed)])
        data = Data(data=z, dim_order="ps")

        emb = _nli_embed(data.to_numpy()[0, :, 0], 5, 1)
        lag = np.abs(_nli_neighbours(emb, 5, 0) - np.arange(len(emb))[:, None])
        assert np.mean(lag <= 5) > 0.1  # the test is not vacuous
        for W in (0, 5, 20):
            lag = np.abs(_nli_neighbours(emb, 5, W) - np.arange(len(emb))[:, None])
            assert lag.min() > W

        L_none.append(NonlinearInterdependence("L", theiler=0).bivariate(data, i=0, j=1))
        L_default.append(NonlinearInterdependence("L").bivariate(data, i=0, j=1))
    assert min(L_none) > 0.1
    assert max(L_default) < 0.05


# --------------------------------------------------------------------------
# Parameters, failures, caching
# --------------------------------------------------------------------------

def test_identifier_parameters_and_labels():
    spi = NonlinearInterdependence("L", m=3, tau=2, k=4)
    assert spi.identifier == "nli_L_m-3_tau-2_k-4"
    assert spi._theiler == 6
    assert NonlinearInterdependence("H", theiler=10).identifier == \
        "nli_H_m-5_tau-1_k-5_theiler-10"
    assert {"directed", "nonlinear", "unsigned"} <= set(spi.labels)
    assert not spi.issigned()
    # The four statistics share one cache; the parameters split it.
    assert (NonlinearInterdependence("S")._cache_subkey
            == NonlinearInterdependence("L", theiler=5)._cache_subkey)
    assert (NonlinearInterdependence("L")._cache_subkey
            != NonlinearInterdependence("L", k=4)._cache_subkey)
    with pytest.raises(ValueError):
        NonlinearInterdependence("M")
    with pytest.raises(ValueError):
        NonlinearInterdependence(k=0)
    with pytest.raises(TypeError):
        NonlinearInterdependence(m=2.5)
    with pytest.raises(ValueError):
        NonlinearInterdependence(theiler=-1)


def test_too_short_series_raise():
    # N = 30 - 4 = 26 delay vectors, k + 2W + 1 = 5 + 20 + 1 = 26: too few.
    data = Data(data=np.random.default_rng(0).standard_normal((2, 30)), dim_order="ps")
    with pytest.raises(ValueError, match="delay vectors"):
        NonlinearInterdependence("L", theiler=10).bivariate(data, i=0, j=1)
    assert np.isfinite(
        NonlinearInterdependence("L", theiler=9).bivariate(data, i=0, j=1))


def test_zero_conditioned_distance_makes_only_that_pair_undefined():
    # Processes 0 and 1 are the same period-10 sequence: each delay vector has
    # exact copies, Y's neighbours of y_n are among them, R^k(X|Y) = 0 and S
    # and H are undefined for that pair only. L (ranks, ties averaged) is not.
    rng = np.random.default_rng(4)
    periodic = np.tile(rng.standard_normal(10), 20)
    z = np.vstack([periodic, periodic, rng.standard_normal(200)])
    data = Data(data=z, dim_order="ps")
    for stat in ("S", "H"):
        with pytest.warns(UserWarning, match="2 of 6 pairs are undefined"):
            A = NonlinearInterdependence(stat).multivariate(data)
        assert np.isnan(A[0, 1]) and np.isnan(A[1, 0])
        assert np.all(np.isfinite(A[[0, 1, 2, 2], [2, 2, 0, 1]]))
        with pytest.raises(ValueError, match="zero mean squared distance"):
            NonlinearInterdependence(stat).bivariate(data, i=0, j=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L = NonlinearInterdependence("L").multivariate(data)
    assert np.all(np.isfinite(L[~np.eye(3, dtype=bool)]))


def test_shared_cache_matches_fresh_data():
    z = np.random.default_rng(5).standard_normal((3, 300))
    z[1, 1:] += 0.7 * z[0, :-1] ** 2

    def fresh():
        return Data(data=z, dim_order="ps")

    makers = [
        lambda: NonlinearInterdependence("L"),
        lambda: NonlinearInterdependence("S"),
        lambda: NonlinearInterdependence("L", k=3),
        lambda: NonlinearInterdependence("L", theiler=0),
        lambda: NonlinearInterdependence("N", m=3),
        lambda: NonlinearInterdependence("H", tau=2),
        lambda: NonlinearInterdependence("L", m=3, tau=2),
    ]
    shared = fresh()
    NonlinearInterdependence("L").bivariate(shared, i=1, j=2)  # partial cache
    for _ in range(2):
        for make in makers:
            got = make().multivariate(shared)
            want = make().multivariate(fresh())
            assert np.array_equal(got, want, equal_nan=True), make().identifier
    assert set(shared.nli) == {(5, 1, 5, 5), (5, 1, 3, 5), (5, 1, 5, 0),
                               (3, 1, 5, 3), (5, 2, 5, 10), (3, 2, 5, 6)}


# --------------------------------------------------------------------------
# Calculator round trip and bundled configs
# --------------------------------------------------------------------------

BUNDLED = {
    "nli_L_m-5_tau-1_k-5": ("L", 5),
    "nli_N_m-5_tau-1_k-5": ("N", 5),
    "nli_H_m-5_tau-1_k-5": ("H", 5),
    "nli_L_m-3_tau-1_k-5": ("L", 3),
}


def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.causal": {"NonlinearInterdependence": {
        "labels": ["directed", "nonlinear", "unsigned", "bivariate",
                   "time-dependent", "M14"],
        "dependencies": None,
        "configs": [{"statistic": s, "m": m, "tau": 1, "k": 5}
                    for s, m in BUNDLED.values()]}}}
    path = tmp_path / "nli.yaml"
    path.write_text(yaml.safe_dump(cfg))
    z = np.random.default_rng(10).standard_normal((3, 200))
    calc = Calculator(dataset=Data(data=z, dim_order="ps"), config=str(path))
    calc.compute()
    assert calc.errors == {}
    assert set(calc.spis) == set(BUNDLED)
    for ident, (stat, m) in BUNDLED.items():
        ref = NonlinearInterdependence(stat, m=m).multivariate(
            Data(data=z, dim_order="ps"))
        got = calc.table[ident].to_numpy()
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12, equal_nan=True)


def test_full_and_fast_configs_construct_nli_entries():
    spis = Calculator(config="full").spis
    for ident, (stat, m) in BUNDLED.items():
        assert ident in spis
        spi = spis[ident]
        assert isinstance(spi, NonlinearInterdependence)
        assert (spi._statistic, spi._m, spi._tau, spi._k, spi._theiler) == \
            (stat, m, 1, 5, m)
        assert {"directed", "time-dependent", "unsigned"} <= set(spi.labels)
    assert "nli_L_m-5_tau-1_k-5" in Calculator(config="fast").spis

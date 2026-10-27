"""CCM convergence statistics and lagged CCM.

``ConvergentCrossMapping(statistic="convergence" | "slope")`` reads CCM's
defining criterion off the cached skill curve: convergence is
``rho(L_max) - rho(L_min)`` and slope the least-squares slope of rho on
``ln L`` (Sugihara et al. 2012, Science 338:496).
``LaggedConvergentCrossMapping`` is the cross-map-lag extension of Ye et al.
(2015), Sci. Rep. 5:14750: skill at the largest library for prediction
horizons ``Tp`` in ``[-k, k]``, whose peak sits at a negative ``Tp`` for a
causal source -> target link.

Oracles: a hand computation from the cached curve (``np.polyfit`` for the
slope), pyEDM's own ``CCM(Tp=...)`` for the lagged skill, a standalone
``tp=0`` cross-map for the ``tp=0`` path, and coupled logistic maps whose
direction and delay are known by construction.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from pyspi.data import Data
from pyspi.statistics.causal import (
    ConvergentCrossMapping,
    LaggedConvergentCrossMapping,
    KDTree,
    _ccm_library_sizes,
    _cross_map_skill,
    _cross_map_skill_dense,
    _optimal_embedding_dimension,
    _pearson,
)

X, Y = 0, 1


def _logistic(T, bxy=0.02, byx=0.1, d=0, burn=300):
    """Sugihara et al.'s (2012) two-species coupled logistic maps, with delay d.

    x(t+1) = x(t) [3.8 - 3.8 x(t) - bxy y(t)]
    y(t+1) = y(t) [3.5 - 3.5 y(t) - byx x(t - d)]

    With the paper's bxy = 0.02, byx = 0.1, x drives y strongly and y drives
    x weakly; bxy = 0 makes the coupling strictly x -> y. x reaches y after
    d + 1 steps.
    """
    n = T + burn
    x, y = np.empty(n), np.empty(n)
    x[:d + 1], y[:d + 1] = 0.4, 0.2
    for t in range(d, n - 1):
        x[t + 1] = x[t] * (3.8 - 3.8 * x[t] - bxy * y[t])
        y[t + 1] = y[t] * (3.5 - 3.5 * y[t] - byx * x[t - d])
    return np.vstack([x[burn:], y[burn:]])


def _fixture(N=150, seed=4):
    """Three processes: the coupled pair plus independent noise."""
    rng = np.random.default_rng(seed)
    return np.vstack([_logistic(N), rng.standard_normal(N)])


def _off(M):
    return ~np.eye(M, dtype=bool)


def _pair_embeddings(data, E):
    """Each pair's E from first principles: the larger simplex-optimal E."""
    z = data.to_numpy(squeeze=True)
    M, N = z.shape
    if E is not None:
        return np.full((M, M), E)
    df = pd.DataFrame(np.column_stack([np.arange(N), z.T]),
                      columns=["index"] + [f"proc{p}" for p in range(M)])
    e = np.array([_optimal_embedding_dimension(df, f"proc{p}", f"10 {N - 10}")
                  for p in range(M)])
    return np.maximum.outer(e, e)


def _library_sizes(N, E):
    """``ceil(2E/10)*10`` to ``floor((N-E-1)/10)*10`` in 21 rounded steps."""
    lo, hi = int(np.ceil(2 * E / 10) * 10), int(np.floor((N - E - 1) / 10) * 10)
    return np.unique(np.rint(np.linspace(lo, hi, 22))).astype(int)


# ---------------------------------------------------------------------------
# convergence and slope: definition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("E", [2, None])
def test_convergence_and_slope_match_the_hand_computation(E):
    """Last finite skill minus the first, and ``np.polyfit`` on ln L, per cell.

    Both through ``multivariate`` (the Calculator's path) and through
    ``bivariate`` on a fresh Data, which computes the pair alone.
    """
    z = _fixture(N=110)
    N = z.shape[1]
    data = Data(z, dim_order="ps")
    conv = ConvergentCrossMapping("convergence", E).multivariate(data)
    slope = ConvergentCrossMapping("slope", E).multivariate(data)
    curves = data.ccm[E]
    pair_E = _pair_embeddings(data, E)

    for i, j in zip(*np.nonzero(_off(3))):
        c = curves[i, j]
        finite = np.isfinite(c)
        assert finite.sum() >= 2
        L = _library_sizes(N, pair_E[i, j])
        assert finite.sum() == len(L)
        expected_conv = c[finite][-1] - c[finite][0]
        expected_slope = np.polyfit(np.log(L), c[finite], 1)[0]
        assert conv[i, j] == pytest.approx(expected_conv, abs=1e-12), (i, j)
        assert slope[i, j] == pytest.approx(expected_slope, abs=1e-12), (i, j)
        for stat, table in (("convergence", conv), ("slope", slope)):
            # from the cached table of a Data that has one
            assert ConvergentCrossMapping(stat, E).bivariate(
                data, i=int(i), j=int(j)) == table[i, j]
    for i, j in ((0, 1), (1, 0), (2, 0)):
        for stat, table in (("convergence", conv), ("slope", slope)):
            # and from the pair alone
            bv = ConvergentCrossMapping(stat, E).bivariate(
                Data(z, dim_order="ps"), i=i, j=j)
            assert bv == pytest.approx(table[i, j], abs=1e-12), (stat, i, j)
    assert np.isnan(np.diag(conv)).all() and np.isnan(np.diag(slope)).all()


def test_existing_ccm_statistics_have_a_bivariate_that_agrees():
    """``bivariate`` gives the multivariate cell for mean/max/diff."""
    z = _fixture(N=100)
    shared = Data(z, dim_order="ps")
    for stat in ("mean", "max", "diff"):
        A = ConvergentCrossMapping(stat, 2).multivariate(shared)
        for i, j in ((0, 1), (1, 2)):
            bv = ConvergentCrossMapping(stat, 2).bivariate(
                Data(z, dim_order="ps"), i=i, j=j)
            assert bv == pytest.approx(A[i, j], abs=1e-12), (stat, i, j)


def test_convergence_ignores_interior_nan_and_needs_two_sizes():
    """The statistic helpers on hand-made curves: NaN gaps and degenerate cells."""
    from pyspi.statistics.causal import _ccm_convergence, _ccm_slope

    curves = np.full((2, 2, 5), np.nan)
    curves[0, 1] = [np.nan, 0.1, np.nan, 0.4, np.nan]
    curves[1, 0, 2] = 0.3                     # one finite size only
    log_L = np.broadcast_to(np.log([10., 20, 30, 40, 50]), curves.shape)
    conv, slope = _ccm_convergence(curves), _ccm_slope(curves, log_L)
    assert conv[0, 1] == pytest.approx(0.3)
    assert slope[0, 1] == pytest.approx(
        np.polyfit(np.log([20., 40]), [0.1, 0.4], 1)[0])
    assert np.isnan(conv[1, 0]) and np.isnan(slope[1, 0])
    assert np.isnan(conv[0, 0]) and np.isnan(slope[0, 0])


# ---------------------------------------------------------------------------
# convergence: the coupled logistic maps of Sugihara et al. (2012)
# ---------------------------------------------------------------------------

def test_convergence_on_sugiharas_coupled_logistic_maps():
    """Real links converge; an absent link does not.

    On the paper's two-species model (x -> y at 0.1, y -> x at 0.02) both
    links are causal and both directions converge (Sugihara et al. 2012,
    Fig. 3). Their ordering is not asserted: the strong direction saturates
    early, and ``max`` separates them instead. With the feedback removed
    (bxy = 0) the reverse direction does not converge.
    """
    def stats(z):
        data = Data(z, dim_order="ps")
        return {s: ConvergentCrossMapping(s, 2).multivariate(data)
                for s in ("convergence", "slope", "max")}

    paper = stats(_logistic(1000))
    for cell in ((X, Y), (Y, X)):
        assert paper["convergence"][cell] > 0.5, cell
        assert paper["slope"][cell] > 0.05, cell
    assert paper["max"][X, Y] > paper["max"][Y, X] + 0.2

    one_way = stats(_logistic(1000, bxy=0.0))
    assert one_way["convergence"][X, Y] > 0.5
    assert one_way["convergence"][Y, X] < 0.05
    assert one_way["slope"][X, Y] > 0.05 > one_way["slope"][Y, X]


# ---------------------------------------------------------------------------
# _cross_map_skill: tp
# ---------------------------------------------------------------------------

def _unshifted_cross_map_skill(library, targets, E, lib_sizes, sample=100, seed=42):
    """Reference ``tp=0`` cross-map: every row from ``E - 1`` on, target unshifted."""
    library = np.asarray(library, dtype=np.float64)
    rows = np.arange(E - 1, library.shape[0])
    embed = np.ascontiguousarray(
        np.column_stack([library[rows - lag] for lag in range(E)]))
    actuals = [np.asarray(y, dtype=np.float64)[rows] for y in targets]
    n_points = embed.shape[0]
    r_M = np.arange(n_points)[:, np.newaxis]

    skill = np.empty((len(targets), len(lib_sizes)))
    rhos = np.empty((len(targets), sample))
    for l, L in enumerate(lib_sizes):
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        L = min(int(L), n_points)
        k = min(E + 1, L - 1)
        if k < 1:
            skill[:, l] = np.nan
            continue
        k_query = min(k + 1, L)
        for s in range(sample):
            lib_idx = rng.choice(n_points, size=L, replace=False)
            lib_idx.sort()
            dist, local = KDTree(embed[lib_idx]).query(embed, k=k_query)
            if dist.ndim == 1:
                dist, local = dist[:, np.newaxis], local[:, np.newaxis]
            neighbours = lib_idx[local]
            valid = ~(neighbours == r_M)
            cs = np.cumsum(valid, axis=1)
            first_k = valid & (cs <= k)
            insufficient = cs[:, -1] < k
            if np.all(insufficient):
                rhos[:, s] = np.nan
                continue
            if np.any(insufficient):
                first_k[insufficient, :] = False
            _, cols = np.where(first_k)
            nn_cols = np.zeros((n_points, k), dtype=np.intp)
            sufficient = ~insufficient
            n_sufficient = sufficient.sum()
            if n_sufficient > 0:
                nn_cols[sufficient] = cols.reshape(n_sufficient, k)
            nn_dist = dist[r_M, nn_cols]
            nn = neighbours[r_M, nn_cols]
            d_min = nn_dist[:, 0:1]
            weights = np.exp(-nn_dist / np.where(d_min > 0.0, d_min, 1.0))
            zero_mask = (d_min == 0.0)
            if np.any(zero_mask):
                weights = np.where(
                    zero_mask, np.where(nn_dist == 0.0, 1.0, 0.0), weights)
            w_sum = weights.sum(axis=1, keepdims=True)
            weights /= np.where(w_sum > 0.0, w_sum, 1.0)
            for t, actual in enumerate(actuals):
                predictions = np.sum(weights * actual[nn], axis=1)
                if np.any(insufficient):
                    predictions[insufficient] = np.nan
                rhos[t, s] = _pearson(predictions, actual)
        for t in range(len(targets)):
            skill[t, l] = np.nanmean(rhos[t])
    return skill


@pytest.mark.parametrize("E", [1, 3, 6])
@pytest.mark.parametrize("N", [40, 133])
def test_cross_map_skill_at_tp0_matches_unshifted_reference(E, N):
    """At ``tp=0`` (default or explicit) the skill equals the unshifted
    reference cross-map bit for bit, with and without tied distances."""
    rng = np.random.default_rng(E * 1000 + N)
    z = rng.standard_normal((3, N))
    z[1] += 0.5 * np.roll(z[0], 1)
    z[2] = np.round(z[2] * 2) / 2            # quantised: tied distances too
    lib_sizes = _ccm_library_sizes(N, E)
    ref = _unshifted_cross_map_skill(z[0], [z[1], z[2]], E, lib_sizes,
                                     sample=12, seed=7)
    for kwargs in ({}, {"tp": 0}, {"tp": np.int64(0)}):
        got = _cross_map_skill(z[0], [z[1], z[2]], E, lib_sizes,
                               sample=12, seed=7, **kwargs)
        assert np.array_equal(ref, got, equal_nan=True)
    quantised = _unshifted_cross_map_skill(z[2], [z[0]], E, lib_sizes, sample=12)
    assert np.array_equal(
        quantised, _cross_map_skill(z[2], [z[0]], E, lib_sizes, sample=12),
        equal_nan=True)


@pytest.mark.parametrize("E", [1, 3])
def test_cross_map_skill_tp_is_pyedms_tp(E):
    """``tp`` shifts the target as pyEDM's ``Tp`` does, bit for bit, both ways.

    Includes a horizon further back than the embedding reaches (-5 < -(E-1)),
    where rows at the start are dropped, and forward horizons, where rows at
    the end are.
    """
    import pyEDM

    rng = np.random.default_rng(2)
    N = 90
    z = np.vstack([_logistic(N)[0], rng.standard_normal(N)])
    df = pd.DataFrame(np.column_stack([np.arange(N), z.T]),
                      columns=["index", "a", "b"])
    sizes = [_ccm_library_sizes(N, E)[0], _ccm_library_sizes(N, E)[-1]]
    for tp in (-5, -1, 2, 4):
        ref = pyEDM.CCM(dataFrame=df, E=E, Tp=tp, columns="a", target="b",
                        libSizes=sizes, sample=10, seed=42, parallel=False)
        ab = _cross_map_skill(z[0], [z[1]], E, sizes, sample=10, tp=tp)[0]
        ba = _cross_map_skill(z[1], [z[0]], E, sizes, sample=10, tp=tp)[0]
        assert np.array_equal(ab, ref["a:b"].to_numpy()), tp
        assert np.array_equal(ba, ref["b:a"].to_numpy()), tp


def test_cross_map_skill_rejects_a_non_integer_horizon():
    z = np.random.default_rng(0).standard_normal((2, 60))
    for bad in (1.5, True, "1"):
        with pytest.raises(TypeError):
            _cross_map_skill(z[0], [z[1]], 2, [40], sample=2, tp=bad)


@pytest.mark.parametrize("E", [1, 2, 5, 10])
@pytest.mark.parametrize("quantised", [False, True])
def test_dense_top_library_equals_the_per_library_tree(E, quantised):
    """One whole-manifold query reproduces a KD-tree per random library.

    Tied distances (quantised data) fall back to the per-library tree, so
    the equality is exact in both cases.
    """
    rng = np.random.default_rng(E)
    N = 160
    z = rng.standard_normal((3, N))
    z[1] = np.sin(0.3 * np.arange(N)) + 0.2 * z[1]
    if quantised:
        z = np.round(z * 2) / 2
    L = _ccm_library_sizes(N, E)[-1]
    for tp in (-4, 0, 3):
        plain = _cross_map_skill(z[0], [z[1], z[2]], E, [L], sample=15, tp=tp)[:, 0]
        dense = _cross_map_skill_dense(z[0], [z[1], z[2]], E, L, sample=15, tp=tp)
        assert np.array_equal(plain, dense, equal_nan=True), tp


# ---------------------------------------------------------------------------
# Lagged CCM
# ---------------------------------------------------------------------------

def test_lagged_ccm_matches_pyedm_tp_at_the_top_library():
    """Every cell and horizon against ``pyEDM.CCM(Tp=...)``, then the statistics.

    ``[i, j, k + Tp]`` is "j xmap i" at Tp, so pyEDM's column ``j:i``. Checked
    through ``multivariate`` and through ``bivariate`` on a fresh Data.
    """
    import pyEDM

    z = _fixture(N=80, seed=1)
    M, N = z.shape
    E, k = 2, 3
    data = Data(z, dim_order="ps")
    zz = data.to_numpy(squeeze=True)
    df = pd.DataFrame(np.column_stack([np.arange(N), zz.T]),
                      columns=["index"] + [f"p{p}" for p in range(M)])
    top = [_library_sizes(N, E)[-1]]
    ref = np.full((M, M, 2 * k + 1), np.nan)
    for i in range(M):
        for j in range(i + 1, M):
            for l, tp in enumerate(range(-k, k + 1)):
                out = pyEDM.CCM(dataFrame=df, E=E, Tp=tp, columns=f"p{j}",
                                target=f"p{i}", libSizes=top, sample=100,
                                seed=42, parallel=False)
                ref[i, j, l] = out[f"p{j}:p{i}"].iloc[0]
                ref[j, i, l] = out[f"p{i}:p{j}"].iloc[0]

    tp = LaggedConvergentCrossMapping("tp", E, max_lag=k).multivariate(data)
    skill = LaggedConvergentCrossMapping("skill", E, max_lag=k).multivariate(data)
    assert np.array_equal(data.ccm[("lagged", E, k)], ref, equal_nan=True)
    off = _off(M)
    np.testing.assert_array_equal(tp[off], np.argmax(ref, axis=2)[off] - k)
    np.testing.assert_allclose(skill[off], np.max(ref, axis=2)[off], atol=1e-12)
    assert np.isnan(np.diag(tp)).all() and np.isnan(np.diag(skill)).all()
    for i, j in zip(*np.nonzero(off)):
        for stat, table in (("tp", tp), ("skill", skill)):
            bv = LaggedConvergentCrossMapping(stat, E, max_lag=k).bivariate(
                Data(z, dim_order="ps"), i=int(i), j=int(j))
            assert bv == pytest.approx(table[i, j], abs=1e-12), (stat, i, j)


@pytest.fixture(scope="module")
def delayed():
    """x drives y through x(t - 2): y(t) has absorbed x only up to t - 3."""
    return Data(_logistic(1000, d=2), dim_order="ps")


def test_lagged_ccm_peaks_at_a_negative_horizon_for_the_true_direction(delayed):
    """Ye et al. (2015): the target's manifold best recovers the source's past.

    With ``y(t+1) = f(y(t), x(t - 2))``, ``y(t)`` carries x up to ``x(t - 3)``,
    so "y xmap x", the ``[x, y]`` cell, peaks at ``Tp = -3`` with E = 2; a
    larger E reaches further back, so with the simplex-optimal embedding only
    the sign and ``<= -1`` are asserted. Without the delay (a lag-1 map) the
    peak is at -1.
    """
    tp_auto = LaggedConvergentCrossMapping("tp").multivariate(delayed)
    assert tp_auto[X, Y] <= -1
    tp2 = LaggedConvergentCrossMapping("tp", 2).multivariate(delayed)
    assert tp2[X, Y] == -3
    undelayed = Data(_logistic(1000, d=0), dim_order="ps")
    assert LaggedConvergentCrossMapping("tp", 2).multivariate(undelayed)[X, Y] == -1
    # The true direction cross-maps far better than the reverse at its peak.
    skill = LaggedConvergentCrossMapping("skill", 2).multivariate(delayed)
    assert skill[X, Y] > 0.9 and skill[X, Y] > skill[Y, X] + 0.2


@pytest.mark.parametrize("E", [2, None])
def test_lagged_tp0_slice_is_the_ccm_top_library_skill(delayed, E):
    """Same seeds and sample count, so ``Tp = 0`` is the last point of the curve."""
    lccm = LaggedConvergentCrossMapping("skill", E).multivariate(delayed)
    ConvergentCrossMapping("max", E).multivariate(delayed)
    curves = delayed.ccm[E]
    lagged = delayed.ccm[("lagged", E, 5)]
    for i, j in ((X, Y), (Y, X)):
        c = curves[i, j]
        top = c[np.isfinite(c)][-1]
        assert lagged[i, j, 5] == pytest.approx(top, abs=1e-12)
        assert lccm[i, j] >= top - 1e-12


def test_lagged_cache_is_keyed_on_embedding_and_max_lag():
    """Alternating (E, k) on one Data reproduces a fresh Data every time,
    and the lagged entries leave the CCM curve cache alone."""
    z = _fixture(N=90, seed=3)

    def fresh(make):
        return make().multivariate(Data(z, dim_order="ps"))

    makers = [lambda: LaggedConvergentCrossMapping("skill", 2, max_lag=2),
              lambda: LaggedConvergentCrossMapping("skill", 2, max_lag=4),
              lambda: LaggedConvergentCrossMapping("skill", 3, max_lag=2),
              lambda: ConvergentCrossMapping("max", 2)]
    truth = [fresh(m) for m in makers]
    shared = Data(z, dim_order="ps")
    for _ in range(2):
        for make, want in zip(makers, truth):
            assert np.array_equal(make().multivariate(shared), want, equal_nan=True)
    assert {2, ("lagged", 2, 2), ("lagged", 2, 4), ("lagged", 3, 2)} <= set(shared.ccm)

    from pyspi._parallel import cache_bucket
    a, b = makers[0](), makers[1]()
    assert cache_bucket(a) != cache_bucket(b) != cache_bucket(makers[3]())
    assert cache_bucket(a) == cache_bucket(LaggedConvergentCrossMapping("tp", 2, max_lag=2))


def test_lagged_parameters_are_validated_and_reach_the_identifier():
    assert LaggedConvergentCrossMapping().identifier == "lccm_E-None_k-5_tp"
    assert (LaggedConvergentCrossMapping("skill", np.int64(3), max_lag=2).identifier
            == "lccm_E-3_k-2_skill")
    with pytest.raises(ValueError):
        LaggedConvergentCrossMapping("mean")
    with pytest.raises(ValueError):
        LaggedConvergentCrossMapping(max_lag=0)
    with pytest.raises(TypeError):
        LaggedConvergentCrossMapping(max_lag=2.0)
    with pytest.raises(ValueError):
        LaggedConvergentCrossMapping(embedding_dimension=0)


# ---------------------------------------------------------------------------
# Calculator and bundled config
# ---------------------------------------------------------------------------

def test_calculator_round_trip():
    from pyspi.calculator import Calculator

    z = _fixture(N=100, seed=6)
    cfg = {".statistics.causal": {
        "ConvergentCrossMapping": {
            "labels": ["directed", "nonlinear", "signed", "bivariate",
                       "time-dependent"],
            "configs": [{"statistic": "convergence", "embedding_dimension": 2},
                        {"statistic": "slope", "embedding_dimension": 2}]},
        "LaggedConvergentCrossMapping": {
            "labels": ["directed", "nonlinear", "signed", "bivariate",
                       "time-dependent"],
            "configs": [{"statistic": "tp", "embedding_dimension": 2, "max_lag": 3},
                        {"statistic": "skill", "embedding_dimension": 2, "max_lag": 3}]},
    }}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        calc = Calculator(dataset=Data(z, dim_order="ps"), config=f.name)
        calc.compute()
    finally:
        os.unlink(f.name)
    expected = {"ccm_E-2_convergence", "ccm_E-2_slope",
                "lccm_E-2_k-3_tp", "lccm_E-2_k-3_skill"}
    assert set(calc.spis) == expected
    assert calc.errors == {}
    off = _off(3)
    direct = {
        "ccm_E-2_convergence": ConvergentCrossMapping("convergence", 2),
        "ccm_E-2_slope": ConvergentCrossMapping("slope", 2),
        "lccm_E-2_k-3_tp": LaggedConvergentCrossMapping("tp", 2, max_lag=3),
        "lccm_E-2_k-3_skill": LaggedConvergentCrossMapping("skill", 2, max_lag=3),
    }
    for ident, spi in direct.items():
        got = calc.table[ident].to_numpy(dtype=float)
        want = spi.multivariate(Data(z, dim_order="ps"))
        assert np.isfinite(got[off]).all(), ident
        np.testing.assert_allclose(got[off], want[off], rtol=0, atol=1e-12)
    tp = calc.table["lccm_E-2_k-3_tp"].to_numpy(dtype=float)[off]
    assert np.all(tp == np.round(tp)) and np.all(np.abs(tp) <= 3)


def test_full_config_bundles_the_new_identifiers():
    from pyspi.calculator import Calculator

    calc = Calculator(config="full")
    new = {"ccm_E-None_convergence", "ccm_E-1_convergence",
           "ccm_E-10_convergence", "ccm_E-None_slope",
           "lccm_E-None_k-5_tp", "lccm_E-None_k-5_skill"}
    assert new <= set(calc.spis)
    for ident in new:
        spi = calc.spis[ident]
        assert spi.issigned()
        assert "directed" in spi.labels and "undirected" not in spi.labels
    assert isinstance(calc.spis["lccm_E-None_k-5_tp"], LaggedConvergentCrossMapping)

"""Liang-Kleeman information flow (`liang_*`, `liang-cond_*`).

Checked against the published estimators written out from scratch (Liang 2014
via Liang 2015 eq. 3 and Liang 2021 eq. 5; the multivariate form with explicit
cofactors, Liang 2021 eq. 14; the normaliser, Liang 2015 eqs. 33-35 and Liang
2021 eqs. 18-20), against Liang's (2014) own example with a known answer, and
on systems whose causal structure is known (a chain, independent series).
Equation numbers are those of the arXiv versions (1501.03548, 2104.11360).
"""
import warnings

import numpy as np
import pytest
import yaml
from scipy.signal import lfilter

from pyspi.calculator import Calculator, load_spis_from_yaml, resolve_config
from pyspi.data import Data
from pyspi.statistics.causal import LiangInformationFlow

VARIANTS = [
    dict(conditional=False, statistic="rate"),
    dict(conditional=False, statistic="normalised"),
    dict(conditional=True, statistic="rate"),
    dict(conditional=True, statistic="normalised"),
]
IDENTIFIERS = ["liang_rate_k-1", "liang_norm_k-1",
               "liang-cond_rate_k-1", "liang-cond_norm_k-1"]


def _offdiag(a):
    return a[~np.eye(a.shape[0], dtype=bool)]


def _fixture(M=4, T=400, seed=0):
    """Correlated, autocorrelated series (a random stable VAR(1))."""
    rng = np.random.default_rng(seed)
    A = 0.6 * np.eye(M) + 0.25 * rng.standard_normal((M, M))
    A *= 0.9 / max(abs(np.linalg.eigvals(A)))
    x = np.zeros((M, T + 100))
    e = rng.standard_normal((M, T + 100))
    for t in range(1, T + 100):
        x[:, t] = A @ x[:, t - 1] + e[:, t]
    return x[:, 100:]


def _euler(z, k=1):
    """Liang's Euler forward difference and the series cut to match it."""
    return z[:, :-k], (z[:, k:] - z[:, :-k]) / k


# --------------------------------------------------------------------------
# 1. Independent references
# --------------------------------------------------------------------------

def _bivariate_reference(z, src, tgt, k=1):
    """Liang 2015 eqs. 33-35 for source `src` -> target `tgt`, with np.cov."""
    x, dx = _euler(z, k)
    # Rows: x1 (target), x2 (source), dx1. np.cov's ddof=1, unlike the SPI's
    # code path, which never calls np.cov.
    c = np.cov(np.vstack([x[tgt], x[src], dx[tgt]]))
    C11, C22, C12 = c[0, 0], c[1, 1], c[0, 1]
    C1d1, C2d1, Cd1d1 = c[0, 2], c[1, 2], c[2, 2]
    T21 = (C11 * C12 * C2d1 - C12 ** 2 * C1d1) / (C11 ** 2 * C22 - C11 * C12 ** 2)
    p = (C22 * C1d1 - C12 * C2d1) / (C11 * C22 - C12 ** 2)
    q = (-C12 * C1d1 + C11 * C2d1) / (C11 * C22 - C12 ** 2)
    noise = k / (2 * C11) * (Cd1d1 + p ** 2 * C11 + q ** 2 * C22
                             - 2 * p * C1d1 - 2 * q * C2d1 + 2 * p * q * C12)
    tau = T21 / (abs(T21) + abs(p) + abs(noise))
    return T21, tau


def _conditional_reference(z, k=1):
    """Liang 2021 eqs. 9-10, 14-15, 18-20 with explicit cofactors.

    The covariances use 1/N (``bias=True``) and the noise variance the
    residual sum of squares of an explicit least-squares fit with intercept,
    divided by N, as in eq. 10 -- the SPI uses 1/(N - 1) throughout, and the
    two must agree because the normalisation cancels.
    """
    x, dx = _euler(z, k)
    M, N = x.shape
    full = np.cov(np.vstack([x, dx]), bias=True)
    C = full[:M, :M]
    Cd = full[:M, M:]                      # Cd[j, i] = C_{j,di}
    detC = np.linalg.det(C)
    cof = np.empty((M, M))
    for a in range(M):
        for b in range(M):
            minor = np.delete(np.delete(C, a, axis=0), b, axis=1)
            cof[a, b] = (-1) ** (a + b) * np.linalg.det(minor)
    T = np.full((M, M), np.nan)
    tau = np.full((M, M), np.nan)
    design = np.column_stack([np.ones(N), x.T])
    for i in range(M):                     # target
        a_i = np.array([sum(cof[j, kk] * Cd[kk, i] for kk in range(M)) / detC
                        for j in range(M)])                       # eq. 9
        resid = dx[i] - design @ np.linalg.lstsq(design, dx[i], rcond=None)[0]
        g_ii = (resid @ resid) * k / N                            # eq. 10
        noise = g_ii / (2 * C[i, i])
        flows = {j: a_i[j] * C[i, j] / C[i, i] for j in range(M) if j != i}  # eq. 14
        Z = abs(a_i[i]) + sum(abs(v) for v in flows.values()) + abs(noise)  # eqs. 15, 20
        for j, v in flows.items():
            T[j, i] = v
            tau[j, i] = v / Z
    return T, tau


@pytest.mark.parametrize("k", [1, 2])
def test_bivariate_estimator_matches_liang_2015_written_out(k):
    data = Data(_fixture(M=4, seed=1), dim_order="ps")
    z = data.to_numpy(squeeze=True)
    rate = LiangInformationFlow(k=k, statistic="rate")
    norm = LiangInformationFlow(k=k, statistic="normalised")
    A_rate, A_norm = rate.multivariate(data), norm.multivariate(data)
    assert np.isnan(np.diag(A_rate)).all() and np.isnan(np.diag(A_norm)).all()
    for src in range(4):
        for tgt in range(4):
            if src == tgt:
                continue
            want_T, want_tau = _bivariate_reference(z, src, tgt, k)
            for spi, A, want in ((rate, A_rate, want_T), (norm, A_norm, want_tau)):
                assert A[src, tgt] == pytest.approx(want, rel=1e-12, abs=1e-14)
                got = spi.bivariate(data, i=src, j=tgt)
                assert got == pytest.approx(want, rel=1e-12, abs=1e-14)


def test_conditional_estimator_matches_liang_2021_cofactor_form():
    data = Data(_fixture(M=4, seed=2), dim_order="ps")
    z = data.to_numpy(squeeze=True)
    want_T, want_tau = _conditional_reference(z)
    for statistic, want in (("rate", want_T), ("normalised", want_tau)):
        spi = LiangInformationFlow(conditional=True, statistic=statistic)
        A = spi.multivariate(data)
        np.testing.assert_allclose(_offdiag(A), _offdiag(want), rtol=1e-10, atol=1e-14)
        for i, j in [(0, 1), (3, 0), (2, 3)]:
            assert spi.bivariate(data, i=i, j=j) == pytest.approx(want[i, j], rel=1e-10)


@pytest.mark.parametrize("statistic", ["rate", "normalised"])
def test_conditional_estimator_with_two_series_is_the_bivariate_one(statistic):
    """Liang 2021: "When d = 2 ... this is reduced to (5)"; so is the normaliser."""
    data = Data(_fixture(M=2, T=600, seed=3), dim_order="ps")
    biv = LiangInformationFlow(conditional=False, statistic=statistic).multivariate(data)
    cond = LiangInformationFlow(conditional=True, statistic=statistic).multivariate(data)
    assert np.all(np.abs(_offdiag(biv)) > 1e-4)  # not vacuous
    np.testing.assert_allclose(_offdiag(cond), _offdiag(biv), rtol=0, atol=1e-12)


@pytest.mark.parametrize("params", VARIANTS)
def test_invariant_to_the_scale_and_offset_of_each_series(params):
    """Every term is a ratio of covariance products: z-scoring changes nothing."""
    raw = _fixture(M=3, seed=4) * np.array([[0.01], [7.0], [300.0]]) + 5.0
    spi = LiangInformationFlow(**params)
    a = spi.multivariate(Data(raw, dim_order="ps", zscore=False))
    b = spi.multivariate(Data(raw, dim_order="ps"))
    np.testing.assert_allclose(_offdiag(a), _offdiag(b), rtol=1e-9)


# --------------------------------------------------------------------------
# 2. Liang's (2014) example: x2 drives x1, x1 does not drive x2
# --------------------------------------------------------------------------

def _liang_2014_path(seed, n_steps, dt=1e-3, b=0.1, spin=5000):
    """dx1 = (-x1 + 0.5 x2) dt + b dW1, dx2 = -x2 dt + b dW2, Euler-Maruyama.

    Each component is a first-order recursion, so `lfilter` integrates it:
    x2[n+1] = (1 - dt) x2[n] + b sqrt(dt) e2[n], and x1 likewise with the
    drive 0.5 dt x2[n]. The stationary flow is T_{2->1} = a12 s12 / s11 =
    0.5 * 0.00125 / 0.005625 = 0.1111 nats per unit time and T_{1->2} = 0
    (Liang 2014, Fig. 1).
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((2, n_steps + spin)) * b * np.sqrt(dt)
    x2 = lfilter([0.0, 1.0], [1.0, -(1.0 - dt)], e[1])
    x1 = lfilter([0.0, 1.0], [1.0, -(1.0 - dt)], e[0] + 0.5 * dt * x2)
    return np.vstack([x1, x2])[:, spin:]


LIANG_T21 = 0.5 * 0.00125 / 0.005625  # 0.1111 nats per unit time


def test_liang_2014_example_recovers_the_one_way_flow():
    """20000 samples, one every h = 0.1 time units (100 Euler steps of 1e-3).

    The rate is per *sample*: about h times the per-unit-time flow, i.e.
    0.0111 here, less a small Euler-difference bias.
    """
    h_steps, h = 100, 0.1
    z = _liang_2014_path(seed=7, n_steps=20000 * h_steps)[:, ::h_steps][:, :20000]
    data = Data(z, dim_order="ps")
    A = LiangInformationFlow().multivariate(data)       # [source, target]
    T21, T12 = A[1, 0], A[0, 1]
    assert T21 > 0, "x2 drives x1: the paper's T_{2->1} = 0.11 > 0"
    assert 0.005 < T21 < 0.016
    assert 0.05 < T21 / h < 0.16                       # per unit time, ~0.1111
    assert abs(T12) < 0.003, "nil causality: x1 does not drive x2"
    assert abs(T12) < 0.25 * T21

    tau = LiangInformationFlow(statistic="normalised").multivariate(data)
    assert tau[1, 0] > 0 and abs(tau[0, 1]) < 0.25 * tau[1, 0]


def test_liang_2014_rate_is_per_sample_and_scales_with_the_sampling_interval():
    """One path of 2000 time units sampled at h = 0.1 and at h = 0.05.

    The per-sample rate is (to first order) h times the per-unit-time flow,
    so halving h about halves it (less the Euler-difference bias).
    """
    path = _liang_2014_path(seed=11, n_steps=2_000_000)
    coarse = LiangInformationFlow().multivariate(Data(path[:, ::100], dim_order="ps"))
    fine = LiangInformationFlow().multivariate(Data(path[:, ::50], dim_order="ps"))
    assert 1.7 < coarse[1, 0] / fine[1, 0] < 2.1
    assert 0.05 < coarse[1, 0] / 0.1 < 0.16
    assert 0.05 < fine[1, 0] / 0.05 < 0.16


# --------------------------------------------------------------------------
# 3. A chain: conditioning removes the indirect flow
# --------------------------------------------------------------------------

def _chain(seed, T=5000, burn=200):
    """x0 -> x1 -> x2, discrete VAR(1); x0 autocorrelated so x0[n] carries
    information about x1[n] and the pairwise 0 -> 2 fit sees the relay."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((3, T + burn))
    x = np.zeros((3, T + burn))
    for t in range(1, T + burn):
        x[0, t] = 0.7 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.5 * x[1, t - 1] + 0.5 * x[0, t - 1] + e[1, t]
        x[2, t] = 0.5 * x[2, t - 1] + 0.5 * x[1, t - 1] + e[2, t]
    return x[:, burn:]


@pytest.mark.parametrize("statistic", ["rate", "normalised"])
def test_conditional_flow_removes_the_indirect_link_of_a_chain(statistic):
    data = Data(_chain(seed=5), dim_order="ps")
    cond = LiangInformationFlow(conditional=True, statistic=statistic).multivariate(data)
    biv = LiangInformationFlow(conditional=False, statistic=statistic).multivariate(data)
    for A in (cond, biv):
        assert A[0, 1] > 0.1 and A[1, 2] > 0.1        # the direct links
    weakest_cond = min(cond[0, 1], cond[1, 2])
    weakest_biv = min(biv[0, 1], biv[1, 2])
    assert abs(cond[0, 2]) < 0.05 * weakest_cond
    assert biv[0, 2] > 0.1 * weakest_biv
    assert biv[0, 2] > 5 * abs(cond[0, 2])


# --------------------------------------------------------------------------
# 4. Nil causality on independent series
# --------------------------------------------------------------------------

@pytest.mark.parametrize("params", VARIANTS)
def test_independent_series_give_no_flow(params):
    """Five independent AR(1) (phi = 0.8), T = 2000."""
    rng = np.random.default_rng(1234)
    z = lfilter([1.0], [1.0, -0.8], rng.standard_normal((5, 2200)), axis=1)[:, 200:]
    A = LiangInformationFlow(**params).multivariate(Data(z, dim_order="ps"))
    bound = 0.015 if params["statistic"] == "rate" else 0.05
    assert np.nanmax(np.abs(_offdiag(A))) < bound


# --------------------------------------------------------------------------
# 5. The normalised flow
# --------------------------------------------------------------------------

@pytest.mark.parametrize("conditional", [False, True])
def test_normalised_flow_is_bounded_and_keeps_the_sign_of_the_rate(conditional):
    data = Data(_fixture(M=6, T=1000, seed=8), dim_order="ps")
    rate = LiangInformationFlow(conditional=conditional).multivariate(data)
    tau = LiangInformationFlow(conditional=conditional,
                               statistic="normalised").multivariate(data)
    r, t = _offdiag(rate), _offdiag(tau)
    assert np.all(np.isfinite(t))
    assert np.all(np.abs(t) <= 1)
    assert np.all(np.abs(t) < 1)  # Z also holds the self and noise terms
    assert np.array_equal(np.sign(t), np.sign(r))
    assert (r > 0).any() and (r < 0).any()  # both signs are exercised
    if conditional:
        # One normaliser per target (column), Liang 2021 eq. 18; pairwise,
        # each pair's own 2-series fit gives its own Z.
        Z = rate / tau
        for i in range(Z.shape[1]):
            col = np.delete(Z[:, i], i)
            assert np.ptp(col) < 1e-10 * np.abs(col).mean()


# --------------------------------------------------------------------------
# Construction, identifiers, caching, degenerate input
# --------------------------------------------------------------------------

def test_identifiers_are_unique_and_parameterised():
    ids = [LiangInformationFlow(**p).identifier for p in VARIANTS]
    assert ids == IDENTIFIERS
    assert LiangInformationFlow(k=2, conditional=True).identifier == "liang-cond_rate_k-2"


@pytest.mark.parametrize("kwargs, err", [
    (dict(k=0), ValueError),
    (dict(k=1.5), TypeError),
    (dict(k=True), TypeError),
    (dict(conditional="yes"), TypeError),
    (dict(statistic="normalized"), ValueError),
])
def test_bad_parameters_are_rejected(kwargs, err):
    with pytest.raises(err):
        LiangInformationFlow(**kwargs)


def test_variants_share_one_cache_keyed_by_k():
    """Alternating k on one Data must reproduce a fresh Data at every step."""
    z = _fixture(M=3, seed=9)

    def fresh():
        return Data(z, dim_order="ps")

    shared = fresh()
    for k in (1, 2, 1, 2):
        for params in VARIANTS:
            got = LiangInformationFlow(k=k, **params).multivariate(shared)
            want = LiangInformationFlow(k=k, **params).multivariate(fresh())
            assert np.array_equal(got, want, equal_nan=True)
    assert sorted(shared.liang) == [1, 2]
    spis = [LiangInformationFlow(k=1, **p) for p in VARIANTS]
    assert {(type(s)._cache_namespace, *s._cache_subkey) for s in spis} == {("liang", 1)}


def test_collinear_pair_is_nan_in_the_bivariate_version_and_refused_when_conditional():
    z = _fixture(M=3, seed=10)
    z[2] = 3.0 * z[0] + 1.0
    data = Data(z, dim_order="ps", zscore=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        A = LiangInformationFlow().multivariate(data)
    assert np.isnan(A[0, 2]) and np.isnan(A[2, 0])
    assert np.isfinite(A[0, 1]) and np.isfinite(A[1, 2])
    assert any("undefined and set to NaN" in str(w.message) for w in caught)
    with pytest.raises(ValueError, match="pair"):
        LiangInformationFlow().bivariate(data, i=0, j=2)
    with pytest.raises(ValueError, match="nonsingular"):
        LiangInformationFlow(conditional=True).multivariate(data)


# --------------------------------------------------------------------------
# 6. Calculator round trip and bundled configs
# --------------------------------------------------------------------------

def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.causal": {"LiangInformationFlow": {
        "labels": ["directed", "linear", "signed", "time-dependent"],
        "configs": [dict(k=1, **p) for p in VARIANTS]}}}
    path = tmp_path / "liang.yaml"
    path.write_text(yaml.safe_dump(cfg))
    z = _fixture(M=4, T=300, seed=12)
    calc = Calculator(dataset=Data(z, dim_order="ps"), config=str(path))
    calc.compute()
    assert calc.errors == {}
    assert sorted(calc.spis) == sorted(IDENTIFIERS)
    for ident, params in zip(IDENTIFIERS, VARIANTS):
        got = calc.table[ident].to_numpy(dtype=float)
        want = LiangInformationFlow(**params).multivariate(Data(z, dim_order="ps"))
        np.testing.assert_allclose(_offdiag(got), _offdiag(want), rtol=1e-12)


def test_bundled_configs_construct_every_variant():
    full = load_spis_from_yaml(resolve_config("full"), quiet=True)
    for ident in IDENTIFIERS:
        assert ident in full
        labels = set(full[ident].labels)
        assert {"directed", "linear", "signed", "time-dependent", "M14"} <= labels
        assert ("multivariate" if "cond" in ident else "bivariate") in labels
        assert not {"bivariate", "multivariate"} <= labels
    assert set(IDENTIFIERS) <= set(Calculator(config="full").spis)
    fast = load_spis_from_yaml(resolve_config("fast"), quiet=True)
    assert [i for i in fast if i.startswith("liang")] == ["liang_norm_k-1"]

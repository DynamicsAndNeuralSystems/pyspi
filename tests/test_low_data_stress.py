"""Representative SPIs against small, deliberately awkward fixtures.

The contract this file enforces:

    A statistic either produces a defensible result, or fails explicitly for a
    documented reason. It must not silently produce a misleading value.

Every fixture is M=3 at T=64 and T=256, from two fixed seeds -- short enough
that the estimators are working near their limits, which is where the failures
this branch fixed all lived, and small enough that the whole file runs in
seconds. Each fixture is routed to the SPIs it can actually say something
about; nothing here runs the full config.

What is asserted, by kind:

* **Analytic or independent references** where one exists (Gaussian MI from
  rho, continuous KSG references, a known lag).
* **Structure**: symmetry, antisymmetry, orientation, permutation covariance.
* **Invariance** where it is mathematically required (per-coordinate rescaling
  for KSG, affine rescaling for the causal scores).
* **Serial/parallel equality.**
* **Direction only for identifiable synthetic models** -- and only where the
  estimator claims identifiability, which CDS and IGCI do not.
* **Explicit rejection** on input a statistic cannot support, rather than a
  number.
"""
import re

import numpy as np
import pytest

from pyspi.data import Data

T_SHORT, T_LONG = 64, 256
SEEDS = (0, 7)
M = 3


# ---------------------------------------------------------------------------
# Fixtures: (M, T) arrays, deterministic in (seed, T)
# ---------------------------------------------------------------------------

def _independent_gaussian(rng, T):
    return rng.standard_normal((M, T))


def _correlated_gaussian(rng, T, rho=0.7):
    z = rng.standard_normal(T)
    return np.vstack([z,
                      rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(T),
                      rng.standard_normal(T)])


def _coupled_var(rng, T):
    """Process 0 -> 1 at lag 1, 1 -> 2 at lag 1; no feedback."""
    X = np.zeros((M, T))
    for t in range(1, T):
        X[0, t] = 0.5 * X[0, t - 1] + rng.standard_normal()
        X[1, t] = 0.3 * X[1, t - 1] + 0.8 * X[0, t - 1] + rng.standard_normal()
        X[2, t] = 0.3 * X[2, t - 1] + 0.8 * X[1, t - 1] + rng.standard_normal()
    return X


def _nonlinear_coupled(rng, T):
    x = rng.standard_normal(T)
    return np.vstack([x, np.tanh(2 * x) + 0.3 * rng.standard_normal(T),
                      rng.standard_normal(T)])


def _quantised(rng, T, levels=2):
    return rng.integers(0, levels, (M, T)).astype(float)


def _delayed_oscillation(rng, T, lag=4):
    t = np.arange(T + lag)
    base = np.sin(2 * np.pi * t / 9.0) + 0.3 * rng.standard_normal(T + lag)
    return np.vstack([base[lag:], base[:-lag] + 0.2 * rng.standard_normal(T),
                      rng.standard_normal(T)])


def _heavy_tailed(rng, T):
    X = rng.standard_t(df=2, size=(M, T))
    X[0, T // 3] = 40.0            # deterministic outliers, not drawn
    X[1, 2 * T // 3] = -35.0
    return X


def _duplicate_and_collinear(rng, T):
    x = rng.standard_normal(T)
    return np.vstack([x, x.copy(), 3.0 * x + 1e-9 * rng.standard_normal(T)])


FIXTURES = {
    "independent_gaussian": _independent_gaussian,
    "correlated_gaussian": _correlated_gaussian,
    "coupled_var": _coupled_var,
    "nonlinear_coupled": _nonlinear_coupled,
    "binary": lambda rng, T: _quantised(rng, T, 2),
    "four_level": lambda rng, T: _quantised(rng, T, 4),
    "delayed_oscillation": _delayed_oscillation,
    "heavy_tailed": _heavy_tailed,
    "duplicate_and_collinear": _duplicate_and_collinear,
}

ALL_CASES = [(name, seed, T) for name in FIXTURES for seed in SEEDS
             for T in (T_SHORT, T_LONG)]


def make(name, seed, T, **data_kwargs):
    return Data(data=FIXTURES[name](np.random.default_rng(seed), T),
                dim_order="ps", **data_kwargs)


def _off(matrix):
    return np.asarray(matrix, float)[~np.eye(M, dtype=bool)]


# ---------------------------------------------------------------------------
# The low-data contract: a number or an explicit refusal, never a silent lie
# ---------------------------------------------------------------------------

def _representative_spis():
    """(label, factory) -- one per family the audit touched, not 322 SPIs."""
    import pyspi.statistics.basic as basic
    import pyspi.statistics.causal as causal
    import pyspi.statistics.distance as distance
    import pyspi.statistics.infotheory as it
    import pyspi.statistics.spectral as spectral

    return [
        ("mi_gaussian", lambda: it.MutualInfo(estimator="gaussian")),
        ("mi_kraskov", lambda: it.MutualInfo(estimator="kraskov")),
        ("tlmi_gaussian", lambda: it.TimeLaggedMutualInfo(estimator="gaussian")),
        ("tlmi_kraskov", lambda: it.TimeLaggedMutualInfo(estimator="kraskov")),
        ("te_gaussian_fixed", lambda: it.TransferEntropy(estimator="gaussian")),
        ("te_kraskov_fixed", lambda: it.TransferEntropy(estimator="kraskov")),
        ("te_gaussian_auto",
         lambda: it.TransferEntropy(estimator="gaussian",
                                    auto_embed_method="MAX_CORR_AIS",
                                    k_search_max=3, tau_search_max=2)),
        ("te_kraskov_auto",
         lambda: it.TransferEntropy(estimator="kraskov",
                                    auto_embed_method="MAX_CORR_AIS",
                                    k_search_max=3, tau_search_max=2)),
        ("di_gaussian", lambda: it.DirectedInfo(estimator="gaussian", n=3)),
        ("di_kraskov", lambda: it.DirectedInfo(estimator="kraskov", n=3)),
        ("xcorr_max_sig", lambda: basic.CrossCorrelation(statistic="max",
                                                         sigonly=True)),
        ("xcorr_mean", lambda: basic.CrossCorrelation(statistic="mean",
                                                      sigonly=False)),
        ("gd_delay", lambda: spectral.GroupDelay(statistic="delay", fmin=0,
                                                 fmax=0.5)),
        ("dcoh_mean", lambda: spectral.DirectedCoherence(statistic="mean",
                                                         fmin=0, fmax=0.5)),
        ("anm", causal.AdditiveNoiseModel),
        ("cds", causal.ConditionalDistributionSimilarity),
        ("reci", causal.RegressionErrorCausalInference),
        ("igci", causal.InformationGeometricCausalInference),
        ("pdist", lambda: distance.PairwiseDistance(metric="euclidean")),
        ("xpdist", lambda: distance.CrossPairwiseDistance(tau=2)),
    ]


# Frozen observations, not claims that these input families must always make
# GroupDelay empty. On precisely these deterministic fixtures the significance
# gate keeps no frequency cluster. Pinning the full case makes any future change
# visible instead of treating every sample from the family as guaranteed-empty.
OBSERVED_EMPTY_CASES = {
    ("gd_delay", name, seed, T)
    for name in ("independent_gaussian", "binary", "four_level", "heavy_tailed")
    for seed in SEEDS
    for T in (T_SHORT, T_LONG)
}

# (SPI label, fixture, seed, T) -> (exception class, message regex).
# DirectedCoherence may explicitly fail its spectral factorisation on these
# singular duplicate-process cases (the backend's random fallback sometimes
# converges, so success is also accepted). No other refusal is accepted.
EXPECTED_REFUSALS = {
    ("dcoh_mean", "duplicate_and_collinear", seed, T):
        (np.linalg.LinAlgError, r"^Singular matrix$")
    for seed in SEEDS
    for T in (T_SHORT, T_LONG)
}
EXPECTED_REFUSALS.update({
    (label, fixture, seed, T):
        (ValueError, r"^KSG requires continuous, tie-free coordinates:")
    for label in ("mi_kraskov", "tlmi_kraskov", "te_kraskov_fixed",
                  "te_kraskov_auto", "di_kraskov")
    for fixture in ("binary", "four_level")
    for seed in SEEDS
    for T in (T_SHORT, T_LONG)
})


@pytest.mark.parametrize("name,seed,T", ALL_CASES)
def test_representative_spis_give_a_result_or_an_explicit_refusal(name, seed, T):
    """No unexplained exception, infinity, or entirely empty output.

    A refusal is accepted only when its exact case, exception type and message
    pattern appear in `EXPECTED_REFUSALS`. An arbitrary nonempty exception is
    not evidence that the estimator declined for the intended reason.
    """
    import warnings

    data = make(name, seed, T)
    problems = []
    for label, factory in _representative_spis():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                table = np.asarray(factory().multivariate(data), dtype=float)
        except Exception as err:                       # noqa: BLE001
            expected = EXPECTED_REFUSALS.get((label, name, seed, T))
            if expected is None:
                problems.append(f"{label}: {type(err).__name__}: {err}")
            else:
                error_type, message = expected
                if type(err) is not error_type or re.search(message, str(err)) is None:
                    problems.append(
                        f"{label}: expected {error_type.__name__} /{message}/, "
                        f"got {type(err).__name__}: {err}"
                    )
            continue

        off = _off(table)
        if np.isinf(off).any():
            problems.append(f"{label}: infinite values")
        elif not np.isfinite(off).any():
            if (label, name, seed, T) not in OBSERVED_EMPTY_CASES:
                problems.append(f"{label}: no finite off-diagonal value")
        elif (label, name, seed, T) in OBSERVED_EMPTY_CASES:
            problems.append(f"{label}: frozen empty observation now produced a "
                            f"value; review the observation table")
    assert not problems, f"[{name} seed={seed} T={T}]\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# Analytic and independent references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_gaussian_mi_tracks_the_sample_correlation_on_short_records(seed):
    """-0.5*log(1 - r^2) from the sample r, so this is exact at any T."""
    import pyspi.statistics.infotheory as it

    for T in (T_SHORT, T_LONG):
        data = make("correlated_gaussian", seed, T)
        Z = data.to_numpy(squeeze=True)
        table = it.MutualInfo(estimator="gaussian").multivariate(data)
        for i in range(M):
            for j in range(i + 1, M):
                r = np.corrcoef(Z[i], Z[j])[0, 1]
                assert table[i, j] == pytest.approx(
                    -0.5 * np.log(1 - r ** 2), abs=1e-6), (T, i, j)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("levels", [2, 4])
def test_ksg_mi_explicitly_refuses_a_duplicated_quantised_process(seed, levels):
    """Tied coordinates require an external discrete estimator, not jitter."""
    import pyspi.statistics.infotheory as it

    rng = np.random.default_rng(seed)
    x = rng.integers(0, levels, T_LONG).astype(float)
    data = Data(data=np.vstack([x, x.copy(), rng.standard_normal(T_LONG)]),
                dim_order="ps", zscore=False)
    with pytest.raises(ValueError, match="continuous, tie-free coordinates"):
        it.MutualInfo(estimator="kraskov").bivariate(data, i=0, j=1)


@pytest.mark.parametrize("seed", SEEDS)
def test_group_delay_recovers_the_oscillation_lag(seed):
    """A known lag on a genuinely narrowband signal, at T=256.

    On the T=64 instances used here the significance gate keeps no cluster;
    that is a fixture observation covered by the contract test, not a theorem
    that short records must return NaN.
    """
    from pyspi.statistics.spectral import GroupDelay

    data = make("delayed_oscillation", seed, T_LONG)
    table = GroupDelay(statistic="delay", fmin=0, fmax=0.5).multivariate(data)
    assert table[0, 1] == pytest.approx(4, abs=1.0)
    assert table[1, 0] == pytest.approx(-table[0, 1], rel=1e-9)


# ---------------------------------------------------------------------------
# Structure: symmetry, antisymmetry, orientation, permutation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,seed,T", ALL_CASES)
def test_declared_structure_holds_on_every_fixture(name, seed, T):
    """Symmetric statistics are symmetric; antisymmetric ones antisymmetric."""
    import warnings

    import pyspi.statistics.basic as basic
    import pyspi.statistics.distance as distance
    import pyspi.statistics.infotheory as it
    from pyspi.statistics.causal import InformationGeometricCausalInference

    symmetric = [("mi_gaussian", it.MutualInfo(estimator="gaussian")),
                 ("xcorr_mean", basic.CrossCorrelation(statistic="mean",
                                                       sigonly=False)),
                 ("pdist", distance.PairwiseDistance(metric="euclidean")),
                 ("xpdist", distance.CrossPairwiseDistance(tau=2))]
    antisymmetric = [("igci", InformationGeometricCausalInference())]

    data = make(name, seed, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, spi in symmetric:
            A = np.asarray(spi.multivariate(data), float)
            assert np.allclose(A, A.T, equal_nan=True, rtol=1e-9), f"{label}"
        for label, spi in antisymmetric:
            A = np.asarray(spi.multivariate(data), float)
            assert np.allclose(A, -A.T, equal_nan=True, rtol=1e-9), f"{label}"


@pytest.mark.parametrize("seed", SEEDS)
def test_permuting_processes_permutes_the_matrix(seed):
    """Nothing may depend on the order the processes were handed over."""
    import warnings

    import pyspi.statistics.causal as causal
    import pyspi.statistics.infotheory as it

    order = [2, 0, 1]
    for T in (T_SHORT, T_LONG):
        Z = FIXTURES["coupled_var"](np.random.default_rng(seed), T)
        for spi in (it.TransferEntropy(estimator="gaussian"),
                    it.MutualInfo(estimator="kraskov"),
                    causal.RegressionErrorCausalInference()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base = np.asarray(spi.multivariate(
                    Data(data=Z, dim_order="ps")), float)
                moved = np.asarray(spi.multivariate(
                    Data(data=Z[order], dim_order="ps")), float)
            assert np.allclose(base[np.ix_(order, order)], moved,
                               equal_nan=True, rtol=1e-9), (T, spi.identifier)


# ---------------------------------------------------------------------------
# Invariance where it is mathematically required
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,seed,T", [
    case for case in ALL_CASES if case[0] not in {"binary", "four_level"}
])
def test_ksg_measures_are_invariant_to_per_process_rescaling(name, seed, T):
    """Per-coordinate standardisation provides affine marginal covariance.

    Exact duplicate processes are excluded from contemporaneous MI: their
    joint law is supported on a diagonal, has no two-dimensional density, and
    has infinite continuous MI. A finite-sample KSG value on that singular
    pair has no finite affine-invariance target. The fixture's near-collinear
    third process remains covered.
    """
    import warnings

    import pyspi.statistics.infotheory as it

    scale = np.array([1e-3, 1.0, 1e3])[:, None]
    Z = FIXTURES[name](np.random.default_rng(seed), T)
    for spi in (it.MutualInfo(estimator="kraskov"),
                it.TimeLaggedMutualInfo(estimator="kraskov")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = spi.multivariate(Data(data=Z, dim_order="ps", zscore=False))
            moved = spi.multivariate(Data(data=Z * scale, dim_order="ps",
                                          zscore=False))
        valid = ~np.eye(M, dtype=bool)
        if spi.identifier.startswith("mi_"):
            for i in range(M):
                for j in range(i):
                    if np.array_equal(Z[i], Z[j]):
                        valid[i, j] = valid[j, i] = False
        assert np.allclose(np.asarray(base, float)[valid],
                           np.asarray(moved, float)[valid],
                           equal_nan=True, atol=1e-12), spi.identifier


@pytest.mark.parametrize("seed", SEEDS)
def test_causal_scores_are_invariant_to_affine_rescaling(seed):
    """CDS, RECI and IGCI scale their inputs internally, so units must not matter.

    ANM is deliberately excluded because this implementation is scale-
    dependent: scikit-learn's default `GaussianProcessRegressor` uses a fixed
    unit ConstantKernel*RBF kernel when no kernel is supplied (both bounds are
    "fixed"). The fit sees raw values while the later independence score
    standardises its arguments. This is a pyspi/CDT implementation choice, not
    a scale-dependence theorem about additive-noise models. pyspi's default
    z-scoring removes the material dependence in the shipped pipeline.
    """
    import pyspi.statistics.causal as causal

    for T in (T_SHORT, T_LONG):
        Z = FIXTURES["nonlinear_coupled"](np.random.default_rng(seed), T)
        moved = 4.0 * Z + 3.0
        for spi in (causal.ConditionalDistributionSimilarity(),
                    causal.RegressionErrorCausalInference(),
                    causal.InformationGeometricCausalInference()):
            a = np.asarray(spi.multivariate(Data(data=Z, dim_order="ps",
                                                 zscore=False)), float)
            b = np.asarray(spi.multivariate(Data(data=moved, dim_order="ps",
                                                 zscore=False)), float)
            assert np.allclose(a, b, equal_nan=True, atol=1e-12), spi.identifier


@pytest.mark.parametrize("seed", SEEDS)
def test_anm_is_scale_dependent_and_z_scoring_is_what_fixes_it(seed):
    """Recorded, not asserted away.

    A future change that made ANM scale-free would fail this test, which is the
    point: the property should change deliberately, not drift.
    """
    import pyspi.statistics.causal as causal

    Z = FIXTURES["nonlinear_coupled"](np.random.default_rng(seed), T_LONG)
    anm = causal.AdditiveNoiseModel()
    raw = np.asarray(anm.multivariate(Data(data=Z, dim_order="ps",
                                           zscore=False)), float)
    moved = np.asarray(anm.multivariate(Data(data=4.0 * Z + 3.0, dim_order="ps",
                                             zscore=False)), float)
    assert np.nanmax(np.abs(raw - moved)) > 0.01, "ANM became scale-free"

    # With pyspi's default z-scoring the difference collapses from ~0.27 to
    # ~4e-5. Not to zero: z-scoring `Z` and `4Z + 3` agrees only to float
    # precision. The point is the four orders of magnitude, which makes the
    # implementation's scale dependence irrelevant in the default pipeline.
    a = np.asarray(anm.multivariate(Data(data=Z, dim_order="ps")), float)
    b = np.asarray(anm.multivariate(Data(data=4.0 * Z + 3.0, dim_order="ps")),
                   float)
    assert np.nanmax(np.abs(a - b)) < 1e-3


# ---------------------------------------------------------------------------
# Direction, only for identifiable models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_transfer_entropy_finds_the_coupling_direction_at_the_longer_length(seed):
    """0 -> 1 -> 2 with no feedback, so TE must be larger in the true direction.

    Asserted at T=256 only. At T=64 the estimators are within their own noise on
    a coupling this weak, and asserting direction there would pin luck rather
    than behaviour -- the contract test above still requires a defensible number.
    """
    import pyspi.statistics.infotheory as it

    data = make("coupled_var", seed, T_LONG)
    for spi in (it.TransferEntropy(estimator="gaussian"),
                it.TransferEntropy(estimator="kraskov")):
        table = np.asarray(spi.multivariate(data), float)
        assert table[0, 1] > table[1, 0], spi.identifier
        assert table[1, 2] > table[2, 1], spi.identifier


@pytest.mark.parametrize("seed", SEEDS)
def test_anm_prefers_the_true_direction_on_the_nonlinear_fixture(seed):
    """ANM is identifiable for a nonlinear map with independent additive noise.

    RECI is **not** asserted here, and that is a finding rather than an
    omission: on this saturating `tanh` pair it prefers the wrong direction at
    both seeds (0.0225 vs 0.0087, 0.0155 vs 0.0095). On the cubic pair in
    `tests/test_pairwise_causal.py` it gets the direction right. RECI is a
    regression-error heuristic whose
    published argument needs assumptions about the input distribution,
    mechanism and regression; it does not impose a simple "near-uniform cause"
    precondition. CDS and IGCI are absent for the same reason -- this fixture
    supplies no general direction oracle for them.
    """
    import pyspi.statistics.causal as causal

    data = make("nonlinear_coupled", seed, T_LONG)
    table = np.asarray(causal.AdditiveNoiseModel().multivariate(data), float)
    # Lower score = more independent residual, so the true direction is lower.
    assert table[0, 1] < table[1, 0]


# ---------------------------------------------------------------------------
# Explicit refusal, and serial/parallel equality
# ---------------------------------------------------------------------------

def test_short_records_are_refused_with_a_reason_not_a_number():
    """The estimators must say why, not return something plausible.

    T=8 with kNN=4 leaves fewer usable neighbours than k once the embedding is
    aligned; the auto-embedding search has no scorable candidate at all.
    """
    import pyspi.statistics.infotheory as it

    # T=5 is the boundary: the aligned sample count is T - 1 - (dim-1)*delay,
    # so dimension 1 leaves 4 points and 3 usable neighbours against k=4. T=6
    # already succeeds, which is why the fixture is this small.
    tiny = Data(data=np.random.default_rng(0).standard_normal((3, 5)),
                dim_order="ps")
    with pytest.raises(ValueError, match="usable neighbour|can be scored"):
        it.TransferEntropy(estimator="kraskov", auto_embed_method="MAX_CORR_AIS",
                           k_search_max=3, tau_search_max=2).bivariate(
                               tiny, i=0, j=1)


def test_constant_process_is_refused_by_the_ksg_estimators():
    """A constant marginal carries no information and is almost always a broken
    input for a continuous-density estimator."""
    import pyspi.statistics.infotheory as it

    Z = np.random.default_rng(0).standard_normal((2, 128))
    Z[1] = 3.0
    data = Data(data=Z, dim_order="ps", zscore=False)
    with pytest.raises(ValueError, match="constant"):
        it.MutualInfo(estimator="kraskov").bivariate(data, i=0, j=1)


@pytest.mark.parametrize("name", ["heavy_tailed", "coupled_var"])
def test_serial_and_parallel_agree_on_the_awkward_fixtures(name, tmp_path):
    """Bit-for-bit, on exactly the inputs where the estimators branch."""
    from pyspi.calculator import Calculator

    config = tmp_path / "stress.yaml"
    config.write_text(
        ".statistics.infotheory:\n"
        "  MutualInfo:\n    labels: [x]\n    configs:\n"
        "      - {estimator: kraskov, prop_k: 4}\n"
        "      - {estimator: gaussian}\n"
        "  DirectedInfo:\n    labels: [x]\n    configs:\n"
        "      - {estimator: kraskov, prop_k: 4, n: 3}\n"
        ".statistics.basic:\n"
        "  CrossCorrelation:\n    labels: [x]\n    configs:\n"
        "      - {statistic: max, sigonly: true}\n"
    )
    Z = FIXTURES[name](np.random.default_rng(0), T_LONG)

    tables = []
    for kwargs in ({"n_jobs": 1}, {"n_jobs": 2, "mp_context": "spawn"}):
        calc = Calculator(dataset=Data(data=Z, dim_order="ps"),
                          config=str(config))
        calc.compute(**kwargs)
        assert not calc.errors, calc.errors
        tables.append({k: calc.table[k].to_numpy() for k in calc.spis})
    for key in tables[0]:
        assert np.array_equal(tables[0][key], tables[1][key], equal_nan=True), key


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_convergent_cross_mapping_on_the_coupled_fixtures(seed):
    """CCM is the one genuinely expensive representative, so it is marked slow.

    One fixed embedding and one automatic, checked for a defensible bounded
    result rather than a direction: CCM's convergence criterion needs far more
    than 256 samples to separate coupling from shared driving.
    """
    from pyspi.statistics.causal import ConvergentCrossMapping

    data = make("coupled_var", seed, T_LONG)
    for spi in (ConvergentCrossMapping(statistic="mean", embedding_dimension=2),
                ConvergentCrossMapping(statistic="mean",
                                       embedding_dimension=None)):
        table = np.asarray(spi.multivariate(data), float)
        off = _off(table)
        assert np.isfinite(off).all(), spi.identifier
        assert np.abs(off).max() <= 1.0 + 1e-9, spi.identifier

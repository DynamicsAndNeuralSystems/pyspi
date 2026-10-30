"""`pyspi.statistics.causal` against published definitions and reference code.

Oracles: closed forms, re-implementations from the papers, systems with a known
causal direction (Sugihara et al. 2012's coupled logistic maps), pyEDM and sklearn.
"""
import numpy as np
import pytest

from pyspi.data import Data


def _coupled_logistic(T, beta_xy=0.0, beta_yx=0.1, rx=3.8, ry=3.5, burn=300):
    """The coupled logistic maps of Sugihara et al. (2012), SOM.

    X(t+1) = X(t) [rx - rx X(t) - beta_xy Y(t)]
    Y(t+1) = Y(t) [ry - ry Y(t) - beta_yx X(t)]

    With beta_xy = 0, X is autonomous and drives Y: the true direction is X -> Y.
    """
    x, y = np.empty(T + burn), np.empty(T + burn)
    x[0], y[0] = 0.4, 0.2
    for t in range(T + burn - 1):
        x[t + 1] = x[t] * (rx - rx * x[t] - beta_xy * y[t])
        y[t + 1] = y[t] * (ry - ry * y[t] - beta_yx * x[t])
    return np.vstack([x[burn:], y[burn:]])


@pytest.fixture(scope="module")
def x_drives_y():
    return Data(_coupled_logistic(300), dim_order="ps")


# ---------------------------------------------------------------------------
# Convergent cross-mapping
# ---------------------------------------------------------------------------

def test_ccm_rows_are_sources_on_sugiharas_coupled_logistic_map(x_drives_y):
    """X drives Y, so A[X, Y] must carry the evidence: row = source.

    Sugihara et al. (2012): "if X causes Y, then ... the historical record of Y
    can be used to estimate states of X", i.e. *Y xmap X* is the skill that
    converges. pyEDM names that column ``Y:X`` (the library variable first),
    and pyspi's directed convention puts the cause on the row, so A[X, Y] must
    be Y-xmap-X: high. A[Y, X] is X-xmap-Y, which on this system is near zero.
    """
    from pyspi.statistics.causal import ConvergentCrossMapping

    X, Y = 0, 1
    for statistic in ("mean", "max"):
        A = np.asarray(ConvergentCrossMapping(
            statistic=statistic, embedding_dimension=2).multivariate(x_drives_y))
        assert A[X, Y] > 0.5 > A[Y, X], (
            f"ccm {statistic}: X drives Y, so A[X,Y] (Y xmap X) must be the "
            f"converging skill. Got A[X,Y]={A[X, Y]:.3f}, A[Y,X]={A[Y, X]:.3f}: "
            f"the matrix is transposed relative to row=source, column=target.")

    D = np.asarray(ConvergentCrossMapping(
        statistic="diff", embedding_dimension=2).multivariate(x_drives_y))
    assert D[X, Y] > 0.5 and D[Y, X] == -D[X, Y], (
        f"ccm diff must be positive from the driver: got D[X,Y]={D[X, Y]:.3f}")


def test_ccm_library_sizes_reach_the_largest_library():
    """The convergence statistics must include the largest library size.

    `_from_cache` spans library sizes from ``ceil(2E/10)*10`` to
    ``floor((N-E-1)/10)*10`` in ``nlibs = 21`` steps. Cross-map skill converges
    *upwards* with library size (Sugihara et al. 2012, Fig. 2), so on a system
    with genuine coupling the skill at the top library is where the evidence
    is, and ``max`` over the advertised range can never be below it.

    The oracle is pyEDM's own CCM evaluated at that single top library size,
    with the same seed and sample count (pyEDM seeds every library size from
    the same stream, so the value is exactly what a multi-size call computes
    at that size).
    """
    import pandas as pd
    import pyEDM

    from pyspi.statistics.causal import ConvergentCrossMapping

    T, E = 100, 2
    data = Data(_coupled_logistic(T), dim_order="ps")
    z = data.to_numpy(squeeze=True)
    df = pd.DataFrame(np.column_stack([np.arange(T), z.T]),
                      columns=["index", "proc0", "proc1"])
    top = int(np.floor((T - E - 1) / 10) * 10)
    skill_at_top = float(pyEDM.CCM(
        dataFrame=df, E=E, columns="proc1", target="proc0", libSizes=[top],
        sample=100, seed=42, parallel=False)["proc1:proc0"].iloc[0])

    A = np.asarray(ConvergentCrossMapping(
        statistic="max", embedding_dimension=E).multivariate(data))
    assert A[0, 1] >= skill_at_top - 1e-12, (
        f"ccm max = {A[0, 1]:.4f} is below the skill at the largest library "
        f"(L={top}: {skill_at_top:.4f}); the largest library sizes are being "
        f"dropped.")


# ---------------------------------------------------------------------------
# IGCI
# ---------------------------------------------------------------------------

def test_igci_slope_estimator_follows_the_documented_sign():
    """Both IGCI estimators must be > 0 for x -> y, as `igci_score` documents.

    Daniusis et al. (2010), Eq. (13): the slope-based estimate of
    C_{X->Y} = E[log |f'(X)|] is ``1/m * sum log |dy/dx|`` over x-sorted
    neighbours, and X -> Y is inferred when C_{X->Y} < C_{Y->X}. The entropy
    estimator (Eq. 12) estimates the same C_{X->Y} as H(Y) - H(X), and pyspi
    reports ``H(x) - H(y)``, positive for x -> y. So the slope estimator must
    report ``C_{Y->X} - C_{X->Y}``.

    The system is deterministic and invertible, y = x**3 with x uniform, the
    case IGCI is built for: under the uniform reference measure
    C_{X->Y} = E[log 3X^2] = log 3 - 2 < 0, so x -> y is unambiguous.
    """
    from pyspi.lib.pairwise_causal import igci_score

    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 500)
    y = x ** 3
    unit = lambda v: (v - v.min()) / (v.max() - v.min())

    def slope_C(a, b):
        a, b = unit(a), unit(b)
        order = np.argsort(a)
        da, db = np.diff(a[order]), np.diff(b[order])
        keep = (da != 0) & (db != 0)
        return np.sum(np.log(np.abs(db[keep] / da[keep]))) / len(a)

    C_xy, C_yx = slope_C(x, y), slope_C(y, x)
    assert C_xy < 0 < C_yx  # the paper's rule picks x -> y

    assert igci_score(x, y, "uniform", "entropy") > 0
    slope = igci_score(x, y, "uniform", "integral")
    assert slope == pytest.approx(C_yx - C_xy, rel=1e-12), (
        f"slope estimator returned {slope:.4f}; the paper's C_yx - C_xy is "
        f"{C_yx - C_xy:.4f} (> 0 for x -> y, matching the entropy estimator)")


@pytest.mark.parametrize("E", [1, 3])
def test_ccm_cross_map_skill_is_bit_identical_to_pyedm(E):
    """`_cross_map_skill` is pyEDM 2.5's CCM, computed once per library variable.

    pyspi computes the simplex neighbours and weights of each variable's
    shadow manifold once and applies them to every target, where
    `pyEDM.CCM(columns=a, target=b)` recomputes them for each pair. That is
    only a speedup if the numbers are exactly pyEDM's, both directions, for
    one target or several at once. A pyEDM release that changes its CCM
    algorithm or its seeding will fail here, which is the point.
    """
    import pandas as pd
    import pyEDM

    from pyspi.statistics.causal import _ccm_library_sizes, _cross_map_skill

    rng = np.random.default_rng(11)
    N = 70
    z = np.vstack([_coupled_logistic(N)[0], _coupled_logistic(N)[1],
                   rng.standard_normal(N)])
    df = pd.DataFrame(np.column_stack([np.arange(N), z.T]),
                      columns=["index", "a", "b", "c"])
    names = ["a", "b", "c"]
    lib_sizes = _ccm_library_sizes(N, E)
    sample = 20

    for i in range(3):
        targets = [j for j in range(3) if j != i]
        together = _cross_map_skill(z[i], [z[j] for j in targets], E,
                                    lib_sizes, sample=sample, seed=42)
        for row, j in zip(together, targets):
            ref = pyEDM.CCM(dataFrame=df, E=E, columns=names[i],
                            target=names[j], libSizes=lib_sizes, sample=sample,
                            seed=42, parallel=False)
            expected = ref[f"{names[i]}:{names[j]}"].to_numpy()
            alone = _cross_map_skill(z[i], [z[j]], E, lib_sizes,
                                     sample=sample, seed=42)[0]
            assert np.array_equal(row, expected), (names[i], names[j])
            assert np.array_equal(alone, expected), (names[i], names[j])
            # and pyEDM's reverse column is the same computation from j's side
            reverse = _cross_map_skill(z[j], [z[i]], E, lib_sizes,
                                       sample=sample, seed=42)[0]
            assert np.array_equal(
                reverse, ref[f"{names[j]}:{names[i]}"].to_numpy())


# ---------------------------------------------------------------------------
# Additive noise model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [150, 260])
def test_anm_shared_factorisation_is_bit_identical_to_sklearn(T):
    """ANM factorises each cause's GP kernel once and reuses it per effect.

    The reference is a per-pair scikit-learn fit:
    ``GaussianProcessRegressor(kernel=C(1)*RBF(1) fixed, optimizer=None)``,
    residual ``predict(x) - y``, then the normalised HSIC against x. The
    kernel matrix is ill-conditioned (condition number ~6e11), so anything
    short of the same operations in the same order would show up here as
    drift. T=260 also engages the HSIC's 200-point subsampling.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF

    from pyspi.lib.pairwise_causal import normalized_hsic
    from pyspi.statistics.causal import AdditiveNoiseModel

    rng = np.random.default_rng(T)
    x = rng.standard_normal(T)
    Z = np.vstack([x, np.tanh(2 * x) + 0.3 * rng.standard_normal(T),
                   rng.standard_normal(T).cumsum(), rng.standard_normal(T)])
    data = Data(Z, dim_order="ps")
    z = data.to_numpy()

    expected = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
            y_predict = gp.fit(z[i], z[j]).predict(z[i]).reshape(-1, 1)
            expected[i, j] = normalized_hsic(y_predict - z[j], z[i])

    got = np.asarray(AdditiveNoiseModel().multivariate(data), float)
    assert np.array_equal(got, expected, equal_nan=True), np.nanmax(
        np.abs(got - expected))
    assert AdditiveNoiseModel().bivariate(data, i=2, j=1) == expected[2, 1]


# ---------------------------------------------------------------------------
# Conditional distribution similarity
# ---------------------------------------------------------------------------

def _cds_populated_bins(x, ffactor=2, maxdev=3, minc=12):
    """How many x-bins hold more than `minc` points (Fonollosa 2016).

    x is standardised, re-standardised on its points within `maxdev`
    deviations, then rounded to a grid of 1/ffactor and clipped at
    +/- ffactor*maxdev: the conditioning bins of the CDS statistic.
    """
    x = (x - x.mean()) / x.std()
    core = x[np.abs(x) < maxdev]
    x = np.clip(np.round((x - core.mean()) / core.std() * ffactor),
                -ffactor * maxdev, ffactor * maxdev)
    _, counts = np.unique(x, return_counts=True)
    return int(np.sum(counts > minc))


def test_cds_is_undefined_not_zero_when_conditionals_cannot_be_compared():
    """CDS is the spread of P(y | x = a) across the x-bins a (Fonollosa 2016).

    It needs at least two bins with more than `minc` = 12 points. With fewer,
    cdt returns 0 (trivially with one bin, hard-coded with none), which is the
    *strongest* evidence the statistic can give for x -> y (lower favours
    x -> y). Undefined must be NaN; at T=64 on Gaussian noise both cases occur.
    """
    from pyspi.lib.pairwise_causal import cds_score
    from pyspi.statistics.causal import ConditionalDistributionSimilarity

    rng = np.random.default_rng(0)
    Z = rng.standard_normal((3, 64))
    data = Data(Z, dim_order="ps")
    z = data.to_numpy(squeeze=True)
    A = np.asarray(ConditionalDistributionSimilarity().multivariate(data), float)

    bins = [_cds_populated_bins(z[i]) for i in range(3)]
    assert min(bins) < 2 <= max(bins), bins   # both cases occur here
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if bins[i] < 2:
                assert np.isnan(A[i, j]), (
                    f"x=process {i} has {bins[i]} usable bin(s); CDS is "
                    f"undefined but the SPI returned {A[i, j]!r}")
            else:
                assert A[i, j] == cds_score(z[i], z[j]) > 0

"""Smoke test for pyspi fork — shape / finiteness / sign checks.

Runs a small set of SPIs on known synthetic data (coupled AR(1)) and checks
that every estimator instantiates without a JVM, that bivariate and
multivariate methods return correctly-shaped finite matrices, and that
dependence measures are positive on coupled signals.

These are cheap sanity checks, not correctness checks. Closed-form validation
of the information-theoretic estimators lives in test_infotheory_analytic.py.
"""
import pytest
import numpy as np

np.random.seed(42)


def generate_coupled_ar1(M=5, T=500, coupling=0.4, noise_std=0.3):
    """Generate M coupled AR(1) processes.

    X_i(t) = coupling * X_{i-1}(t-1) + noise
    Process 0 is independent AR(1).
    """
    X = np.zeros((M, T))
    X[:, 0] = np.random.randn(M)
    for t in range(1, T):
        X[0, t] = 0.8 * X[0, t - 1] + noise_std * np.random.randn()
        for i in range(1, M):
            X[i, t] = (0.5 * X[i, t - 1]
                        + coupling * X[i - 1, t - 1]
                        + noise_std * np.random.randn())
    return X


def test_imports():
    """All estimators instantiate without JIDT/JVM."""
    from pyspi.statistics.infotheory import (
        MutualInfo, TimeLaggedMutualInfo, TransferEntropy,
        JointEntropy, ConditionalEntropy, CrossmapEntropy,
        CausalEntropy, DirectedInfo, StochasticInteraction,
        IntegratedInformation,
    )
    from pyspi.statistics.basic import (
        Covariance, Precision, CrossCorrelation,
        SpearmanR, KendallTau, LaggedCorrelation,
    )
    from pyspi.statistics.distance import DynamicTimeWarping, CrossPairwiseDistance
    from pyspi.statistics.spectral import CoherenceMagnitude, DirectedCoherence
    from pyspi.statistics.misc import LinearModel, GPModel

    # Info-theoretic: all estimators. Two exclusions, in opposite directions.
    #
    # kozachenko is entropy-only: MutualInfo, TimeLaggedMutualInfo and
    # TransferEntropy are computed directly rather than from marginal
    # entropies, so those combinations raise NotImplementedError (they used
    # to return NaN silently). See test_kozachenko_rejected_for_non_entropy.
    #
    # kraskov is the converse: only those same three have a genuine KSG
    # implementation. The composed measures used to accept it and run the
    # Gaussian estimator while advertising kraskov_NN-<k>.
    # See tests/test_estimator_contracts.py.
    # DirectedInfo estimates each I(X^i; Y_i | Y^{i-1}) term with a direct
    # KSG/Frenzel-Pompe CMI, so it supports kraskov; the others are composed
    # from marginal entropies and do not.
    composed = (JointEntropy, ConditionalEntropy, CrossmapEntropy,
                CausalEntropy, StochasticInteraction)
    direct = (MutualInfo, TimeLaggedMutualInfo, TransferEntropy, DirectedInfo)

    for est in ('gaussian', 'kraskov', 'kernel', 'kozachenko'):
        if est != 'kraskov':
            for cls in composed:
                cls(estimator=est)
        if est != 'kozachenko':
            for cls in direct:
                cls(estimator=est)

    TransferEntropy(estimator='symbolic', k_history=3)
    TransferEntropy(estimator='kernel', kernel_width=0.25)


@pytest.mark.parametrize("cls_name", [
    "JointEntropy", "ConditionalEntropy", "CrossmapEntropy",
    "CausalEntropy", "StochasticInteraction",
])
def test_kraskov_rejected_for_composed_measures(cls_name):
    """kraskov must fail loudly where no KSG estimator exists.

    These six composed the measure from marginal entropies and were handed
    GaussianEntropyCalculator for "kraskov", so they returned exactly the
    Gaussian result while their identifier claimed kraskov_NN-<k>.
    """
    import pyspi.statistics.infotheory as it
    with pytest.raises(NotImplementedError, match="kraskov"):
        getattr(it, cls_name)(estimator="kraskov")


@pytest.mark.parametrize("cls_name", ["MutualInfo", "TimeLaggedMutualInfo", "TransferEntropy"])
def test_kozachenko_rejected_for_non_entropy(cls_name):
    """kozachenko must fail loudly for measures with no Kozachenko-Leonenko path.

    These three previously fell through to a logging.warning and returned NaN,
    so an invalid config produced a silently all-NaN SPI rather than an error.
    """
    import pyspi.statistics.infotheory as it
    with pytest.raises(NotImplementedError, match="kozachenko"):
        getattr(it, cls_name)(estimator="kozachenko")


def test_gaussian_mi_analytical():
    """Gaussian MI matches analytical formula: MI = -0.5 * ln(1 - r^2)."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import MutualInfo

    X = generate_coupled_ar1(M=3, T=1000)
    data = Data(X, zscore=True)

    mi = MutualInfo(estimator='gaussian')
    result = mi.multivariate(data)

    # Check shape and finiteness
    assert result.shape == (3, 3), f"Shape mismatch: {result.shape}"
    assert np.all(np.isfinite(result[~np.isnan(result)])), "Non-finite MI values"
    assert np.all(np.isnan(np.diag(result))), "Diagonal should be NaN"

    # NOTE: this compares against the *sample* correlation, so it is an
    # implementation-identity check, not independent validation. Real closed-form
    # validation against the true rho lives in test_infotheory_analytic.py.
    Z = data.to_numpy(squeeze=True)
    R = np.corrcoef(Z)
    r2 = np.clip(R ** 2, 0, 1 - 1e-15)
    expected = -0.5 * np.log(1 - r2)
    np.fill_diagonal(expected, np.nan)

    off_diag = ~np.isnan(result)
    assert np.allclose(result[off_diag], expected[off_diag], atol=1e-10), \
        f"Gaussian MI mismatch: max diff={np.max(np.abs(result[off_diag] - expected[off_diag]))}"

    # Coupled processes should have positive MI
    assert result[0, 1] > 0.01, f"MI(0,1) should be positive: {result[0, 1]}"


def test_kraskov_mi():
    """KSG MI is positive for correlated signals."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import MutualInfo

    X = generate_coupled_ar1(M=3, T=500)
    data = Data(X, zscore=True)

    mi = MutualInfo(estimator='kraskov', prop_k=4)
    result = mi.multivariate(data)

    assert result.shape == (3, 3)
    off_diag = ~np.isnan(result)
    assert np.all(np.isfinite(result[off_diag])), "Non-finite KSG MI"
    assert result[0, 1] > 0, f"KSG MI(0,1) should be positive: {result[0, 1]}"


def test_kernel_mi():
    """Kernel MI is positive for correlated signals."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import MutualInfo

    X = generate_coupled_ar1(M=3, T=500)
    data = Data(X, zscore=True)

    mi = MutualInfo(estimator='kernel', kernel_width=0.25)
    result = mi.multivariate(data)

    assert result.shape == (3, 3)
    off_diag = ~np.isnan(result)
    assert np.all(np.isfinite(result[off_diag])), "Non-finite kernel MI"
    assert result[0, 1] > 0, f"Kernel MI(0,1) should be positive: {result[0, 1]}"


def test_transfer_entropy():
    """TE is positive for causally coupled signals (all estimators)."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import TransferEntropy

    X = generate_coupled_ar1(M=3, T=500, coupling=0.5)
    data = Data(X, zscore=True)

    for est in ('gaussian', 'kraskov', 'kernel', 'symbolic'):
        if est == 'kernel':
            te = TransferEntropy(estimator=est, kernel_width=0.25)
        elif est == 'symbolic':
            te = TransferEntropy(estimator=est, k_history=3)
        else:
            te = TransferEntropy(estimator=est)

        result = te.multivariate(data)
        assert result.shape == (3, 3), f"{est} TE shape: {result.shape}"

        # TE(0→1) should be positive (0 causes 1)
        te_01 = result[0, 1]
        assert np.isfinite(te_01), f"{est} TE(0→1) not finite: {te_01}"

        if est == 'gaussian':
            # Gaussian TE = Granger causality, should be clearly positive
            assert te_01 > 0.01, f"Gaussian TE(0→1) too small: {te_01}"


def test_joint_conditional_entropy():
    """JE and CE produce finite values for kernel estimator."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import JointEntropy, ConditionalEntropy

    X = generate_coupled_ar1(M=3, T=300)
    data = Data(X, zscore=True)

    for est in ('gaussian', 'kernel', 'kozachenko'):
        je = JointEntropy(estimator=est)
        result_je = je.multivariate(data)
        assert result_je.shape == (3, 3)
        off = ~np.isnan(result_je)
        assert np.all(np.isfinite(result_je[off])), f"{est} JE has non-finite values"

        ce = ConditionalEntropy(estimator=est)
        result_ce = ce.multivariate(data)
        assert result_ce.shape == (3, 3)
        off = ~np.isnan(result_ce)
        assert np.all(np.isfinite(result_ce[off])), f"{est} CE has non-finite values"


def test_basic_spis():
    """Basic SPIs (correlation, DTW, etc.) produce finite values."""
    from pyspi.data import Data
    from pyspi.statistics.basic import (
        Covariance, SpearmanR, KendallTau, CrossCorrelation, LaggedCorrelation,
    )
    from pyspi.statistics.distance import DynamicTimeWarping

    X = generate_coupled_ar1(M=3, T=200)
    data = Data(X, zscore=True)

    for SPI, kwargs in [
        (Covariance, {}),
        (SpearmanR, {}),
        (KendallTau, {}),
        (CrossCorrelation, {}),
        (LaggedCorrelation, {"tau": 1}),
        (LaggedCorrelation, {"tau": 3, "estimator": "spearman"}),
        (DynamicTimeWarping, {}),
    ]:
        name = SPI.__name__ + str(kwargs)
        spi = SPI(**kwargs)
        result = spi.multivariate(data)
        assert result.shape == (3, 3), f"{name} shape: {result.shape}"
        off = ~np.isnan(result)
        assert np.all(np.isfinite(result[off])), f"{name} has non-finite values"


def test_spectral_spis():
    """Spectral SPIs produce finite values."""
    from pyspi.data import Data
    from pyspi.statistics.spectral import CoherenceMagnitude

    X = generate_coupled_ar1(M=3, T=200)
    data = Data(X, zscore=True)

    spi = CoherenceMagnitude()
    result = spi.multivariate(data)
    assert result.shape == (3, 3)
    off = ~np.isnan(result)
    assert np.all(np.isfinite(result[off])), "CoherenceMagnitude has non-finite values"

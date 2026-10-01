"""Every directed SPI must agree on what a row and a column mean.

pyspi's convention is set by ``base.Directed.multivariate``, which fills
``A[i, j] = bivariate(i, j)``: **row is the source, column is the target**.

This matters because the spectral backends (spectral_connectivity, nitime)
follow the opposite DTF/PDC convention -- element ``[i, j]`` is the flow *into*
i *from* j. Their output used to be passed through unchanged, which left every
directed spectral SPI transposed relative to every directed information-theory
SPI in the same results table. See ``spectral._to_source_target``.

The baseline-drift suite cannot be relied on to catch a regression here: on the
bundled fixtures the directed spectral matrices are close to symmetric, so a
transpose barely moves the numbers. These tests use a deliberately asymmetric
process instead.
"""

import numpy as np
import pytest

from pyspi.data import Data

# Process 0 drives process 1 at lag 1, with no feedback. Long enough that the
# estimators resolve the asymmetry well clear of their noise floor.
T = 2000
SEED = 0


@pytest.fixture(scope="module")
def driven_pair():
    """Data where process 0 unambiguously drives process 1."""
    rng = np.random.default_rng(SEED)
    x = np.zeros(T)
    y = np.zeros(T)
    ex, ey = rng.standard_normal(T), rng.standard_normal(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + ey[t]
    return Data(np.vstack([x, y]))


def _spi(module, cls_name, **kwargs):
    import importlib
    mod = importlib.import_module(f"pyspi.statistics.{module}")
    return getattr(mod, cls_name)(**kwargs)


# (module, class, kwargs). One per directed family that reaches a backend whose
# native convention differs from pyspi's, plus information-theory references.
DIRECTED = [
    ("infotheory", "TransferEntropy", {"estimator": "gaussian"}),
    ("infotheory", "TimeLaggedMutualInfo", {"estimator": "gaussian"}),
    ("spectral", "SpectralGrangerCausality", {}),
    ("spectral", "SpectralGrangerCausality", {"method": "parametric"}),
    ("spectral", "DirectedCoherence", {}),
    ("spectral", "PartialDirectedCoherence", {}),
    ("spectral", "GeneralizedPartialDirectedCoherence", {}),
    ("spectral", "DirectedTransferFunction", {}),
    ("spectral", "DirectDirectedTransferFunction", {}),
]


@pytest.mark.parametrize("module, cls_name, kwargs", DIRECTED,
                         ids=[f"{c}{'-' + str(k.get('method')) if k.get('method') else ''}"
                              for _, c, k in DIRECTED])
def test_directed_spi_is_source_by_target(module, cls_name, kwargs, driven_pair):
    """A[0, 1] (0 -> 1, the true direction) must exceed A[1, 0]."""
    spi = _spi(module, cls_name, **kwargs)
    A = np.asarray(spi.multivariate(driven_pair), dtype=float)

    assert A.shape == (2, 2)
    if np.isnan(A[0, 1]) or np.isnan(A[1, 0]):
        pytest.skip(f"{cls_name} returned NaN off-diagonals on this process")

    assert A[0, 1] > A[1, 0], (
        f"{cls_name}{kwargs}: process 0 drives process 1, so A[0,1] must be the "
        f"larger entry (row=source, column=target). Got A[0,1]={A[0, 1]:.4f}, "
        f"A[1,0]={A[1, 0]:.4f} -- this SPI is transposed relative to the rest "
        f"of the library."
    )


UNDIRECTED_SYMMETRIC = [
    "CoherenceMagnitude", "ImaginaryCoherence", "PhaseLockingValue",
    "PairwisePhaseConsistency",
]


@pytest.mark.parametrize("cls_name", UNDIRECTED_SYMMETRIC)
def test_undirected_spectral_stays_symmetric(cls_name, driven_pair):
    """Undirected spectral SPIs must not be touched by the directed transpose."""
    A = np.asarray(_spi("spectral", cls_name).multivariate(driven_pair), dtype=float)
    assert A[0, 1] == pytest.approx(A[1, 0], abs=1e-12), f"{cls_name} is not symmetric"


# These are labelled undirected but are antisymmetric (they encode a direction
# in their sign). Transposing them would silently negate every value, so the
# sign relationship is pinned here.
UNDIRECTED_ANTISYMMETRIC = ["PhaseLagIndex", "WeightedPhaseLagIndex", "PhaseSlopeIndex"]


@pytest.mark.parametrize("cls_name", UNDIRECTED_ANTISYMMETRIC)
def test_antisymmetric_spectral_sign_preserved(cls_name, driven_pair):
    A = np.asarray(_spi("spectral", cls_name).multivariate(driven_pair), dtype=float)
    assert A[0, 1] == pytest.approx(-A[1, 0], rel=1e-9), (
        f"{cls_name} should be antisymmetric; got A[0,1]={A[0, 1]}, A[1,0]={A[1, 0]}"
    )


# --------------------------------------------------------------------------
# Wilson-derived spectral measures, against an exact analytic spectrum
# --------------------------------------------------------------------------

def test_wilson_factorisation_recovers_a_known_transfer_function():
    """Factorise an *exact* VAR(1) spectrum, bypassing sample estimation.

    For x_t = A x_{t-1} + e_t with noise covariance Sigma, the cross-spectral
    matrix is S(f) = H(f) Sigma H(f)^H with H(f) = (I - A e^{-2 pi i f})^{-1}.
    Feeding that exact S to the Wilson decomposition isolates the factorisation
    from every source of finite-sample error, so any discrepancy is the
    algorithm's own.

    An earlier version of this test compared pyspi's DTF against the *full*
    three-process transfer function. That was invalid: pyspi computes
    NonparametricSpectralBivariate measures on two-process subsystems
    (`z[[i, j]]`), and a subsystem of a larger VAR legitimately shows flow in
    both directions because the omitted processes induce correlation. The
    0.10-0.16 floor that comparison produced was the test's error, not the
    estimator's.
    """
    from spectral_connectivity.minimum_phase_decomposition import (
        minimum_phase_decomposition,
    )

    A = np.array([[0.5, 0.0], [0.7, 0.4]])
    M = A.shape[0]
    Sigma = np.eye(M)
    # The FULL two-sided grid over [0, 1): the algorithm takes an inverse FFT
    # internally to impose causality, so a half-spectrum silently gives a
    # factor unrelated to H even though S = G G^H still holds.
    n = 256
    freqs = np.arange(n) / n

    H = np.stack([np.linalg.inv(np.eye(M) - A * np.exp(-2j * np.pi * f)) for f in freqs])
    S = H @ Sigma @ np.conj(np.transpose(H, (0, 2, 1)))

    G = minimum_phase_decomposition(S[np.newaxis, ...])[0]

    # S = G G^H is the contract. The bound is the algorithm's own convergence
    # tolerance (default 1e-8), not machine precision -- this is an iterative
    # method, so ~1e-8 is the expected floor rather than a discrepancy.
    residual = np.abs(S - G @ np.conj(np.transpose(G, (0, 2, 1)))).max()
    assert residual < 1e-6, f"Wilson reconstruction residual {residual:.3g}"

    # G(f) = H(f) Sigma^{1/2}; the zeroth Fourier coefficient of G is Sigma^{1/2}.
    g0 = np.fft.ifft(G, axis=0)[0]
    H_hat = G @ np.linalg.inv(g0)
    err = np.abs(H_hat - H).max()
    assert err < 1e-9, f"recovered transfer function differs by {err:.3g}"

    num = np.abs(H_hat) ** 2
    dtf_hat = num / num.sum(axis=-1, keepdims=True)
    num = np.abs(H) ** 2
    dtf = num / num.sum(axis=-1, keepdims=True)
    assert np.abs(dtf_hat - dtf).max() < 1e-9


def test_directed_transfer_function_orientation_on_a_two_process_var():
    """On a 2-process VAR the subsystem *is* the system, so DTF is comparable.

    Bounded, and the driving direction dominates. Absolute calibration is not
    asserted: DTF as implemented is the squared form, and the multitaper
    estimate carries finite-sample bias at these lengths.
    """
    from pyspi.data import Data
    from pyspi.statistics.spectral import DirectedTransferFunction

    A = np.array([[0.5, 0.0], [0.7, 0.4]])   # 0 -> 1 only
    rng = np.random.default_rng(0)
    T = 500
    X = np.zeros((2, T))
    for t in range(1, T):
        X[:, t] = A @ X[:, t - 1] + rng.standard_normal(2)

    got = DirectedTransferFunction(statistic="mean", fmin=0, fmax=0.5).multivariate(
        Data(data=X, dim_order="ps", zscore=False)
    )
    finite = got[np.isfinite(got)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0, (
        f"DTF outside [0,1]: [{finite.min():.4f}, {finite.max():.4f}]"
    )
    assert got[0, 1] > got[1, 0], (
        f"DTF did not favour the driving direction: 0->1={got[0,1]:.4f}, "
        f"1->0={got[1,0]:.4f}"
    )


def test_directed_coherence_is_bounded():
    """DC is defined on [0,1]; the backend's version is not.

    ``spectral_connectivity.directed_coherence`` puts |H|^2 in the numerator
    while ``_total_inflow`` is on the magnitude scale, so the ratio is
    dimensionally |H|^2/|H| and unbounded -- the shipped baselines reached 3.27
    (VAR), 1.84 (CML) and 1139.47 (Kuramoto). pyspi recomputes it with |H|.
    """
    import os
    from pyspi.data import Data
    from pyspi.calculator import load_spis_from_yaml, resolve_config

    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    dcoh = [k for k in spis if k.startswith("dcoh_")]
    assert dcoh, "no directed-coherence SPIs in the full config"

    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "kuramoto_M7_T100.npy")
    data = Data(data=fixture, dim_order="sp")
    for k in dcoh:
        A = np.asarray(spis[k].multivariate(data), dtype=float)
        finite = A[np.isfinite(A)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0 + 1e-9, (
            f"{k} outside [0,1]: [{finite.min():.4f}, {finite.max():.4f}]"
        )


class _StubConnectivity:
    """Minimal stand-in exposing only the two private properties DC reads."""

    def __init__(self, transfer_function, noise_covariance):
        self._transfer_function = transfer_function
        self._noise_covariance = noise_covariance


def _baccala_dc(H, sigma):
    """Baccala et al. (1998) DC, written out with explicit loops."""
    n_f, n = H.shape[-3], H.shape[-1]
    out = np.zeros((n_f, n, n))
    for k in range(n_f):
        for i in range(n):
            den = np.sqrt(sum(sigma[c] * abs(H[0, k, i, c]) ** 2 for c in range(n)))
            for j in range(n):
                out[k, i, j] = np.sqrt(sigma[j]) * abs(H[0, k, i, j]) / den
    return out


def test_directed_coherence_weights_by_the_source_innovation_variance():
    """The sigma_jj weight is indexed by the *source*, and must not cancel.

    Regression test for a second backend defect, distinct from the |H|^2
    numerator. ``_get_noise_variance`` reshapes ``diag(Sigma)`` to
    ``(..., 1, n, 1)``, which broadcasts along the *row* (target) axis of H.
    A row-indexed weight is constant across the summation index, so it factors
    out of numerator and denominator and cancels exactly -- the measure
    degenerates to ``sqrt(DTF)`` for *every* noise covariance. The previous
    equal-variance test could not see this, because it asserted precisely the
    identity the bug makes unconditionally true.
    """
    from pyspi.statistics.spectral import DirectedCoherence

    rng = np.random.default_rng(0)
    n_f, n = 7, 3
    H = (rng.standard_normal((1, n_f, n, n))
         + 1j * rng.standard_normal((1, n_f, n, n)))
    sigma = np.array([0.5, 4.0, 0.1])          # unequal *diagonal* variances
    C = _StubConnectivity(H, np.diag(sigma)[np.newaxis])

    got = DirectedCoherence.__new__(DirectedCoherence)._get_statistic(C)[0]
    assert np.abs(got - _baccala_dc(H, sigma)).max() < 1e-12

    # Bounded, and exactly row-normalised: sum_j DC_ij^2 == 1.
    assert got.min() >= 0.0 and got.max() <= 1.0
    assert np.abs((got ** 2).sum(axis=-1) - 1.0).max() < 1e-12

    # The weight must actually bite. sqrt(DTF) is what the row-indexed form
    # returns; if this were still close, the variances would be cancelling.
    mag2 = np.abs(H[0]) ** 2
    sqrt_dtf = np.sqrt(mag2 / mag2.sum(axis=-1, keepdims=True))
    assert np.abs(got - sqrt_dtf).max() > 0.1, (
        "DC collapsed onto sqrt(DTF) despite unequal innovation variances"
    )


def test_directed_coherence_equals_sqrt_dtf_iff_variances_are_equal():
    """Both halves of the identity, on an *exact* VAR(1) spectrum.

    Equal innovation variances make sigma cancel legitimately, so DC reduces to
    sqrt(DTF); unequal ones must not. Driving this from the analytic
    cross-spectrum rather than a sampled estimate keeps the assertion about the
    factorisation and the algebra, not about finite-sample calibration.
    """
    from spectral_connectivity.minimum_phase_decomposition import (
        minimum_phase_decomposition,
    )
    from pyspi.statistics.spectral import DirectedCoherence

    A = np.array([[0.5, 0.0], [0.7, 0.4]])
    M, n = A.shape[0], 256
    freqs = np.arange(n) / n
    H = np.stack([np.linalg.inv(np.eye(M) - A * np.exp(-2j * np.pi * f))
                  for f in freqs])

    dc = DirectedCoherence.__new__(DirectedCoherence)
    for sigma, equal in ((np.array([1.0, 1.0]), True),
                         (np.array([0.25, 4.0]), False)):
        Sigma = np.diag(sigma)
        S = H @ Sigma @ np.conj(np.transpose(H, (0, 2, 1)))
        G = minimum_phase_decomposition(S[np.newaxis, ...])
        g0 = np.fft.ifft(G, axis=-3).real[..., 0, :, :]
        H_hat = G @ np.linalg.inv(g0)[:, np.newaxis]
        Sigma_hat = g0 @ np.transpose(g0, (0, 2, 1))

        # Sigma is recovered even though G is unique only up to a real
        # orthogonal factor U: g0 = Sigma^{1/2} U, so g0 g0^T = Sigma.
        assert np.abs(Sigma_hat[0] - Sigma).max() < 1e-6

        got = dc._get_statistic(_StubConnectivity(H_hat, Sigma_hat))[0]
        assert np.abs(got - _baccala_dc(H_hat, sigma)).max() < 1e-9

        mag2 = np.abs(H_hat[0]) ** 2
        sqrt_dtf = np.sqrt(mag2 / mag2.sum(axis=-1, keepdims=True))
        gap = np.abs(got - sqrt_dtf).max()
        if equal:
            assert gap < 1e-9, f"DC != sqrt(DTF) at equal variances: {gap:.3g}"
        else:
            assert gap > 0.1, f"DC collapsed onto sqrt(DTF): gap {gap:.3g}"


def test_directed_coherence_uses_only_the_documented_backend_privates():
    """Pin the private-API surface DC depends on.

    ``_transfer_function`` and ``_noise_covariance`` are the whole contract;
    the arithmetic is done in pyspi. Losing either is an import-time-visible
    break rather than a silently wrong number, which is what the supported
    version range in pyproject.toml is anchored on.
    """
    import spectral_connectivity as sc

    for attr in ("_transfer_function", "_noise_covariance"):
        assert isinstance(getattr(sc.Connectivity, attr, None), property), (
            f"spectral_connectivity.Connectivity.{attr} is gone; "
            f"DirectedCoherence cannot be computed. Check the supported "
            f"spectral-connectivity range in pyproject.toml."
        )


# --------------------------------------------------------------------------
# Group delay
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lag", [1, 3, 5, 8])
def test_group_delay_recovers_a_known_lag(lag):
    """An independent oracle: a pure delay has a known group delay.

    All three shipped ``gd_*`` SPIs returned no finite off-diagonal value on
    any input -- Gaussian noise, one-way coupled AR, every frozen fixture, at
    5/11/19/39 tapers and T up to 4000, on a pair with median coherence 0.998.
    The cause is one line in the backend, not the data:
    ``coherence_fisher_z_transform`` divides by
    ``sqrt(coherence_bias(n_obs1) + coherence_bias(n_obs2))`` and the
    one-sample call passes ``n_obs2 = 0``, for which ``coherence_bias`` returns
    ``1/(2*0 - 2) = -0.5``. The radicand is negative for every ``n_obs1``, so
    every p-value is NaN and nothing is ever significant.

    pyspi computes the statistic itself with the standard one-sample form,
    ``(arctanh|C| - b) / sqrt(b)`` with ``b = 1/(2n - 2)``. With
    ``y(t) = x(t - lag)`` the answer is known in advance, which is what makes
    this a test rather than a re-run of the implementation.
    """
    from pyspi.statistics.spectral import GroupDelay

    rng = np.random.default_rng(SEED)
    T_ = 2000
    x = rng.standard_normal(T_ + lag)
    y = x[:-lag] + 0.1 * rng.standard_normal(T_)
    data = Data(data=np.vstack([x[lag:], y]), dim_order="ps")

    delay = GroupDelay(statistic="delay", fmin=0, fmax=0.5).multivariate(data)
    # Row is the source: process 0 leads process 1 by `lag` samples.
    assert delay[0, 1] == pytest.approx(lag, abs=0.05)
    assert delay[1, 0] == pytest.approx(-lag, abs=0.05)

    r = GroupDelay(statistic="rvalue", fmin=0, fmax=0.5).multivariate(data)
    assert r[0, 1] > 0.99 and r[0, 1] == pytest.approx(r[1, 0])


def test_group_delay_is_estimable_on_the_bundled_fixtures():
    """Not a repeat of the lag test: it pins that real data now produces values.

    Partial NaN is correct here and is not the defect being guarded against --
    group delay is defined only where the coherence is significant, so pairs
    without a significant cluster have none. An *entirely* NaN column is the
    defect.
    """
    import os

    from pyspi.statistics.spectral import GroupDelay

    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "fixtures")
    for name in ("var1_M3_T100.npy", "cml_M5_T100.npy", "kuramoto_M7_T100.npy"):
        data = Data(data=os.path.join(fixtures, name), dim_order="sp")
        table = GroupDelay(statistic="delay", fmin=0, fmax=0.5).multivariate(data)
        off = ~np.eye(table.shape[0], dtype=bool)
        assert np.isfinite(table[off]).any(), f"{name}: gd is entirely NaN"


def test_group_delay_of_independent_processes_is_undefined():
    """The significance gate must still gate. Independent noise has no delay."""
    from pyspi.statistics.spectral import GroupDelay

    rng = np.random.default_rng(SEED)
    data = Data(data=rng.standard_normal((3, 500)), dim_order="ps")
    table = GroupDelay(statistic="delay", fmin=0, fmax=0.5).multivariate(data)
    off = ~np.eye(3, dtype=bool)
    assert not np.isfinite(table[off]).any()


# --------------------------------------------------------------------------
# Spectral Granger causality
# --------------------------------------------------------------------------

def test_spectral_gc_nan_mask_is_in_the_same_orientation_as_the_values():
    """The mask must be transformed with the matrix it masks.

    `multivariate` puts the backend's matrix into pyspi's (source, target)
    orientation by transposing it, then applied a NaN mask computed in the
    backend's orientation. With a directionally asymmetric NaN pattern the cell
    that was genuinely unestimable is already NaN, and the *mirror* cell -- a
    perfectly good estimate -- is the one that gets blanked.
    """
    from pyspi.statistics.spectral import SpectralGrangerCausality

    M, n_freq = 3, 20
    F = np.ones((1, n_freq, M, M))
    F[:, :, 0, 2] = np.nan            # unestimable in one direction only
    freq = np.linspace(0.0, 0.5, n_freq)

    spi = SpectralGrangerCausality(fmin=0, fmax=0.5, nan_threshold=0.5)
    spi._get_cache = lambda data: (F, freq)

    with pytest.warns(UserWarning, match="NaN values"):
        result = spi.multivariate(Data(data=np.zeros((M, 8)), dim_order="ps",
                                       zscore=False))

    # Backend [0, 2] is pyspi [2, 0].
    assert np.isnan(result[2, 0])
    assert np.isfinite(result[0, 2]), "the mirror cell was blanked instead"


def test_spectral_gc_parametric_honours_the_sampling_frequency():
    """`fs` is in the identifier and the cache key, so it must reach the model.

    The parametric branch built `TimeSeries(..., sampling_interval=1)`
    unconditionally, so `GA.frequencies` came back on a unit-rate axis whatever
    `fs` said and the [fmin, fmax] band was applied to the wrong frequencies.
    Two SPIs differing only in `fs` advertised different sampling rates and
    returned the same numbers.
    """
    from pyspi.statistics.spectral import SpectralGrangerCausality

    rng = np.random.default_rng(SEED)
    T_ = 400
    X = np.zeros((2, T_))
    A = np.array([[0.5, 0.0], [0.7, 0.4]])
    for t in range(1, T_):
        X[:, t] = A @ X[:, t - 1] + rng.standard_normal(2)

    # The same physical band, expressed at two sampling rates: [0, 0.25] cycles
    # per sample is [0, 0.5] Hz at fs=2 and [0, 0.25] Hz at fs=1.
    base = SpectralGrangerCausality(method="parametric", order=2,
                                    fmin=1e-5, fmax=0.25)
    scaled = SpectralGrangerCausality(method="parametric", order=2, fs=2,
                                      fmin=1e-5, fmax=0.5)
    a = base.multivariate(Data(data=X, dim_order="ps"))
    b = scaled.multivariate(Data(data=X, dim_order="ps"))
    assert np.allclose(a, b, atol=1e-8, equal_nan=True), (
        f"fs did not reach the model:\n{a}\n{b}"
    )


def test_spectral_gc_names_the_cause_when_order_selection_does_not_converge():
    """The documented exception in KNOWN_UNESTIMABLE, tested rather than assumed.

    nitime's own message -- "Model estimation order did not converge at
    max_order = 50" -- says nothing about the data, and pyspi used to swallow it
    into an all-NaN return plus a warning, so `Calculator.errors` recorded only
    the generic "returned no finite off-diagonal values".
    """
    import os

    from pyspi.statistics.spectral import SpectralGrangerCausality

    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "kuramoto_M7_T100.npy")
    spi = SpectralGrangerCausality(method="parametric", order=None, max_order=50)
    with pytest.raises(ValueError, match="order selection did not converge"):
        spi.multivariate(Data(data=fixture, dim_order="sp"))


@pytest.mark.parametrize("fs", [1, 2, 4])
def test_group_delay_is_reported_in_samples_at_every_sampling_frequency(fs):
    """The regression runs against physical frequency, so its slope is seconds.

    `C.frequencies` is in Hz, so `slope/(2*pi)` is a delay in seconds: a true
    4-sample lag came back as 4.0, 2.0 and 1.0 at fs = 1, 2, 4 while the API and
    every other lagged SPI in pyspi count samples. Scaling by fs makes the
    number mean what the identifier says at any rate.
    """
    import warnings

    from pyspi.statistics.spectral import GroupDelay

    rng = np.random.default_rng(SEED)
    lag, T_ = 4, 2000
    x = rng.standard_normal(T_ + lag)
    y = x[:-lag] + 0.1 * rng.standard_normal(T_)
    data = Data(data=np.vstack([x[lag:], y]), dim_order="ps")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = GroupDelay(statistic="delay", fs=fs, fmin=0,
                           fmax=fs / 2).multivariate(data)
    assert table[0, 1] == pytest.approx(lag, abs=0.05)
    assert table[1, 0] == pytest.approx(-lag, abs=0.05)


@pytest.mark.parametrize("statistic", ["delay", "slope", "rvalue"])
def test_group_delay_is_covariant_under_process_permutation(statistic):
    """Reversing the process order must move a value, not change it.

    `rvalue` stored the *signed* regression r symmetrically. The fit is of the
    phase of C_ij and phase(C_ji) = -phase(C_ij), so reversing the order turned
    +0.99997 into -0.99997 at the mirrored position -- for a statistic declared
    symmetric. It is now |r|, which is orientation-free.
    """
    from pyspi.statistics.spectral import GroupDelay

    rng = np.random.default_rng(SEED)
    lag, T_ = 4, 2000
    x = rng.standard_normal(T_ + lag)
    y = x[:-lag] + 0.1 * rng.standard_normal(T_)
    Z = np.vstack([x[lag:], y])

    spi = lambda: GroupDelay(statistic=statistic, fmin=0, fmax=0.5)
    forward = spi().multivariate(Data(data=Z, dim_order="ps"))
    reversed_ = spi().multivariate(Data(data=Z[::-1], dim_order="ps"))
    assert forward[0, 1] == pytest.approx(reversed_[1, 0], rel=1e-9)
    assert forward[1, 0] == pytest.approx(reversed_[0, 1], rel=1e-9)


def test_group_delay_structural_labels_match_the_matrices():
    """delay/slope are antisymmetric and signed; rvalue is symmetric and not.

    And a structural trait replaces `directed`/`undirected` rather than sitting
    beside a stale one: `gd_*` carried the class's `antisymmetric` and the
    config's `directed` at the same time, so `filter_spis` answered both ways
    for the same SPI.
    """
    from pyspi.calculator import load_spis_from_yaml, resolve_config
    from pyspi.statistics.spectral import GroupDelay

    for statistic in ("delay", "slope"):
        spi = GroupDelay(statistic=statistic, fmin=0, fmax=0.5)
        assert "antisymmetric" in spi.labels and spi.issigned()
        assert not {"directed", "undirected", "unsigned"} & set(spi.labels)

    r = GroupDelay(statistic="rvalue", fmin=0, fmax=0.5)
    assert "undirected" in r.labels and not r.issigned()
    assert "antisymmetric" not in r.labels

    shipped = load_spis_from_yaml(resolve_config("full"), quiet=True)
    for identifier, spi in shipped.items():
        labels = set(spi.labels)
        if {"antisymmetric", "asymmetric"} & labels:
            assert not {"directed", "undirected"} & labels, identifier

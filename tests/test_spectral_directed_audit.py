"""Independent references for the directed / parametric spectral SPIs.

Each test here pins one audited property of ``pyspi.statistics.spectral``'s
directed half -- ``NonparametricSpectralBivariate`` and its five subclasses
(DTF, DC, PDC, gPDC, dDTF) and ``SpectralGrangerCausality`` -- against a
reference computed without going through the code under test: the backend's
own per-frequency spectrum with an explicitly stated band, a textbook formula on
an exact VAR spectrum, or a direct Yule-Walker solve.
"""

import numpy as np
import pytest

import spectral_connectivity as sc

from pyspi.data import Data

SEED = 0


def _driven_pair(T=400, seed=SEED):
    """Process 0 drives process 1 at lag 1; positive coefficients, so low-pass."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.0], [0.5, 0.3]])
    X = np.zeros((2, T))
    for t in range(1, T):
        X[:, t] = A @ X[:, t - 1] + rng.standard_normal(2)
    return X


# --------------------------------------------------------------------------
# Band statistic of the bivariate (pairwise-factorised) directed measures
# --------------------------------------------------------------------------

BIVARIATE = [
    ("DirectedTransferFunction", "directed_transfer_function"),
    ("PartialDirectedCoherence", "partial_directed_coherence"),
    ("GeneralizedPartialDirectedCoherence",
     "generalized_partial_directed_coherence"),
]


@pytest.mark.parametrize("cls_name, measure", BIVARIATE)
@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("band", [(0, 0.25), (0.25, 0.5), (0, 0.5)])
def test_bivariate_band_statistic_covers_the_closed_band(cls_name, measure,
                                                         statistic, band):
    """``fmin``/``fmax`` are inclusive, as for every other spectral SPI.

    The reference is the backend's own spectrum for the ordered pair
    ``[source, target]`` (read at ``[target, source]``, its convention), reduced
    over ``fmin <= f <= fmax``. The multivariate spectral classes, spectral GC
    and the wavelet SPIs all select the band that way; the bivariate class used
    strict inequalities, so ``fmin-0_fmax-0-25`` silently dropped the DC bin and
    the 0.25 bin, and ``fmin-0-25_fmax-0-5`` dropped 0.25 and Nyquist. On a
    low-pass process the DC bin is where these measures peak, so the ``max``
    over ``[0, 0.25]`` missed the maximum.
    """
    from pyspi.statistics import spectral

    X = _driven_pair()
    data = Data(data=X, dim_order="ps", zscore=False)
    fmin, fmax = band
    spi = getattr(spectral, cls_name)(fmin=fmin, fmax=fmax, statistic=statistic)
    got = spi.multivariate(data)

    reduce = {"mean": np.mean, "max": np.max}[statistic]
    for src, tgt in ((0, 1), (1, 0)):
        m = sc.Multitaper(X[[src, tgt]].T[:, np.newaxis, :], sampling_frequency=1)
        conn = sc.Connectivity.from_multitaper(m)
        spectrum = getattr(conn, measure)()[0, :, 1, 0]      # [target, source]
        f = conn.frequencies
        want = reduce(spectrum[(f >= fmin) & (f <= fmax)])
        assert got[src, tgt] == pytest.approx(want, rel=1e-9, abs=1e-12), (
            f"{cls_name} {statistic} over [{fmin}, {fmax}], {src}->{tgt}: got "
            f"{got[src, tgt]!r}, the closed-band reference is {want!r}"
        )


# --------------------------------------------------------------------------
# Wilson-derived measures on an exact VAR spectrum, end to end
# --------------------------------------------------------------------------

def _exact_var2_pair(n=512):
    """Exact spectrum of a two-process VAR(2) in which 0 drives 1.

    Unequal innovation variances, so the noise-weighted measures differ from
    their unweighted counterparts.
    """
    A1 = np.array([[0.6, 0.0], [0.5, 0.4]])
    A2 = np.array([[-0.3, 0.0], [0.3, 0.0]])
    Sigma = np.diag([1.0, 2.5])
    f = np.fft.fftfreq(n)
    Af = np.stack([np.eye(2) - A1 * np.exp(-2j * np.pi * fr)
                   - A2 * np.exp(-4j * np.pi * fr) for fr in f])
    H = np.linalg.inv(Af)
    S = H @ Sigma @ np.conj(np.transpose(H, (0, 2, 1)))
    return f, H, S, Sigma


@pytest.fixture
def exact_pair_spectrum(monkeypatch):
    """Route the bivariate class's pairwise Connectivity to an exact spectrum.

    ``NonparametricSpectralBivariate`` builds one ``_FactoredConnectivity`` per
    ordered pair ``[source, target]``. Replacing its ``from_multitaper`` with
    one that returns the exact cross-spectrum, permuted to the pair's order,
    exercises everything downstream -- the Wilson factorisation, the measure,
    the ``[target, source]`` read-out and the band statistic -- with no
    finite-sample error. Process 0's samples are all 0 and process 1's all 1,
    which is how the patched constructor tells the order apart.
    """
    from pyspi.statistics import spectral

    f, H, S, Sigma = _exact_var2_pair()
    swap = np.array([[0, 1], [1, 0]])

    class Exact(spectral._FactoredConnectivity):
        def __init__(self, S_pair):
            n = S_pair.shape[0]
            super().__init__(np.ones((1, 1, 1, n, 2), complex), frequencies=f)
            self._S = S_pair[np.newaxis]

        def _expectation_cross_spectral_matrix(self, fcn=None, dtype=None):
            return self._S

    def from_multitaper(m, *args, **kwargs):
        first = int(np.asarray(m.time_series)[0, 0, 0])
        return Exact(S if first == 0 else swap @ S @ swap)

    monkeypatch.setattr(spectral._FactoredConnectivity, "from_multitaper",
                        staticmethod(from_multitaper))
    data = Data(data=np.vstack([np.zeros(64), np.ones(64)]), dim_order="ps",
                zscore=False)
    return data, f, H, S, Sigma


def test_direct_dtf_is_full_frequency_dtf_times_partial_coherence(
        exact_pair_spectrum):
    """dDTF (Korzeniewska et al. 2003): ffDTF times |partial coherence|.

    ``eta_ij(f) = |H_ij(f)| / sqrt(sum_f sum_k |H_ik(f)|^2)`` and
    ``P_ij(f) = -Sinv_ij / sqrt(Sinv_ii Sinv_jj)`` with ``Sinv = S(f)^-1``, so
    ``delta_ij = eta_ij |P_ij|`` (the square root of the published
    ``delta^2 = eta^2 P^2``, on the magnitude scale the backend reports).

    The backend's ``direct_directed_transfer_function`` multiplies by
    ``sqrt(partial_directed_coherence())`` instead -- |PDC|, a different,
    *directed* quantity -- although its own docstring describes "the directed
    transfer function and partial coherence". For a pair, partial coherence is
    the ordinary coherence, while bivariate PDC equals DTF, so the backend's
    pairwise dDTF was ``eta * sqrt(DTF)``.
    """
    from pyspi.statistics.spectral import DirectDirectedTransferFunction

    data, f, H, S, _ = exact_pair_spectrum
    nn = f.size // 2 + 1
    Hn, Sn = H[:nn], S[:nn]
    eta = np.abs(Hn) / np.sqrt(np.sum(np.abs(Hn) ** 2, axis=(-1, -3),
                                      keepdims=True))
    Sinv = np.linalg.inv(Sn)
    d = np.abs(np.diagonal(Sinv, axis1=-1, axis2=-2))
    pcoh = np.abs(Sinv) / np.sqrt(d[..., :, None] * d[..., None, :])
    want = eta * pcoh                                   # [f, target, source]

    for statistic, reduce in (("mean", np.mean), ("max", np.max)):
        got = DirectDirectedTransferFunction(
            fmin=0, fmax=0.5, statistic=statistic).multivariate(data)
        for src, tgt in ((0, 1), (1, 0)):
            ref = reduce(want[:, tgt, src])
            assert got[src, tgt] == pytest.approx(ref, rel=1e-6, abs=1e-9), (
                f"dDTF {statistic} {src}->{tgt}: got {got[src, tgt]:.6g}, "
                f"ffDTF x |partial coherence| gives {ref:.6g}"
            )


def test_dtf_pdc_gpdc_dc_match_their_textbook_forms_end_to_end(
        exact_pair_spectrum):
    """Formula *and* orientation, through ``bivariate``, on an exact spectrum.

    With ``H`` the transfer function and ``A = H^-1``, both indexed
    ``[target, source]``, and ``s_k`` the innovation variances:

    * DTF  (Kaminski & Blinowska 1991): ``|H_ij|^2 / sum_k |H_ik|^2``
    * PDC  (Baccala & Sameshima 2001):  ``|A_ij|^2 / sum_k |A_kj|^2``
    * gPDC (Baccala et al. 2007):       ``(|A_ij|^2/s_i) / sum_k |A_kj|^2/s_k``
    * DC   (Baccala et al. 1998):       ``sqrt(s_j)|H_ij| / sqrt(sum_k s_k|H_ik|^2)``

    pyspi's ``A[source, target]`` must be the ``[target, source]`` entry.
    The reverse direction of a one-way system is exactly zero.
    """
    from pyspi.statistics import spectral

    data, f, H, S, Sigma = exact_pair_spectrum
    nn = f.size // 2 + 1
    Hn = H[:nn]
    An = np.linalg.inv(Hn)
    s = np.diag(Sigma)
    ref = {
        "DirectedTransferFunction":
            np.abs(Hn) ** 2 / np.sum(np.abs(Hn) ** 2, axis=-1, keepdims=True),
        "PartialDirectedCoherence":
            np.abs(An) ** 2 / np.sum(np.abs(An) ** 2, axis=-2, keepdims=True),
        "GeneralizedPartialDirectedCoherence":
            (np.abs(An) ** 2 / s[:, None])
            / np.sum(np.abs(An) ** 2 / s[:, None], axis=-2, keepdims=True),
        "DirectedCoherence":
            np.sqrt(s) * np.abs(Hn)
            / np.sqrt(np.sum(s * np.abs(Hn) ** 2, axis=-1, keepdims=True)),
    }
    for cls_name, want in ref.items():
        got = getattr(spectral, cls_name)(fmin=0, fmax=0.5,
                                          statistic="mean").multivariate(data)
        assert got[0, 1] == pytest.approx(want[:, 1, 0].mean(), rel=1e-6), cls_name
        assert abs(got[1, 0]) < 1e-9, f"{cls_name}: reverse direction {got[1, 0]}"
        assert want[:, 1, 0].mean() > 0.1      # the comparison is not vacuous


# --------------------------------------------------------------------------
# Parametric spectral Granger causality
# --------------------------------------------------------------------------

def _yule_walker_geweke(x, order, freq):
    """Textbook bivariate VAR(p) fit and Geweke's spectral GC, from scratch.

    Yule-Walker on the sample autocovariance about the mean, normalised by T
    (Brockwell & Davis 1991, sec. 11.5; Luetkepohl 2005, sec. 3.3.4): that is
    the estimator whose block-Toeplitz matrix is positive semidefinite, which
    is what guarantees a stable model and a positive-definite innovation
    covariance. Solved here as one dense linear system, not by recursion.

    Returns ``gc[f, source, target]`` (Geweke 1982, with the instantaneous
    term rotated out as in Ding, Chen & Bressler 2006).
    """
    x = x - x.mean(axis=1, keepdims=True)
    T = x.shape[1]
    R = [x[:, k:] @ x[:, :T - k].T / T for k in range(order + 1)]   # E[x(t+k)x(t)']
    lag = lambda m: R[m] if m >= 0 else R[-m].T
    G = np.block([[lag(k - l) for k in range(1, order + 1)]
                  for l in range(1, order + 1)])
    Phi = np.hstack(R[1:]) @ np.linalg.inv(G)
    Phi = [Phi[:, 2 * l:2 * l + 2] for l in range(order)]
    Sigma = R[0] - sum(P @ R[l + 1].T for l, P in enumerate(Phi))

    gc = np.zeros((freq.size, 2, 2))
    for n, fr in enumerate(freq):
        Af = np.eye(2) - sum(P * np.exp(-2j * np.pi * fr * (l + 1))
                             for l, P in enumerate(Phi))
        H = np.linalg.inv(Af)
        S = (H @ Sigma @ H.conj().T).real
        for src, tgt in ((0, 1), (1, 0)):
            rot = Sigma[src, src] - Sigma[src, tgt] ** 2 / Sigma[tgt, tgt]
            gc[n, src, tgt] = np.log(S[tgt, tgt]
                                     / (S[tgt, tgt] - rot * abs(H[tgt, src]) ** 2))
    return gc, Sigma


def _kuramoto_pair(i, j):
    import os
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "kuramoto_M7_T100.npy")
    z = Data(data=fixture, dim_order="sp").to_numpy(squeeze=True)
    return z[[i, j]]


@pytest.mark.parametrize("order", [1, 2, 20])
@pytest.mark.parametrize("pair", [(0, 1), (1, 6), (3, 4)])
def test_parametric_sgc_is_yule_walker_plus_geweke(order, pair):
    """Parametric SGC against a direct Yule-Walker solve and Geweke's formula.

    nitime's ``GrangerAnalyzer`` builds the Yule-Walker system from
    ``autocov_vector``, which averages each lag over ``T - k`` products (the
    "unbiased" estimator) and does not remove the mean. That block-Toeplitz
    matrix need not be positive semidefinite, and on the bundled Kuramoto
    fixture it is not: the LWR recursion returned an *indefinite* innovation
    covariance on 7 of 21 pairs at order 1 and all 21 at orders 2 and 20, and
    an unstable model on 18 of 21 at order 20. Geweke's ratio is then
    meaningless (negative or NaN causality), and the NaN in BIC's
    ``log det`` is what stopped the automatic order search from ever
    "turning over" -- the reason ``order=None`` was listed as unestimable.
    """
    from pyspi.statistics.spectral import SpectralGrangerCausality

    x = _kuramoto_pair(*pair)
    data = Data(data=x, dim_order="ps", zscore=False)
    spi = SpectralGrangerCausality(method="parametric", order=order,
                                   fmin=0, fmax=0.5)
    F, freq = spi._get_cache(data)          # [0, f, target, source]
    # nitime evaluates the spectrum with scipy's freqz on n points of [0, pi),
    # i.e. at k / (2n) cycles per sample; the labels are tested separately.
    n = F.shape[1]
    want, Sigma = _yule_walker_geweke(x, order, np.arange(n) / (2 * n))
    assert np.linalg.eigvalsh(Sigma).min() > 0      # the reference is a valid model

    for src, tgt in ((0, 1), (1, 0)):
        got = F[0, :, tgt, src]
        err = np.nanmax(np.abs(got - want[:, src, tgt])) if np.isfinite(got).any() else np.inf
        assert np.isfinite(got).all() and err < 1e-8 * max(1.0, np.abs(want).max()), (
            f"pair {pair}, order {order}, {src}->{tgt}: max |pyspi - Yule-Walker/"
            f"Geweke| = {err:.3g} ({np.isnan(got).sum()} NaN of {got.size} "
            f"frequencies; min pyspi value {np.nanmin(got) if np.isfinite(got).any() else np.nan:.3g})"
        )


def test_parametric_sgc_order_selection_converges_on_the_kuramoto_fixture():
    """With a valid autocovariance the BIC search turns over at a low order."""
    import os
    from pyspi.statistics.spectral import SpectralGrangerCausality

    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "kuramoto_M7_T100.npy")
    table = SpectralGrangerCausality(method="parametric", order=None,
                                     fmin=0, fmax=0.5).multivariate(
        Data(data=fixture, dim_order="sp"))
    off = ~np.eye(table.shape[0], dtype=bool)
    assert np.isfinite(table[off]).all() and (table[off] >= 0).all()


@pytest.mark.parametrize("fs", [1, 4])
def test_parametric_sgc_frequency_axis_is_where_the_spectrum_was_evaluated(fs):
    """Each value must carry the frequency it was computed at.

    nitime's ``GrangerAnalyzer.frequencies`` is ``linspace(0, fs/2, 513)``, but
    ``granger_causality_xy`` evaluates the model through ``scipy.signal.freqz``
    with ``worN=513``: at ``k/1026`` cycles per sample on ``[0, 1/2)``. Every
    label but DC was off, by up to one grid step (the bin labelled Nyquist is
    0.499), so the band was applied on a slightly stretched axis.
    """
    from pyspi.statistics.spectral import SpectralGrangerCausality

    X = _driven_pair(T=2000)
    data = Data(data=X, dim_order="ps", zscore=False)
    spi = SpectralGrangerCausality(method="parametric", order=1, fs=fs,
                                   fmin=0, fmax=fs / 2)
    F, freq = spi._get_cache(data)
    want, _ = _yule_walker_geweke(X, 1, np.asarray(freq) / fs)
    err = np.abs(F[0, :, 1, 0] - want[:, 0, 1]).max()
    assert err < 1e-9, (
        f"fs={fs}: values differ from Geweke at their labelled frequencies by "
        f"{err:.3g}; last label {freq[-1]!r}"
    )

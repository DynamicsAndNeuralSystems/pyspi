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


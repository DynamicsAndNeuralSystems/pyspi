import logging
import numpy as np
from contextlib import contextmanager
from copy import deepcopy

import spectral_connectivity as sc  # For directed spectral statistics (excl. spectral GC)
from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
import nitime.analysis as nta
import nitime.timeseries as ts
import warnings
from pyspi.utils import fmt_param

try:
    from spectral_connectivity.transforms import prepare_time_series
except ImportError:
    def prepare_time_series(time_series, axis="signals"):
        if axis != "signals":
            raise ImportError(
                "spectral_connectivity is missing prepare_time_series; "
                "upgrade the package or ensure axis='signals'."
            )
        return time_series[:, np.newaxis, :]


@contextmanager
def _surface_backend_log_warnings():
    """Re-emit ``spectral_connectivity``'s log warnings as Python warnings.

    Wilson's factorisation is iterative. When it hits its iteration cap it
    reports that through ``logging.Logger.warning`` -- "Maximum iterations
    reached. 0 of 1 converged" -- and then *returns the unconverged factor
    anyway*. Every Wilson-derived measure (directed coherence, DTF, dDTF, PDC,
    gPDC, nonparametric spectral GC) is computed from that factor.

    pyspi records per-SPI diagnostics from the ``warnings`` channel only
    (``_parallel.run_spi``), and logging is a different channel, so an
    unconverged factorisation reached the results table with nothing recorded
    against it. On the bundled ``kuramoto_M7_T100`` fixture that is 2 of 21
    pairs; the relative factorisation residual ``max|S - GG^H| / max|S|`` there
    runs 0.14-6.0 across pairs, so these are not marginal numbers.

    This bridges the two channels for the duration of one backend call. It does
    not change any value: the point is that a caller inspecting
    ``calc.errors``/warnings can now see which pairs the factorisation failed
    on. Improving the estimate itself (longer series, a parametric VAR fit, or
    a tighter multitaper configuration) is the user's call, not something pyspi
    can do behind their back.
    """
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    backend = logging.getLogger("spectral_connectivity")
    backend.addHandler(handler)
    try:
        yield
    finally:
        backend.removeHandler(handler)
    # dict.fromkeys: one warning per distinct message, not per frequency bin.
    for message in dict.fromkeys(records):
        warnings.warn(f"spectral_connectivity: {message}", RuntimeWarning)


def _ensure_time_series_3d(z):
    """Ensure time-series array follows (n_time, n_trials, n_signals)."""
    if z.ndim == 2:
        return prepare_time_series(z, axis="signals")
    return z


def _circular_nanmean(angles, axis):
    """Circular mean angle, omitting NaNs along ``axis``.

    Wrapped phase is averaged through its unit phasors, not as real numbers;
    e.g. values just below +pi and just above -pi average near pi rather than
    zero. A zero resultant has no circular mean. An antipodal mean (+/-pi) has
    a circular location but no unique sign in pyspi's ordinary-float,
    numerically antisymmetric representation, so both cases return NaN.

    For ``n`` unit phasors, the normalised error of their floating-point sum is
    O(n * eps). Eight times that bound covers the complex exponential plus the
    real and imaginary reductions without treating a resolvable resultant as
    zero. The same bound decides whether an antipodal resultant's imaginary
    component is distinguishable from round-off.
    """
    phasors = np.exp(1j * angles)
    count = np.sum(~np.isnan(angles), axis=axis)
    total = np.nansum(phasors, axis=axis)
    mean = np.full(np.shape(total), np.nan + 0j, dtype=complex)
    np.divide(total, count, out=mean, where=count > 0)

    roundoff = 8.0 * np.finfo(float).eps * np.maximum(count, 1)
    zero_resultant = np.abs(mean) <= roundoff
    antipodal = (mean.real < 0) & (np.abs(mean.imag) <= roundoff)
    undefined = (count == 0) | zero_resultant | antipodal
    return np.where(undefined, np.nan, np.angle(mean))


class NonparametricSpectral(Unsigned):
    """Base class for the nonparametric spectral methods from the Eden-Kramer repo"""

    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean"):
        if fmax is None:
            fmax = fs / 2

        self._fs = fs
        if fs != 1:
            warnings.warn("Multiple sampling frequencies not yet handled.")
        self._fmin = fmin
        self._fmax = fmax
        if statistic == "mean":
            self._statfn = np.nanmean
        elif statistic == "max":
            self._statfn = np.nanmax
        elif statistic not in ["delay", "slope", "rvalue"]:
            raise NameError(f"Unknown statistic: {statistic}")
        else:
            self._statfn = None
        self._statistic = statistic

        # Structural trait, derived from the implementation rather than assumed.
        # PLI, wPLI, PSI and coherence phase carry a sign that encodes lead/lag:
        # A[i,j] == -A[j,i]. That is a third category, neither "undirected"
        # (which implies symmetry) nor "directed" (which implies the two
        # orientations are independent quantities). Labelling them undirected
        # misdescribes them for any downstream filtering.
        #
        # Only band statistics that are linear in the spectrum preserve
        # antisymmetry: mean does, max does not (max(-x) != -max(x)), so a
        # 'max' variant of an antisymmetric measure is genuinely neither and is
        # left unlabelled here.
        if getattr(self, "_antisymmetric_spectrum", False):
            trait = "antisymmetric" if statistic == "mean" else "asymmetric"
            self.labels = [
                l for l in self.labels
                if l not in ("undirected", "directed", "unsigned")
            ] + [trait, "signed"]
            # And genuinely signed, not just labelled so. `issigned()` is not
            # cosmetic: `Calculator._rmmin` subtracts the minimum from every
            # SPI it reports as unsigned, which on an antisymmetric matrix
            # shifts A[i,j] and A[j,i] by the same amount and destroys the
            # antisymmetry that carries the lead/lag; and `set_group`
            # correlates unsigned SPIs through `abs()`, folding lead onto lag.
            # True for the `max` variants too: the maximum of a signed
            # spectrum is still signed.
            self.issigned = lambda: True

        paramstr = (
            f"_multitaper_{statistic}_fs-{fmt_param(fs)}_fmin-{fmt_param(fmin)}"
            f"_fmax-{fmt_param(fmax)}".replace(
                ".", "-"
            )
        )
        self.identifier += paramstr

    @property
    def key(self):
        """Cache key: every parameter that changes the cached result.

        ``fs`` is part of the key because it is passed to ``sc.Multitaper`` and
        therefore changes both the connectivity estimate and the frequency grid.
        Omitting it let two SPIs differing only in sampling frequency collide.

        GroupDelay and PhaseSlopeIndex additionally cache a band-dependent
        statistic, so their key carries fmin/fmax as well.
        """
        base = (self.measure, self._fs)
        if isinstance(self, (GroupDelay, PhaseSlopeIndex)):
            return base + (self._fmin, self._fmax)
        return base

    @property
    def _freq_key(self):
        """Frequency grid depends on fs, so it cannot live under a bare 'freq'."""
        return ("freq", self._fs)

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f"Include measure for {self.identifier}")

    def _get_statistic(self, C):
        raise NotImplementedError


def _to_source_target(spi, adj):
    """Put a spectral adjacency matrix into pyspi's (source, target) convention.

    pyspi's convention is set by ``base.Directed.multivariate``, which fills
    ``A[i, j] = bivariate(i, j)`` -- row is the source, column the target. The
    spectral backends (spectral_connectivity, nitime) instead follow the
    DTF/PDC literature, where element ``[i, j]`` is the flow *into* i *from* j
    (Kaminski & Blinowska 1991; the library normalises "by inflow"). Passing
    their output through unchanged left every directed spectral SPI transposed
    relative to every other directed SPI in the library.

    Only directed SPIs are transposed. The undirected spectral measures share
    this code path, and several of them (PhaseLagIndex, WeightedPhaseLagIndex,
    PhaseSlopeIndex) are antisymmetric rather than symmetric, so transposing
    them would silently negate their values.

    Note the test is ``not isinstance(spi, Undirected)``, not
    ``isinstance(spi, Directed)``: ``Undirected`` subclasses ``Directed`` (it
    reuses its multivariate loop and then mirrors), so the latter is true for
    every SPI here.
    """
    return adj if isinstance(spi, Undirected) else adj.T


class NonparametricSpectralMultivariate(NonparametricSpectral):
    _cache_namespace = "spectral_mv"

    @property
    def _cache_subkey(self):
        # Default: one cache entry per (class, fs). GroupDelay and
        # PhaseSlopeIndex override this — their cache also keys on fmin/fmax.
        # fs mirrors `key`: SPIs at different sampling frequencies do not share
        # a cache entry, so they must not share an amortization bucket either.
        return (type(self).__name__, self._fs)

    def _get_cache(self, data):
        # One key type throughout. The previous version created the dict with a
        # *string* key (self.measure) but read with a *tuple* key (self.key), so
        # the first write was unreachable, the second call recomputed and stored
        # under the tuple, and only from the third call did the cache hit --
        # which is why a two-call probe showed no problem.
        cache = getattr(data, "spectral_mv", None)
        if cache is None:
            cache = data.spectral_mv = {}

        if self.key in cache:
            return cache[self.key], cache[self._freq_key]

        z = np.transpose(data.to_numpy(squeeze=True))
        z = _ensure_time_series_3d(z)
        m = sc.Multitaper(z, sampling_frequency=self._fs)
        conn = sc.Connectivity.from_multitaper(m)
        # `_recompute` lets a subclass override a backend measure outright, not
        # just when the backend raises. DirectedCoherence needs this: the
        # backend's method returns fine, it is simply unbounded.
        # The Connectivity object is lazy, so the factorisation happens inside
        # this block, not at construction -- which is why the bridge wraps the
        # measure extraction rather than `from_multitaper`.
        with _surface_backend_log_warnings():
            if getattr(self, "_recompute", False):
                res = self._get_statistic(conn)
            else:
                try:
                    res = getattr(conn, self.measure)()
                except TypeError:
                    res = self._get_statistic(conn)

        freq = conn.frequencies
        cache[self.key] = res
        cache[self._freq_key] = freq
        return res, freq

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        try:
            adj = self._statfn(adj_freq[0, freq_id, :, :], axis=0)
        except IndexError:  # For phase slope index
            adj = adj_freq[0]
        except TypeError:  # For group delay
            stat_id = [
                i
                for i, s in enumerate(["delay", "slope", "rvalue"])
                if self._statistic == s
            ][0]
            adj = adj_freq[stat_id][0]
        adj = _to_source_target(self, adj)
        np.fill_diagonal(adj, np.nan)
        return adj


class NonparametricSpectralBivariate(NonparametricSpectral):
    _cache_namespace = "spectral_bv"

    @property
    def _cache_subkey(self):
        # Per-pair Connectivity is shared across classes, but the per-class
        # measure extraction (DTF, dDTF, dCoh, etc.) dominates in practice
        # (~10s per class at M=16,T=800 vs <1s for the shared Multitaper).
        # Bucket amortization by class so variants of one measure share, but
        # different measures don't get cross-amortized. fs mirrors `key`.
        return (type(self).__name__, self._fs)

    def _get_cache(self, data, i, j):
        """Cache Connectivity object per (i,j) pair, not per (measure,i,j).

        Multiple directed spectral SPIs (DirectedCoherence, PartialDirectedCoherence,
        etc.) share the same Multitaper+Connectivity for a given (i,j). The expensive
        part is building the Multitaper — each measure extraction is cheap.
        """
        # fs belongs in both keys: it is passed to Multitaper, so it changes the
        # Connectivity object and the frequency grid as well as the measure.
        measure_key = (self.measure, self._fs, i, j)
        cache = getattr(data, "spectral_bv", None)
        if cache is None:
            cache = data.spectral_bv = {}

        if measure_key in cache:
            return cache[measure_key], cache[self._freq_key]

        conn_key = (self._fs, i, j)
        conns = getattr(data, "_spectral_bv_conn", None)
        if conns is None:
            conns = data._spectral_bv_conn = {}
        conn = conns.get(conn_key)
        if conn is None:
            z = np.transpose(data.to_numpy(squeeze=True)[[i, j]])
            z = _ensure_time_series_3d(z)
            m = sc.Multitaper(z, sampling_frequency=self._fs)
            conn = sc.Connectivity.from_multitaper(m)
            conns[conn_key] = conn

        # `_recompute` lets a subclass override a backend measure outright, not
        # just when the backend raises. DirectedCoherence needs this: the
        # backend's method returns fine, it is simply unbounded.
        with _surface_backend_log_warnings():
            if getattr(self, "_recompute", False):
                res = self._get_statistic(conn)
            else:
                try:
                    res = getattr(conn, self.measure)()
                except TypeError:
                    res = self._get_statistic(conn)

        freq = conn.frequencies
        cache[measure_key] = res
        cache[self._freq_key] = freq
        return res, freq

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        bv_freq, freq = self._get_cache(data, i, j)
        freq_id = np.where((freq > self._fmin) * (freq < self._fmax))[0]

        # [1, 0] not [0, 1]: the sub-system is built as [i, j] (local 0 = i,
        # local 1 = j) and the backend indexes [target, source], so the i->j
        # flow is at [1, 0]. See _to_source_target.
        return self._statfn(bv_freq[0, freq_id, 1, 0])


class CoherenceMagnitude(NonparametricSpectralMultivariate, Undirected):
    name = "Coherence magnitude"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "cohmag"
        super().__init__(**kwargs)
        self._measure = "coherence_magnitude"


class CoherencePhase(NonparametricSpectralMultivariate, Undirected):
    # Antisymmetric in (i, j): phase difference: phi(i,j) = -phi(j,i).
    _antisymmetric_spectrum = True
    name = "Coherence phase"
    labels = ["signed", "spectral", "antisymmetric"]

    def __init__(self, **kwargs):
        if kwargs.get("statistic", "mean") != "mean":
            raise ValueError(
                "CoherencePhase supports only statistic='mean'. Wrapped phase "
                "has no branch-cut-independent ordinary maximum; use a "
                "different circular summary with an explicit interpretation."
            )
        self.identifier = "phase"
        super().__init__(**kwargs)
        self.labels = ["signed", "spectral", "antisymmetric"]
        self._measure = "coherence_phase"

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]
        adj = _circular_nanmean(adj_freq[0, freq_id, :, :], axis=0)

        # Coherency phase is antisymmetric. Select one computed orientation and
        # construct its opposite explicitly so branch-cut representations at
        # +/-pi and backend round-off cannot violate the declared structure.
        ui = np.triu_indices(data.n_processes, 1)
        adj[ui] = -adj.T[ui]
        np.fill_diagonal(adj, np.nan)
        return adj


class ImaginaryCoherence(NonparametricSpectralMultivariate, Undirected):
    name = "Imaginary coherence"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "icoh"
        super().__init__(**kwargs)
        self._measure = "imaginary_coherence"


class PhaseLockingValue(NonparametricSpectralMultivariate, Undirected):
    name = "Phase locking value"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "plv"
        super().__init__(**kwargs)
        self._measure = "phase_locking_value"

        myfn = deepcopy(self._statfn)
        self._statfn = lambda x, **kwargs: myfn(np.absolute(x), **kwargs)


class PhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    # Antisymmetric in (i, j): sign of the imaginary coherency.
    _antisymmetric_spectrum = True
    name = "Phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "pli"
        super().__init__(**kwargs)
        self._measure = "phase_lag_index"


class WeightedPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    # Antisymmetric in (i, j): imaginary-coherency weighted sign.
    _antisymmetric_spectrum = True
    name = "Weighted phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "wpli"
        super().__init__(**kwargs)
        self._measure = "weighted_phase_lag_index"


class DebiasedSquaredPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Debiased squared phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dspli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_phase_lag_index"


class DebiasedSquaredWeightedPhaseLagIndex(
    NonparametricSpectralMultivariate, Undirected
):
    name = "Debiased squared weighted phase-lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dswpli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_weighted_phase_lag_index"


class PairwisePhaseConsistency(NonparametricSpectralMultivariate, Undirected):
    name = "Pairwise phase consistency"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "ppc"
        super().__init__(**kwargs)
        self._measure = "pairwise_phase_consistency"


"""
    These next several seem to segfault for large vector autoregressive processes (something to do with np.linalg solver).
    Switched them to bivariate for now until the issue is resolved
"""


class DirectedCoherence(NonparametricSpectralBivariate, Directed):
    """Directed coherence (Baccala et al. 1998).

    ``DC_ij(f) = sqrt(sigma_jj) |H_ij(f)| / sqrt(sum_k sigma_kk |H_ik(f)|^2)``,
    where ``H`` is the transfer function in the backend's ``[target, source]``
    convention and ``sigma_kk`` is the innovation variance of process ``k``.
    Bounded in [0, 1], with ``sum_j DC_ij^2 == 1`` by construction.

    Two things are wrong with the backend's ``directed_coherence()``:

    1. It puts the *squared* magnitude in the numerator while the denominator
       stays on the magnitude scale, so the ratio is dimensionally |H|^2 / |H|
       and unbounded -- the shipped baselines reached 3.27 (VAR), 1.84 (CML)
       and 1139.47 (Kuramoto).
    2. ``_get_noise_variance`` reshapes ``diag(Sigma)`` to ``(..., 1, n, 1)``,
       which broadcasts the variance along the *row* (target) axis of ``H``.
       Baccala's weight is indexed by the *source*. With the weight on the row
       it factors out of numerator and denominator alike and cancels exactly,
       so the innovation variances have no effect at all: the measure silently
       degenerates to ``sqrt(directed_transfer_function())`` for *every* noise
       covariance, not just the equal-variance case. Verified: with innovation
       standard deviations (1, 3, 0.2) the row-indexed form still reproduces
       sqrt(DTF) to 4e-16.

    Both are corrected here by recomputing from the same transfer function with
    ``|H|`` in the numerator and the variance broadcast along the source axis.
    The equal-variance identity ``DC == sqrt(DTF)`` now *discriminates*: it
    holds only when the innovation variances are in fact equal.

    Correlated innovations
    ----------------------
    Baccala's formula uses only ``diag(Sigma)``. Two properties are unaffected
    by off-diagonal innovation covariance: the value stays in [0, 1] (the
    denominator contains the numerator's term), and ``sum_j DC_ij^2 == 1``
    holds identically. What does *not* survive is the reading of ``DC_ij^2`` as
    the fraction of process ``i``'s spectral power arriving from ``j``: that
    requires ``S_ii = sum_j sigma_jj |H_ij|^2``, which holds only for diagonal
    ``Sigma``. This is not academic -- the Wilson-estimated innovation
    correlation on the bundled fixtures reaches 0.14 (VAR), 0.64 (CML) and
    1.00 (Kuramoto).

    Whitening is *not* applied. The minimum-phase factor ``G = H L`` with
    ``L = g_0`` triangular would give an exactly power-decomposing variant, but
    a triangular factor is order-dependent: recomputing the same pair as
    ``[j, i]`` yields a different ``L`` (measured, not permutation-related), so
    a pairwise SPI built on it would depend on process order -- the defect that
    made ``coint_aeg`` wrong. Baccala's published diagonal form is order-free,
    so it is what is computed, with the assumption stated rather than hidden.
    """

    name = "Directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "dcoh"
        super().__init__(**kwargs)
        self._measure = "directed_coherence"
        self._recompute = True

    def _get_statistic(self, C):
        # Deliberately does not use the backend's _get_noise_variance /
        # _total_inflow helpers: the first carries the source/target axis bug
        # described above, and doing the algebra here keeps the private-API
        # surface down to the two properties (see test_backend_private_api).
        H = C._transfer_function                       # (..., f, target, source)
        sigma = np.diagonal(C._noise_covariance, axis1=-1, axis2=-2)   # (..., n)
        sigma = sigma[..., np.newaxis, np.newaxis, :]  # broadcast along `source`
        mag2 = np.abs(H) ** 2
        inflow = np.sqrt(np.sum(sigma * mag2, axis=-1, keepdims=True))
        return np.sqrt(sigma) * np.abs(H) / inflow


class PartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "pdcoh"
        super().__init__(**kwargs)
        self._measure = "partial_directed_coherence"


class GeneralizedPartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Generalized partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "gpdcoh"
        super().__init__(**kwargs)
        self._measure = "generalized_partial_directed_coherence"


class DirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    name = "Directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "dtf"
        super().__init__(**kwargs)
        self._measure = "directed_transfer_function"


class DirectDirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    name = "Direct directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "ddtf"
        super().__init__(**kwargs)
        self._measure = "direct_directed_transfer_function"


class PhaseSlopeIndex(NonparametricSpectralMultivariate, Undirected):
    # Antisymmetric in (i, j): slope of the phase spectrum.
    _antisymmetric_spectrum = True
    name = "Phase slope index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        self._measure = "phase_slope_index"

    @property
    def _cache_subkey(self):
        # Narrower cache: (class, fs, fmin, fmax) per the key property override.
        return (type(self).__name__, self._fs, self._fmin, self._fmax)

    def _get_statistic(self, C):
        return C.phase_slope_index(
            frequencies_of_interest=[self._fmin, self._fmax],
            frequency_resolution=(self._fmax - self._fmin) / 50,
        )


def _independent_significant_frequencies(p_values, step, alpha=0.05,
                                         min_group_size=3):
    """Indices of the largest significant coherence cluster, thinned to `step`.

    Benjamini-Hochberg over the in-band frequencies, then the longest
    contiguous run of significant points, then every `step`-th point of that
    run so the regression is not fitted to correlated estimates. Returns None
    if fewer than `min_group_size` independent points survive -- the phase
    slope is not identifiable from one or two points, and a two-point "fit" has
    an r-value of exactly 1 whatever the data.
    """
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    passed = ranked <= alpha * np.arange(1, n + 1) / n
    significant = np.zeros(n, dtype=bool)
    if passed.any():
        significant[order[: np.flatnonzero(passed)[-1] + 1]] = True
    if not significant.any():
        return None

    # Longest contiguous run of True.
    padded = np.concatenate([[False], significant, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[::2], edges[1::2]
    longest = np.argmax(ends - starts)
    keep = np.arange(starts[longest], ends[longest], step)
    return keep if keep.size >= min_group_size else None


class GroupDelay(NonparametricSpectralMultivariate, Directed):
    name = "Group delay"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "gd"
        super().__init__(**kwargs)
        self._measure = "group_delay"
        # `delay` and `slope` are antisymmetric by construction (the phase of
        # C_ji is the negative of the phase of C_ij, so the fitted slopes are
        # exact negatives) and signed -- a negative delay means the row lags
        # the column. `rvalue` is the fit quality and is symmetric.
        if self._statistic in ("delay", "slope"):
            self.labels = [
                l for l in self.labels
                if l not in ("undirected", "directed", "unsigned")
            ] + ["antisymmetric", "signed"]
            self.issigned = lambda: True
        else:
            # |r| is symmetric by construction and non-negative, so it is
            # neither directed nor signed.
            self.labels = [
                l for l in self.labels if l not in ("directed", "antisymmetric")
            ] + ["undirected"]
        # Always use `_get_statistic`. The dispatcher's fallback is
        # `except TypeError`, and `Connectivity.group_delay()` accepts a
        # no-argument call, so the backend's (all-NaN, see below) result was
        # returned without the band arguments ever being passed.
        self._recompute = True

    @property
    def _cache_subkey(self):
        # Narrower cache: (class, fs, fmin, fmax) per the key property override.
        return (type(self).__name__, self._fs, self._fmin, self._fmax)

    def _get_statistic(self, C):
        """Gotman (1983) group delay, computed here rather than in the backend.

        ``spectral_connectivity.Connectivity.group_delay`` returns all-NaN for
        every input, on every dataset, at every length. Not a power problem --
        the defect is one line in its significance test.
        ``coherence_fisher_z_transform`` divides by
        ``sqrt(coherence_bias(n_obs1) + coherence_bias(n_obs2))`` and, in the
        one-sample case, is called with ``n_obs2 = 0``. ``coherence_bias(0)`` is
        ``1 / (2*0 - 2) = -0.5``, so the argument of the square root is
        ``1/(2n-2) - 0.5``, negative for every ``n``: every p-value is NaN,
        nothing is ever significant, the phase regression runs on a fully
        masked array, and the result is NaN. Confirmed at 5, 11, 19 and 39
        tapers, T up to 4000, on a pair with median coherence 0.998.

        The one-sample test is the standard Enochson-Goodman/Bokil form: for
        coherence estimated from ``n`` independent spectral estimates,
        ``arctanh|C|`` is approximately normal with mean ``arctanh|Gamma|`` plus
        a bias ``b = 1/(2n - 2)`` and *variance* ``b``, so the null z-score is
        ``(arctanh|C| - b) / sqrt(b)``. That is what the second ``coherence_bias``
        term was meant to be and is what is used here.

        The rest follows the backend's own recipe: Benjamini-Hochberg over the
        in-band frequencies, keep the largest contiguous significant run,
        subsample it to statistically independent points, require at least
        three, and regress the unwrapped coherence phase on frequency.
        ``fs * slope / (2*pi)`` is the delay in samples; the unscaled
        ``slope/(2*pi)`` would be in seconds, because the regression runs
        against physical frequency. ``rvalue`` is |r|, which is orientation-
        free; the signed r is not, since phase(C_ji) = -phase(C_ij).
        """
        from scipy import stats

        coherency = np.asarray(C.coherency())
        freqs = np.asarray(C.frequencies)
        in_band = (freqs >= self._fmin) & (freqs <= self._fmax)
        f = freqs[in_band]
        coh = coherency[:, in_band]
        n_trials, _, M, _ = coh.shape

        # Independent frequency spacing, as the backend defines it: the band is
        # resolved into 50 pieces and points closer than that are not
        # independent estimates.
        df = float(freqs[1] - freqs[0])
        resolution = (self._fmax - self._fmin) / 50
        step = max(1, int(np.ceil(resolution / df)))

        bias = 1.0 / (2 * C.n_observations - 2)
        magnitude = np.minimum(np.abs(coh), 1 - np.finfo(float).eps)
        z = (np.arctanh(magnitude) - bias) / np.sqrt(bias)
        p_values = stats.norm.sf(z)
        phase = np.unwrap(np.angle(coh), axis=1)

        slope = np.full((n_trials, M, M), np.nan)
        r_value = np.full((n_trials, M, M), np.nan)
        for t in range(n_trials):
            for i in range(M):
                for j in range(i + 1, M):
                    keep = _independent_significant_frequencies(
                        p_values[t, :, i, j], step
                    )
                    if keep is None:
                        continue
                    fit = stats.linregress(f[keep], phase[t, keep, i, j])
                    # Sign fixed by measurement, not by assumption: with
                    # y(t) = x(t - L), this orientation makes the delay
                    # +L at [source, target] after `_to_source_target`, i.e.
                    # positive means the row leads the column. Verified for
                    # L in {1, 3, 5, 8} to within 0.005 samples.
                    slope[t, i, j] = -fit.slope
                    slope[t, j, i] = fit.slope
                    # |r|, not r. The fit is of the phase of C_ij, and
                    # phase(C_ji) = -phase(C_ij), so the signed correlation
                    # flips with the orientation -- while this matrix is
                    # written symmetrically. Reversing the process order
                    # therefore turned +0.99997 into -0.99997 at the mirrored
                    # position, for a statistic declared symmetric. The
                    # magnitude is what "fit quality" means here, and it is
                    # orientation-free.
                    r_value[t, i, j] = r_value[t, j, i] = abs(fit.rvalue)

        # Samples, not seconds. `C.frequencies` is in Hz, so the fitted slope
        # is radians per Hz and slope/(2*pi) is a delay in *seconds*: at fs=4 a
        # true 4-sample lag came back as 0.9999. The API and every other lagged
        # SPI in pyspi count samples, so the delay is scaled by fs. No change at
        # the shipped fs=1.
        return self._fs * slope / (2 * np.pi), slope, r_value


class SpectralGrangerCausality(NonparametricSpectralMultivariate, Directed, Unsigned):
    name = "Spectral Granger causality"
    identifier = "sgc"
    labels = ["unsigned", "embedding", "spectral", "directed", "lagged"]

    def __init__(
        self,
        fs=1,
        fmin=1e-5,
        fmax=0.5,
        method="nonparametric",
        order=None,
        max_order=50,
        statistic="mean",
        ignore_nan=True,
        nan_threshold=0.5,
    ):
        self._fs = fs  # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        self.ignore_nan = ignore_nan
        self.nan_threshold = nan_threshold

        if self._fmin <= 0.0:
            warnings.warn(f"Frequency minimum set to {self._fmin}; overriding to 1e-5.")
            self._fmin = 1e-5

        if statistic == "mean":
            if self.ignore_nan:
                self._statfn = np.nanmean
            else:
                self._statfn = np.mean
        elif statistic == "max":
            if self.ignore_nan:
                self._statfn = np.nanmax
            else:
                self._statfn = np.max
        else:
            raise NameError(f"Unknown statistic {statistic}")

        self._method = method
        # `self._fmin`, not the `fmin` argument. Spectral GC is undefined at
        # zero frequency, so `fmin=0` is overridden to 1e-5 above -- but the
        # identifier was built from the argument, so twelve shipped SPIs said
        # `fmin-0` while integrating from 1e-5, i.e. excluding the DC bin. Two
        # configs differing only in `fmin: 0` versus `fmin: 1e-5` would have
        # produced two identifiers for one computation.
        if self._method == "nonparametric":
            self._measure = "pairwise_spectral_granger_prediction"
            paramstr = (f"_nonparametric_{statistic}_fs-{fmt_param(fs)}"
                        f"_fmin-{fmt_param(self._fmin)}"
                        f"_fmax-{fmt_param(fmax)}").replace(
                ".", "-"
            )
        else:
            self._order = order
            self._max_order = max_order
            paramstr = (f"_parametric_{statistic}_fs-{fmt_param(fs)}"
                        f"_fmin-{fmt_param(self._fmin)}"
                        f"_fmax-{fmt_param(fmax)}_order-{order}").replace(
                ".", "-"
            )

        self.identifier = self.identifier + paramstr

    def _getkey(self):
        # fs is passed to the spectral transform, so it must key the cache.
        # Without it, reusing one Data at another sampling frequency returned
        # the first computation (0.2037 for fs=1 vs 0.2497 fresh at fs=4).
        if self._method == "nonparametric":
            return (self._method, self._fs, -1, -1)
        else:
            return (self._method, self._fs, self._order, self._max_order)

    def _get_cache(self, data):
        key = self._getkey()

        try:
            F = data.spectral_gc[key]["F"]
            freq = data.spectral_gc[key]["freq"]
        except (AttributeError, KeyError):

            if self._method == "nonparametric":
                F, freq = super()._get_cache(data)
            else:
                z = data.to_numpy(squeeze=True)
                # 1/fs, not 1. `fs` is in this SPI's identifier and in its
                # cache key, but the parametric branch hard-coded a unit
                # sampling interval, so `GA.frequencies` came back in units of
                # 1/1 whatever `fs` said and the [fmin, fmax] band was applied
                # on the wrong axis. Two SPIs differing only in `fs` therefore
                # advertised different sampling rates and computed the same
                # numbers. No effect at the shipped fs=1.
                time_series = ts.TimeSeries(z, sampling_interval=1.0 / self._fs)
                GA = nta.GrangerAnalyzer(
                    time_series, order=self._order, max_order=self._max_order
                )

                triu_id = np.triu_indices(data.n_processes)

                try:
                    causality_xy = GA.causality_xy
                    causality_yx = GA.causality_yx
                except ValueError as err:
                    # nitime's order search walks the lag up to `max_order` and
                    # raises if the information criterion never turns over.
                    # That is the signature of over-fitting at this record
                    # length, not of a transient numerical problem, and its own
                    # message ("Model estimation order did not converge at
                    # max_order = 50") says nothing about the data. Relaying it
                    # as an all-NaN return buried the cause under pyspi's
                    # generic "returned no finite off-diagonal values".
                    if "did not converge" not in str(err):
                        raise
                    raise ValueError(
                        f"Parametric spectral Granger causality: automatic AR "
                        f"order selection did not converge at "
                        f"max_order={self._max_order} on "
                        f"{data.n_observations} observations -- the "
                        f"information criterion improved all the way to the "
                        f"cap, which at this length means the model is "
                        f"over-fitting rather than that the true order is "
                        f"high. Set an explicit `order`, reduce `max_order`, "
                        f"or use a longer series."
                    ) from err

                F = np.full(causality_xy.shape, np.nan)
                F[triu_id[0], triu_id[1], :] = causality_xy[
                    triu_id[0], triu_id[1], :
                ]
                F[triu_id[1], triu_id[0], :] = causality_yx[
                    triu_id[0], triu_id[1], :
                ]

                F = np.transpose(np.expand_dims(F, axis=3), axes=[3, 2, 1, 0])
                freq = GA.frequencies
            try:
                data.spectral_gc[key] = {"freq": freq, "F": F}
            except AttributeError:
                data.spectral_gc = {key: {"freq": freq, "F": F}}

        return F, freq

    @parse_multivariate
    def multivariate(self, data):
        try:
            cache, freq = self._get_cache(data)
            freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

            result = _to_source_target(self, self._statfn(cache[0, freq_id, :, :], axis=0))

            # Transformed the same way as the values it masks. The result is
            # put into pyspi's (source, target) orientation while the mask was
            # left in the backend's, so a directionally asymmetric NaN pattern
            # blanked the *mirror* of the affected pair: the cell that was
            # actually unestimable was already NaN, and a perfectly good one
            # next to it was destroyed.
            nan_pct = _to_source_target(
                self, np.isnan(cache[0, freq_id, :, :]).mean(axis=0))
            np.fill_diagonal(nan_pct, 0.0)

            isna = nan_pct > self.nan_threshold
            if isna.any():
                warnings.warn(
                    f"Spectral GC: the following processes have >{self.nan_threshold*100:.1f}% "
                    f"NaN values:\n{np.transpose(np.where(isna))}\nThese indices will be set to NaN. "
                    "Set ignore_nan to False or modify nan_threshold parameter if required."
                )
                result[isna] = np.nan

            return result
        except ValueError as err:
            # Not swallowed into an all-NaN table: a ValueError here means the
            # model could not be fitted, which the caller needs in
            # `Calculator.errors` with its cause attached, not as a silent
            # empty column plus a warning.
            raise

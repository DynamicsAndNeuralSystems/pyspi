import functools
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
import nitime.algorithms as nt_alg
import nitime.utils as nt_utils
import warnings
from scipy import signal
from pyspi.utils import fmt_param, require_int

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

    # The `statistic` values a class can apply. A band statistic reduces a
    # spectrum over frequency; GroupDelay's three are outputs of a regression.
    # Accepting one of GroupDelay's elsewhere used to fall through the
    # dispatcher in `multivariate` and return the spectrum's 0 Hz bin.
    _statistics = ("mean", "max")

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
        if statistic not in self._statistics:
            raise ValueError(
                f"{type(self).__name__} does not support statistic="
                f"{statistic!r}; it supports {', '.join(self._statistics)}.")
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


class _FactoredConnectivity(sc.Connectivity):
    """A pair's ``Connectivity``, reduced to its spectral factorisation.

    The directed measures are all read off the Wilson minimum-phase factor,
    through the transfer function, the innovation covariance and the MVAR
    coefficients. The backend derives each of those afresh on every access --
    two to four factorisations for a single measure, fourteen for the five
    directed measures of one pair -- and the factorisation is the expensive
    step. Here it is done once. Each quantity is a deterministic function of
    the same spectrum, so keeping the first result changes no value.

    Once factorised, the Fourier coefficients and the factor itself are let go:
    one of these objects is held per ordered pair, and they were most of it.
    """

    @functools.cached_property
    def _minimum_phase_factor(self):
        return super()._minimum_phase_factor

    def _factorise(self):
        held = vars(self)
        if "_noise_covariance" not in held:
            # The parent's properties read the factor through `self`, so it is
            # computed by the cached property above exactly once for the two.
            transfer = np.array(super()._transfer_function)
            held["_noise_covariance"] = np.array(super()._noise_covariance)
            held["_transfer_function"] = transfer
            del held["_minimum_phase_factor"]
            self.fourier_coefficients = None
        return held

    @functools.cached_property
    def _transfer_function(self):
        return self._factorise()["_transfer_function"]

    @functools.cached_property
    def _noise_covariance(self):
        return self._factorise()["_noise_covariance"]

    @functools.cached_property
    def _MVAR_Fourier_coefficients(self):
        return super()._MVAR_Fourier_coefficients


class NonparametricSpectralBivariate(NonparametricSpectral):
    _cache_namespace = "spectral_bv"

    @property
    def _cache_subkey(self):
        # One bucket for every directed measure at a sampling frequency: they
        # share the pair's Connectivity and, through _FactoredConnectivity, its
        # factorisation, which is nearly all of the cost. Split by class, each
        # worker repeated that factorisation for every pair. fs mirrors `key`.
        return (self._fs,)

    def _get_cache(self, data, i, j):
        """Cache Connectivity object per (i,j) pair, not per (measure,i,j).

        Multiple directed spectral SPIs (DirectedCoherence, PartialDirectedCoherence,
        etc.) share the same Multitaper+Connectivity for a given (i,j), and with
        it the spectral factorisation every one of them is derived from.
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
            conn = _FactoredConnectivity.from_multitaper(m)
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

        # Only the i->j spectrum is kept. [1, 0] not [0, 1]: the sub-system is
        # built as [i, j] (local 0 = i, local 1 = j) and the backend indexes
        # [target, source], so the i->j flow is at [1, 0]. See _to_source_target.
        res = np.array(res[0, :, 1, 0])
        freq = conn.frequencies
        cache[measure_key] = res
        cache[self._freq_key] = freq
        return res, freq

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        bv_freq, freq = self._get_cache(data, i, j)
        # Closed band, as in NonparametricSpectralMultivariate, spectral GC and
        # the wavelet SPIs. Strict inequalities here dropped the DC, 0.25 and
        # Nyquist bins from bands whose identifiers include them.
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]
        return self._statfn(bv_freq[freq_id])


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
    """Debiased squared phase lag index (Vinck et al. 2011, NeuroImage 55, 1548).

    The mean of ``sign(Im S_k) sign(Im S_l)`` over pairs ``k != l`` of the
    ``n`` tapers (and trials), ``S_k`` being taper ``k``'s cross-spectrum:
    ``((sum s)^2 - sum s^2) / (n (n - 1))``.

    Computed here rather than by the backend, whose
    ``(n PLI^2 - 1) / (n - 1)`` is the same expression only when every sign is
    nonzero. At 0 Hz and at the Nyquist frequency the Fourier coefficients of a
    real signal are real, so every ``Im S_k`` is exactly zero: the estimator is
    0 there, and the backend's shortcut returned ``-1/(n - 1)`` (-0.25 at the
    default five tapers) for every input.
    """

    name = "Debiased squared phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dspli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_phase_lag_index"
        self._recompute = True

    def _get_statistic(self, C):
        X = np.asarray(C.fourier_coefficients)  # (window, trial, taper, fft, signal)
        X = X[..., : X.shape[-2] // 2 + 1, :]   # the backend's non-negative bins
        sign = np.sign(np.imag(X[..., :, np.newaxis] * np.conj(X[..., np.newaxis, :])))
        n = sign.shape[1] * sign.shape[2]
        total = sign.sum(axis=(1, 2))
        squares = np.abs(sign).sum(axis=(1, 2))
        return (total ** 2 - squares) / (n * (n - 1))


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
    """Direct directed transfer function (Korzeniewska et al. 2003).

    ``dDTF_ij(f) = eta_ij(f) |P_ij(f)|``: the full-frequency DTF
    ``eta_ij(f) = |H_ij(f)| / sqrt(sum_f sum_k |H_ik(f)|^2)`` times the
    magnitude of the partial coherence ``P_ij = -Q_ij / sqrt(Q_ii Q_jj)``,
    ``Q = S(f)^-1``, with ``H`` in the backend's ``[target, source]``
    convention. This is the square root of the published
    ``delta_ij^2 = eta_ij^2 P_ij^2``, i.e. the magnitude scale the backend
    reports.

    The backend's ``direct_directed_transfer_function()`` multiplies ffDTF by
    ``sqrt(partial_directed_coherence())`` -- |PDC|, a different and itself
    directed quantity -- although its docstring describes "the directed
    transfer function and partial coherence". ``S`` here is the spectrum of
    the fitted model, ``H Sigma H^H``, so like the other four measures dDTF is
    read entirely off the pair's spectral factorisation.

    On a pair there is nothing to partial out: ``P`` is the ordinary
    coherency, and dDTF is ffDTF weighted by the pair's coherence.
    """

    name = "Direct directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "ddtf"
        super().__init__(**kwargs)
        self._measure = "direct_directed_transfer_function"
        self._recompute = True

    def _get_statistic(self, C):
        # Only the two private properties DirectedCoherence already depends on
        # (see test_directed_coherence_uses_only_the_documented_backend_privates).
        H = C._transfer_function                       # (..., f, target, source)
        Sigma = C._noise_covariance[..., np.newaxis, :, :]
        S = H @ Sigma @ np.conj(np.swapaxes(H, -1, -2))
        Q = np.linalg.inv(S)
        q = np.abs(np.diagonal(Q, axis1=-1, axis2=-2))
        partial_coherence = np.abs(Q) / np.sqrt(
            q[..., :, np.newaxis] * q[..., np.newaxis, :])
        mag = np.abs(H)
        ffdtf = mag / np.sqrt(np.sum(mag ** 2, axis=(-1, -3), keepdims=True))
        return ffdtf * partial_coherence


class PhaseSlopeIndex(NonparametricSpectralMultivariate, Undirected):
    """Phase slope index (Nolte et al. 2008, Phys. Rev. Lett. 100, 234101).

    ``Psi_ij = Im sum_f conj(C_ij(f)) C_ij(f + df)`` over the band, with ``C``
    the complex coherency and ``df`` the frequency resolution of its estimate:
    a coherence-weighted phase difference between neighbouring frequencies. It
    grows with the lag between two processes and its sign says which leads.

    ``df`` is the multitaper bandwidth, ``_FREQUENCY_STEP`` FFT bins: closer
    frequencies share tapers and carry no independent phase. The sign is then
    reliable for lags up to ``T / (2 * _FREQUENCY_STEP)`` samples, beyond which
    the phase difference wraps.

    This is the unnormalised index. Nolte et al. divide it by a jackknife
    standard deviation over epochs, which a single recording does not provide,
    so the value scales with the number of frequencies in the band.

    Computed here rather than by ``Connectivity.phase_slope_index``. In
    spectral-connectivity 2.0 that method sums over *all* pairs of frequencies,
    which is not this index -- it fell as the lag grew and changed sign near 50
    samples -- and later releases change it; doing the sum here keeps the value
    independent of the backend version.
    """

    # Antisymmetric in (i, j): slope of the phase spectrum.
    _antisymmetric_spectrum = True
    name = "Phase slope index"
    labels = ["unsigned", "spectral", "undirected"]
    # A sum over the band, not a band statistic of a spectrum: `max` was
    # accepted and ignored, returning the default variant's matrix under a
    # `_max_` identifier. `mean` is kept as the name the identifiers carry.
    _statistics = ("mean",)

    # Frequency resolution of the multitaper estimate in FFT bins: twice the
    # backend's default time-halfbandwidth product of 3.
    _FREQUENCY_STEP = 6

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        self._measure = "phase_slope_index"
        # Always use `_get_statistic`, as GroupDelay does: the backend method
        # of this name is not the published index (see the class docstring).
        self._recompute = True

    @property
    def _cache_subkey(self):
        # Narrower cache: (class, fs, fmin, fmax) per the key property override.
        return (type(self).__name__, self._fs, self._fmin, self._fmax)

    def _get_statistic(self, C):
        freq = C.frequencies
        # Strictly inside the band, as the backend's own band selection is.
        inside = np.flatnonzero((self._fmin < freq) & (freq < self._fmax))
        coherency = C.coherency()[..., inside[::self._FREQUENCY_STEP], :, :]
        if coherency.shape[-3] < 2:
            raise ValueError(
                f"Phase slope index needs at least two independent frequencies "
                f"inside ({self._fmin}, {self._fmax}); the series is too short "
                f"for this band.")
        return np.imag(np.sum(
            np.conj(coherency[..., :-1, :, :]) * coherency[..., 1:, :, :], axis=-3))


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
    _statistics = ("delay", "slope", "rvalue")

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


# nitime.GrangerAnalyzer's frequency grid: n_freqs // 2 + 1 points on [0, fs/2].
_GC_N_FREQS = 1024


def _fit_bivariate_var(x, order=None, max_order=50):
    """Yule-Walker fit of the two-process VAR behind parametric spectral GC.

    This is ``nitime.analysis.granger.fit_model`` -- Whittle's (LWR)
    recursion, and with ``order=None`` the lowest order at which nitime's BIC
    first stops improving -- with one difference: the autocovariance. nitime's
    ``autocov_vector`` averages lag ``k`` over ``T - k`` products and does not
    remove the mean. That "unbiased" block-Toeplitz matrix is not guaranteed to
    be positive semidefinite, and when it is not the recursion returns an
    indefinite innovation covariance and an unstable model: on the bundled
    ``kuramoto_M7_T100`` fixture, 7 of 21 pairs at order 1 and all 21 at
    orders 2 and 20, so Geweke's ratio came out negative or NaN, and the NaN
    in BIC's ``log det`` stopped the order search from ever turning over.

    The Yule-Walker estimator is defined on the sample autocovariance about
    the mean normalised by ``T`` (Brockwell & Davis 1991, sec. 11.5;
    Luetkepohl 2005, sec. 3.3.4), whose Toeplitz matrix is positive
    semidefinite by construction; that is what makes the fitted model stable
    with a positive-definite innovation covariance. It is computed once, to
    the largest lag needed, and sliced for each candidate order.

    ``max_order`` is the largest order tried (nitime stopped at
    ``max_order - 2``). Returns ``(coef, ecov)`` in nitime's convention,
    ``X[t] + sum_k coef[k] X[t-k] = E[t]``.
    """
    x = x - x.mean(axis=-1, keepdims=True)
    n_obs = x.shape[-1]
    n_lags = (max_order if order is None else order) + 1
    # R[k] = (1/T) sum_t x(t+k) x(t)^T: nitime's lag orientation.
    R = np.stack([x[:, k:] @ x[:, :n_obs - k].T for k in range(n_lags)]) / n_obs
    if order is not None:
        return nt_alg.lwr_recursion(R)

    best = None
    for p in range(max_order + 1):
        coef, ecov = nt_alg.lwr_recursion(R[: p + 1])
        # nitime's criterion and arguments: (ecov, n_process, order, n_process * T).
        crit = nt_utils.bayesian_information_criterion(ecov, 2, p, 2 * n_obs)
        if best is not None and crit > best[0]:
            return best[1:]
        best = (crit, coef, ecov)
    raise ValueError(
        f"Model estimation order did not converge at max_order = {max_order}"
    )


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
                n_proc = data.n_processes
                # F[source, target, f]. nitime's `granger_causality_xy(...)`
                # returns (w, f_x_on_y, f_y_on_x, ...) for the pair (x, y) =
                # (z[i], z[j]), i.e. i -> j and then j -> i.
                F = np.full((n_proc, n_proc, _GC_N_FREQS // 2 + 1), np.nan)
                # freqz's grid, which granger_causality_xy returns as `w`.
                w = np.linspace(0, np.pi, _GC_N_FREQS // 2 + 1, endpoint=False)
                try:
                    for i in range(n_proc):
                        for j in range(i + 1, n_proc):
                            coef, ecov = _fit_bivariate_var(
                                z[[i, j]], self._order, self._max_order)
                            w, F[i, j], F[j, i], _, _ = nt_alg.granger_causality_xy(
                                coef, ecov, n_freqs=_GC_N_FREQS)
                except ValueError as err:
                    # The order search walks the lag up to `max_order` and
                    # raises if the information criterion never turns over.
                    # That is the signature of over-fitting at this record
                    # length, not of a transient numerical problem, and the
                    # bare message ("Model estimation order did not converge at
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

                F = np.transpose(np.expand_dims(F, axis=3), axes=[3, 2, 1, 0])
                # The frequencies the model was evaluated at, in Hz: `w` is
                # scipy.signal.freqz's grid, k*pi/513 rad on [0, pi). nitime's
                # GrangerAnalyzer.frequencies labels the same values
                # linspace(0, fs/2, 513), up to one grid step off, so the band
                # was applied on a stretched axis. And Hz, not cycles/sample:
                # `fs` is in this SPI's identifier and cache key.
                freq = w * self._fs / (2 * np.pi)
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


def _welch_cross_spectra(z, nperseg):
    """Hann-window Welch spectral matrices ``S[f, j, k]`` of ``z`` (processes x time).

    One-sided rfft bins; segments of ``nperseg`` samples start every
    ``nperseg // 2`` (a remainder shorter than a segment is dropped), and each
    segment is demeaned before windowing. Unscaled: the window and sampling
    normalisations cancel in any coherence-type ratio.
    """
    starts = np.arange(0, z.shape[1] - nperseg + 1, nperseg // 2)
    segments = z[:, starts[:, None] + np.arange(nperseg)]          # (M, K, n)
    segments = segments - segments.mean(axis=-1, keepdims=True)
    X = np.fft.rfft(segments * signal.get_window("hann", nperseg), axis=-1)
    X = np.transpose(X, (2, 0, 1))                                  # (F, M, K)
    return X @ np.conj(np.swapaxes(X, 1, 2)) / len(starts)


class PhaseAlignedCorrelation(Undirected, Unsigned):
    """Phase-aligned correlation: zero-lag correlation maximised over all-pass filters.

    ``P_jk = sum_f w_f |S_jk(f)| / sqrt(sum_f w_f S_jj(f) * sum_f w_f S_kk(f))``

    from Welch cross-spectra (``_welch_cross_spectra``), where ``w_f`` counts
    each one-sided bin's multiplicity on the two-sided grid: 1 at DC and, for
    even ``nperseg``, at Nyquist; 2 elsewhere. The denominator is then the
    product of the two signals' total powers.

    In the population, ``S_jk = |S_jk| exp(i phi_jk)`` gives the zero-lag
    correlation ``r = int |S_jk| cos(phi_jk) / N`` and the lag-``h``
    cross-correlation ``int |S_jk| cos(phi_jk + 2 pi f h) / N``. Filtering one
    process by an all-pass filter shifts ``phi_jk`` by any phase per frequency,
    and the best choice cancels it, so ``P`` is the supremum over such filters
    and ``r <= max_h xcorr(h) <= P``: no realignment, one global delay, any phase
    per frequency. A pure delay with positive gain has linear phase, so there
    ``max xcorr == P``; ``max xcorr < P`` flags dispersion (frequency-dependent
    delay). Undirected, unsigned (invariant to the sign of either process) and in
    [0, 1] by Cauchy-Schwarz.

    Equivalently ``P`` is the power-weighted mean coherence magnitude,
    ``sum_f w_f sqrt(S_jj S_kk) |C_jk| / N``. ``cohmag`` is instead the
    unweighted mean *squared* coherence magnitude, so it scores a pair coupled
    only in a narrow high-power band as weakly as one coupled in a narrow
    low-power band, which ``P`` does not.

    Estimation. On a raw periodogram every bin is perfectly coherent and ``P`` is
    about ``pi / 4`` for independent noise. Averaging ``K`` segments restricts
    the realignment to phases smooth on the scale ``1 / nperseg`` and leaves a
    null floor of about ``sqrt(pi / (4 K))`` (measured 0.074 against 0.071 at
    ``T = 20000``, ``nperseg = 256``, ``K = 155``). The cost of a short segment
    is resolution: a delay of ``d`` samples is only partly inside one segment,
    and ``P`` of a pure delay is the Hann window's normalised autocorrelation at
    ``d`` -- 0.98 at ``d = nperseg / 20``, 0.94 at ``nperseg / 10`` and 0.47 at
    ``nperseg / 3``.

    ``nperseg=None`` picks the largest power of two not above ``T / 8``, within
    [8, 256]. From ``T = 64`` to 2047 that keeps ``K`` between 15 and 31 (floor
    0.16-0.23; ``T = 1000``: 64 samples, ``K = 30``); from ``T = 2048`` the
    segment is 256 samples and ``K`` grows with ``T``. Below ``T = 64`` the
    segment stays at 8 samples and the floor rises (``T = 30``: ``K = 6``,
    measured 0.37), and the automatic choice refuses a record with fewer than
    four segments (``T < 20``). Pass an integer to fix the resolution across
    lengths; the identifier records ``auto`` or the integer.
    """

    name = "Phase-aligned correlation"
    identifier = "phasealigned"
    labels = ["unsigned", "spectral", "undirected"]

    _MAX_AUTO_NPERSEG = 256
    _MIN_AUTO_NPERSEG = 8
    _MIN_AUTO_SEGMENTS = 4

    def __init__(self, nperseg=None):
        if nperseg is not None:
            nperseg = require_int("nperseg", nperseg, minimum=2)
        self._nperseg = nperseg
        self.identifier = (f"{self.identifier}_welch_nperseg-"
                           f"{'auto' if nperseg is None else nperseg}")

    def _segment_length(self, T):
        if self._nperseg is not None:
            if self._nperseg > T:
                raise ValueError(f"nperseg={self._nperseg} exceeds the series "
                                 f"length T={T}.")
            return self._nperseg
        n = max(self._MIN_AUTO_NPERSEG, 1 << max((T // 8).bit_length() - 1, 0))
        n = min(self._MAX_AUTO_NPERSEG, n)
        shortest = n + (self._MIN_AUTO_SEGMENTS - 1) * (n // 2)
        if T < shortest:
            raise ValueError(
                f"Phase-aligned correlation needs T >= {shortest} for "
                f"{self._MIN_AUTO_SEGMENTS} segments of the automatic length "
                f"{n} (got T={T}); pass nperseg to override.")
        return n

    def _matrix(self, z):
        nperseg = self._segment_length(z.shape[1])
        S = _welch_cross_spectra(z, nperseg)
        w = np.full(S.shape[0], 2.0)
        w[0] = 1.0
        if nperseg % 2 == 0:
            w[-1] = 1.0
        power = np.einsum("f,fjj->j", w, S).real
        return np.einsum("f,fjk->jk", w, np.abs(S)) / np.sqrt(np.outer(power, power))

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return float(self._matrix(data.to_numpy(squeeze=True)[[i, j]])[0, 1])

    @parse_multivariate
    def multivariate(self, data):
        P = self._matrix(data.to_numpy(squeeze=True))
        # |S_jk| and |S_kj| are summed in different orders by the matmul, so
        # mirror one triangle, as Undirected.multivariate does.
        lower = np.tril_indices(P.shape[0], -1)
        P[lower] = P.T[lower]
        np.fill_diagonal(P, np.nan)
        return P

import functools
import itertools
import logging
import numbers
import warnings
from contextlib import contextmanager
from copy import deepcopy

import nitime.algorithms as nt_alg
import nitime.utils as nt_utils
import numpy as np
import spectral_connectivity as sc
from scipy import signal
from scipy.fft import ifft
from spectral_connectivity.connectivity import _estimate_spectral_granger_prediction
from spectral_connectivity.transforms import prepare_time_series

from pyspi.base import (
    Directed,
    NotAppropriate,
    Undirected,
    Unsigned,
    fill_pairwise,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import (
    empty_band_error,
    fmt_param,
    require_band,
    require_int,
    require_positive_float,
    singular_covariance_error,
    unit_diagonal_ratio,
)


@contextmanager
def _collect_backend_log():
    """
    Collect the warnings ``spectral_connectivity`` logs inside the block.

    Yields a list that fills with the message of each record, in order. When
    Wilson's factorisation reaches its iteration cap, the backend logs a
    warning and returns the unconverged factor, from which every
    Wilson-derived measure (directed coherence, DTF, dDTF, PDC, gPDC,
    nonparametric spectral GC) is computed. The messages are kept with the
    cached result, so that every SPI that reads it can warn for itself
    (``_warn_factorisation_log``).
    """
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    backend = logging.getLogger("spectral_connectivity")
    backend.addHandler(handler)
    try:
        yield records
    finally:
        backend.removeHandler(handler)


# The start of the message spectral_connectivity logs when Wilson's algorithm
# stops at its iteration cap.
_WILSON_ITERATION_CAP = "Maximum iterations reached"


def _list_pairs(pairs, limit=10):
    """``(0, 1), (1, 0)``: the first ``limit`` pairs, then an ellipsis."""
    shown = ", ".join(f"({i}, {j})" for i, j in pairs[:limit])
    return shown + (", ..." if len(pairs) > limit else "")


def _warn_factorisation_log(records, n_pairs, unit):
    """
    Warn about what the backend logged while factorising.

    ``records`` holds ``(pair, message)`` for each logged message, with
    ``pair`` the process indices ``(i, j)``, or None where the backend does
    not say which pair it was factorising. A factorisation that reached
    Wilson's iteration cap gives one warning listing the pairs; any other
    message is passed on as it is, once. pyspi records per-SPI diagnostics
    from the ``warnings`` channel only (``_parallel.execute_spi``), which
    attributes each warning to the SPI being computed, and an SPI that reads
    a cached factorisation warns as the one that computed it did. It only
    warns; it changes no value.
    """
    capped = [(pair, m) for pair, m in records
              if m.startswith(_WILSON_ITERATION_CAP)]
    if capped:
        pairs = sorted(pair for pair, _ in capped if pair is not None)
        named = f": {_list_pairs(pairs)}" if pairs else ""
        detail = "; ".join(dict.fromkeys(m for _, m in capped))
        warnings.warn(
            f"Wilson's spectral factorisation did not "
            f"converge for {len(capped)} of {n_pairs} {unit}{named}. Their "
            f"values come from the unconverged factor (spectral_connectivity: "
            f"{detail}).", RuntimeWarning)
    for message in dict.fromkeys(m for _, m in records
                                 if not m.startswith(_WILSON_ITERATION_CAP)):
        warnings.warn(f"spectral_connectivity: {message}", RuntimeWarning)


def _dependent_spectrum_error(csm, i, j):
    """
    The ``NotAppropriate`` for a pair whose spectrum cannot be factorised, or None.

    ``csm`` is the pair's ``2 x 2`` cross-spectral matrix at every frequency.
    The backend starts Wilson's iteration from the Cholesky factor of the
    zero-lag covariance, the real part of the inverse FFT of ``csm`` at lag 0,
    and draws a random start from NumPy's global stream when that Cholesky
    fails. That covariance is formed directly from the spectrum, and the pair
    is refused when it is singular by ``pyspi.utils.SINGULAR_RCOND``
    (``singular_covariance_error``). At that tolerance the Cholesky of every
    pair that is kept succeeds, so the random start is never reached. The
    real part is the backend's: the inverse FFT of a Hermitian spectrum is
    real up to rounding.
    """
    lag0 = ifft(csm, axis=-3)[..., 0, :, :].real.reshape(2, 2)
    return singular_covariance_error(
        "Wilson's spectral factorisation", i, j, lag0,
        "their cross-spectral matrix is singular and has no spectral factor",
        of="their zero-lag covariance")


def _ensure_time_series_3d(z):
    """Ensure time-series array follows (n_time, n_trials, n_signals)."""
    if z.ndim == 2:
        return prepare_time_series(z, axis="signals")
    return z


def _circular_nanmean(angles, axis):
    """
    Circular mean angle, omitting NaNs along ``axis``.

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


def _real_valued_bins(freq, fs):
    """
    Mask of the bins where the cross-spectrum of real signals is real.

    0 Hz always, and ``fs / 2`` when it is on the grid (an even FFT length),
    compared with a relative tolerance because the backend's grid is
    ``k * fs / nfft`` in floating point.
    """
    return (freq == 0) | np.isclose(freq, fs / 2, rtol=1e-9, atol=0)


def _closed_band(freq, fmin, fmax):
    """
    Mask of ``fmin <= f <= fmax`` with a relative 1e-9 tolerance at both edges.

    The tolerance keeps edge bins such as Nyquist, which the backend's
    floating-point grid ``k * fs / nfft`` can place just past ``fs / 2``.
    """
    freq = np.asarray(freq)
    above = (freq >= fmin) | np.isclose(freq, fmin, rtol=1e-9, atol=0)
    below = (freq <= fmax) | np.isclose(freq, fmax, rtol=1e-9, atol=0)
    return above & below


# spectral_connectivity's default time-halfbandwidth product, which the bundled
# identifiers leave unnamed.
_DEFAULT_NW = 3

# PhaseSlopeIndex's default group width is _PSI_WIDTH * fs / sqrt(T): it
# narrows with T, so the longest lag whose sign is right grows as sqrt(T),
# while each group averages about _PSI_WIDTH * sqrt(T) FFT bins.
_PSI_WIDTH = 0.4


def _check_time_halfbandwidth_product(nw):
    """
    Validated ``NW``: a real number of at least 1.5, so that there are two or
    more tapers (``floor(2 NW) - 1``); integral values are returned as ``int``.
    """
    if (isinstance(nw, bool) or not isinstance(nw, numbers.Real)
            or not np.isfinite(nw) or nw < 1.5):
        raise ValueError(
            f"time_halfbandwidth_product must be a real number >= 1.5 (at least "
            f"two tapers), got {nw!r}.")
    return int(nw) if float(nw).is_integer() else float(nw)


def _nw_suffix(nw):
    """Identifier suffix for a non-default time-halfbandwidth product."""
    if nw == _DEFAULT_NW:
        return ""
    return f"_nw-{fmt_param(nw)}".replace(".", "-")


class NonparametricSpectral(Unsigned):
    """
    Base class of the multitaper spectral SPIs, computed with spectral_connectivity.

    The spectral matrix is a whole-record multitaper estimate [1]_ with
    ``floor(2 NW) - 1`` DPSS tapers, on the grid ``k fs / n_fft``, with a
    frequency resolution of ``2 NW / T`` cycles per sample. A larger ``NW``
    averages more tapers, which lowers the variance and the upward bias of
    coherence-type measures on independent series, at the cost of frequency
    resolution. GroupDelay spaces its independent frequencies one bandwidth,
    ``2 NW`` FFT bins, apart, and PhaseSlopeIndex averages the spectra over
    groups of frequencies about one bandwidth wide or wider.

    Each subclass reduces its spectrum over the
    closed band ``fmin <= f <= fmax``, compared with a relative tolerance of
    1e-9 at both edges (``_closed_band``) so that a grid point such as Nyquist
    is kept. Phase measures (``_phase_based``) also leave out the real-valued
    bins, 0 Hz and ``fs / 2``: there every taper's Fourier coefficient of a
    real signal is real, so the cross-spectral phase is exactly 0 or pi and
    carries no estimate, and a phase measure's band of zero width at 0 or
    ``fs / 2`` is refused at construction. A band left with no frequency of
    the series' grid, which depends on ``T``, raises ``NotAppropriate`` when
    the SPI is computed, which the Calculator records as NaN with the
    reason.

    Frequencies are in the units of the sampling frequency ``fs``: at the
    default ``fs = 1``, normalised frequency in cycles per sample, with the
    Nyquist frequency at 0.5. ``fs`` only relabels the frequency axis, so
    ``fs = 250`` with the band [8, 12] gives the same values as ``fs = 1``
    with [0.032, 0.048], up to floating-point rounding; GroupDelay's
    ``slope``, in radians per unit of frequency, is the one output that
    scales with it. At construction ``fs`` must be positive and finite and
    ``0 <= fmin <= fmax <= fs / 2`` (the upper edge with a relative tolerance
    of 1e-9), or ``ValueError`` is raised.

    For a measure that is antisymmetric at every frequency
    (``_antisymmetric_spectrum``), ``statistic='mean'`` keeps
    ``A[i, j] = -A[j, i]`` and ``'max'`` does not (``max(-x) != -max(x)``),
    so the ``max`` variants are labelled asymmetric. Both are signed.

    References
    ----------
    .. [1] D.J. Thomson, "Spectrum estimation and harmonic analysis",
           Proceedings of the IEEE 70(9), 1055-1096 (1982).

    Parameters
    ----------
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5 (two
        or more tapers). Every multitaper class takes it, including
        SpectralGrangerCausality with ``method="nonparametric"``; the
        parametric method refuses a value other than 3. A value other than 3
        is appended to the identifier as ``_nw-<value>``, with dots written as
        dashes (``_nw-8``, ``_nw-2-5``). Default: 3.
    """

    # The `statistic` values a class can apply. A band statistic reduces a
    # spectrum over frequency; GroupDelay's three are outputs of a regression.
    _statistics = ("mean", "max")

    # A function of the phase of the cross-spectrum: its band leaves out the
    # real-valued bins (see the class docstring).
    _phase_based = False

    # Needs a band of positive width (`fmin < fmax`) whatever the series.
    _positive_width = False

    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean",
                 time_halfbandwidth_product=_DEFAULT_NW):
        require_positive_float("fs", fs)
        if fmax is None:
            fmax = fs / 2
        require_band(fs, fmin, fmax, strict=self._positive_width)
        # A single real-valued bin is all such a band can hold at any length.
        if self._phase_based and fmin == fmax and (
                fmin == 0 or np.isclose(fmax, fs / 2, rtol=1e-9, atol=0)):
            raise ValueError(
                f"{type(self).__name__}: the band [{fmin}, {fmax}] holds only "
                f"{'0 Hz' if fmin == 0 else 'fs / 2'}, where the cross-spectrum "
                f"of real signals has no phase; widen it.")

        # `fs` reaches the multitaper estimate, the frequency grid, every cache
        # key and GroupDelay's conversion to samples.
        self._fs = fs
        self._nw = _check_time_halfbandwidth_product(time_halfbandwidth_product)
        self._fmin = fmin
        self._fmax = fmax
        if statistic not in self._statistics:
            raise ValueError(
                f"{type(self).__name__} does not support statistic="
                f"{statistic!r}; it supports {', '.join(self._statistics)}.")
        self._statfn = {"mean": np.nanmean, "max": np.nanmax}.get(statistic)
        self._statistic = statistic

        # PLI, wPLI, PSI and coherence phase carry lead/lag in a sign,
        # A[i, j] == -A[j, i]: neither undirected (symmetric) nor directed. Only
        # the mean keeps the antisymmetry (max(-x) != -max(x)).
        if getattr(self, "_antisymmetric_spectrum", False):
            trait = "antisymmetric" if statistic == "mean" else "asymmetric"
            self.labels = [
                l for l in self.labels
                if l not in ("undirected", "directed")
            ] + [trait, "signed"]
            # Signed in behaviour too: `Calculator._rmmin` shifts SPIs that are
            # not signed by their minimum, which would destroy the lead/lag
            # sign.
            self.issigned = lambda: True

        paramstr = (
            f"_multitaper_{statistic}_fs-{fmt_param(fs)}_fmin-{fmt_param(fmin)}"
            f"_fmax-{fmt_param(fmax)}".replace(
                ".", "-"
            )
        )
        self.identifier += paramstr + _nw_suffix(self._nw)

    @property
    def _frequency_step(self):
        """
        One multitaper bandwidth, ``2 NW`` FFT bins: estimates closer than
        this share the tapers' main lobes and carry no independent phase.
        """
        return max(1, int(round(2 * self._nw)))

    @property
    def key(self):
        """
        Cache key: every parameter that changes the cached result.

        ``fs`` is part of the key because it is passed to ``sc.Multitaper`` and
        therefore changes both the connectivity estimate and the frequency grid;
        ``NW`` because it sets the tapers.

        GroupDelay and PhaseSlopeIndex additionally cache a band-dependent
        statistic, so their key carries fmin/fmax as well.
        """
        base = (self.measure, self._fs, self._nw)
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

    def _band(self, freq):
        """
        Indices of the frequencies the band statistic reduces over.

        The closed band, less the real-valued bins for a phase measure (see
        ``NonparametricSpectral``). A band left with no frequency raises
        ``NotAppropriate``.
        """
        in_band = _closed_band(freq, self._fmin, self._fmax)
        keep = in_band & ~_real_valued_bins(freq, self._fs) if self._phase_based else in_band
        if not keep.any():
            what = ("only the real-valued bins (0 Hz and fs/2), where the "
                    "cross-spectrum of real signals has no phase"
                    if in_band.any() else "no frequency of the series' grid")
            step = f" (grid step {freq[1] - freq[0]:.4g})" if freq.size > 1 else ""
            raise empty_band_error(type(self).__name__, self._fmin, self._fmax,
                                   what + step)
        return np.flatnonzero(keep)

    def _band_statistic(self, spectra):
        """
        The band statistic of ``spectra[f, i, j]`` for each off-diagonal cell.

        The diagonal is NaN: several backend measures are NaN there at every
        frequency, and a NaN-ignoring statistic of an all-NaN column would
        warn on every call.
        """
        n = spectra.shape[-1]
        off = ~np.eye(n, dtype=bool)
        values = self._statfn(spectra[:, off], axis=0)
        adj = np.full((n, n), np.nan, dtype=np.result_type(values, float))
        adj[off] = values
        return adj


def _to_source_target(spi, adj):
    """
    Put a spectral adjacency matrix into pyspi's (source, target) convention.

    pyspi fills ``A[i, j] = bivariate(i, j)``, row = source, column = target
    (``base.Directed.multivariate``). The spectral backends (spectral_connectivity,
    nitime) follow the DTF/PDC literature, where ``[i, j]`` is the flow into
    ``i`` from ``j``, so directed SPIs are transposed. The others are not:
    several (PhaseLagIndex, WeightedPhaseLagIndex, PhaseSlopeIndex) are
    antisymmetric, and transposing them would negate them. The test is
    ``not isinstance(spi, Undirected)`` because ``Undirected`` subclasses
    ``Directed``.
    """
    return adj if isinstance(spi, Undirected) else adj.T


class NonparametricSpectralMultivariate(NonparametricSpectral):
    """
    Measures read from one multitaper ``Connectivity`` of all processes.

    Each measure's full-spectrum result is cached on the dataset under
    ``(measure, fs, NW)``, so configurations that differ only in band or
    statistic compute it once; ``GroupDelay`` and ``PhaseSlopeIndex`` also key
    on the band. A subclass names the backend measure in ``_measure`` or, with
    ``_recompute``, computes its own in ``_get_statistic``. The band statistic
    is applied in ``multivariate``, and directed results are then put into
    pyspi's (source, target) orientation by ``_to_source_target``.
    """

    _cache_namespace = "spectral_mv"

    @property
    def _cache_subkey(self):
        # One entry per (class, fs, NW); GroupDelay and PhaseSlopeIndex also
        # key on the band. fs and NW mirror `key`.
        return (type(self).__name__, self._fs, self._nw)

    def _get_cache(self, data):
        # One key type throughout: written and read under the tuple self.key.
        cache = getattr(data, "spectral_mv", None)
        if cache is None:
            cache = data.spectral_mv = {}

        if self.key in cache:
            return cache[self.key], cache[self._freq_key]

        z = np.transpose(data.to_numpy(squeeze=True))
        z = _ensure_time_series_3d(z)
        m = sc.Multitaper(z, sampling_frequency=self._fs,
                          time_halfbandwidth_product=self._nw)
        conn = sc.Connectivity.from_multitaper(m)
        # `_recompute` replaces a backend measure outright, not only when the
        # backend raises. Connectivity is lazy, so the factorisation (and its
        # log warnings) happens inside this block.
        with _collect_backend_log() as log:
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
        cache[self._log_key] = tuple(log)
        return res, freq

    @property
    def _log_key(self):
        """Cache key of the backend's log messages from computing ``self.key``."""
        return ("backend_log",) + self.key

    def _backend_log(self, data):
        """The messages the backend logged while computing this SPI's cached result."""
        return getattr(data, "spectral_mv", {}).get(self._log_key, ())

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        freq_id = self._band(freq)

        try:
            adj = self._band_statistic(adj_freq[0, freq_id, :, :])
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
        M = data.n_processes
        _warn_factorisation_log([(None, m) for m in self._backend_log(data)],
                                M * (M - 1) // 2, "pairs")
        return adj


class _FactoredConnectivity(sc.Connectivity):
    """
    A pair's ``Connectivity``, reduced to its spectral factorisation.

    The directed measures are read off the Wilson minimum-phase factor through
    the transfer function, the innovation covariance and the MVAR
    coefficients, which the backend factorises afresh on every access. Here the
    factorisation is done once; each quantity is a deterministic function of
    the same spectrum, so no value changes. The Fourier coefficients and the
    factor are then released, since one object is held per ordered pair.
    What the backend logged while factorising (an iteration cap reached, say)
    is kept as ``factorisation_log``. A pair of linearly dependent series
    (``_dependent_spectrum_error``) is not factorised: every quantity read
    from the factor raises ``NotAppropriate`` with the reason, which is kept
    as ``refusal``. ``pair`` holds the processes ``(i, j)`` of the dataset,
    for the message; by default those of the Connectivity itself.
    """

    factorisation_log = ()
    pair = (0, 1)
    refusal = None

    @functools.cached_property
    def _minimum_phase_factor(self):
        return super()._minimum_phase_factor

    def _factorise(self):
        held = vars(self)
        if self.refusal is not None:
            raise NotAppropriate(self.refusal)
        if "_noise_covariance" not in held:
            err = _dependent_spectrum_error(
                self._expectation_cross_spectral_matrix(), *self.pair)
            if err is not None:
                self.refusal = str(err)
                self.fourier_coefficients = None
                raise err
            # The parent's properties read the factor through `self`, so it is
            # computed by the cached property above exactly once for the two.
            with _collect_backend_log() as log:
                transfer = np.array(super()._transfer_function)
                held["_noise_covariance"] = np.array(super()._noise_covariance)
            held["_transfer_function"] = transfer
            held["factorisation_log"] = tuple(log)
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


@functools.lru_cache(maxsize=8)
def _whole_record_tapers(n_time_samples, fs, nw):
    """
    The DPSS tapers ``sc.Multitaper`` uses for one whole-record window.

    They depend on the record length, sampling rate and time-halfbandwidth
    product only, so they are computed once rather than for every pair's
    Multitaper.
    """
    tapers = sc.Multitaper(np.zeros((n_time_samples, 1, 1)),
                           sampling_frequency=fs,
                           time_halfbandwidth_product=nw).tapers
    tapers.setflags(write=False)
    return tapers


class NonparametricSpectralBivariate(NonparametricSpectral):
    """
    Measures read from a pair's multitaper ``Connectivity`` and its factorisation.

    The directed spectral measures (DC, PDC, gPDC, DTF, dDTF) need the pair's
    transfer function ``H(f)``, obtained by Wilson's factorisation of the
    pair's spectral matrix. It is computed once per ``(fs, NW, i, j)`` and
    shared by every subclass. The measures are evaluated pair by pair, so they
    are bivariate, not conditioned on the other processes. Where Wilson's
    algorithm reaches its iteration cap, the values come from the unconverged
    factor, and every SPI that reads it warns, naming the ordered pairs. A
    pair of linearly dependent series, whose zero-lag covariance scaled to
    unit diagonal has a smallest/largest eigenvalue of at most 1e-12
    (``pyspi.utils.SINGULAR_RCOND``), has a singular spectral matrix and no
    factor: it raises ``NotAppropriate``, which the Calculator records as NaN
    for that pair with the reason, and the other pairs keep their values. A
    copy delayed by a lag is not refused: its measures are finite estimates
    of an infinite quantity.
    """

    _cache_namespace = "spectral_bv"

    @property
    def _cache_subkey(self):
        # One bucket for every directed measure at a sampling frequency: they
        # share the pair's factorisation, which is nearly all of the cost.
        return (self._fs,)

    def _get_cache(self, data, i, j):
        """
        Cache Connectivity object per (i,j) pair, not per (measure,i,j).

        Multiple directed spectral SPIs (DirectedCoherence, PartialDirectedCoherence,
        etc.) share the same Multitaper+Connectivity for a given (i,j), and with
        it the spectral factorisation every one of them is derived from.
        """
        # fs and NW belong in both keys: they are passed to Multitaper, so they
        # change the Connectivity object as well as the measure.
        measure_key = (self.measure, self._fs, self._nw, i, j)
        cache = getattr(data, "spectral_bv", None)
        if cache is None:
            cache = data.spectral_bv = {}

        if measure_key in cache:
            return cache[measure_key], cache[self._freq_key]

        conn_key = (self._fs, self._nw, i, j)
        conns = getattr(data, "_spectral_bv_conn", None)
        if conns is None:
            conns = data._spectral_bv_conn = {}
        conn = conns.get(conn_key)
        if conn is None:
            z = np.transpose(data.to_numpy(squeeze=True)[[i, j]])
            z = _ensure_time_series_3d(z)
            m = sc.Multitaper(
                z, sampling_frequency=self._fs,
                time_halfbandwidth_product=self._nw,
                tapers=_whole_record_tapers(z.shape[0], self._fs, self._nw))
            conn = _FactoredConnectivity.from_multitaper(m)
            conn.pair = (i, j)
            conns[conn_key] = conn

        # `_recompute` replaces a backend measure outright, not only when the
        # backend raises. The backend logs only while factorising, and
        # `_FactoredConnectivity` keeps those messages.
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

    def _pair_value(self, data, i, j):
        bv_freq, freq = self._get_cache(data, i, j)
        # Closed band, as in NonparametricSpectralMultivariate.
        return self._statfn(bv_freq[self._band(freq)])

    def _warn_factorisation(self, data, pairs, n_pairs):
        """Warn about what the backend logged while factorising ``pairs``."""
        conns = getattr(data, "_spectral_bv_conn", {})
        records = []
        for i, j in pairs:
            conn = conns.get((self._fs, self._nw, i, j))
            if conn is not None:
                records += [((i, j), m) for m in conn.factorisation_log]
        _warn_factorisation_log(records, n_pairs, "ordered pairs")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        value = self._pair_value(data, i, j)
        self._warn_factorisation(data, [(i, j)], 1)
        return value

    @parse_multivariate
    def multivariate(self, data):
        """
        The ``M x M`` matrix of every ordered pair, with one convergence warning.

        Computed pair by pair as in ``Directed``; the warning lists every
        ordered pair whose factorisation did not converge.
        """
        M = data.n_processes
        A = np.full((M, M), np.nan)
        pairs = [(i, j) for j in range(M) for i in range(M) if i != j]
        fill_pairwise(self.identifier, A, pairs,
                      lambda i, j: self._pair_value(data, i, j))
        self._warn_factorisation(data, pairs, len(pairs))
        return A


class CoherenceMagnitude(NonparametricSpectralMultivariate, Undirected):
    """
    Magnitude-squared coherence from a multitaper spectral estimate [1]_.

    ``|S_ij(f)|^2 / (S_ii(f) S_jj(f))``, the squared modulus of the coherency
    (the backend's ``coherence_magnitude``, whose name the identifier
    ``cohmag`` keeps), reduced over the band. Undirected and unsigned, in
    [0, 1].

    References
    ----------
    .. [1] D.J. Thomson, "Spectrum estimation and harmonic analysis",
           Proc. IEEE 70(9), 1055-1096 (1982).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    name = "Magnitude-squared coherence"
    labels = ["spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "cohmag"
        super().__init__(**kwargs)
        self._measure = "coherence_magnitude"


class CoherencePhase(NonparametricSpectralMultivariate, Undirected):
    """
    Phase of the coherency, circular-averaged over the band [1]_.

    ``arg S_ij(f)``, the phase of the multitaper cross-spectrum, in radians,
    averaged over the band through unit phasors (a circular mean).
    Antisymmetric and signed: ``A[i, j] = -A[j, i]``; positive means process
    ``i`` leads. NaN where the mean phasor is zero or points to +/-pi, which
    has no sign. The band leaves out 0 Hz and ``fs / 2``, where the phase of
    real signals is exactly 0 or pi. A delay winds the phase with frequency,
    so over a band wide enough for it to wrap the mean loses the lead/lag.

    References
    ----------
    .. [1] D.J. Thomson, "Spectrum estimation and harmonic analysis",
           Proc. IEEE 70(9), 1055-1096 (1982).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        Only ``'mean'``, the circular mean: wrapped phase has no ordinary
        maximum. Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    # Antisymmetric in (i, j): phi(i, j) = -phi(j, i).
    _antisymmetric_spectrum = True
    _phase_based = True
    name = "Coherence phase"
    labels = ["signed", "timing", "spectral", "antisymmetric"]

    def __init__(self, **kwargs):
        if kwargs.get("statistic", "mean") != "mean":
            raise ValueError(
                "CoherencePhase supports only statistic='mean'. Wrapped phase "
                "has no branch-cut-independent ordinary maximum; use a "
                "different circular summary with an explicit interpretation."
            )
        self.identifier = "phase"
        super().__init__(**kwargs)
        self.labels = ["signed", "timing", "spectral", "antisymmetric"]
        self._measure = "coherence_phase"

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        adj = _circular_nanmean(adj_freq[0, self._band(freq), :, :], axis=0)

        # Coherency phase is antisymmetric. Select one computed orientation and
        # construct its opposite explicitly so branch-cut representations at
        # +/-pi and backend round-off cannot violate the declared structure.
        ui = np.triu_indices(data.n_processes, 1)
        adj[ui] = -adj.T[ui]
        np.fill_diagonal(adj, np.nan)
        return adj


class ImaginaryCoherence(NonparametricSpectralMultivariate, Undirected):
    """
    Magnitude of the imaginary part of coherency [1]_.

    ``|Im S_ij(f)| / sqrt(S_ii(f) S_jj(f))``, with the cross-spectrum averaged
    over tapers, reduced over the band. Zero for coupling at zero lag, whose
    coherency is real, so instantaneous mixing (such as volume conduction)
    does not contribute. Undirected and unsigned, in [0, 1]: the backend takes
    the absolute value, so the sign that says which process leads is not
    kept. The band leaves out 0 Hz and ``fs / 2``.

    References
    ----------
    .. [1] G. Nolte, O. Bai, L. Wheaton, Z. Mari, S. Vorbach, M. Hallett,
           "Identifying true brain interaction from EEG data using the
           imaginary part of coherency", Clin. Neurophysiol. 115(10),
           2292-2307 (2004).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    _phase_based = True
    name = "Imaginary coherence"
    labels = ["spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "icoh"
        super().__init__(**kwargs)
        self._measure = "imaginary_coherence"


class PhaseLockingValue(NonparametricSpectralMultivariate, Undirected):
    """
    Phase locking value across tapers [1]_.

    At each frequency, ``|mean_k exp(i arg S_k(f))|``, ``S_k`` being taper
    ``k``'s cross-spectrum, then the band statistic. Lachaux et al. average
    over trials; a single record has one, so the average here runs over the
    tapers. Undirected and unsigned, in [0, 1]. With few tapers it is biased
    upward for independent processes; :class:`PairwisePhaseConsistency` is
    the bias-free counterpart. The band leaves out 0 Hz and ``fs / 2``.

    References
    ----------
    .. [1] J.-P. Lachaux, E. Rodriguez, J. Martinerie, F.J. Varela, "Measuring
           phase synchrony in brain signals", Hum. Brain Mapp. 8(4), 194-208
           (1999).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    _phase_based = True
    name = "Phase locking value"
    labels = ["spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "plv"
        super().__init__(**kwargs)
        self._measure = "phase_locking_value"

        myfn = deepcopy(self._statfn)
        self._statfn = lambda x, **kwargs: myfn(np.absolute(x), **kwargs)


class PhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    """
    Phase lag index, signed [1]_.

    At each frequency, the mean over tapers of ``sign(Im S_k(f))``, ``S_k``
    being taper ``k``'s cross-spectrum, then the band statistic. This is the
    backend's signed index, not Stam et al.'s ``|PLI|``: it is in [-1, 1]
    with mean 0 under independence, whereas ``|PLI|`` is biased upward at a
    finite number of tapers. With ``statistic='mean'``,
    ``A[i, j] = -A[j, i]``; positive means process ``i`` leads. The ``max``
    variants are signed but not antisymmetric.

    A delay of ``d`` samples winds the cross-spectral phase by ``2 pi f d``.
    Where the phase passes pi inside the band the per-frequency signs cancel,
    so the band mean can fall, flip sign or vanish as the delay grows; the
    debiased squared forms discard the sign and avoid this, at the cost of the
    direction. The band leaves out 0 Hz and ``fs / 2``.

    References
    ----------
    .. [1] C.J. Stam, G. Nolte, A. Daffertshofer, "Phase lag index:
           assessment of functional connectivity from multi channel EEG and
           MEG with diminished bias from common sources", Hum. Brain Mapp.
           28(11), 1178-1193 (2007).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` (antisymmetric) or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    # Antisymmetric in (i, j): sign of the imaginary coherency.
    _antisymmetric_spectrum = True
    _phase_based = True
    name = "Phase lag index"
    labels = ["spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "pli"
        super().__init__(**kwargs)
        self._measure = "phase_lag_index"


class WeightedPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    """
    Weighted phase lag index, signed [1]_.

    At each frequency, ``E[Im S_k(f)] / E[|Im S_k(f)|]`` over the tapers,
    ``S_k`` being taper ``k``'s cross-spectrum, then the band statistic. In
    [-1, 1] with mean 0 under independence. With ``statistic='mean'``,
    ``A[i, j] = -A[j, i]``; positive means process ``i`` leads. The ``max``
    variants are signed but not antisymmetric. As for
    :class:`PhaseLagIndex`, where the cross-spectral phase passes pi inside
    the band the band mean can fall or cancel as the delay grows. The band
    leaves out 0 Hz and ``fs / 2``.

    References
    ----------
    .. [1] M. Vinck, R. Oostenveld, M. van Wingerden, F. Battaglia,
           C.M.A. Pennartz, "An improved index of phase-synchronization for
           electrophysiological data in the presence of volume-conduction,
           noise and sample-size bias", NeuroImage 55(4), 1548-1565 (2011).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` (antisymmetric) or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    # Antisymmetric in (i, j): imaginary-coherency weighted sign.
    _antisymmetric_spectrum = True
    _phase_based = True
    name = "Weighted phase lag index"
    labels = ["spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "wpli"
        super().__init__(**kwargs)
        self._measure = "weighted_phase_lag_index"


class DebiasedSquaredPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    """
    Debiased squared phase lag index [1]_.

    The mean of ``sign(Im S_k) sign(Im S_l)`` over pairs ``k != l`` of the
    ``n`` tapers (and trials), ``S_k`` being taper ``k``'s cross-spectrum:
    ``((sum s)^2 - sum s^2) / (n (n - 1))``, then the band statistic. An
    estimator of ``PLI^2`` without the upward bias of squaring a finite-taper
    estimate, so it can be negative. Undirected (symmetric). The band leaves
    out 0 Hz and ``fs / 2``.

    Computed here rather than by the backend, whose shortcut
    ``(n PLI^2 - 1) / (n - 1)`` equals this only when every sign is nonzero
    (it gives ``-1/(n - 1)`` where the cross-spectrum is real).

    References
    ----------
    .. [1] M. Vinck, R. Oostenveld, M. van Wingerden, F. Battaglia,
           C.M.A. Pennartz, "An improved index of phase-synchronization for
           electrophysiological data in the presence of volume-conduction,
           noise and sample-size bias", NeuroImage 55(4), 1548-1565 (2011).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    _phase_based = True
    name = "Debiased squared phase lag index"
    labels = ["spectral", "undirected"]

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
    """
    Debiased squared weighted phase lag index [1]_.

    ``((sum_k Im S_k)^2 - sum_k (Im S_k)^2) / ((sum_k |Im S_k|)^2 - sum_k
    (Im S_k)^2)`` over the tapers, ``S_k`` being taper ``k``'s
    cross-spectrum, then the band statistic. An estimator of ``wPLI^2``
    without the upward bias of squaring a finite-taper estimate, so it can be
    slightly negative. Undirected (symmetric). The band leaves out 0 Hz and
    ``fs / 2``.

    References
    ----------
    .. [1] M. Vinck, R. Oostenveld, M. van Wingerden, F. Battaglia,
           C.M.A. Pennartz, "An improved index of phase-synchronization for
           electrophysiological data in the presence of volume-conduction,
           noise and sample-size bias", NeuroImage 55(4), 1548-1565 (2011).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    _phase_based = True
    name = "Debiased squared weighted phase-lag index"
    labels = ["spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dswpli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_weighted_phase_lag_index"


class PairwisePhaseConsistency(NonparametricSpectralMultivariate, Undirected):
    """
    Pairwise phase consistency across tapers [1]_.

    The mean of ``cos(phi_k - phi_l)`` over pairs ``k != l`` of the ``n``
    tapers (and trials), ``phi_k`` being taper ``k``'s cross-spectral phase,
    equivalently ``(n^2 PLV^2 - n) / (n (n - 1))``, then the band statistic.
    An unbiased estimator of ``PLV^2``, so it can be negative. Undirected
    (symmetric). The band leaves out 0 Hz and ``fs / 2``.

    References
    ----------
    .. [1] M. Vinck, M. van Wingerden, T. Womelsdorf, P. Fries,
           C.M.A. Pennartz, "The pairwise phase consistency: a bias-free
           measure of rhythmic neuronal synchronization", NeuroImage 51(1),
           112-122 (2010).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    _phase_based = True
    name = "Pairwise phase consistency"
    labels = ["spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "ppc"
        super().__init__(**kwargs)
        self._measure = "pairwise_phase_consistency"


# The directed measures below are computed pair by pair: the backend's
# factorisation of a large multivariate spectral matrix can crash.


class DirectedCoherence(NonparametricSpectralBivariate, Directed):
    """
    Directed coherence of a pair, from its spectral factorisation [1]_.

    ``DC_ij(f) = sqrt(sigma_jj) |H_ij(f)| / sqrt(sum_k sigma_kk |H_ik(f)|^2)``
    in the literature's ``[target i, source j]`` notation, where ``H`` is the
    transfer function from Wilson's factorisation of the pair's multitaper
    spectral matrix and ``sigma_kk`` the innovation variance of process
    ``k``; then the band statistic. ``A[i, j]`` is the flow from row ``i``
    (source) to column ``j`` (target). Directed and unsigned, in [0, 1], with
    ``sum_j DC_ij^2 = 1`` by construction.

    pyspi recomputes it from the backend's transfer function and innovation
    covariance, because the backend's ``directed_coherence()`` differs in two
    ways: it puts the squared magnitude ``|H|^2`` in the numerator, so its
    ratio is unbounded; and it broadcasts ``diag(Sigma)`` along the target
    axis, where it cancels, so its value reduces to the square root of the
    directed transfer function. Here ``DC = sqrt(DTF)`` only when the
    innovation variances are equal.

    Only ``diag(Sigma)`` enters, as in the published form. With correlated
    innovations the value stays in [0, 1] and the squares still sum to 1, but
    ``DC_ij^2`` is not then the fraction of process ``i``'s power arriving
    from ``j``. A whitened (triangular-factor) variant would restore that
    reading but would depend on the order of the two processes.

    A pair of linearly dependent series has no spectral factor and is NaN,
    labelled not appropriate (see ``NonparametricSpectralBivariate``).

    References
    ----------
    .. [1] L.A. Baccalá, K. Sameshima, G. Ballester, A.C. do Valle,
           C. Timo-Iaria, "Studying the interaction between brain structures
           via directed coherence and Granger causality", Appl. Signal
           Process. 5(1), 40-48 (1998).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    name = "Directed coherence"
    labels = ["spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "dcoh"
        super().__init__(**kwargs)
        self._measure = "directed_coherence"
        self._recompute = True

    def _get_statistic(self, C):
        # Not the backend's _get_noise_variance / _total_inflow helpers: the
        # first broadcasts along the wrong axis. Only two private properties
        # of the backend are used.
        H = C._transfer_function                       # (..., f, target, source)
        sigma = np.diagonal(C._noise_covariance, axis1=-1, axis2=-2)   # (..., n)
        sigma = sigma[..., np.newaxis, np.newaxis, :]  # broadcast along `source`
        mag2 = np.abs(H) ** 2
        inflow = np.sqrt(np.sum(sigma * mag2, axis=-1, keepdims=True))
        return np.sqrt(sigma) * np.abs(H) / inflow


class PartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    """
    Partial directed coherence of a pair, from its spectral factorisation [1]_.

    ``A[i, j]`` is the band statistic of ``|A_ji(f)|^2 / sum_k |A_ki(f)|^2``
    for source ``i`` (row) and target ``j`` (column), where
    ``A(f) = H(f)^-1`` is the inverse of the pair's transfer function from
    Wilson's factorisation of its multitaper spectral matrix: the share of the
    source's outflow that goes to the target. The squared magnitude, as the
    backend reports it. Directed and unsigned, in [0, 1]. On a pair there is
    nothing to partial out; ``ConditionalPartialDirectedCoherence`` conditions
    on every other process.

    References
    ----------
    .. [1] L.A. Baccalá, K. Sameshima, "Partial directed coherence: a new
           concept in neural structure determination", Biol. Cybern. 84(6),
           463-474 (2001).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    name = "Partial directed coherence"
    labels = ["spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "pdcoh"
        super().__init__(**kwargs)
        self._measure = "partial_directed_coherence"


class GeneralizedPartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    """
    Generalized partial directed coherence of a pair [1]_.

    :class:`PartialDirectedCoherence` with each target weighted by its
    innovation variance: the band statistic of
    ``(|A_ji(f)|^2 / sigma_j^2) / sum_k (|A_ki(f)|^2 / sigma_k^2)`` for source
    ``i`` (row) and target ``j`` (column), with ``sigma_k^2`` the innovation
    variance of process ``k``. The weighting makes it invariant to rescaling
    either process. The squared magnitude, as the backend reports it.
    Directed and unsigned, in [0, 1].

    References
    ----------
    .. [1] L.A. Baccalá, K. Sameshima, D.Y. Takahashi, "Generalized partial
           directed coherence", 15th International Conference on Digital
           Signal Processing, 163-166 (2007).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    name = "Generalized partial directed coherence"
    labels = ["spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "gpdcoh"
        super().__init__(**kwargs)
        self._measure = "generalized_partial_directed_coherence"


class DirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    """
    Directed transfer function of a pair, from its spectral factorisation [1]_.

    ``A[i, j]`` is the band statistic of ``|H_ji(f)|^2 / sum_k |H_jk(f)|^2``,
    the share of the inflow to the target ``j`` (column) that comes from the
    source ``i`` (row), with ``H`` the transfer function from Wilson's
    factorisation of the pair's multitaper spectral matrix. The squared
    magnitude, as the backend reports it. Directed and unsigned, in [0, 1].

    It is not weighted by the innovation variances, so on a pair it is about
    ``DC^2 sigma_jj / sigma_ii`` for weak coupling, with ``DC`` the directed
    coherence and ``sigma_kk`` the innovation variance of process ``k``: its
    direction follows which process is less predictable as well as the
    coupling.
    :class:`DirectedCoherence` applies that weighting.

    References
    ----------
    .. [1] M.J. Kamiński, K.J. Blinowska, "A new method of the description of
           the information flow in the brain structures", Biol. Cybern. 65(3),
           203-210 (1991).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    name = "Directed transfer function"
    labels = ["spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "dtf"
        super().__init__(**kwargs)
        self._measure = "directed_transfer_function"


class DirectDirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    """
    Direct directed transfer function of a pair [1]_.

    ``dDTF_ij(f) = eta_ij(f) |P_ij(f)|`` in the literature's
    ``[target i, source j]`` notation: the full-frequency DTF
    ``eta_ij(f) = |H_ij(f)| / sqrt(sum_f sum_k |H_ik(f)|^2)`` times the
    magnitude of the partial coherence ``P_ij = -Q_ij / sqrt(Q_ii Q_jj)``,
    ``Q = S(f)^-1``, where ``H`` is the transfer function from Wilson's
    factorisation of the pair's multitaper spectral matrix and
    ``S = H Sigma H^H`` the spectrum of the factorised model; then the band
    statistic. ``A[i, j]`` is the flow from row ``i`` (source) to column
    ``j`` (target). Directed and unsigned, in [0, 1]. This is the square root
    of the published ``delta_ij^2 = eta_ij^2 P_ij^2``, the magnitude scale
    the backend reports. On a pair, ``P`` is the ordinary coherency.

    pyspi computes it here because the backend's
    ``direct_directed_transfer_function()`` uses ``sqrt(PDC)`` in place of the
    partial coherence. Its DTF factor is not weighted by the innovation
    variances, so its direction follows which process is less predictable as
    well as the coupling (see :class:`DirectedTransferFunction`).

    References
    ----------
    .. [1] A. Korzeniewska, M. Mańczak, M. Kamiński, K.J. Blinowska,
           S. Kasicki, "Determination of information flow direction among
           brain structures by a modified directed transfer function (dDTF)
           method", J. Neurosci. Methods 125(1-2), 195-207 (2003).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band, ignoring NaN.
        Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    name = "Direct directed transfer function"
    labels = ["spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "ddtf"
        super().__init__(**kwargs)
        self._measure = "direct_directed_transfer_function"
        self._recompute = True

    def _get_statistic(self, C):
        # Only the two private properties DirectedCoherence already uses.
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
    """
    Phase slope index from the multitaper coherency [1]_.

    ``Psi_ij = Im sum_k conj(C_ij(k)) C_ij(k + 1)``, a coherence-weighted
    phase difference between neighbouring groups of frequencies, which grows
    with the lag between two processes. ``C_ij(k)`` is the coherency of group
    ``k``'s mean cross- and auto-spectra over its frequencies and tapers,
    ``mean S_ij / sqrt(mean S_ii mean S_jj)``. Antisymmetric and signed:
    ``A[i, j] = -A[j, i]``; positive means process ``i`` leads.

    The groups have a target width ``w``: ``0.4 fs / sqrt(T)`` for a series
    of length ``T``, never less than one multitaper bandwidth
    ``2 NW fs / T`` (the two are equal at ``T = 225``), or ``resolution``
    when given. With ``x = (fmax - fmin) / w`` and ``K = floor(x)`` (at least
    two), the band is cut at fixed frequencies into ``K`` equal groups and,
    separately, into ``K + 1``; each frequency of the series' grid in the
    band, less 0 Hz and ``fs / 2``, joins the group it falls in, and the
    index is the average of the two sums weighted ``1 - (x - K)`` and
    ``x - K``. The weighting makes the index change continuously as ``w``
    changes with ``T``, rather than jumping when a group is added.

    The default ``w`` narrows as ``T`` grows while each group still averages
    more estimates, about ``0.4 sqrt(T)`` FFT bins, so the spread of the
    index between independent series falls with ``T`` and the index of a
    given lag approaches the integral of the coherence-weighted phase slope
    over the band, about ``2 pi lag (fmax - fmin) / fs`` times the band's
    mean squared coherence. It approaches that integral from below, the
    more slowly the longer the lag, so values at lags that are not small
    against the sign limit below compare across datasets only at similar
    lengths.

    Neighbouring groups are ``d = (fmax - fmin) / K`` apart or less, so a lag
    of ``L`` samples turns the phase by at most ``2 pi L d / fs`` between
    them. The sign is right for lags below ``fs / (2 d)`` samples, about
    ``1.25 sqrt(T)`` at the default (19 at ``T = 250``, 39 at 1000, 79 at
    4000, 158 at 16000) and ``T / (4 NW)`` on series shorter than 225
    samples. Near the limit the index approaches zero, so noise decides its
    sign there. Between that limit and twice it the index has the opposite
    sign: a process that leads by such a lag reads as following. Beyond
    twice the limit the averaging within each group shrinks the index, and
    its sign keeps alternating with period ``fs / d`` samples. On series
    longer than 225 observations the limit is below the ``T / (4 NW)``
    samples at which estimates one bandwidth apart wrap, so a lag between
    the two is reported with the opposite sign; a smaller ``resolution``
    extends the range of lags at the cost of power.

    This is the unnormalised index; Nolte et al. divide it by a jackknife
    standard deviation over epochs, which a single recording does not
    provide. It is computed here rather than by the backend's
    ``Connectivity.phase_slope_index``, which in spectral_connectivity 2.0
    sums over all pairs of frequencies rather than neighbouring ones and
    whose definition differs between releases. The band must have
    ``fmin < fmax`` (checked at construction) and be at least two
    bandwidths wide, ``T >= 4 NW fs / (fmax - fmin)`` (24 observations for
    [0, 0.5] and 48 for [0, 0.25] at ``fs = 1``, ``NW = 3``), or the class
    raises ``NotAppropriate`` (NaN with the reason in a Calculator).

    References
    ----------
    .. [1] G. Nolte, A. Ziehe, V.V. Nikulin, A. Schlögl, N. Krämer,
           T. Brismar, K.-R. Müller, "Robustly estimating the flow direction
           of information in complex physical systems", Phys. Rev. Lett.
           100(23), 234101 (2008).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        Only ``'mean'``, the name the identifiers (``psi_multitaper_mean_*``)
        give to the sum over the band. Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    resolution : float or None, optional
        Target width ``w`` of the groups of frequencies, in the units of
        ``fs``, positive and at most ``fs / 2``, used at every length (never
        below one multitaper bandwidth); the sign is then right for lags
        below about ``fs / (2 resolution)`` samples. None is
        ``0.4 fs / sqrt(T)``. A value is appended to the identifier as
        ``_res-<value>``, with dots written as dashes; None adds nothing.
        Default: None.
    """

    # Antisymmetric in (i, j): slope of the phase spectrum.
    _antisymmetric_spectrum = True
    name = "Phase slope index"
    labels = ["spectral", "undirected"]
    # A sum over the band, not a band statistic; `mean` is the name the
    # identifiers carry.
    _statistics = ("mean",)
    _phase_based = True
    # Two groups of frequencies: a single frequency never has them.
    _positive_width = True

    def __init__(self, resolution=None, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        if resolution is not None:
            # A lower-precision float (np.float32(0.02)) is read as the
            # value it prints as, so its identifier is `_res-0-02`.
            if isinstance(resolution, np.floating):
                resolution = float(str(resolution))
            resolution = require_positive_float("resolution", resolution)
            if resolution > self._fs / 2:
                raise ValueError(
                    f"resolution must be at most fs / 2 = {self._fs / 2}, got "
                    f"{resolution!r}.")
            self.identifier += f"_res-{fmt_param(resolution)}".replace(".", "-")
        self._resolution = resolution
        self._measure = "phase_slope_index"
        # Always `_get_statistic`: the backend method of this name is not the
        # published index.
        self._recompute = True

    @property
    def key(self):
        return super().key + (self._resolution,)

    @property
    def _cache_subkey(self):
        # (class, fs, NW, fmin, fmax, resolution), as in `key`.
        return (type(self).__name__, self._fs, self._nw, self._fmin, self._fmax,
                self._resolution)

    def _get_cache(self, data):
        # The default group width depends on the length of the series.
        self._n_time = data.n_observations
        return super()._get_cache(data)

    def _splits(self):
        """
        The two group counts and the weight of the second (see the class docstring).

        ``x = (fmax - fmin) / w``; the counts are ``K = floor(x)`` and ``K + 1``
        and the weight is ``x - K``, so the index changes continuously as ``w``
        does.
        """
        T, span = self._n_time, self._fmax - self._fmin
        bandwidth = 2 * self._nw * self._fs / T
        if span < 2 * bandwidth * (1 - 1e-9):
            shortest = int(np.ceil(4 * self._nw * self._fs / span - 1e-9))
            raise NotAppropriate(
                f"Phase slope index needs the band [{self._fmin}, {self._fmax}] to "
                f"hold two groups one multitaper bandwidth (2 NW fs / T = "
                f"{bandwidth:.4g}) wide, so at least {shortest} observations; the "
                f"series has {T}.")
        target = (_PSI_WIDTH * self._fs / np.sqrt(T) if self._resolution is None
                  else self._resolution)
        x = span / max(target, bandwidth)
        # The offset keeps a width that divides the band exactly (0.5 / 0.01)
        # from rounding down.
        K = max(2, int(np.floor(x + 1e-9)))
        return K, min(max(x - K, 0.0), 1.0)

    def _groups(self, freq, n_groups):
        """Indices of the grid frequencies in each of ``n_groups`` equal groups."""
        inside = self._band(freq)
        span = self._fmax - self._fmin
        # Fixed edges fmin + k * span / n_groups; a frequency on an inner edge
        # joins the group above it.
        position = (freq[inside] - self._fmin) * n_groups / span
        k = np.minimum(n_groups - 1, np.floor(position + 1e-9).astype(int))
        groups = [inside[k == g] for g in range(n_groups)]
        if min(g.size for g in groups) == 0:
            raise NotAppropriate(
                f"Phase slope index: a group of [{self._fmin}, {self._fmax}] holds "
                f"no frequency of the series' grid.")
        return groups

    @staticmethod
    def _nolte_sum(coefficients, groups):
        """Nolte's sum over neighbouring groups, from the tapered coefficients."""
        # (windows, trials, tapers, frequencies, signals). Each group's mean
        # cross-spectral matrix over its tapers and frequencies, without the
        # full per-frequency matrix.
        n_windows, n_signals = coefficients.shape[0], coefficients.shape[-1]
        mean = np.empty((n_windows, len(groups), n_signals, n_signals), dtype=complex)
        for k, g in enumerate(groups):
            X = coefficients[..., g, :].reshape(n_windows, -1, n_signals)
            mean[:, k] = np.einsum("wni,wnj->wij", X, X.conj()) / X.shape[1]
        power = np.real(np.diagonal(mean, axis1=-2, axis2=-1))
        # The backend's floor on the normalisation, so a constant process
        # gives 0 rather than 0 / 0.
        norm = np.maximum(np.sqrt(power[..., :, None] * power[..., None, :]),
                          np.finfo(float).eps)
        grouped = mean / norm
        return np.imag(np.sum(
            np.conj(grouped[..., :-1, :, :]) * grouped[..., 1:, :, :], axis=-3))

    def _get_statistic(self, C):
        freq = np.asarray(C.frequencies)
        K, weight = self._splits()
        coefficients = np.asarray(C.fourier_coefficients)
        value = self._nolte_sum(coefficients, self._groups(freq, K))
        if weight > 0:
            value = ((1 - weight) * value
                     + weight * self._nolte_sum(coefficients, self._groups(freq, K + 1)))
        return value


def _independent_significant_frequencies(p_values, step, alpha=0.05,
                                         min_group_size=3):
    """
    Indices of the largest significant coherence cluster, thinned to `step`.

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
    """
    Group delay from the slope of the coherency phase [1]_.

    The multitaper coherency is screened bin by bin for significance with the
    Fisher-z approximation [2]_: for coherence from ``n`` independent spectral
    estimates, ``arctanh|C|`` is taken as normal with bias and variance
    ``b = 1/(2n - 2)``, so the null z-score is ``(arctanh|C| - b) / sqrt(b)``.
    Benjamini-Hochberg is applied over the in-band frequencies, the longest
    contiguous run of significant bins is kept and thinned to one bin per
    multitaper bandwidth (``2 NW`` FFT bins), and, if at least three bins
    remain, the unwrapped coherency phase is regressed on frequency. The band leaves
    out 0 Hz and ``fs / 2``, where the phase is forced to 0 or pi. The
    per-bin screen is liberal with few tapers, so false estimates are
    controlled by the run and three-point requirements. NaN when no such run
    exists.

    ``delay`` and ``slope`` are antisymmetric and signed:
    ``A[i, j] = -A[j, i]``; positive means process ``i`` leads. ``rvalue`` is
    symmetric and unsigned.

    The backend's ``Connectivity.group_delay`` is not used: its significance
    test passes ``n_obs2 = 0`` to ``coherence_bias``, so the square root in
    its z-score has a negative argument, every p-value is NaN and every
    result is NaN.

    References
    ----------
    .. [1] J. Gotman, "Measurement of small time differences between EEG
           channels: method and application to epileptic seizure
           propagation", Electroencephalogr. Clin. Neurophysiol. 56(5),
           501-514 (1983).
    .. [2] H. Bokil, K. Purpura, J.-M. Schoffelen, D. Thomson, P. Mitra,
           "Comparing spectra and coherences for groups of unequal size",
           J. Neurosci. Methods 159(2), 337-345 (2007).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str
        ``'delay'`` (the slope converted to a delay in samples,
        ``fs * slope / (2 pi)``), ``'slope'`` (radians per unit of
        frequency) or ``'rvalue'`` (``|r|`` of the regression). Required in
        practice: the inherited default ``'mean'`` raises ``ValueError``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    name = "Group delay"
    labels = ["spectral", "directed", "lagged"]
    _statistics = ("delay", "slope", "rvalue")
    _phase_based = True

    def __init__(self, **kwargs):
        self.identifier = "gd"
        super().__init__(**kwargs)
        self._measure = "group_delay"
        # `delay` and `slope` are antisymmetric (phase(C_ji) = -phase(C_ij))
        # and signed; `rvalue` is symmetric.
        if self._statistic in ("delay", "slope"):
            self.labels = [
                l for l in self.labels
                if l not in ("undirected", "directed")
            ] + ["antisymmetric", "signed"]
            self.issigned = lambda: True
            # Both are the same time offset, as a delay or as a phase slope.
            self.labels.append("timing")
        else:
            # |r| is symmetric by construction and non-negative, so it is
            # neither directed nor signed.
            self.labels = [
                l for l in self.labels if l not in ("directed", "antisymmetric")
            ] + ["undirected"]
        # Always `_get_statistic`: `Connectivity.group_delay()` accepts a
        # no-argument call, so the dispatcher's TypeError fallback would
        # otherwise return the backend's all-NaN result.
        self._recompute = True

    @property
    def _cache_subkey(self):
        # (class, fs, NW, fmin, fmax), as in `key`.
        return (type(self).__name__, self._fs, self._nw, self._fmin, self._fmax)

    def _get_statistic(self, C):
        """
        ``(delay, slope, |r|)`` of Gotman's group delay (see the class docstring).

        The regression runs against frequency in the units of ``fs``, so
        ``slope / (2 pi)`` is in seconds and ``fs * slope / (2 pi)`` in
        samples. ``|r|`` is orientation-free; the signed ``r`` is not, since
        ``phase(C_ji) = -phase(C_ij)``.
        """
        from scipy import stats

        coherency = np.asarray(C.coherency())
        freqs = np.asarray(C.frequencies)
        # The closed band less the real-valued bins, where the phase is forced
        # to 0 or pi whatever the delay.
        in_band = self._band(freqs)
        f = freqs[in_band]
        coh = coherency[:, in_band]
        n_trials, _, M, _ = coh.shape

        # Independent frequency spacing: one multitaper bandwidth, as for
        # PhaseSlopeIndex (closer estimates share the tapers' main lobes).
        step = self._frequency_step

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
                    # With y(t) = x(t - L), this orientation makes the delay
                    # +L at [source, target] after `_to_source_target`, i.e.
                    # positive means the row leads the column.
                    slope[t, i, j] = -fit.slope
                    slope[t, j, i] = fit.slope
                    # |r|: the signed r flips with the orientation, and this
                    # matrix is written symmetrically.
                    r_value[t, i, j] = r_value[t, j, i] = abs(fit.rvalue)

        # A delay in samples, as every other lagged SPI in pyspi reports.
        return self._fs * slope / (2 * np.pi), slope, r_value


# Spectral GC is undefined at 0 Hz, so a band given from fmin = 0 starts at
# fs / _SGC_EDGE_DIVISOR, 1e-5 cycles per sample: above 0 Hz and below the
# first positive frequency of either method's grid on up to 1e5 samples. A
# division, so that the edge is the double nearest fs / 1e5 (1e-05 at fs = 1).
_SGC_EDGE_DIVISOR = 100_000

# nitime.GrangerAnalyzer's grid size: granger_causality_xy evaluates the model at
# n_freqs // 2 + 1 points on [0, fs/2) (scipy.signal.freqz, endpoint excluded).
_GC_N_FREQS = 1024


# The shortest series the parametric method takes, at any order.
_GC_MIN_OBSERVATIONS = 20


def _var_min_observations(order):
    """The shortest series an order ``order`` VAR of a pair is fitted to."""
    return max(_GC_MIN_OBSERVATIONS, 3 * order + 1)


def _parametric_gc_frequencies(fs):
    """The grid ``granger_causality_xy`` evaluates on, ``k fs / 1026`` for k = 0 to 512."""
    w = np.linspace(0, np.pi, _GC_N_FREQS // 2 + 1, endpoint=False)
    return w * fs / (2 * np.pi)


class _SingularPairFit(NotAppropriate):
    """The VAR of one pair cannot be fitted: that pair alone is refused."""


def _yule_walker_condition(Rn, p):
    """
    Condition number of the order ``p`` Yule-Walker system of a pair.

    ``Rn`` holds the autocovariances scaled to unit variance; the system's
    matrix is the ``2p x 2p`` block Toeplitz matrix of ``Rn[0]`` to
    ``Rn[p - 1]``. Infinite when it is singular.
    """
    blocks = [[Rn[b - a] if b >= a else Rn[a - b].T for b in range(p)]
              for a in range(p)]
    sv = np.linalg.svd(np.block(blocks), compute_uv=False)
    return np.inf if sv[-1] == 0 else sv[0] / sv[-1]


def _fit_bivariate_var(x, order=None, max_order=50, pair=(0, 1)):
    """
    Yule-Walker fit of the two-process VAR behind parametric spectral GC.

    As ``nitime.analysis.granger.fit_model`` (Whittle's LWR recursion; with
    ``order=None``, the lowest order at which BIC stops improving), but on the
    Yule-Walker autocovariance: about the mean and normalised by ``T``
    (Brockwell & Davis 1991, sec. 11.5; Lütkepohl 2005, sec. 3.3.4). Its
    Toeplitz matrix is positive semidefinite, so the fitted model is stable;
    nitime's ``autocov_vector`` (normalised by ``T - k``, not demeaned) need
    not be.

    An order ``p`` fit needs ``max(20, 3p + 1)`` observations, or
    ``NotAppropriate`` is raised. Each equation of the model has ``2p``
    coefficients and ``T - p`` observations to fit them, so ``3p + 1`` is the
    shortest series with more observations than coefficients; and on fewer
    than 20 observations the estimate does not separate coupled from
    uncoupled simulated pairs at any order. Automatic selection therefore
    tries orders up to ``min(max_order, (T - 1) // 3)``, and raises
    ``NotAppropriate`` if BIC is still improving at the last of them.
    Returns ``(coef, ecov)`` in nitime's convention,
    ``X[t] + sum_k coef[k] X[t-k] = E[t]``.

    Every order fitted is checked. The order ``p`` system, with condition
    number ``kappa`` (``_yule_walker_condition``), determines the innovation
    covariance ``ecov`` only to about ``eps * kappa`` of its largest
    eigenvalue. An order whose system is singular to working precision
    (``eps * kappa >= 1``), or whose ``ecov`` is singular by
    ``singular_covariance_error`` with that ``kappa``, cannot be fitted: the
    two series are linearly dependent, or so nearly that Geweke's ratio of
    the fit would be set by rounding error. ``eps * kappa`` bounds the error
    from above, so near the boundary the check is conservative by a small
    factor (a pair whose fit is accurate to about 1% can be refused). A
    fixed ``order`` that cannot be fitted raises ``_SingularPairFit``, a
    ``NotAppropriate`` that names the numbers. The BIC search stops at the
    first order that cannot be fitted and selects the best order below it,
    even if BIC was still improving: the Yule-Walker system of a higher
    order contains that of a lower one, so it is no better conditioned, and
    the orders that can be fitted are fitted accurately. It raises when
    order 0 or 1 cannot be fitted, since then no order with lags is left and
    order 0 gives no causality by construction, not by estimate. A search
    that reaches the record-length limit while still improving raises, since
    a longer series could fit the higher orders.
    """
    i, j = pair
    owner = "Parametric spectral Granger causality"
    x = x - x.mean(axis=-1, keepdims=True)
    n_obs = x.shape[-1]
    if order is not None and n_obs < _var_min_observations(order):
        raise NotAppropriate(
            f"Parametric spectral Granger causality: order {order} needs at "
            f"least {_var_min_observations(order)} observations, but the "
            f"series has {n_obs}. Set a lower `order`, or use a longer series.")
    if n_obs < _GC_MIN_OBSERVATIONS:
        raise NotAppropriate(
            f"Parametric spectral Granger causality needs at least "
            f"{_GC_MIN_OBSERVATIONS} observations, but the series has {n_obs}.")
    supported = (n_obs - 1) // 3
    largest = order if order is not None else min(max_order, supported)
    # R[k] = (1/T) sum_t x(t+k) x(t)^T: nitime's lag orientation.
    R = np.stack([x[:, k:] @ x[:, :n_obs - k].T
                  for k in range(largest + 1)]) / n_obs
    d = np.diag(R[0])
    Rn = R / np.sqrt(np.outer(d, d)) if (d > 0).all() else None

    def fit(p):
        kappa = 1.0
        if p > 0:
            kappa = np.inf if Rn is None else _yule_walker_condition(Rn, p)
            if np.finfo(float).eps * kappa >= 1:
                raise _SingularPairFit(
                    f"{owner}: processes {i} and {j} are linearly dependent: "
                    f"the order {p} Yule-Walker equations are singular to "
                    f"working precision (condition number {kappa:.2g}), so "
                    f"the VAR cannot be fitted.")
        coef, ecov = nt_alg.lwr_recursion(R[: p + 1])
        err = singular_covariance_error(
            owner, i, j, ecov,
            "Geweke's ratio of the fitted VAR is not defined",
            of=f"the VAR({p}) innovation covariance", condition=kappa)
        if err is not None:
            raise _SingularPairFit(str(err))
        return coef, ecov

    if order is not None:
        return fit(order)

    best = None
    for p in range(largest + 1):
        try:
            coef, ecov = fit(p)
        except _SingularPairFit:
            # An order that cannot be fitted to working precision is not a
            # candidate, nor is any higher one: the search selects among the
            # orders below it. If order 0 or 1 cannot be fitted no order with
            # lags is left, and order 0 would give GC = 0 by construction.
            if p <= 1:
                raise
            return best[1:]
        # nitime's criterion and arguments: (ecov, n_process, order, n_process * T).
        crit = nt_utils.bayesian_information_criterion(ecov, 2, p, 2 * n_obs)
        if best is not None and crit > best[0]:
            return best[1:]
        best = (crit, coef, ecov)
    if largest < max_order:
        raise NotAppropriate(
            f"Parametric spectral Granger causality: automatic AR order "
            f"selection was still improving at order {largest}, the largest "
            f"that {n_obs} observations support. Set an explicit `order`, or "
            f"use a longer series.")
    raise NotAppropriate(
        f"Model estimation order did not converge at max_order = {max_order}"
    )


class SpectralGrangerCausality(NonparametricSpectralMultivariate, Directed, Unsigned):
    """
    Spectral Granger causality of each pair, reduced over the band [1]_.

    Geweke's frequency-domain measure ``ln(S_jj(f) / S_jj^(i)(f))`` in nats,
    with ``S_jj^(i)`` the target's spectrum net of the part driven by the
    source's innovation. ``A[i, j]`` is the causality from row ``i``
    (source) to column ``j`` (target). Directed and unsigned. Each pair is
    modelled on its own, so the measure is bivariate, not conditioned on the
    other processes.

    With ``method='nonparametric'`` the spectral factors come from Wilson's
    factorisation of the pair's multitaper spectral matrix [2]_ (the backend's
    ``pairwise_spectral_granger_prediction``). With ``'parametric'`` a VAR is
    fitted to each pair by Yule-Walker (Whittle's recursion, as nitime's
    ``fit_model`` but with the autocovariance taken about the mean and
    normalised by ``T``, so the fitted model is stable) and evaluated by
    nitime's ``granger_causality_xy`` on 513 frequencies on ``[0, fs / 2)``.

    Frequencies are in the units of ``fs``, at the default ``fs = 1`` in
    cycles per sample; ``fs`` only relabels the frequency axis (see
    ``NonparametricSpectral``). At construction ``fs`` must be positive and
    finite and ``0 <= fmin <= fmax <= fs / 2``, or ``ValueError`` is raised.
    The parametric method's grid does not depend on the series, so a band
    holding none of its frequencies raises ``ValueError`` at construction;
    the nonparametric grid does, and a band holding none of it raises
    ``NotAppropriate`` when the SPI is computed. Spectral GC is undefined at
    0 Hz, so ``fmin = 0`` sets the lower edge to ``fs / 1e5``, which the
    identifier names (``_fmin-1e-05`` at ``fs = 1``). The edge is a fixed
    fraction of ``fs``, so rescaling ``fs`` with the band gives the same
    value; it leaves out 0 Hz and, on a series of at most 1e5 samples, no
    other frequency of either method's grid. On a longer series of the
    nonparametric method it also leaves out the positive frequencies below
    ``fs / 1e5`` (10 of them at 1e6 samples).

    A pair of linearly dependent series has no spectral factor or no VAR
    fit and raises ``NotAppropriate``, which the Calculator records as NaN
    for that pair with the reason; the other pairs keep their values. The
    nonparametric method refuses a pair whose zero-lag covariance is
    singular (``pyspi.utils.SINGULAR_RCOND``); the parametric method one
    whose Yule-Walker equations or fitted innovation covariance are singular
    to the precision their conditioning allows (``_fit_bivariate_var``). A
    copy of a series delayed by a lag is refused by neither: its spectral GC
    is infinite in theory, and both methods return a large finite value set
    by the estimator. Every other NaN cell is also not appropriate,
    with the fraction of its in-band frequencies that are NaN: a cell is
    NaN when more than ``nan_threshold`` of them are, or when any is and
    ``ignore_nan`` is False.

    With ``method='nonparametric'``, where
    Wilson's factorisation of a pair reaches its iteration cap, the value
    comes from the unconverged factor and the SPI warns. The parametric
    method fits an order ``p`` model only to a series of at least
    ``max(20, 3p + 1)`` observations: more observations than each equation
    has coefficients, and at least 20, below which the estimate does not
    separate coupled from uncoupled simulated pairs. A shorter series raises
    ``NotAppropriate``. Automatic selection tries orders up to
    ``min(max_order, (T - 1) // 3)`` on ``T`` observations, and raises
    ``NotAppropriate`` if BIC is still improving at the last of them: the
    order cannot be selected at that record length.

    References
    ----------
    .. [1] J. Geweke, "Measurement of linear dependence and feedback between
           multiple time series", J. Am. Stat. Assoc. 77(378), 304-313
           (1982).
    .. [2] M. Dhamala, G. Rangarajan, M. Ding, "Analyzing information flow in
           brain networks with nonparametric Granger causality", NeuroImage
           41(2), 354-362 (2008).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin`` and ``fmax`` are
        in its units; at the default 1 they are in cycles per sample.
        Default: 1.
    fmin : float, optional
        Lower edge of the band, in [0, ``fs / 2``]; 0 becomes ``fs / 1e5``,
        just above 0 Hz. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in [``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    method : str, optional
        ``'nonparametric'`` (multitaper and Wilson's factorisation) or
        ``'parametric'`` (a VAR fitted to each pair); any other value raises
        ``ValueError``. Default: ``'nonparametric'``.
    order : int or None, optional
        VAR order of the parametric method, at least 0; None selects it per
        pair as the lowest order at which BIC stops improving. Ignored by the
        nonparametric method. Default: None.
    max_order : int, optional
        Largest order the parametric selection tries, at least 1; a value other than 50
        with ``order=None`` is appended to the identifier as
        ``_maxorder-<n>``. Default: 50.
    statistic : str, optional
        ``'mean'`` or ``'max'`` over the band. Default: ``'mean'``.
    ignore_nan : bool, optional
        Ignore NaN frequencies in the band statistic; False lets a NaN
        frequency make the value NaN, and appends ``_ignorenan-False`` to the
        identifier. Default: True.
    nan_threshold : float, optional
        Largest fraction of NaN in-band frequencies a cell may have before it
        is set to NaN; a value other than 0.5 is appended to the identifier
        as ``_nanthr-<x>``. Default: 0.5.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5,
        for ``method='nonparametric'`` (see ``NonparametricSpectral``); the
        parametric method refuses a value other than 3. Default: 3.
    """

    name = "Spectral Granger causality"
    identifier = "sgc"
    labels = ["embedding", "spectral", "directed", "lagged"]

    def __init__(
        self,
        fs=1,
        fmin=0,
        fmax=None,
        method="nonparametric",
        order=None,
        max_order=50,
        statistic="mean",
        ignore_nan=True,
        nan_threshold=0.5,
        time_halfbandwidth_product=_DEFAULT_NW,
    ):
        if method not in ("nonparametric", "parametric"):
            raise ValueError(
                f"method must be 'nonparametric' or 'parametric', got {method!r}.")
        require_positive_float("fs", fs)
        if fmax is None:
            fmax = fs / 2
        require_band(fs, fmin, fmax)
        self._fs = fs  # Reaches the multitaper estimate and the frequency labels.
        self._nw = _check_time_halfbandwidth_product(time_halfbandwidth_product)
        if method != "nonparametric" and self._nw != _DEFAULT_NW:
            raise ValueError(
                "time_halfbandwidth_product sets the multitaper estimate, which "
                "method='parametric' does not use.")
        # Spectral GC is undefined at 0 Hz, so a band from 0 starts at
        # fs / 1e5, which the identifier names: the band is the same fraction
        # of the spectrum whatever the units of fs.
        self._fmin = fs / _SGC_EDGE_DIVISOR if fmin == 0 else fmin
        self._fmax = fmax
        if self._fmin > fmax:
            raise ValueError(
                f"With fmin=0 the band starts at fs / 1e5 = "
                f"{fmt_param(self._fmin)} (spectral GC is undefined at 0 Hz), "
                f"so fmax must be at least that, got {fmax!r}.")
        if method == "parametric" and not _closed_band(
                _parametric_gc_frequencies(fs), self._fmin, fmax).any():
            raise ValueError(
                f"SpectralGrangerCausality: the band [{self._fmin}, {fmax}] "
                f"holds no frequency of the parametric method's grid, "
                f"k fs / {_GC_N_FREQS + 2} for k = 0 to {_GC_N_FREQS // 2}; "
                f"widen it.")
        self.ignore_nan = ignore_nan
        self.nan_threshold = nan_threshold

        if statistic not in ("mean", "max"):
            raise ValueError(
                f"SpectralGrangerCausality does not support statistic="
                f"{statistic!r}; it supports mean, max.")
        if statistic == "mean":
            self._statfn = np.nanmean if self.ignore_nan else np.mean
        else:
            self._statfn = np.nanmax if self.ignore_nan else np.max

        self._method = method
        # `self._fmin`, not the `fmin` argument: the identifier names the band
        # actually integrated.
        if self._method == "nonparametric":
            self._measure = "pairwise_spectral_granger_prediction"
            paramstr = (f"_nonparametric_{statistic}_fs-{fmt_param(fs)}"
                        f"_fmin-{fmt_param(self._fmin)}"
                        f"_fmax-{fmt_param(fmax)}").replace(
                ".", "-"
            ) + _nw_suffix(self._nw)
        else:
            if order is not None:
                require_int("order", order, minimum=0)
            require_int("max_order", max_order)
            self._order = order
            self._max_order = max_order
            paramstr = (f"_parametric_{statistic}_fs-{fmt_param(fs)}"
                        f"_fmin-{fmt_param(self._fmin)}"
                        f"_fmax-{fmt_param(fmax)}_order-{fmt_param(order)}").replace(
                ".", "-"
            )
            # max_order bounds the order search, so it matters only when the
            # order is selected rather than given.
            if order is None and max_order != 50:
                paramstr += f"_maxorder-{fmt_param(max_order)}"
        # Non-default NaN handling changes the value, so it is named; defaults
        # are not, so bundled identifiers are unchanged.
        if not ignore_nan:
            paramstr += "_ignorenan-False"
        if nan_threshold != 0.5:
            paramstr += f"_nanthr-{fmt_param(nan_threshold)}".replace(".", "-")

        self.identifier = self.identifier + paramstr

    def _getkey(self):
        # fs is passed to the spectral transform, so it must key the cache.
        if self._method == "nonparametric":
            return (self._method, self._fs, self._nw, -1, -1)
        else:
            return (self._method, self._fs, None, self._order, self._max_order)

    def _nonparametric_gc(self, data):
        """
        ``(F, freq, refused, log)`` of the nonparametric method.

        The backend's pairwise spectral GC (``Connectivity.
        pairwise_spectral_granger_prediction``) factorises each pair's
        sub-matrix of one cross-spectral matrix in turn. The same computation
        is run here on the pairs that can be factorised: a pair of linearly
        dependent series is left out and named in ``refused``, with the
        reason, before the backend reaches it (``_dependent_spectrum_error``).
        ``log`` holds what the backend logged while factorising.
        """
        z = _ensure_time_series_3d(np.transpose(data.to_numpy(squeeze=True)))
        m = sc.Multitaper(z, sampling_frequency=self._fs,
                          time_halfbandwidth_product=self._nw)
        conn = sc.Connectivity.from_multitaper(m)
        csm = conn._expectation_cross_spectral_matrix()
        pairs, refused = [], {}
        for i, j in itertools.combinations(range(data.n_processes), 2):
            err = _dependent_spectrum_error(csm[..., [i, j], :][..., [i, j]], i, j)
            if err is None:
                pairs.append((i, j))
            else:
                refused[(i, j)] = refused[(j, i)] = str(err)
        with _collect_backend_log() as log:
            F = _estimate_spectral_granger_prediction(conn._power, csm, pairs)
        return np.asarray(F), conn.frequencies, refused, tuple(log)

    def _parametric_gc(self, data):
        """
        ``(F, freq, refused)`` of the parametric method.

        A pair whose VAR cannot be fitted (``_fit_bivariate_var``) is named in
        ``refused``, with the reason.
        """
        z = data.to_numpy(squeeze=True)
        n_proc = data.n_processes
        # F[source, target, f]. nitime's `granger_causality_xy(...)` returns
        # (w, f_x_on_y, f_y_on_x, ...) for the pair (x, y) = (z[i], z[j]),
        # i.e. i -> j and then j -> i.
        F = np.full((n_proc, n_proc, _GC_N_FREQS // 2 + 1), np.nan)
        # freqz's grid, which granger_causality_xy returns as `w`.
        w = np.linspace(0, np.pi, _GC_N_FREQS // 2 + 1, endpoint=False)
        refused = {}
        try:
            for i in range(n_proc):
                for j in range(i + 1, n_proc):
                    try:
                        coef, ecov = _fit_bivariate_var(
                            z[[i, j]], self._order, self._max_order, (i, j))
                    except _SingularPairFit as err:
                        refused[(i, j)] = refused[(j, i)] = str(err)
                        continue
                    w, F[i, j], F[j, i], _, _ = nt_alg.granger_causality_xy(
                        coef, ecov, n_freqs=_GC_N_FREQS)
        except NotAppropriate as err:
            # BIC still improving at `max_order` signals over-fitting at this
            # record length; say so rather than relay the bare message.
            if "did not converge" not in str(err):
                raise
            raise NotAppropriate(
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
        # The frequencies the model was evaluated at, in Hz (`w` is
        # scipy.signal.freqz's grid, k*pi/513 rad on [0, pi)).
        freq = w * self._fs / (2 * np.pi)
        return F, freq, refused

    def _get_cache(self, data):
        """
        ``(F, freq)``: ``F[0, f, target, source]``, NaN for a refused pair.

        Cached on the dataset with ``refused``, which maps each refused
        ordered pair to its reason, and ``log``, what the backend logged.
        """
        key = self._getkey()
        cached = getattr(data, "spectral_gc", {}).get(key)
        if cached is not None:
            return cached["F"], cached["freq"]
        if self._method == "nonparametric":
            F, freq, refused, log = self._nonparametric_gc(data)
        else:
            F, freq, refused = self._parametric_gc(data)
            log = ()
        cached = {"freq": freq, "F": F, "refused": refused, "log": log}
        try:
            data.spectral_gc[key] = cached
        except AttributeError:
            data.spectral_gc = {key: cached}
        return F, freq

    @parse_multivariate
    def multivariate(self, data):
        F, freq = self._get_cache(data)
        cached = getattr(data, "spectral_gc", {}).get(self._getkey(), {})
        refused = cached.get("refused", {})
        freq_id = self._band(freq)
        spectra = F[0, freq_id, :, :]
        M = data.n_processes
        dependent = np.zeros((M, M), dtype=bool)
        for i, j in refused:
            dependent[i, j] = True
        if dependent.any():
            # A refused pair has no spectrum; 0 in its place keeps it out of
            # the NaN-fraction test, and `fill_pairwise` then sets it NaN.
            # `dependent` is symmetric, so either orientation masks it.
            spectra = np.where(dependent, 0.0, spectra)

        result = _to_source_target(self, self._band_statistic(spectra))

        # Oriented as the values it masks, so it blanks the cell and not
        # its mirror.
        nan_pct = _to_source_target(self, np.isnan(spectra).mean(axis=0))
        np.fill_diagonal(nan_pct, 0.0)

        # A cell that is NaN and not refused for its pair takes its reason
        # from the NaN frequencies of its spectrum.
        reasons = dict(refused)
        n_band = spectra.shape[0]
        result[nan_pct > self.nan_threshold] = np.nan
        for i, j in zip(*np.nonzero(np.isnan(result))):
            if i == j or (i, j) in reasons:
                continue
            why = (f"more than nan_threshold = {self.nan_threshold}"
                   if nan_pct[i, j] > self.nan_threshold
                   else "and ignore_nan is False" if not self.ignore_nan
                   else "so the band statistic is undefined")
            reasons[(i, j)] = (
                f"Spectral GC: {nan_pct[i, j]:.0%} of the {n_band} in-band "
                f"frequencies of ({i}, {j}) give NaN values, {why}; the "
                f"measure is undefined at those frequencies.")

        if reasons:
            # Each such cell is NaN, labelled not appropriate with its reason;
            # every other cell keeps its value.
            def value(i, j):
                if (i, j) in reasons:
                    raise NotAppropriate(reasons[(i, j)])
                return result[i, j]

            pairs = [(i, j) for j in range(M) for i in range(M) if i != j]
            result = fill_pairwise(self.identifier, result, pairs, value)

        _warn_factorisation_log([(None, m) for m in cached.get("log", ())],
                                M * (M - 1) // 2, "pairs")
        return result


def _welch_cross_spectra(z, nperseg):
    """
    Hann-window Welch spectral matrices ``S[f, j, k]`` of ``z`` (processes x time).

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
    """
    Phase-aligned correlation: zero-lag correlation maximised over all-pass filters.

    ``P_jk = sum_f w_f |S_jk(f)| / sqrt(sum_f w_f S_jj(f) * sum_f w_f S_kk(f))``
    from Welch cross-spectra [1]_ (Hann window, segments of ``nperseg``
    samples overlapping by half, each demeaned), where ``w_f`` counts each
    one-sided bin's multiplicity on the two-sided grid: 1 at 0 Hz and, for
    even ``nperseg``, at Nyquist; 2 elsewhere. The denominator is then the
    product of the two processes' total powers. Undirected and unsigned
    (invariant to the sign of either process), in [0, 1] by Cauchy-Schwarz.

    In the population, filtering one process by an all-pass filter shifts the
    cross-spectral phase by any amount at each frequency, and the best choice
    cancels it, so ``P`` is the supremum of the zero-lag correlation over such
    filters, and ``r <= max_h xcorr(h) <= P``. A pure delay with positive gain
    has linear phase, so there ``max xcorr = P``; ``max xcorr < P`` flags
    dispersion (frequency-dependent delay). Equivalently ``P`` is the
    power-weighted mean coherence magnitude, whereas ``cohmag`` is the
    unweighted mean of the squared coherence, which scores coupling in a
    high-power band no higher than the same coupling in a low-power one.

    On a raw periodogram every bin is perfectly coherent, so ``P`` is about
    ``pi / 4`` for independent noise. Averaging ``K`` segments leaves a null
    floor of about ``sqrt(pi / (4 K))``; the cost of a short segment is
    resolution, since ``P`` of a pure delay of ``d`` samples is the Hann
    window's normalised autocorrelation at ``d``.

    References
    ----------
    .. [1] P.D. Welch, "The use of fast Fourier transform for the estimation
           of power spectra: a method based on time averaging over short,
           modified periodograms", IEEE Trans. Audio Electroacoust. 15(2),
           70-73 (1967).

    Parameters
    ----------
    nperseg : int or None, optional
        Welch segment length in samples, at least 2 and at most ``T``. None
        picks the largest power of two not above ``T / 8``, within [8, 256],
        which keeps ``K`` between 15 and 31 for ``T`` from 64 to 2047, and
        refuses a record too short for four segments (``T < 20``), as an
        integer above ``T`` does, with ``NotAppropriate``. An integer
        fixes the resolution across lengths. The identifier records ``auto``
        or the integer. Default: None.
    """

    name = "Phase-aligned correlation"
    identifier = "phasealigned"
    labels = ["spectral", "undirected"]

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
                raise NotAppropriate(f"nperseg={self._nperseg} exceeds the series "
                                     f"length T={T}.")
            return self._nperseg
        n = max(self._MIN_AUTO_NPERSEG, 1 << max((T // 8).bit_length() - 1, 0))
        n = min(self._MAX_AUTO_NPERSEG, n)
        shortest = n + (self._MIN_AUTO_SEGMENTS - 1) * (n // 2)
        if T < shortest:
            raise NotAppropriate(
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


def _flattened_power(x):
    """
    Multitaper power of a demeaned series divided by its log-log linear fit.

    Returns the frequencies (cycles/sample) from the multitaper half-bandwidth
    ``3 / T`` up to (not including) Nyquist, and the power there divided by the
    least-squares ``1/f^a`` fit, so a peak above the aperiodic background
    stands out whatever the background's slope.
    """
    T = x.size
    m = sc.Multitaper(x[:, np.newaxis, np.newaxis], sampling_frequency=1,
                      tapers=_whole_record_tapers(T, 1, _DEFAULT_NW))
    n = m.n_fft_samples
    S = np.mean(np.abs(m.fft()[0, 0, :, :, 0]) ** 2, axis=0)[: n // 2 + 1]
    f = np.arange(S.size) / n
    keep = (f >= 3 / T) & (f < 0.5)
    a, b = np.polyfit(np.log(f[keep]), np.log(S[keep]), 1)
    return f[keep], S[keep] / np.exp(b + a * np.log(f[keep]))


def _segment_ppc(dphi, L):
    """
    Pairwise phase consistency of ``dphi`` over samples in non-adjacent segments.

    ``dphi`` is cut into ``M`` segments of ``L`` samples (a remainder is
    dropped), and the value is the mean of ``cos(dphi_s - dphi_t)`` over the
    sample pairs whose segments are neither the same nor adjacent. NaN for
    fewer than three segments.
    """
    M = dphi.size // L
    if M < 3:
        return np.nan
    s = np.exp(1j * dphi[: M * L]).reshape(M, L).sum(axis=1)
    num = (np.abs(s.sum()) ** 2 - np.sum(np.abs(s) ** 2)
           - 2 * np.sum((s[1:] * np.conj(s[:-1])).real))
    return num / ((M * L) ** 2 - M * L * L - 2 * (M - 1) * L * L)


class AdaptivePhaseConsistency(Undirected, Unsigned):
    """
    Spectrally weighted Hilbert phase consistency [1]_.

    Each process is band-passed (zero-phase, second-order Butterworth) in
    bands ``[f_c / 2, 3 f_c / 2]``, the upper edge capped at 0.49, with centres
    ``f_c`` a factor ``sqrt(2)`` apart from ``max(4 / T, 0.005)`` to 0.4
    cycles/sample, and its instantaneous phase is taken from the analytic
    signal. In each band the phase difference ``dphi = phi_i - phi_j`` is cut
    into segments of ``L = ceil(2 / (f_hi - f_lo))`` samples, and the pairwise
    phase consistency is the mean of ``cos(dphi_s - dphi_t)`` over sample pairs
    in segments that are neither the same nor adjacent, which leaves out the
    dependence between nearby samples of a band-passed signal. The bands are
    averaged with weights ``w(f_c)``, the mean of ``G_i(f) G_j(f)`` over the
    band, where ``G`` is a process's multitaper power spectrum (time-bandwidth
    product 3) divided by its log-log linear (``1/f^a``) fit over ``3 / T`` to
    Nyquist. The weights follow the rhythmic power the
    two processes share, so no band is fixed in advance.

    Undirected and unsigned (invariant to the sign and scale of either
    process), in [-1, 1]: about 0 under independence and 1 for a constant
    phase lag, anti-phase included. Coupling in a rhythm that stands little
    above either process's ``1/f`` background gets little weight. It has no
    parameters: the bands are in cycles per sample, so the value does not
    depend on the sampling frequency. The first and last 5% of samples are
    left out of every band to avoid filter edge effects, and a band that then
    holds fewer than six segments is skipped, so the lowest band used is
    centred at about 13 cycles per record or above and slower rhythms are not
    seen. Because sample pairs in the same or adjacent segments are left out,
    a phase difference that drifts by about a cycle across the record reads
    negative, more so the fewer segments a band holds. NaN for a constant
    process. When every band is skipped (``T < 46``) the series cannot
    support the estimate and ``NotAppropriate`` is raised, which the
    Calculator records as NaN with its reason.

    References
    ----------
    .. [1] M. Vinck, M. van Wingerden, T. Womelsdorf, P. Fries,
           C.M.A. Pennartz, "The pairwise phase consistency: a bias-free
           measure of rhythmic neuronal synchronization", NeuroImage 51(1),
           112-122 (2010).
    """

    name = "Spectrally weighted Hilbert phase consistency"
    identifier = "ppc_hilbert_adaptive"
    labels = ["spectral", "undirected"]

    _FC_MIN = 0.005
    _FC_MAX = 0.4
    _F_TOP = 0.49
    _MIN_SEGMENTS = 6

    def _bands(self, T):
        """``(sos, lo, hi, L)`` for each band whose trimmed series holds six segments."""
        n = T - 2 * (T // 20)
        bands = []
        fc = max(4 / T, self._FC_MIN)
        while fc <= self._FC_MAX:
            lo, hi = fc / 2, min(1.5 * fc, self._F_TOP)
            L = int(np.ceil(2 / (hi - lo)))
            if n // L >= self._MIN_SEGMENTS:
                sos = signal.butter(2, [lo, hi], btype="band", fs=1, output="sos")
                bands.append((sos, lo, hi, L))
            fc *= np.sqrt(2)
        return bands

    @staticmethod
    def _process(x, bands):
        """Flattened power spectrum and per-band phase of one process; None if constant."""
        if np.ptp(x) == 0:
            return None
        x = x - x.mean()
        T = x.size
        trim = slice(T // 20, T - T // 20)
        f, G = _flattened_power(x)
        phases = [np.angle(signal.hilbert(signal.sosfiltfilt(sos, x)))[trim]
                  for sos, *_ in bands]
        return f, G, phases

    @staticmethod
    def _pair(a, b, bands):
        if a is None or b is None:
            return np.nan
        f, Ga, pa = a
        _, Gb, pb = b
        G = Ga * Gb
        num = den = 0.0
        for k, (_, lo, hi, L) in enumerate(bands):
            p = _segment_ppc(pa[k] - pb[k], L)
            in_band = (f >= lo) & (f <= hi)
            if np.isfinite(p) and in_band.any():
                w = G[in_band].mean()
                num += w * p
                den += w
        return num / den if den > 0 else np.nan

    def _matrix(self, z):
        M, T = z.shape
        A = np.full((M, M), np.nan)
        bands = self._bands(T)
        if not bands:
            raise NotAppropriate(
                f"Spectrally weighted phase consistency: no band holds "
                f"{self._MIN_SEGMENTS} segments in a series of {T} samples "
                f"(T >= 46 is needed).")
        processes = [self._process(x, bands) for x in z]
        for j in range(M):
            for i in range(j):
                A[i, j] = A[j, i] = self._pair(processes[i], processes[j], bands)
        return A

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return float(self._matrix(data.to_numpy(squeeze=True)[[i, j]])[0, 1])

    @parse_multivariate
    def multivariate(self, data):
        return self._matrix(data.to_numpy(squeeze=True))

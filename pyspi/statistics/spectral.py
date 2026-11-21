import functools
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

from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
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
    """
    Re-emit ``spectral_connectivity``'s log warnings as Python warnings.

    When Wilson's factorisation reaches its iteration cap, the backend logs a
    warning and returns the unconverged factor, from which every
    Wilson-derived measure (directed coherence, DTF, dDTF, PDC, gPDC,
    nonparametric spectral GC) is computed. pyspi records per-SPI diagnostics
    from the ``warnings`` channel only (``_parallel.execute_spi``), so this
    bridges the two for one backend call. It changes no value.
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
    resolution. PhaseSlopeIndex and GroupDelay space their independent
    frequencies one bandwidth, ``2 NW`` FFT bins, apart.

    Each subclass reduces its spectrum over the
    closed band ``fmin <= f <= fmax``, compared with a relative tolerance of
    1e-9 at both edges (``_closed_band``) so that a grid point such as Nyquist
    is kept. Phase measures (``_phase_based``) also leave out the real-valued
    bins, 0 Hz and ``fs / 2``: there every taper's Fourier coefficient of a
    real signal is real, so the cross-spectral phase is exactly 0 or pi and
    carries no estimate. A band left with no frequency raises ``ValueError``.

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

    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean",
                 time_halfbandwidth_product=_DEFAULT_NW):
        if fmax is None:
            fmax = fs / 2

        # `fs` reaches the multitaper estimate, the frequency grid, every cache
        # key and GroupDelay's conversion to samples.
        self._fs = fs
        self._nw = _check_time_halfbandwidth_product(time_halfbandwidth_product)
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

        # PLI, wPLI, PSI and coherence phase carry lead/lag in a sign,
        # A[i, j] == -A[j, i]: neither undirected (symmetric) nor directed. Only
        # the mean keeps the antisymmetry (max(-x) != -max(x)).
        if getattr(self, "_antisymmetric_spectrum", False):
            trait = "antisymmetric" if statistic == "mean" else "asymmetric"
            self.labels = [
                l for l in self.labels
                if l not in ("undirected", "directed")
            ] + [trait, "signed"]
            # Signed in behaviour too: `Calculator._rmmin` shifts unsigned SPIs
            # by their minimum and `set_group` takes their abs(), and either
            # would destroy the lead/lag sign.
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

    def _band(self, freq):
        """
        Indices of the frequencies the band statistic reduces over.

        The closed band, less the real-valued bins for a phase measure (see
        ``NonparametricSpectral``).
        """
        in_band = _closed_band(freq, self._fmin, self._fmax)
        if not self._phase_based:
            return np.flatnonzero(in_band)
        real = _real_valued_bins(freq, self._fs)
        if not (in_band & ~real).any():
            what = ("only the real-valued bins (0 Hz and fs/2), where the "
                    "cross-spectrum of real signals has no phase"
                    if in_band.any() else "no frequency of the series' grid")
            step = f" (grid step {freq[1] - freq[0]:.4g})" if freq.size > 1 else ""
            raise ValueError(
                f"{type(self).__name__}: the band [{self._fmin}, {self._fmax}] "
                f"holds {what}{step}; widen it.")
        return np.flatnonzero(in_band & ~real)

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        freq_id = self._band(freq)

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
    """
    A pair's ``Connectivity``, reduced to its spectral factorisation.

    The directed measures are read off the Wilson minimum-phase factor through
    the transfer function, the innovation covariance and the MVAR
    coefficients, which the backend factorises afresh on every access. Here the
    factorisation is done once; each quantity is a deterministic function of
    the same spectrum, so no value changes. The Fourier coefficients and the
    factor are then released, since one object is held per ordered pair.
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
    pair's spectral matrix. It is computed once per ``(fs, i, j)`` and shared
    by every subclass. The measures are evaluated pair by pair, so they are
    bivariate, not conditioned on the other processes.
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
            conns[conn_key] = conn

        # `_recompute` replaces a backend measure outright, not only when the
        # backend raises.
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
        # Closed band, as in NonparametricSpectralMultivariate.
        freq_id = np.flatnonzero(_closed_band(freq, self._fmin, self._fmax))
        return self._statfn(bv_freq[freq_id])


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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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

    References
    ----------
    .. [1] L.A. Baccalá, K. Sameshima, G. Ballester, A.C. do Valle,
           C. Timo-Iaria, "Studying the interaction between brain structures
           via directed coherence and Granger causality", Appl. Signal
           Process. 5(1), 40-48 (1998).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
    ``i`` (row) and target ``j`` (column), which makes it invariant to
    rescaling either process. The squared magnitude, as the backend reports
    it. Directed and unsigned, in [0, 1].

    References
    ----------
    .. [1] L.A. Baccalá, K. Sameshima, D.Y. Takahashi, "Generalized partial
           directed coherence", 15th International Conference on Digital
           Signal Processing, 163-166 (2007).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
    ``DC^2 sigma_j / sigma_i`` for weak coupling, with ``DC`` the directed
    coherence and ``sigma`` the innovation standard deviations: its direction
    follows which process is less predictable as well as the coupling.
    :class:`DirectedCoherence` applies that weighting.

    References
    ----------
    .. [1] M.J. Kamiński, K.J. Blinowska, "A new method of the description of
           the information flow in the brain structures", Biol. Cybern. 65(3),
           203-210 (1991).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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

    ``Psi_ij = Im sum_f conj(C_ij(f)) C_ij(f + df)`` over the band, with ``C``
    the complex coherency and ``df`` the frequency resolution of its estimate:
    a coherence-weighted phase difference between neighbouring frequencies. It
    grows with the lag between two processes. Antisymmetric and signed:
    ``A[i, j] = -A[j, i]``; positive means process ``i`` leads.

    ``df`` is the multitaper bandwidth, ``2 NW`` FFT bins (six at the default
    ``NW = 3``): closer frequencies share tapers and carry no independent
    phase. The sign is reliable for lags up to ``T / (4 NW)`` samples, beyond
    which the phase difference wraps. The band leaves out 0 Hz and ``fs / 2``
    and must hold at least two frequencies one bandwidth apart, or the class
    raises ``ValueError``.

    This is the unnormalised index. Nolte et al. divide it by a jackknife
    standard deviation over epochs, which a single recording does not provide,
    so the value scales with the number of frequencies in the band. It is
    computed here rather than by the backend's
    ``Connectivity.phase_slope_index``, which in spectral_connectivity 2.0
    sums over all pairs of frequencies rather than neighbouring ones and
    whose definition differs between releases.

    References
    ----------
    .. [1] G. Nolte et al., "Robustly estimating the flow direction of
           information in complex physical systems", Phys. Rev. Lett.
           100(23), 234101 (2008).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
    statistic : str, optional
        Only ``'mean'``, the name the identifiers (``psi_multitaper_mean_*``)
        give to the sum over the band. Default: ``'mean'``.
    time_halfbandwidth_product : float, optional
        Time-halfbandwidth product ``NW`` of the DPSS tapers, at least 1.5
        (see ``NonparametricSpectral``). Default: 3.
    """

    # Antisymmetric in (i, j): slope of the phase spectrum.
    _antisymmetric_spectrum = True
    name = "Phase slope index"
    labels = ["spectral", "undirected"]
    # A sum over the band, not a band statistic; `mean` is the name the
    # identifiers carry.
    _statistics = ("mean",)
    _phase_based = True

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        self._measure = "phase_slope_index"
        # Always `_get_statistic`: the backend method of this name is not the
        # published index.
        self._recompute = True

    @property
    def _cache_subkey(self):
        # (class, fs, NW, fmin, fmax), as in `key`.
        return (type(self).__name__, self._fs, self._nw, self._fmin, self._fmax)

    def _get_statistic(self, C):
        freq = np.asarray(C.frequencies)
        # The closed band less the real-valued bins, thinned to one bin per
        # multitaper bandwidth.
        inside = self._band(freq)
        coherency = C.coherency()[..., inside[::self._frequency_step], :, :]
        if coherency.shape[-3] < 2:
            raise ValueError(
                f"Phase slope index needs at least two independent frequencies "
                f"in [{self._fmin}, {self._fmax}] besides 0 Hz and fs/2; the "
                f"series is too short for this band.")
        return np.imag(np.sum(
            np.conj(coherency[..., :-1, :, :]) * coherency[..., 1:, :, :], axis=-3))


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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
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
            # The delay is a time offset.
            if self._statistic == "delay":
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
        in_band = (_closed_band(freqs, self._fmin, self._fmax)
                   & ~_real_valued_bins(freqs, self._fs))
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


# nitime.GrangerAnalyzer's grid size: granger_causality_xy evaluates the model at
# n_freqs // 2 + 1 points on [0, fs/2) (scipy.signal.freqz, endpoint excluded).
_GC_N_FREQS = 1024


def _fit_bivariate_var(x, order=None, max_order=50):
    """
    Yule-Walker fit of the two-process VAR behind parametric spectral GC.

    As ``nitime.analysis.granger.fit_model`` (Whittle's LWR recursion; with
    ``order=None``, the lowest order at which BIC stops improving), but on the
    Yule-Walker autocovariance: about the mean and normalised by ``T``
    (Brockwell & Davis 1991, sec. 11.5; Lütkepohl 2005, sec. 3.3.4). Its
    Toeplitz matrix is positive semidefinite, so the fitted model is stable;
    nitime's ``autocov_vector`` (normalised by ``T - k``, not demeaned) need
    not be.

    ``max_order`` is the largest order tried. Returns ``(coef, ecov)`` in
    nitime's convention, ``X[t] + sum_k coef[k] X[t-k] = E[t]``.
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

    Spectral GC is undefined at 0 Hz, so an ``fmin`` of 0 or below is raised
    to 1e-5 with a warning, and the identifier names the band actually used.
    A cell with more than ``nan_threshold`` of its in-band frequencies NaN is
    set to NaN, with a warning. Automatic order selection that has not
    converged by ``max_order`` raises ``ValueError``: the model is
    over-fitting at that record length.

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
        Sampling frequency; ``fmin`` and ``fmax`` are in the same units.
        Default: 1.
    fmin : float, optional
        Lower edge of the band; values at or below 0 become 1e-5.
        Default: 1e-5.
    fmax : float, optional
        Upper edge of the band. Default: 0.5 (Nyquist at ``fs = 1``).
    method : str, optional
        ``'nonparametric'`` (multitaper and Wilson's factorisation) or
        ``'parametric'`` (a VAR fitted to each pair).
        Default: ``'nonparametric'``.
    order : int or None, optional
        VAR order of the parametric method; None selects it per pair as the
        lowest order at which BIC stops improving. Ignored by the
        nonparametric method. Default: None.
    max_order : int, optional
        Largest order the parametric selection tries; a value other than 50
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
        fmin=1e-5,
        fmax=0.5,
        method="nonparametric",
        order=None,
        max_order=50,
        statistic="mean",
        ignore_nan=True,
        nan_threshold=0.5,
        time_halfbandwidth_product=_DEFAULT_NW,
    ):
        self._fs = fs  # Reaches the multitaper estimate and the frequency labels.
        self._nw = _check_time_halfbandwidth_product(time_halfbandwidth_product)
        if method != "nonparametric" and self._nw != _DEFAULT_NW:
            raise ValueError(
                "time_halfbandwidth_product sets the multitaper estimate, which "
                "method='parametric' does not use.")
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
        # `self._fmin`, not the `fmin` argument: spectral GC is undefined at
        # zero frequency, so `fmin=0` is overridden to 1e-5 above, and the
        # identifier names the band actually integrated.
        if self._method == "nonparametric":
            self._measure = "pairwise_spectral_granger_prediction"
            paramstr = (f"_nonparametric_{statistic}_fs-{fmt_param(fs)}"
                        f"_fmin-{fmt_param(self._fmin)}"
                        f"_fmax-{fmt_param(fmax)}").replace(
                ".", "-"
            ) + _nw_suffix(self._nw)
        else:
            self._order = order
            self._max_order = max_order
            paramstr = (f"_parametric_{statistic}_fs-{fmt_param(fs)}"
                        f"_fmin-{fmt_param(self._fmin)}"
                        f"_fmax-{fmt_param(fmax)}_order-{order}").replace(
                ".", "-"
            )
            # max_order bounds the order search, so it matters only when the
            # order is selected rather than given.
            if order is None and max_order != 50:
                paramstr += f"_maxorder-{max_order}"
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
                    # BIC still improving at `max_order` signals over-fitting
                    # at this record length; say so rather than relay the
                    # bare message.
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
                # The frequencies the model was evaluated at, in Hz (`w` is
                # scipy.signal.freqz's grid, k*pi/513 rad on [0, pi)).
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
            freq_id = np.flatnonzero(_closed_band(freq, self._fmin, self._fmax))

            result = _to_source_target(self, self._statfn(cache[0, freq_id, :, :], axis=0))

            # Oriented as the values it masks, so it blanks the cell and not
            # its mirror.
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
            # The model could not be fitted: the error reaches
            # `Calculator.errors` with its cause, not an all-NaN table.
            raise


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
        refuses a record too short for four segments (``T < 20``). An integer
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
    process, or when every band is skipped (``T < 46``).

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
            return A
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

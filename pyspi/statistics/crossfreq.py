"""Cross-frequency coupling statistics.

Every spectral SPI in ``spectral.py`` and ``wavelet.py`` relates a frequency of
one process to the *same* frequency of another. The statistics here relate
different frequencies: the phase of one process in a low band to the amplitude
envelope of another process in a high band.

Frequencies are in cycles per sample (``fs = 1``, Nyquist 0.5), as everywhere
in ``spectral.py``.
"""
import functools
import warnings

import numpy as np
from scipy import signal, special

from pyspi.base import (
    Directed,
    Unsigned,
    fill_pairwise,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import fmt_param, require_int

# Butterworth order of each band-pass. Applied forwards and backwards
# (`sosfiltfilt`): zero phase, -6 dB at the band edges, ~48 dB per octave outside.
_ORDER = 4

# A filter's "length" is the number of samples holding this fraction of the
# energy of its one-pass impulse response.
_ENERGY_FRACTION = 0.99

# Samples discarded at each end of the record, in filter lengths.
_EDGE_FILTER_LENGTHS = 2

# Relative slack on the bandwidth rule, so an amplitude band exactly twice the
# upper phase frequency wide is not refused over a rounding error in f4 - f3.
_RULE_RTOL = 1e-9

# Seed of the surrogates' circular shifts: fixed, so the shifts depend only on
# the window length and n_surr, and a pair's value not on the other processes.
_SURROGATE_SEED = 0


@functools.lru_cache(maxsize=None)
def _shift_index(n, n_surr):
    """``(n_surr, n)`` gather index: row ``k`` circularly delays a length-``n``
    series by ``s_k`` samples, ``s_k`` uniform on the integers in
    ``[n // 10, n - n // 10]``. Read-only and cached per ``(n, n_surr)``."""
    rng = np.random.default_rng(_SURROGATE_SEED)
    shifts = rng.integers(n // 10, n - n // 10, size=n_surr, endpoint=True)
    index = (np.arange(n)[None, :] - shifts[:, None]) % n
    index.setflags(write=False)
    return index


def _validate_band(name, band):
    """``band`` as a ``(low, high)`` tuple of floats with 0 < low < high < 0.5."""
    try:
        values = tuple(band)
    except TypeError:
        values = ()
    if len(values) != 2 or not all(
        isinstance(f, (int, float, np.integer, np.floating))
        and not isinstance(f, bool) for f in values
    ):
        raise ValueError(
            f"{name} must be a pair (low, high) of frequencies in cycles per "
            f"sample, got {band!r}."
        )
    lo, hi = (float(f) for f in values)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi < 0.5):
        raise ValueError(
            f"{name} must satisfy 0 < low < high < 0.5 cycles per sample (the "
            f"Nyquist frequency at fs = 1), got {band!r}."
        )
    return lo, hi


@functools.lru_cache(maxsize=None)
def _bandpass(band):
    """Second-order sections of the order-``_ORDER`` Butterworth band-pass."""
    return signal.butter(_ORDER, band, btype="bandpass", fs=1.0, output="sos")


@functools.lru_cache(maxsize=None)
def _filter_length(band):
    """Samples holding 99% of the energy of the one-pass impulse response.

    The response rings for a time set by the lower band edge and the
    bandwidth, so it is evaluated over ``100 / min(low, high - low)`` samples,
    far past where its energy has converged.
    """
    lo, hi = band
    n = int(np.ceil(100.0 / min(lo, hi - lo)))
    impulse = np.zeros(n)
    impulse[0] = 1.0
    response = signal.sosfilt(_bandpass(band), impulse)
    energy = np.cumsum(response * response)
    return int(np.searchsorted(energy, _ENERGY_FRACTION * energy[-1])) + 1


def _band_series(data, band, kind):
    """Phase or amplitude of every process in ``band``, full length, cached.

    ``data.crossfreq[(band, "phase")]`` is the angle and
    ``data.crossfreq[(band, "amp")]`` the modulus of the analytic signal
    (``scipy.signal.hilbert``) of the zero-phase band-passed series, each an
    ``(M, T)`` read-only array. They are stored untrimmed: the edge discarded
    depends on the *pair* of bands, so it is applied by the caller.
    """
    try:
        cache = data.crossfreq
    except AttributeError:
        cache = data.crossfreq = {}
    key = (band, kind)
    series = cache.get(key)
    if series is None:
        z = np.squeeze(data.to_numpy(), axis=2)
        analytic = signal.hilbert(
            signal.sosfiltfilt(_bandpass(band), z, axis=-1), axis=-1)
        series = np.angle(analytic) if kind == "phase" else np.abs(analytic)
        series.setflags(write=False)
        cache[key] = series
    return series


class PhaseAmplitudeCoupling(Directed, Unsigned):
    """Phase-amplitude coupling (PAC) from the source's phase to the target's amplitude.

    ``A[i, j]`` measures how strongly the amplitude envelope of process ``j``
    (column, target) in a high band ``amp_band = (f3, f4)`` is modulated by the
    phase of process ``i`` (row, source) in a low band ``phase_band = (f1,
    f2)``. Bands are in cycles per sample. The two orientations are different
    quantities (``A[j, i]`` takes the *phase* of ``j``), so the SPI is
    directed. The two raw statistics are in [0, 1] and unsigned; their
    surrogate-normalised versions (``_excess``, ``_z``) are signed.

    Signals
    -------
    Each process is band-passed with a zero-phase (forward-backward,
    ``scipy.signal.sosfiltfilt``) order-4 Butterworth filter, and the analytic
    signal is taken with ``scipy.signal.hilbert``:

    ``phi_t = angle(H[bp_{f1,f2}(x_i)]_t)``, ``A_t = |H[bp_{f3,f4}(x_j)]_t|``.

    Order 4 per pass (about 48 dB per octave after the two passes) separates
    the bands while keeping the ringing short. Filter transients and the
    Hilbert transform's wrap-around are confined to the ends of the record, so
    ``2 L`` samples are discarded at each end, where ``L`` is the larger of the
    two filters' lengths, a filter's length being the number of samples that
    hold 99% of its impulse response's energy. For the bundled pairs that is
    200 samples at each end for ``p = (0.01, 0.05), a = (0.15, 0.4)`` and 100
    for ``p = (0.05, 0.1), a = (0.2, 0.45)``. On a record shorter than ``8 L``
    the edge is capped at ``T // 4`` samples, with a warning.
    ``edge_samples(T)`` returns the number of samples discarded at each end.

    Statistics
    ----------
    With ``n`` retained samples:

    - ``"mi"``: the modulation index of Tort et al. (2010). The source phase is
      divided into ``n_bins`` equal bins on [-pi, pi), ``m_k`` is the mean
      target amplitude over the samples whose phase falls in bin ``k``, ``P_k
      = m_k / sum_l m_l``, and ``MI = (log N - H(P)) / log N`` with ``H(P) =
      -sum_k P_k log P_k`` and ``N = n_bins``: the Kullback-Leibler divergence
      of ``P`` from the uniform distribution, normalised by its maximum. 0 when
      the mean amplitude is the same in every phase bin, 1 when all amplitude
      is in one bin. Every bin must be occupied; a pair whose source phase
      leaves a bin empty (a record too short for the phase band) is undefined
      and NaN.
    - ``"mvl"``: the normalised mean vector length (direct PAC estimate) of
      Özkurt & Schnitzler (2011),
      ``|sum_t A_t exp(i phi_t)| / (sqrt(n) sqrt(sum_t A_t^2))``,
      i.e. ``|mean(A exp(i phi))| / sqrt(mean(A^2))``, which is at most 1.
      Unlike the raw mean vector length of Canolty et al. (2006), it does not
      scale with the target's amplitude. For ``A_t = 1 + m cos(phi_t)`` with
      the phase uniformly distributed it is ``(m / 2) / sqrt(1 + m^2 / 2)``.

    Both are invariant to rescaling either process (the source enters only
    through its phase, and ``P`` and the MVL are ratios in ``A``), so they do
    not depend on z-scoring.

    Surrogate normalisation
    -----------------------
    Neither raw statistic is zero for independent processes, and their chance
    level grows with the target's own amplitude modulation, which a slow
    rhythm of the source can align with by chance over a finite record. The
    normalised statistics compare the index with its values on ``n_surr``
    surrogates, each a circular shift of the target's envelope within the
    ``n`` retained samples by ``s`` samples, ``s`` uniform on the integers in
    ``[n // 10, n - n // 10]`` (Canolty et al. 2006; Tort et al. 2010). A
    shift keeps the envelope and its spectrum and destroys its timing relative
    to the source's phase:

    - ``"<stat>_excess"``: the index minus the mean of its surrogates, in the
      index's units. About 0 for independent processes whatever their own
      modulation, and it grows with the coupling as the raw index does.
    - ``"<stat>_z"``: ``(index - mean) / sd`` over the surrogates (``sd`` with
      ``ddof = 0``), a significance score for comparing pairs whose targets
      differ in their own modulation. It is not an effect size: the
      surrogates' spread grows with the target's modulation about as fast as
      the index, so it saturates as the coupling grows, and it grows with the
      record length. Under independence it is centred near 0 but is not
      standard normal. Undefined (NaN) if every surrogate gives the same value.

    The shifts come from a fixed seed and depend only on ``n`` and
    ``n_surr``, so the values are deterministic and a pair's value does not
    depend on the other processes. Time-shift surrogates fail for a strictly
    periodic source: a shift only rotates the preferred phase, so the
    surrogates keep the coupling and both normalised values are near 0.

    Neither normalisation removes spurious PAC from sharp, non-sinusoidal
    waveforms, whose harmonics in the amplitude band are locked to their own
    fundamental in the phase band (Kramer, Tort & Kopell 2008); read PAC on
    such data together with the waveform shape.

    Validity of the bands
    ---------------------
    The envelope of the target can follow a modulation at frequency ``f``
    only if the amplitude band passes the sidebands at ``carrier +/- f``, so
    its bandwidth ``f4 - f3`` must be at least twice the highest phase
    frequency ``f2`` (Aru et al. 2015). A narrower amplitude band filters the
    modulation out and biases the coupling towards zero, so construction
    raises ``ValueError`` for such a band pair. It also raises if the amplitude
    band starts below the top of the phase band (``f3 < f2``).

    Parameters
    ----------
    phase_band : (float, float)
        ``(f1, f2)``, the band of the source's phase, cycles per sample.
    amp_band : (float, float)
        ``(f3, f4)``, the band of the target's amplitude, cycles per sample.
        Requires ``f3 >= f2`` and ``f4 - f3 >= 2 f2``.
    statistic : {"mi", "mvl", "mi_excess", "mvl_excess", "mi_z", "mvl_z"}
        Tort modulation index or Özkurt-Schnitzler normalised MVL: raw, as the
        excess over their time-shift surrogates' mean, or as a z-score against
        them.
    n_bins : int
        Phase bins of the modulation index (Tort et al. use 18 of 20 degrees).
        Ignored by the MVL statistics. A value other than 18 is appended to the
        identifier as ``_nbins-<n>``.
    n_surr : int
        Number of time-shift surrogates (at least 2) of the ``_excess`` and
        ``_z`` statistics; ignored by the raw ones. A value other than 100 is
        appended to the identifier as ``_nsurr-<n>``.

    The identifier is ``pac_<statistic>_p-<f1>-<f2>_a-<f3>-<f4>`` with decimal
    points written as dashes, as in ``spectral.py``: ``pac_mi_p-0-01-0-05_a-
    0-15-0-4`` is the modulation index with phase band (0.01, 0.05) and
    amplitude band (0.15, 0.4), and ``pac_mi_excess_p-0-01-0-05_a-0-15-0-4``
    its excess over the surrogates.

    Its direction is a role, which process supplies the phase and which the
    amplitude, not temporal precedence as in the Granger-type SPIs: ``A[i, j]
    > A[j, i]`` does not by itself mean that ``i`` drives ``j``.

    References
    ----------
    Tort, Komorowski, Eichenbaum & Kopell (2010). Measuring phase-amplitude
    coupling between neuronal oscillations of different frequencies.
    J Neurophysiol 104:1195-1210.

    Özkurt & Schnitzler (2011). A critical note on the definition of
    phase-amplitude cross-frequency coupling. J Neurosci Methods 201:438-443.

    Canolty et al. (2006). High gamma power is phase-locked to theta
    oscillations in human neocortex. Science 313:1626-1628.

    Aru et al. (2015). Untangling cross-frequency coupling in neuroscience.
    Curr Opin Neurobiol 31:51-61.

    Kramer, Tort & Kopell (2008). Sharp edge artifacts and spurious coupling
    in EEG frequency comodulation measures. J Neurosci Methods 170:352-357.
    """

    name = "Phase-amplitude coupling"
    identifier = "pac"
    labels = ["unsigned", "directed", "nonlinear", "spectral"]

    # All statistics of one band pair share data.crossfreq[(band, kind)].
    _cache_namespace = "crossfreq"

    _statistics = ("mi", "mvl", "mi_excess", "mvl_excess", "mi_z", "mvl_z")
    _DEFAULT_N_BINS = 18
    _DEFAULT_N_SURR = 100

    def __init__(self, phase_band=(0.01, 0.05), amp_band=(0.15, 0.4),
                 statistic="mi", n_bins=18, n_surr=100):
        self._phase_band = _validate_band("phase_band", phase_band)
        self._amp_band = _validate_band("amp_band", amp_band)
        if statistic not in self._statistics:
            raise ValueError(
                f"Unknown statistic {statistic!r}; PhaseAmplitudeCoupling "
                f"supports {', '.join(self._statistics)}.")
        # "mi_z" -> ("mi", "z"): the raw index and its normalisation ("" if raw).
        self._statistic, _, self._normalise = statistic.partition("_")
        self._surrogate = bool(self._normalise)
        self._n_bins = require_int("n_bins", n_bins, minimum=2)
        self._n_surr = require_int("n_surr", n_surr, minimum=2)
        if self._surrogate:
            # Negative when the index is below its surrogates' mean.
            self.labels = ["signed" if label == "unsigned" else label
                           for label in PhaseAmplitudeCoupling.labels]

        f1, f2 = self._phase_band
        f3, f4 = self._amp_band
        if f3 < f2:
            raise ValueError(
                f"amp_band {self._amp_band} starts below the top of phase_band "
                f"{self._phase_band}; phase-amplitude coupling needs the "
                f"amplitude band above the phase band (f3 >= f2).")
        if f4 - f3 < 2 * f2 * (1 - _RULE_RTOL):
            raise ValueError(
                f"amp_band {self._amp_band} is {f4 - f3:g} wide, less than twice "
                f"the highest phase frequency ({2 * f2:g}): its envelope cannot "
                f"carry a modulation at the phase frequencies, so the coupling "
                f"would be underestimated (Aru et al. 2015). Widen the amplitude "
                f"band or lower the phase band.")

        suffix = (f"_{statistic}_p-{fmt_param(f1)}-{fmt_param(f2)}"
                  f"_a-{fmt_param(f3)}-{fmt_param(f4)}")
        if self._statistic == "mi" and self._n_bins != self._DEFAULT_N_BINS:
            suffix += f"_nbins-{self._n_bins}"
        if self._surrogate and self._n_surr != self._DEFAULT_N_SURR:
            suffix += f"_nsurr-{self._n_surr}"
        self.identifier += suffix.replace(".", "-")

    @property
    def _cache_subkey(self):
        return (self._phase_band, self._amp_band)

    @property
    def edge(self):
        """Nominal samples discarded at each end: ``2 L`` of the longer filter."""
        return _EDGE_FILTER_LENGTHS * max(_filter_length(self._phase_band),
                                          _filter_length(self._amp_band))

    def edge_samples(self, n_observations):
        """Samples discarded at each end of a record of ``n_observations``."""
        return min(self.edge, n_observations // 4)

    def _window(self, data):
        """Source phase and target amplitude, ``(M, n)`` each, edges discarded."""
        T = data.n_observations
        edge = self.edge_samples(T)
        if edge < self.edge:
            warnings.warn(
                f'SPI "{self.identifier}": T={T} is shorter than {4 * self.edge} '
                f'samples, so only {edge} (not {self.edge}) samples are '
                f'discarded at each end; values near the ends of the retained '
                f'window carry filter transients.')
        phase = _band_series(data, self._phase_band, "phase")[:, edge:T - edge]
        amp = _band_series(data, self._amp_band, "amp")[:, edge:T - edge]
        return phase, amp

    def _phase_bins(self, phase):
        """Bin index in [0, n_bins) of each phase; bin k is [-pi + k w, -pi + (k+1) w)."""
        nb = self._n_bins
        bins = np.floor((phase + np.pi) * (nb / (2 * np.pi))).astype(np.intp)
        # np.angle returns pi itself for a negative real value: fold it into
        # the last bin rather than open a bin of its own.
        np.clip(bins, 0, nb - 1, out=bins)
        return bins

    def _modulation_index(self, bins, counts, amp):
        nb = self._n_bins
        empty = np.count_nonzero(counts == 0)
        if empty:
            raise ValueError(
                f"{empty} of {nb} phase bins are empty in the {bins.size}-sample "
                f"window; the modulation index needs every bin occupied (use a "
                f"longer record or fewer bins).")
        mean_amp = np.bincount(bins, weights=amp, minlength=nb) / counts
        total = mean_amp.sum()
        if not total > 0:
            raise ValueError("the target has no amplitude in amp_band.")
        p = mean_amp / total
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        log_n = np.log(nb)
        return max(0.0, float((log_n - entropy) / log_n))

    @staticmethod
    def _mean_vector_length(phasor, amp):
        power = np.dot(amp, amp)
        if not power > 0:
            raise ValueError("the target has no amplitude in amp_band.")
        return float(np.abs(np.dot(phasor, amp)) / np.sqrt(amp.size * power))

    def issigned(self):
        return self._surrogate

    def _surrogate_mi(self, onehot, counts, shifted):
        """Modulation index of each row of ``shifted`` (``(n_surr, n)``)."""
        mean_amp = (shifted @ onehot) / counts
        p = mean_amp / mean_amp.sum(axis=1, keepdims=True)
        log_n = np.log(self._n_bins)
        return (log_n + np.sum(special.xlogy(p, p), axis=1)) / log_n

    def _normalised(self, observed, surrogates):
        """``observed`` against its surrogates: excess over their mean, or z-score."""
        excess = observed - surrogates.mean()
        if self._normalise == "excess":
            return float(excess)
        sd = surrogates.std()
        if not sd > 0:
            raise ValueError(
                "the statistic is identical on every time-shifted surrogate, "
                "so its z-score is undefined.")
        return float(excess / sd)

    def _pair_function(self, phase, amp):
        """``compute(i, j)``: the statistic of source ``i`` and target ``j``.

        Each source's bins (or phasors) are formed once, then the raw index is
        O(n) per pair. For the surrogate statistics, each target's ``n_surr``
        shifted envelopes and each source's one-hot bin matrix are formed once,
        on first use, and a pair's surrogates are one matrix product.
        """
        if self._statistic == "mi":
            bins = self._phase_bins(phase)
            counts = [np.bincount(b, minlength=self._n_bins) for b in bins]
            index = lambda i, j: self._modulation_index(bins[i], counts[i], amp[j])
        else:
            phasors = np.exp(1j * phase)
            index = lambda i, j: self._mean_vector_length(phasors[i], amp[j])
        if not self._surrogate:
            return index

        n = amp.shape[1]
        shift = _shift_index(n, self._n_surr)
        shifted, onehot = {}, {}

        def compute(i, j):
            observed = index(i, j)  # raises first for an undefined pair
            if j not in shifted:
                shifted[j] = amp[j][shift]
            if self._statistic == "mi":
                if i not in onehot:
                    onehot[i] = np.eye(self._n_bins)[bins[i]]
                surrogates = self._surrogate_mi(onehot[i], counts[i], shifted[j])
            else:
                surrogates = (np.abs(shifted[j] @ phasors[i])
                              / np.sqrt(n * np.dot(amp[j], amp[j])))
            return self._normalised(observed, surrogates)

        return compute

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        phase, amp = self._window(data)
        return self._pair_function(phase[[i]], amp[[j]])(0, 0)

    @parse_multivariate
    def multivariate(self, data):
        # A pair with an empty phase bin is NaN.
        phase, amp = self._window(data)
        M = phase.shape[0]
        A = np.full((M, M), np.nan)
        pairs = [(i, j) for i in range(M) for j in range(M) if i != j]
        return fill_pairwise(self.identifier, A, pairs,
                             self._pair_function(phase, amp))

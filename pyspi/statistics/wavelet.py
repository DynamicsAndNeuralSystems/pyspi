import numpy as np
from mne_connectivity import phase_slope_index

from pyspi.base import NotAppropriate, Undirected, Unsigned, parse_multivariate
from pyspi.utils import (
    empty_band_error,
    fmt_param,
    require_band,
    require_positive_float,
)


class mne(Unsigned):
    """
    Base class of the Morlet-wavelet SPIs computed with mne-connectivity.

    Holds the sampling frequency, the band ``[fmin, fmax]`` and the statistic
    that reduces the result, and appends them to the identifier. The band is
    checked at construction: ``fs`` positive and finite, and
    ``0 <= fmin < fmax <= fs / 2``.
    """

    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean"):
        require_positive_float("fs", fs)
        if fmax is None:
            fmax = fs / 2
        # Strict: the wavelet frequencies are spaced linearly from fmin to fmax.
        require_band(fs, fmin, fmax, strict=True)

        # fs is passed to mne as sfreq and sets PhaseSlopeIndex's frequency floor.
        self._fs = fs
        self._fmin = fmin
        # At most fs / 2 exactly, which mne requires of every wavelet frequency.
        self._fmax = min(fmax, fs / 2)
        if statistic not in ("mean", "max"):
            raise ValueError(
                f"{type(self).__name__} does not support statistic="
                f"{statistic!r}; it supports mean, max.")
        self._statfn = np.nanmean if statistic == "mean" else np.nanmax

        self._statistic = statistic

        paramstr = (
            f"_wavelet_{statistic}_fs-{fmt_param(fs)}_fmin-{fmt_param(fmin)}"
            f"_fmax-{fmt_param(fmax)}".replace(
                ".", "-"
            )
        )
        self.identifier += paramstr


class PhaseSlopeIndex(mne, Undirected):
    """
    Time-averaged instantaneous phase slope from Morlet wavelets.

    Not Nolte et al.'s phase slope index [1]_, which weights each phase
    difference by the coherence; that index is ``statistics.spectral``'s
    ``PhaseSlopeIndex``. This class computes, through mne-connectivity's
    ``phase_slope_index(mode="cwt_morlet")`` on the record as one epoch:

    - Morlet coefficients ``W_i(f, t)`` at 10 frequencies spaced linearly from
      ``max(fmin, 5 fs / T)`` to ``fmax`` (7 cycles, fewer where the wavelet
      would not fit the record);
    - the coherency at every time point. mne-connectivity averages the
      cross-spectrum over epochs and nothing else, so with one epoch
      ``C(f, t) = exp(i dphi(f, t))`` with ``dphi`` the phase of
      ``W_i W_j*``: ``|C| = 1`` and no coherence weighting;
    - ``psi(t) = sum_k sin(dphi(f_{k+1}, t) - dphi(f_k, t))`` over the 8
      interior frequencies (the backend's band is open, so both ends drop out),
      bounded by 7 in magnitude.

    ``mean`` is the time average of ``psi(t)``, zero in expectation for
    independent series, with a standard deviation that falls as
    ``1 / sqrt(T)``. It is antisymmetric and signed: ``A[i, j] = -A[j, i]``;
    positive means process ``i`` leads. Neighbouring frequencies are
    ``df = (fmax - fmin) / 9`` apart, so for a delay longer than about
    ``1 / (2 df)`` samples the sign reverses (about 9 samples at ``fs = 1``
    on [0, 0.5]). The value is not proportional to the delay.

    ``max`` is the maximum of ``psi(t)`` over time. It tends to the bound
    whatever the data, so it carries no lead/lag information; it is signed
    but not antisymmetric.

    Frequencies are in the units of ``fs``, at the default ``fs = 1`` in
    cycles per sample; ``fs`` only relabels the frequency axis, so
    ``fs = 250`` with the band [8, 12] gives the same values as ``fs = 1``
    with [0.032, 0.048], up to floating-point rounding. At construction
    ``fs`` must be positive and finite and ``0 <= fmin < fmax <= fs / 2``, or
    ``ValueError`` is raised; a band whose upper edge is at or below the floor
    ``5 fs / T`` raises ``NotAppropriate`` when the SPI is computed, which the
    Calculator records as NaN with the reason.

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
        Lower edge of the band, in [0, ``fmax``), raised to ``5 fs / T``
        (five cycles over the record) when below it. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, in (``fmin``, ``fs / 2``]; None is the
        Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        ``'mean'`` (antisymmetric) or ``'max'`` of ``psi(t)`` over time.
        Default: ``'mean'``.
    """

    name = "Phase slope index (wavelet)"
    labels = ["wavelet"]

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        # Only a statistic that commutes with negation keeps the matrix
        # antisymmetric: mean does, max does not.
        trait = "antisymmetric" if self._statistic == "mean" else "asymmetric"
        self.labels = [l for l in self.labels
                       if l not in ("undirected", "directed")
                       ] + [trait, "signed"]
        # Signed in behaviour too, as in spectral.NonparametricSpectral:
        # `_rmmin` would shift an unsigned SPI and destroy the antisymmetry.
        self.issigned = lambda: True
        self.identifier += f"_{self._statistic}"

    def _get_psi(self, data):
        """PSI over this class's ``[fmin, fmax]`` band, cached per band."""
        # Frequencies below five cycles over the record are not resolved.
        floor = 5.0 * self._fs / data.n_observations
        fmin = max(self._fmin, floor)
        if fmin >= self._fmax:
            raise empty_band_error(
                "PhaseSlopeIndex", self._fmin, self._fmax,
                f"no frequency of the series' grid (the lowest it resolves is "
                f"5 fs / T = {floor:.4g} for T = {data.n_observations})")
        # Keyed on the resolved fmin, so bands floored alike share an entry.
        key = (self._fs, fmin, self._fmax)
        try:
            return data.mne_psi[key]
        except (AttributeError, KeyError):
            pass

        z = np.moveaxis(data.to_numpy(), 2, 0)
        cwt_freqs = np.linspace(max(fmin, 1e-6), self._fmax, 10)
        # MNE's Morlet wavelet spans about 1.59 n_cycles / f samples, so at its
        # default 7 cycles the low-frequency wavelets can outlast the record.
        # Cap n_cycles (at least 1) so every wavelet fits, with a 5% margin.
        n_obs = data.n_observations
        max_cycles = 0.95 * n_obs * cwt_freqs / (1.59 * self._fs)
        cwt_n_cycles = np.clip(max_cycles, 1.0, 7.0)
        psi_obj = phase_slope_index(
            data=z,
            mode="cwt_morlet",
            sfreq=self._fs,
            mt_adaptive=True,
            fmin=fmin,
            fmax=self._fmax,
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles,
            verbose=False,
        )
        psi = psi_obj.get_data(output="dense")

        try:
            data.mne_psi[key] = psi
        except AttributeError:
            data.mne_psi = {key: psi}
        return psi

    @parse_multivariate
    def multivariate(self, data):
        psi = np.real(self._get_psi(data))

        # mne_connectivity fills only the lower triangle. PSI is antisymmetric
        # at every frequency and time, so the upper triangle is its negative,
        # filled before the statistic (max does not commute with negation).
        # The upper triangle is zero, so subtracting the transpose does both.
        psi = psi - np.swapaxes(psi, 0, 1)

        adj = self._statfn(psi, axis=(2, 3))
        np.fill_diagonal(adj, np.nan)
        return adj


# Number of linearly spaced bin centres on (0, fs/2) the wavelet grid starts from.
_N_WAVELET_FREQS = 64


def _efolding_time(freqs, fs, n_cycles):
    """
    Morlet e-folding time in samples, ``sqrt(2) sigma_t`` with
    ``sigma_t = n_cycles / (2 pi f)`` seconds.
    """
    return np.sqrt(2) * n_cycles / (2 * np.pi * freqs) * fs


def _morlet_frequencies(n_observations, fs, n_cycles):
    """
    Bin centres ``(k + 1/2) fs / (2 * 64)`` on ``(0, fs/2)`` whose cone of
    influence leaves at least half the record (``T - 2 tau >= T / 2``).
    """
    freqs = (np.arange(_N_WAVELET_FREQS) + 0.5) * (fs / 2) / _N_WAVELET_FREQS
    tau = _efolding_time(freqs, fs, n_cycles)
    return freqs[n_observations - 2 * tau >= n_observations / 2]


def _morlet_coefficients(z, freqs, fs, n_cycles):
    """
    Analytic Morlet coefficients ``W[i, f, t]`` of the rows of ``z``.

    Each wavelet is a Gaussian in frequency centred on ``f`` with standard
    deviation ``f / n_cycles`` and unit peak gain, applied by FFT with zero
    padding to at least twice the record, so ``W`` is the analytic signal of
    the series band-passed around ``f``.
    """
    n_obs = z.shape[1]
    nfft = int(2 ** np.ceil(np.log2(2 * n_obs)))
    spectrum = np.fft.fft(z - z.mean(axis=1, keepdims=True), n=nfft, axis=1)
    nu = np.fft.fftfreq(nfft, d=1 / fs)
    gain = 2.0 * np.exp(
        -0.5 * ((nu[None, :] - freqs[:, None]) / (freqs[:, None] / n_cycles)) ** 2)
    gain[:, nu < 0] = 0.0
    return np.fft.ifft(spectrum[:, None, :] * gain[None], axis=2)[..., :n_obs]


class WaveletCoherence(Unsigned, Undirected):
    """
    Smoothed wavelet coherence of each pair, averaged over a frequency band [1]_.

    ``R^2_ij(f, t) = |S(W_i W_j*)|^2 / (S(|W_i|^2) S(|W_j|^2))`` is the local
    squared coherence of the analytic Morlet coefficients ``W(f, t)``, where
    ``S`` smooths in time with a Gaussian and then in scale with a boxcar in
    log frequency [1]_ [2]_. ``R^2`` is averaged over time and then over the
    grid frequencies in ``[fmin, fmax]``: 64 bin centres on ``(0, fs/2)``,
    less those whose cone of influence (one e-folding time
    ``sqrt(2) n_cycles / (2 pi f)`` from either end [3]_) covers more than
    half the record. The Morlet coefficients are computed by FFT as a Gaussian
    band-pass of standard deviation ``f / n_cycles`` around each frequency.
    The scale boxcar is uniform in log frequency, so on the linear grid each
    frequency is weighted by ``1 / f``.

    Undirected and unsigned: symmetric, in [0, 1], and 1 for identical
    series. Because the coherence is estimated locally before it is averaged,
    it stays high when the coupling is transient or its sign or phase lag
    changes during the record, where the coherence of the whole record
    cancels.

    Each local estimate pools few independent samples, so ``R^2`` is biased
    upwards: independent series give values well above zero, by an amount
    set by the smoothing rather than by the record length. Compare values
    with each other or with surrogates, not with zero.

    The defaults smooth twice as widely in time and in scale as the minimal
    windows of [1]_ (one wavelet width in time, 0.6 octave in scale). That
    lowers the bias and variance of ``R^2`` while the time window, with a
    standard deviation of about two cycles, still follows coupling that
    changes over tens of cycles.
    Settings other than the defaults are appended to the identifier
    (``_cycles-8``, ``_tsmooth-1``, ``_ssmooth-0-6``, ``_coi-False``).

    Frequencies are in the units of ``fs``, at the default ``fs = 1`` in
    cycles per sample; ``fs`` only relabels the frequency axis, so
    ``fs = 250`` with the band [8, 12] gives the same values as ``fs = 1``
    with [0.032, 0.048], up to floating-point rounding. At construction
    ``fs`` must be positive and finite and ``0 <= fmin <= fmax <= fs / 2``,
    or ``ValueError`` is raised. A band holding no frequency of the grid,
    which depends on ``T``, and a series too short for any wavelet to fit
    raise ``NotAppropriate`` when the SPI is computed, which the Calculator
    records as NaN with the reason.

    References
    ----------
    .. [1] C. Torrence, P.J. Webster, "Interdecadal changes in the
           ENSO-monsoon system", Journal of Climate 12(8), 2679-2690 (1999).
    .. [2] A. Grinsted, J.C. Moore, S. Jevrejeva, "Application of the cross
           wavelet transform and wavelet coherence to geophysical time
           series", Nonlinear Processes in Geophysics 11(5/6), 561-566 (2004).
    .. [3] C. Torrence, G.P. Compo, "A practical guide to wavelet analysis",
           Bulletin of the American Meteorological Society 79(1), 61-78 (1998).

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, positive and finite. ``fmin``, ``fmax`` and the
        grid are in its units; at the default 1 they are in cycles per
        sample. Default: 1.
    fmin : float, optional
        Lower edge of the band, inclusive, in [0, ``fs / 2``]. Default: 0.
    fmax : float or None, optional
        Upper edge of the band, inclusive, in [``fmin``, ``fs / 2``]; None is
        the Nyquist frequency ``fs / 2``. Default: None.
    statistic : str, optional
        Band statistic; only ``'mean'`` is accepted. Default: ``'mean'``.
    n_cycles : float, optional
        Morlet cycles, setting the wavelet's time spread
        ``n_cycles / (2 pi f)``; 6 corresponds to ``omega_0 = 6``.
        Default: 6.
    time_smoothing : float, optional
        Standard deviation of the Gaussian time window, in units of the
        wavelet's time spread. Default: 2.
    scale_smoothing : float, optional
        Width of the scale boxcar in octaves; 0 smooths in time only.
        Default: 1.2.
    coi : bool, optional
        Average only the samples outside the cone of influence. Default: True.
    """

    name = "Wavelet coherence"
    labels = ["wavelet", "undirected"]
    _cache_namespace = "wavelet_coherence"

    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean", n_cycles=6,
                 time_smoothing=2, scale_smoothing=1.2, coi=True):
        require_positive_float("fs", fs)
        if fmax is None:
            fmax = fs / 2
        require_band(fs, fmin, fmax)
        if statistic != "mean":
            raise ValueError(
                f"WaveletCoherence supports only statistic='mean', got {statistic!r}.")
        if not isinstance(coi, (bool, np.bool_)):
            raise TypeError(f"coi must be True or False, got {coi!r}.")
        scale_smoothing = (0.0 if scale_smoothing == 0 else
                           require_positive_float("scale_smoothing", scale_smoothing))
        self._fs = fs
        self._fmin = fmin
        self._fmax = fmax
        self._statistic = statistic
        self._n_cycles = _integral(require_positive_float("n_cycles", n_cycles))
        self._time_smoothing = _integral(
            require_positive_float("time_smoothing", time_smoothing))
        self._scale_smoothing = _integral(scale_smoothing)
        self._coi = bool(coi)

        paramstr = (
            f"_wavelet_{statistic}_fs-{fmt_param(fs)}_fmin-{fmt_param(fmin)}"
            f"_fmax-{fmt_param(fmax)}")
        # Settings other than the defaults are named; the defaults are not.
        if self._n_cycles != 6:
            paramstr += f"_cycles-{fmt_param(self._n_cycles)}"
        if self._time_smoothing != 2:
            paramstr += f"_tsmooth-{fmt_param(self._time_smoothing)}"
        if self._scale_smoothing != 1.2:
            paramstr += f"_ssmooth-{fmt_param(self._scale_smoothing)}"
        if not self._coi:
            paramstr += "_coi-False"
        self.identifier = "wtc" + paramstr.replace(".", "-")

    @property
    def _cache_subkey(self):
        return (self._fs, self._n_cycles, self._time_smoothing,
                self._scale_smoothing, self._coi)

    def _smooth(self, a, freqs, scale_kernel):
        """Gaussian time smoothing of ``a[..., f, t]``, then the scale boxcar."""
        n_obs = a.shape[-1]
        sigma = self._time_smoothing * self._n_cycles / (2 * np.pi * freqs) * self._fs
        # Padding beyond four standard deviations keeps the circular
        # convolution from wrapping one end of the record onto the other.
        nfft = int(2 ** np.ceil(np.log2(n_obs + 4 * sigma.max())))
        nu = np.fft.fftfreq(nfft)
        kernel = np.exp(-2 * (np.pi * nu[None, :] * sigma[:, None]) ** 2)
        a = np.fft.ifft(np.fft.fft(a, n=nfft, axis=-1) * kernel, axis=-1)[..., :n_obs]
        return scale_kernel @ a

    def _scale_kernel(self, freqs):
        """
        Boxcar of ``scale_smoothing`` octaves, uniform in log frequency.

        The grid is linear, so each frequency is weighted by ``1 / f``, the
        width of its bin in log frequency.
        """
        if self._scale_smoothing == 0:
            return np.eye(freqs.size)
        octave = np.log2(freqs)
        kernel = ((np.abs(octave[:, None] - octave[None, :])
                   <= self._scale_smoothing / 2) / freqs[None, :])
        return kernel / kernel.sum(axis=1, keepdims=True)

    def _coherence_by_frequency(self, data):
        """Time-averaged ``R^2`` at every grid frequency, ``(F, M, M)``, cached."""
        key = self._cache_subkey
        cache = getattr(data, "wavelet_coherence", None)
        if cache is None:
            cache = data.wavelet_coherence = {}
        if key in cache:
            return cache[key]

        z = data.to_numpy(squeeze=True)
        n_proc, n_obs = z.shape
        freqs = _morlet_frequencies(n_obs, self._fs, self._n_cycles)
        if freqs.size == 0:
            raise NotAppropriate(
                f"WaveletCoherence: a series of {n_obs} samples is too short for "
                f"{self._n_cycles}-cycle wavelets.")
        W = _morlet_coefficients(z, freqs, self._fs, self._n_cycles)
        K = self._scale_kernel(freqs)
        if self._coi:
            tau = _efolding_time(freqs, self._fs, self._n_cycles)
            t = np.arange(n_obs)
            weight = ((t[None, :] >= tau[:, None])
                      & (t[None, :] <= n_obs - 1 - tau[:, None])).astype(float)
        else:
            weight = np.ones((freqs.size, n_obs))
        weight /= weight.sum(axis=1, keepdims=True)

        # Bounded memory: smooth a block of series or pairs at a time.
        block = max(1, 2 ** 20 // (freqs.size * n_obs))
        power = np.concatenate([
            self._smooth(np.abs(W[k:k + block]) ** 2, freqs, K).real
            for k in range(0, n_proc, block)])
        rows, cols = np.triu_indices(n_proc, 1)
        r2 = np.full((freqs.size, n_proc, n_proc), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in range(0, rows.size, block):
                i, j = rows[k:k + block], cols[k:k + block]
                cross = self._smooth(W[i] * W[j].conj(), freqs, K)
                local = np.abs(cross) ** 2 / (power[i] * power[j])
                r2[:, i, j] = r2[:, j, i] = np.sum(local * weight, axis=-1).T
        cache[key] = (freqs, r2)
        return cache[key]

    @parse_multivariate
    def multivariate(self, data):
        freqs, r2 = self._coherence_by_frequency(data)
        in_band = (freqs >= self._fmin) & (freqs <= self._fmax)
        if not in_band.any():
            raise empty_band_error(
                "WaveletCoherence", self._fmin, self._fmax,
                f"no frequency of the series' grid (the wavelet grid for "
                f"T = {data.n_observations})")
        adj = np.mean(r2[in_band], axis=0)
        np.fill_diagonal(adj, np.nan)
        return adj


def _integral(x):
    """
    ``x`` as an ``int`` when it is integral, so ``6`` and ``6.0`` name the
    same identifier.
    """
    return int(x) if float(x).is_integer() else float(x)

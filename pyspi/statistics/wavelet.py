import numpy as np
from mne_connectivity import phase_slope_index

from pyspi.base import Undirected, Unsigned, parse_multivariate
from pyspi.utils import fmt_param


class mne(Unsigned):
    """
    Base class of the Morlet-wavelet SPIs computed with mne-connectivity.

    Holds the sampling frequency, the band ``[fmin, fmax]`` and the statistic
    that reduces the result, and appends them to the identifier.
    """

    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean"):
        if fmax is None:
            fmax = fs / 2

        # fs is passed to mne as sfreq and sets PhaseSlopeIndex's frequency floor.
        self._fs = fs
        self._fmin = fmin
        self._fmax = fmax
        if statistic == "mean":
            self._statfn = np.nanmean
        elif statistic == "max":
            self._statfn = np.nanmax
        else:
            raise NameError(f"Unknown statistic {statistic}")

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
        Lower edge of the band, raised to ``5 fs / T`` (five cycles over the
        record) when below it. Default: 0.
    fmax : float or None, optional
        Upper edge of the band; None is the Nyquist frequency ``fs / 2``.
        Default: None.
    statistic : str, optional
        ``'mean'`` (antisymmetric) or ``'max'`` of ``psi(t)`` over time.
        Default: ``'mean'``.
    """

    name = "Phase slope index (wavelet)"
    labels = ["unsigned", "wavelet"]

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        # Only a statistic that commutes with negation keeps the matrix
        # antisymmetric: mean does, max does not.
        trait = "antisymmetric" if self._statistic == "mean" else "asymmetric"
        self.labels = [l for l in self.labels
                       if l not in ("undirected", "directed", "unsigned")
                       ] + [trait, "signed"]
        # Signed in behaviour too, as in spectral.NonparametricSpectral:
        # `_rmmin` would shift an unsigned SPI and destroy the antisymmetry.
        self.issigned = lambda: True
        self.identifier += f"_{self._statistic}"

    def _get_psi(self, data):
        """PSI over this class's ``[fmin, fmax]`` band, cached per band."""
        # Frequencies below five cycles over the record are not resolved.
        fmin = max(self._fmin, 5.0 * self._fs / data.n_observations)
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

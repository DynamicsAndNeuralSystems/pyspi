from mne_connectivity import phase_slope_index
from pyspi.base import Undirected, Unsigned, parse_multivariate
import numpy as np
from pyspi.utils import fmt_param


class mne(Unsigned):
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
    """Time-averaged instantaneous phase slope from Morlet wavelets.

    Not Nolte et al.'s (2008) phase slope index, which weights each phase
    difference by the coherence; that index is ``statistics.spectral``'s
    ``PhaseSlopeIndex``. What this class computes, through mne-connectivity's
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

    ``mean`` is the time average of ``psi(t)``: positive when ``i`` leads
    ``j`` and zero in expectation for independent series, with a standard
    deviation that falls as ``1 / sqrt(T)``. Neighbouring
    frequencies are ``df = (fmax - fmin) / 9`` apart, so for a delay longer
    than about ``1 / (2 df)`` samples the sign reverses (about 9 samples at
    fs=1 on [0, 0.5]). The value is not proportional to the delay.

    ``max`` is the maximum of ``psi(t)`` over time. It tends to the bound
    whatever the data, so it carries no lead/lag information.
    """

    name = "Phase slope index (wavelet)"
    labels = ["unsigned", "wavelet"]

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        # The per-frequency tensor is antisymmetric, but only a statistic that
        # commutes with negation keeps the matrix antisymmetric. mean does;
        # max does not (max of the negated band is -min, not -max), so the max
        # variants are genuinely asymmetric.
        trait = "antisymmetric" if self._statistic == "mean" else "asymmetric"
        self.labels = [l for l in self.labels
                       if l not in ("undirected", "directed", "unsigned")
                       ] + [trait, "signed"]
        # Signed in behaviour, not only in label: see the same note on
        # statistics/spectral.py's NonparametricSpectral -- `_rmmin` shifts
        # every "unsigned" SPI by its minimum, which destroys antisymmetry.
        self.issigned = lambda: True
        self.identifier += f"_{self._statistic}"

    def _get_psi(self, data):
        """Compute PSI over this class's [fmin, fmax] band.

        mne_connectivity's phase_slope_index integrates cwt_freqs that fall
        inside [fmin, fmax], so each band needs its own call. Cached per band
        on the dataset to share across PSI variants with the same band.
        """
        # Floor fmin at five cycles over the record (5 * fs / T Hz), as in
        # mne._get_cache: lower frequencies are not resolved by the data.
        fmin = max(self._fmin, 5.0 * self._fs / data.n_observations)
        # Key on the resolved fmin, not the requested one, so two bands that
        # collapse onto the same floor share correctly.
        key = (self._fs, fmin, self._fmax)
        try:
            return data.mne_psi[key]
        except (AttributeError, KeyError):
            pass

        z = np.moveaxis(data.to_numpy(), 2, 0)
        cwt_freqs = np.linspace(max(fmin, 1e-6), self._fmax, 10)
        # A Morlet wavelet spans n_cycles/f seconds. At MNE's default of 7
        # cycles the low-frequency wavelets are longer than the data (223
        # samples for T=100), which MNE warns about and which makes those bands
        # meaningless. Shorten the wavelet at low frequencies so it always fits:
        # the usual time-frequency trade, resolution given up to stay estimable.
        # MNE's Morlet length is ~1.59 * n_cycles / f samples. Cap n_cycles so
        # the wavelet never exceeds the signal, with a small margin.
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

        # mne_connectivity returns a *lower-triangular* dense tensor: the upper
        # triangle is zero and must be filled in. PSI is antisymmetric per
        # frequency -- psi[i,j,f] = -psi[j,i,f], the sign being the entire
        # lead/lag content -- so the fill must negate, and it must happen
        # BEFORE the band statistic is applied.
        #
        # Negating after reduction is only valid for a statistic that commutes
        # with negation. mean does; max does not:
        #     max_f psi[i,j,f] = max_f(-psi[j,i,f]) = -min_f psi[j,i,f]
        # which is not -max_f psi[j,i,f].
        #
        # Subtracting the transpose fills both triangles in one step, since the
        # upper triangle is zero: lower keeps psi[i,j,f], upper becomes
        # -psi[j,i,f], diagonal cancels to zero.
        psi = psi - np.swapaxes(psi, 0, 1)

        adj = self._statfn(psi, axis=(2, 3))
        np.fill_diagonal(adj, np.nan)
        return adj

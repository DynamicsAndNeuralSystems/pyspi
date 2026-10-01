from mne_connectivity import spectral_connectivity_epochs, phase_slope_index
from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
import numpy as np
import warnings
from pyspi.utils import fmt_param
from pyspi.statistics.spectral import _circular_nanmean


class mne(Unsigned):
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

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f"Include measure for {self.identifier}")

    def _get_cache(self, data):
        try:
            conn, freq = data.mne[(self.measure, self._fs)]
        except (KeyError, AttributeError):
            z = np.moveaxis(data.to_numpy(), 2, 0)

            cwt_freqs = np.linspace(0.2, 0.5, 125)
            cwt_n_cycles = cwt_freqs / 7.0
            con = spectral_connectivity_epochs(
                data=z,
                method=self.measure,
                mode="cwt_morlet",
                sfreq=self._fs,
                mt_adaptive=True,
                fmin=5 / data.n_observations,
                fmax=self._fs / 2,
                cwt_freqs=cwt_freqs,
                cwt_n_cycles=cwt_n_cycles,
                verbose=False,
            )
            conn = con.get_data(output="dense")
            freq = np.asarray(con.freqs)

            try:
                data.mne[(self.measure, self._fs)] = (conn, freq)
            except AttributeError:
                data.mne = {(self.measure, self._fs): (conn, freq)}

        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        return conn, freq_id

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq_id = self._get_cache(data)
        try:
            adj = self._statfn(adj_freq[..., freq_id, :], axis=(2, 3))
        except np.AxisError:
            adj = self._statfn(adj_freq[..., freq_id], axis=2)
        ui = np.triu_indices(data.n_processes, 1)
        adj[ui] = adj.T[ui]
        np.fill_diagonal(adj, np.nan)
        return adj


class CoherenceMagnitude(mne, Undirected):
    name = "Coherence magnitude (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "cohmag"
        self._measure = "coh"
        super().__init__(**kwargs)


class CoherencePhase(mne, Directed):
    name = "Coherence phase (wavelet)"
    labels = ["signed", "wavelet", "antisymmetric"]

    def __init__(self, **kwargs):
        if kwargs.get("statistic", "mean") != "mean":
            raise ValueError(
                "CoherencePhase supports only statistic='mean'. Wrapped phase "
                "has no branch-cut-independent ordinary maximum; use a "
                "different circular summary with an explicit interpretation."
            )
        self.identifier = "phase"
        self._measure = "cohy"
        super().__init__(**kwargs)
        self.labels = ["signed", "wavelet", "antisymmetric"]

    def issigned(self):
        return True

    @parse_multivariate
    def multivariate(self, data):
        """Reduce the complete signed phase spectrum over the selected band.

        MNE returns one triangle.  Coherency phase obeys
        ``phase(i, j) = -phase(j, i)``, so reconstruct that triangle before
        reducing. Wrapped angles are combined with a circular mean.
        """
        adj_freq, freq_id = self._get_cache(data)
        phase = np.angle(adj_freq).copy()
        ui = np.triu_indices(data.n_processes, 1)
        phase[ui[0], ui[1], ...] = -phase[ui[1], ui[0], ...]
        if phase.ndim == 4:
            adj = _circular_nanmean(phase[..., freq_id, :], axis=(2, 3))
        elif phase.ndim == 3:
            adj = _circular_nanmean(phase[..., freq_id], axis=2)
        else:
            raise ValueError(
                f"Expected a 3D or 4D wavelet connectivity tensor, got "
                f"shape {phase.shape}."
            )
        adj[ui] = -adj.T[ui]
        np.fill_diagonal(adj, np.nan)
        return adj


class ImaginaryCoherence(mne, Undirected):
    name = "Imaginary coherency (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "icoh"
        self._measure = "imcoh"
        super().__init__(**kwargs)


class PhaseLockingValue(mne, Undirected):
    name = "Phase locking value (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "plv"
        self._measure = "plv"
        super().__init__(**kwargs)


class PairwisePhaseConsistency(mne, Undirected):
    name = "Pairwise phase consistency (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "ppc"
        self._measure = "ppc"
        super().__init__(**kwargs)


class PhaseLagIndex(mne, Undirected):
    name = "Phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "pli"
        self._measure = "pli"
        super().__init__(**kwargs)


class DebiasedSquaredWeightedPhaseLagIndex(mne, Undirected):
    name = "Debiased squared phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dspli"
        self._measure = "pli2_unbiased"
        super().__init__(**kwargs)


class weighted_PhaseLagIndex(mne, Undirected):
    name = "Weighted squared phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "wspli"
        self._measure = "wpli"
        super().__init__(**kwargs)


class debiased_weighted_squared_PhaseLagIndex(mne, Undirected):
    name = "Debiased weighted squared phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dwspli"
        self._measure = "wpli2_debiased"
        super().__init__(**kwargs)


class PhaseSlopeIndex(mne, Undirected):
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
        # Key on the resolved fmin (below), not the requested one, so two
        # bands that collapse onto the same floor share correctly.
        key = (self._fs, max(self._fmin, 5.0 / data.n_observations), self._fmax)
        try:
            return data.mne_psi[key]
        except (AttributeError, KeyError):
            pass

        z = np.moveaxis(data.to_numpy(), 2, 0)
        # Resolve fmin against the frequency the data can actually support.
        # fmin=0 asks for an unbounded period: MNE reports an unreliable
        # spectrum and builds an ~11.1-million-sample Morlet wavelet for T=100.
        # The five-cycle floor is the same criterion the sibling wavelet path
        # already applies (see mne._get_cache), so this makes the two
        # consistent rather than inventing a new policy.
        fmin = max(self._fmin, 5.0 / data.n_observations)
        cwt_freqs = np.linspace(max(fmin, 1e-6), self._fmax, 10)
        # A Morlet wavelet spans n_cycles/f seconds. At MNE's default of 7
        # cycles the low-frequency wavelets are longer than the data (223
        # samples for T=100), which MNE warns about and which makes those bands
        # meaningless. Shorten the wavelet at low frequencies so it always fits:
        # the usual time-frequency trade, resolution given up to stay estimable.
        # MNE's Morlet length is ~1.59 * n_cycles / f samples (measured: 159
        # samples at f=0.05, n_cycles=5). Cap n_cycles so the wavelet never
        # exceeds the signal, with a small margin.
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
        # which is not -max_f psi[j,i,f]. Reducing first and negating second
        # made the max variants fail a process-permutation test by up to 11.5.
        #
        # Subtracting the transpose fills both triangles in one step, since the
        # upper triangle is zero: lower keeps psi[i,j,f], upper becomes
        # -psi[j,i,f], diagonal cancels to zero.
        psi = psi - np.swapaxes(psi, 0, 1)

        adj = self._statfn(psi, axis=(2, 3))
        np.fill_diagonal(adj, np.nan)
        return adj

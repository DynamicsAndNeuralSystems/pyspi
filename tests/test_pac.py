"""Phase-amplitude coupling (`pac_<statistic>_p-<f1>-<f2>_a-<f3>-<f4>`,
`pyspi.statistics.crossfreq.PhaseAmplitudeCoupling`).

A[i, j]: phase of process i (row, source) in the phase band against the
amplitude envelope of process j (column, target) in the amplitude band.
"mi" is Tort et al. (2010)'s modulation index, (log N - H(P)) / log N over N
phase bins; "mvl" is Ozkurt & Schnitzler (2011)'s normalised mean vector
length, |sum A exp(i phi)| / (sqrt(n) sqrt(sum A^2)).
"""

import warnings

import numpy as np
import pytest
import yaml
from scipy import signal

from pyspi._parallel import cache_bucket
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.crossfreq import PhaseAmplitudeCoupling, _filter_length

# The two bundled band pairs: (phase_band, amp_band).
BANDS = [((0.01, 0.05), (0.15, 0.4)), ((0.05, 0.1), (0.2, 0.45))]
IDENTIFIERS = [
    "pac_mi_p-0-01-0-05_a-0-15-0-4",
    "pac_mvl_p-0-01-0-05_a-0-15-0-4",
    "pac_mi_p-0-05-0-1_a-0-2-0-45",
    "pac_mvl_p-0-05-0-1_a-0-2-0-45",
]


def _data(arr, zscore=True):
    return Data(data=np.asarray(arr, dtype=float), dim_order="ps", zscore=zscore)


def _pac_pair(seed, T=4000, depth=0.8, noise=0.5, phase_diffusion=0.0):
    """Source: a 0.03 cycles/sample oscillation plus noise. Target: a 0.25
    cycles/sample carrier with amplitude ``1 + depth * cos(source phase)``
    plus noise. ``phase_diffusion`` adds a random walk with that step (rad) to
    the source phase."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phase = (2 * np.pi * 0.03 * t + rng.uniform(0, 2 * np.pi)
             + np.cumsum(phase_diffusion * rng.standard_normal(T)))
    source = np.cos(phase) + noise * rng.standard_normal(T)
    target = ((1 + depth * np.cos(phase)) * np.cos(2 * np.pi * 0.25 * t)
              + noise * rng.standard_normal(T))
    return source, target, rng


# --------------------------------------------------------------------------
# Independent reference
# --------------------------------------------------------------------------

def _reference_series(z, band):
    """Zero-phase order-4 Butterworth band-pass, then the analytic signal."""
    sos = signal.butter(4, [band[0] / 0.5, band[1] / 0.5], btype="bandpass",
                        output="sos")
    return signal.hilbert(signal.sosfiltfilt(sos, z, axis=1), axis=1)


def _reference_mi(phase, amp, n_bins=18):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    which = np.digitize(phase, edges[1:-1])
    mean_amp = np.array([amp[which == k].mean() for k in range(n_bins)])
    p = mean_amp / mean_amp.sum()
    entropy = -np.sum(p * np.log(p))
    return (np.log(n_bins) - entropy) / np.log(n_bins)


def _reference_mvl(phase, amp):
    n = phase.size
    return abs(np.sum(amp * np.exp(1j * phase))) / (np.sqrt(n) * np.sqrt(np.sum(amp ** 2)))


def _reference_fixture():
    src, tgt, rng = _pac_pair(seed=7, T=1500)
    return _data(np.vstack([src, tgt, rng.standard_normal(1500)]))


@pytest.mark.parametrize("phase_band, amp_band", BANDS)
def test_cached_series_are_the_band_passed_analytic_signal(phase_band, amp_band):
    data = _reference_fixture()
    PhaseAmplitudeCoupling(phase_band, amp_band, "mi").multivariate(data)
    z = data.to_numpy(squeeze=True)
    np.testing.assert_allclose(data.crossfreq[(phase_band, "phase")],
                               np.angle(_reference_series(z, phase_band)),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(data.crossfreq[(amp_band, "amp")],
                               np.abs(_reference_series(z, amp_band)),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("statistic", ["mi", "mvl"])
@pytest.mark.parametrize("phase_band, amp_band", BANDS)
def test_matches_reference_through_multivariate_and_bivariate(
        phase_band, amp_band, statistic):
    data = _reference_fixture()
    spi = PhaseAmplitudeCoupling(phase_band, amp_band, statistic)
    A = spi.multivariate(data)

    T = data.n_observations
    edge = spi.edge_samples(T)
    phase = data.crossfreq[(phase_band, "phase")][:, edge:T - edge]
    amp = data.crossfreq[(amp_band, "amp")][:, edge:T - edge]
    reference = _reference_mi if statistic == "mi" else _reference_mvl

    M = data.n_processes
    for i in range(M):
        assert np.isnan(A[i, i])
        for j in range(M):
            if i == j:
                continue
            want = reference(phase[i], amp[j])
            assert A[i, j] == pytest.approx(want, rel=0, abs=1e-12), (i, j)
            assert spi.bivariate(data, i=i, j=j) == pytest.approx(want, rel=0, abs=1e-12)


def test_edge_is_two_filter_lengths_of_the_longer_filter():
    # Samples holding 99% of the one-pass impulse response's energy.
    assert [_filter_length(b) for b in ((0.01, 0.05), (0.05, 0.1),
                                        (0.15, 0.4), (0.2, 0.45))] == [100, 50, 14, 19]
    assert PhaseAmplitudeCoupling(*BANDS[0]).edge == 200
    assert PhaseAmplitudeCoupling(*BANDS[1]).edge == 100
    # Capped at T // 4 on a short record, with a warning.
    spi = PhaseAmplitudeCoupling(*BANDS[0], statistic="mvl")
    assert spi.edge_samples(10_000) == 200
    assert spi.edge_samples(300) == 75
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="only 75 \\(not 200\\) samples"):
        spi.multivariate(_data(rng.standard_normal((2, 300))))


def test_closed_form_on_a_noise_free_pair():
    """A = 1 + m cos(phi), phi uniform: MVL = (m/2) / sqrt(1 + m^2/2), and
    the binned MI is the KL divergence of the bin means of 1 + m cos(phi)."""
    m, n_bins = 0.8, 18
    t = np.arange(4000)
    phase = 2 * np.pi * 0.03 * t + 0.3
    data = _data(np.vstack([np.cos(phase),
                            (1 + m * np.cos(phase)) * np.cos(2 * np.pi * 0.25 * t)]))
    mvl = PhaseAmplitudeCoupling(*BANDS[0], statistic="mvl").bivariate(data, i=0, j=1)
    assert mvl == pytest.approx((m / 2) / np.sqrt(1 + m ** 2 / 2), abs=2e-3)

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_mass = np.diff(edges) + m * np.diff(np.sin(edges))
    p = bin_mass / bin_mass.sum()
    mi_theory = (np.log(n_bins) + np.sum(p * np.log(p))) / np.log(n_bins)
    mi = PhaseAmplitudeCoupling(*BANDS[0], statistic="mi").bivariate(data, i=0, j=1)
    assert mi == pytest.approx(mi_theory, rel=0.02)


# --------------------------------------------------------------------------
# Synthetic coupling: direction, independence, time-shift control
# --------------------------------------------------------------------------

@pytest.mark.parametrize("statistic, ratio", [("mi", 10), ("mvl", 3)])
def test_coupling_is_seen_from_the_phase_source_to_the_amplitude_target(statistic, ratio):
    src, tgt, rng = _pac_pair(seed=0)
    data = _data(np.vstack([src, tgt, rng.standard_normal(src.size)]))
    A = PhaseAmplitudeCoupling(*BANDS[0], statistic=statistic).multivariate(data)
    forward = A[0, 1]
    # Reverse orientation, and the source/target paired with an independent process.
    others = [A[1, 0], A[0, 2], A[2, 1]]
    assert forward > {"mi": 0.02, "mvl": 0.2}[statistic]
    assert forward > ratio * max(others), (forward, others)


def test_time_shifted_target_loses_the_coupling():
    """Shifting the target by several hundred samples destroys the phase
    relation when the source's phase wanders (a random walk with 0.1 rad
    steps decorrelates it over about 200 samples). With a strictly periodic
    source a shift only rotates the preferred phase, which MI and MVL ignore,
    so the control is uninformative there."""
    spi = PhaseAmplitudeCoupling(*BANDS[0], statistic="mi")
    for diffusion, check in ((0.1, lambda r: r < 0.4), (0.0, lambda r: r > 0.8)):
        src, tgt, rng = _pac_pair(seed=3, phase_diffusion=diffusion)
        shift = int(rng.integers(300, 900))
        original = spi.bivariate(_data(np.vstack([src, tgt])), i=0, j=1)
        shifted = spi.bivariate(_data(np.vstack([src, np.roll(tgt, shift)])), i=0, j=1)
        assert check(shifted / original), (diffusion, shift, original, shifted)


@pytest.mark.parametrize("phase_band, amp_band", BANDS)
def test_independent_white_noise_gives_small_values(phase_band, amp_band):
    """Two independent white-noise series, T=4000."""
    for seed in range(5):
        data = _data(np.random.default_rng(1000 + seed).standard_normal((2, 4000)))
        mi = PhaseAmplitudeCoupling(phase_band, amp_band, "mi").multivariate(data)
        mvl = PhaseAmplitudeCoupling(phase_band, amp_band, "mvl").multivariate(data)
        off = ~np.eye(2, dtype=bool)
        assert (mi[off] >= 0).all() and (mi[off] < 0.002).all(), mi
        assert (mvl[off] >= 0).all() and (mvl[off] < 0.15).all(), mvl


@pytest.mark.parametrize("statistic", ["mi", "mvl"])
def test_invariant_to_rescaling_either_process_without_zscoring(statistic):
    src, tgt, _ = _pac_pair(seed=1, T=2000)
    spi = PhaseAmplitudeCoupling(*BANDS[0], statistic=statistic)
    base = spi.bivariate(_data(np.vstack([src, tgt]), zscore=False), i=0, j=1)
    for a, b in ((1, 10), (10, 1), (0.01, 1e3)):
        scaled = spi.bivariate(_data(np.vstack([a * src, b * tgt]), zscore=False), i=0, j=1)
        assert scaled == pytest.approx(base, rel=1e-10), (a, b)


# --------------------------------------------------------------------------
# Construction: the band-validity rule and parameters
# --------------------------------------------------------------------------

def test_amplitude_band_narrower_than_twice_the_phase_frequency_is_refused():
    """Aru et al. (2015): the amplitude band must pass carrier +/- f_phase."""
    with pytest.raises(ValueError, match="less than twice"):
        PhaseAmplitudeCoupling(phase_band=(0.05, 0.1), amp_band=(0.25, 0.4))
    # Exactly twice is accepted.
    PhaseAmplitudeCoupling(phase_band=(0.05, 0.1), amp_band=(0.2, 0.4))
    with pytest.raises(ValueError, match="above the phase band"):
        PhaseAmplitudeCoupling(phase_band=(0.05, 0.1), amp_band=(0.08, 0.45))


@pytest.mark.parametrize("kwargs", [
    dict(phase_band=(0.05, 0.01)),
    dict(phase_band=(0, 0.05)),
    dict(amp_band=(0.15, 0.5)),
    dict(amp_band=(0.15,)),
    dict(amp_band="0.15-0.4"),
    dict(statistic="plv"),
])
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        PhaseAmplitudeCoupling(**kwargs)


def test_identifier_labels_and_cache_bucket():
    spis = [PhaseAmplitudeCoupling(p, a, s) for p, a in BANDS for s in ("mi", "mvl")]
    assert [s.identifier for s in spis] == IDENTIFIERS
    assert PhaseAmplitudeCoupling(n_bins=12).identifier.endswith("_nbins-12")
    assert PhaseAmplitudeCoupling(statistic="mvl", n_bins=12).identifier == IDENTIFIERS[1]
    assert {"directed", "nonlinear", "unsigned"} <= set(spis[0].labels)
    assert not spis[0].issigned()
    # Both statistics of one band pair share a parallel bucket; band pairs do not.
    assert cache_bucket(spis[0]) == cache_bucket(spis[1]) != cache_bucket(spis[2])


def test_shared_cache_matches_a_fresh_data():
    src, tgt, rng = _pac_pair(seed=4, T=1500)
    arr = np.vstack([src, tgt, rng.standard_normal(1500)])
    shared = _data(arr)
    for p, a in BANDS + BANDS:
        for s in ("mi", "mvl"):
            got = PhaseAmplitudeCoupling(p, a, s).multivariate(shared)
            want = PhaseAmplitudeCoupling(p, a, s).multivariate(_data(arr))
            assert np.array_equal(got, want, equal_nan=True)
    assert set(shared.crossfreq) == {((0.01, 0.05), "phase"), ((0.15, 0.4), "amp"),
                                     ((0.05, 0.1), "phase"), ((0.2, 0.45), "amp")}


def test_empty_phase_bin_makes_only_that_source_undefined():
    """A source whose phase leaves a bin empty has no modulation index.

    On the frozen var1_M3_T100 fixture the edge is capped at 25 samples and
    the 50 kept are half a period of the 0.01-0.05 band's lower edge: sources
    0 and 2 leave a bin empty, source 1 does not. Their rows are NaN with one
    warning, the MVL (no bins) is defined everywhere, and `bivariate` raises
    for an undefined pair."""
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "fixtures", "var1_M3_T100.npy")
    data = Data(data=path, dim_order="sp")
    spi = PhaseAmplitudeCoupling(*BANDS[0], statistic="mi")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        A = spi.multivariate(data)
    off = ~np.eye(3, dtype=bool)
    assert np.isnan(A[[0, 2]][off[[0, 2]]]).all()
    assert np.isfinite(A[1][off[1]]).all()
    assert sum("undefined and set to NaN" in str(w.message) for w in caught) == 1
    with pytest.raises(ValueError, match="phase bins are empty"), \
            pytest.warns(UserWarning, match="samples are discarded"):
        spi.bivariate(data, i=0, j=1)
    with pytest.warns(UserWarning, match="samples are discarded"):
        mvl = PhaseAmplitudeCoupling(*BANDS[0], statistic="mvl").multivariate(data)
    assert np.isfinite(mvl[off]).all()


# --------------------------------------------------------------------------
# Calculator round trip and bundled config
# --------------------------------------------------------------------------

def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.crossfreq": {"PhaseAmplitudeCoupling": {
        "labels": ["directed", "nonlinear", "unsigned", "bivariate",
                   "frequency-dependent", "M11"],
        "dependencies": None,
        "configs": [{"phase_band": list(p), "amp_band": list(a), "statistic": s}
                    for p, a in BANDS for s in ("mi", "mvl")],
    }}}
    path = tmp_path / "pac.yaml"
    path.write_text(yaml.safe_dump(cfg))
    src, tgt, rng = _pac_pair(seed=5, T=1500)
    arr = np.vstack([src, tgt, rng.standard_normal(1500)])
    calc = Calculator(dataset=Data(data=arr, dim_order="ps"), config=str(path))
    assert sorted(calc.spis) == sorted(IDENTIFIERS)
    calc.compute()
    assert calc.errors == {}
    for (p, a), s in [(b, s) for b in BANDS for s in ("mi", "mvl")]:
        spi = PhaseAmplitudeCoupling(p, a, s)
        want = spi.multivariate(Data(data=arr, dim_order="ps"))
        got = calc.table[spi.identifier].to_numpy(dtype=float)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)


def test_full_config_constructs_pac_entries():
    spis = Calculator(config="full").spis
    for ident in IDENTIFIERS:
        assert ident in spis
        assert {"directed", "nonlinear", "unsigned", "bivariate",
                "frequency-dependent"} <= set(spis[ident].labels)
    assert not any(k.startswith("pac_") for k in Calculator(config="fast").spis)

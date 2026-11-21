"""Smoothed wavelet coherence (``WaveletCoherence``): definition, bias, time-varying coupling, identifiers."""
import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.wavelet import WaveletCoherence

BANDS = ((0, 0.5), (0, 0.25), (0.25, 0.5))


def _ar1(rng, n, a=0.6):
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + e[t]
    return x


def _data(rows):
    return Data(np.vstack(rows), dim_order="ps")


def _sign_flipping_pair(rng, n=1000):
    """y follows x at lag 1 with weight +0.8 in the first half and -0.8 in the second."""
    x = _ar1(rng, n)
    w = np.where(np.arange(n) < n // 2, 0.8, -0.8)
    y = 0.5 * rng.standard_normal(n)
    y[1:] += w[1:] * x[:-1]
    return x, y


@pytest.mark.parametrize("band", BANDS)
def test_identical_series_have_coherence_one(band):
    x = _ar1(np.random.default_rng(1), 600)
    adj = WaveletCoherence(fmin=band[0], fmax=band[1]).multivariate(_data([x, x, -x]))
    off = ~np.eye(3, dtype=bool)
    assert np.allclose(adj[off], 1.0, atol=1e-12)
    assert np.isnan(adj[np.diag_indices(3)]).all()


def test_values_are_symmetric_in_the_unit_interval_and_permutation_invariant():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((4, 500))
    z[1, 2:] += 0.7 * z[0, :-2]
    adj = WaveletCoherence().multivariate(Data(z, dim_order="ps"))
    off = ~np.eye(4, dtype=bool)
    assert np.allclose(adj, adj.T, equal_nan=True)
    assert np.all((adj[off] > 0) & (adj[off] < 1))
    perm = [3, 1, 0, 2]
    inv = np.argsort(perm)
    permuted = WaveletCoherence().multivariate(Data(z[perm], dim_order="ps"))
    assert np.allclose(permuted[np.ix_(inv, inv)], adj, rtol=1e-12, equal_nan=True)


def test_labels_and_sign_agree_with_the_values():
    spi = WaveletCoherence()
    assert {"undirected", "unsigned"} <= set(spi.labels)
    assert not {"directed", "signed", "antisymmetric"} & set(spi.labels)
    assert not spi.issigned()


def test_independent_series_are_biased_upwards_but_below_coupled_ones():
    """The local estimate pools few samples, so independent series sit well
    above zero, at a level that does not fall with the record length."""
    off = ~np.eye(4, dtype=bool)
    level = {}
    for n in (500, 4000):
        z = np.random.default_rng(3).standard_normal((4, n))
        level[n] = WaveletCoherence().multivariate(Data(z, dim_order="ps"))[off].mean()
    assert 0.1 < level[500] < 0.6
    assert level[4000] == pytest.approx(level[500], abs=0.05)

    rng = np.random.default_rng(4)
    x = _ar1(rng, 1000)
    coupled = WaveletCoherence().multivariate(
        _data([x, np.roll(x, 2) + rng.standard_normal(1000)]))[0, 1]
    assert coupled > level[500] + 0.2


def test_coupling_that_changes_sign_is_detected_where_whole_record_coherence_cancels():
    """Coherence of the whole record averages the +w and -w halves to about
    zero; the local coherence does not."""
    from pyspi.statistics.spectral import CoherenceMagnitude

    rng = np.random.default_rng(5)
    x, y = _sign_flipping_pair(rng)
    d = _data([x, y, _ar1(rng, 1000), _ar1(rng, 1000)])
    wtc = WaveletCoherence().multivariate(d)
    mt = CoherenceMagnitude().multivariate(d)
    null_pairs = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert wtc[0, 1] > max(wtc[p] for p in null_pairs) + 0.1
    assert mt[0, 1] < max(mt[p] for p in null_pairs) + 0.05


@pytest.mark.parametrize("band", BANDS)
def test_wavelet_coherence_is_invariant_to_the_unit_of_frequency(band):
    rng = np.random.default_rng(6)
    x, y = _sign_flipping_pair(rng, 400)
    d = _data([x, y, rng.standard_normal(400)])
    ref = WaveletCoherence(fmin=band[0], fmax=band[1]).multivariate(d)
    fs = 250.0
    got = WaveletCoherence(fs=fs, fmin=band[0] * fs, fmax=band[1] * fs).multivariate(d)
    assert np.allclose(got, ref, rtol=1e-9, equal_nan=True)


def test_shared_cache_gives_the_values_of_a_fresh_dataset():
    """Bands share one cached spectrum; different settings must not."""
    rng = np.random.default_rng(7)
    z = rng.standard_normal((3, 300))
    z[1] += 0.5 * np.roll(z[0], 1)
    makers = [lambda: WaveletCoherence(fmin=0, fmax=0.25),
              lambda: WaveletCoherence(fmin=0.25, fmax=0.5),
              lambda: WaveletCoherence(n_cycles=8),
              lambda: WaveletCoherence(time_smoothing=1),
              lambda: WaveletCoherence(scale_smoothing=0),
              lambda: WaveletCoherence(coi=False),
              lambda: WaveletCoherence()]
    shared = Data(z, dim_order="ps")
    for make in makers + makers[::-1]:
        got = make().multivariate(shared)
        want = make().multivariate(Data(z, dim_order="ps"))
        assert np.array_equal(got, want, equal_nan=True), make().identifier
    values = {make().identifier: make().multivariate(shared)[0, 1] for make in makers}
    assert len(set(np.round(list(values.values()), 12))) == len(makers)


def test_identifier_names_the_band_and_non_default_settings():
    assert WaveletCoherence().identifier == "wtc_wavelet_mean_fs-1_fmin-0_fmax-0-5"
    assert (WaveletCoherence(fmin=0.25, fmax=0.5).identifier
            == "wtc_wavelet_mean_fs-1_fmin-0-25_fmax-0-5")
    assert WaveletCoherence(n_cycles=6.0, time_smoothing=2.0).identifier == \
        WaveletCoherence().identifier
    assert (WaveletCoherence(n_cycles=8, time_smoothing=1, scale_smoothing=0.6,
                             coi=False).identifier
            == "wtc_wavelet_mean_fs-1_fmin-0_fmax-0-5_cycles-8_tsmooth-1"
               "_ssmooth-0-6_coi-False")


@pytest.mark.parametrize("kwargs", [dict(statistic="max"), dict(n_cycles=0),
                                    dict(time_smoothing=-1), dict(scale_smoothing=-0.1),
                                    dict(coi="yes")])
def test_invalid_settings_are_refused(kwargs):
    with pytest.raises((ValueError, TypeError)):
        WaveletCoherence(**kwargs)


def test_short_series_and_empty_bands_are_refused():
    with pytest.raises(ValueError, match="too short"):
        WaveletCoherence().multivariate(Data(np.random.default_rng(8).standard_normal((2, 8)),
                                             dim_order="ps"))
    d = Data(np.random.default_rng(9).standard_normal((2, 100)), dim_order="ps")
    with pytest.raises(ValueError, match="no wavelet frequency"):
        WaveletCoherence(fmin=0, fmax=0.01).multivariate(d)


def test_wavelet_coherence_is_bundled_in_full_only():
    from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config

    for name in bundled_configs():
        wtc = sorted(k for k in load_spis_from_yaml(resolve_config(name), quiet=True)
                     if k.startswith("wtc_"))
        if name == "full":
            assert wtc == ["wtc_wavelet_mean_fs-1_fmin-0-25_fmax-0-5",
                           "wtc_wavelet_mean_fs-1_fmin-0_fmax-0-25",
                           "wtc_wavelet_mean_fs-1_fmin-0_fmax-0-5"]
        else:
            assert not wtc, name

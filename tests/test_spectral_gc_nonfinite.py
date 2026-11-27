"""An infinite in-band spectral GC value counts as missing, like NaN.

nitime's parametric log ratio has no guard against a zero denominator, so a
frequency can be +/-Inf. ``SpectralGrangerCausality.multivariate`` treats it
as NaN: past ``nan_threshold`` (or any, with ``ignore_nan=False``) the cell is
NaN with its reason in ``calc.pair_not_appropriate``; under the threshold the
band statistic is taken over the finite frequencies. Inf is injected into one
direction of one pair of a real fit, so the mirror cell and the other pairs
must keep their values.
"""
import warnings

import numpy as np
import pytest
import yaml

from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics import spectral
from pyspi.statistics.spectral import SpectralGrangerCausality

M, T = 3, 300
KWARGS = dict(method="parametric", order=2)
PAIR = ("proc-0", "proc-1")  # 0 -> 1, by process name


def _data():
    return Data(np.random.default_rng(0).standard_normal((M, T)), dim_order="ps")


def _compute(tmp_path, **extra):
    config = tmp_path / "sgc.yaml"
    config.write_text(yaml.safe_dump({".statistics.spectral": {
        "SpectralGrangerCausality": {"labels": [], "configs": [{**KWARGS, **extra}]}}}))
    calc = Calculator(dataset=_data(), config=str(config))
    (key,) = calc.spis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    return calc, key


def _inject(monkeypatch, value, n_bad):
    """Set the first ``n_bad`` in-band frequencies of 0 -> 1 to ``value``."""
    original = spectral.nt_alg.granger_causality_xy
    band = SpectralGrangerCausality(**KWARGS)._band(
        spectral._parametric_gc_frequencies(1))

    def patched(coef, ecov, n_freqs=spectral._GC_N_FREQS):
        w, f_x_on_y, f_y_on_x, f_xy, S = original(coef, ecov, n_freqs=n_freqs)
        if patched.first:  # the (0, 1) pair comes first; f_x_on_y is 0 -> 1
            patched.first = False
            f_x_on_y = f_x_on_y.copy()
            f_x_on_y[band[:n_bad]] = value
        return w, f_x_on_y, f_y_on_x, f_xy, S

    patched.first = True
    monkeypatch.setattr(spectral.nt_alg, "granger_causality_xy", patched)
    return band


def _baseline(tmp_path, **extra):
    calc, key = _compute(tmp_path, **extra)
    return calc.table[key].to_numpy(dtype=float)


def _off(A):
    return A[~np.eye(M, dtype=bool)]


@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_an_inf_spectrum_past_the_threshold_is_not_appropriate(
        tmp_path, monkeypatch, value, statistic):
    base = _baseline(tmp_path, statistic=statistic)
    band = _inject(monkeypatch, value, n_bad=10**6)  # every in-band frequency
    calc, key = _compute(tmp_path, statistic=statistic)
    got = calc.table[key].to_numpy(dtype=float)

    assert key not in calc.errors and key not in calc.pair_errors
    reasons = {(a, b): msg for a, b, msg in calc.pair_not_appropriate[key]}
    assert set(reasons) == {PAIR}
    assert (f"100% of the {band.size} in-band frequencies of (0, 1) give "
            f"non-finite values") in reasons[PAIR]
    assert np.isnan(got[0, 1])
    keep = ~np.eye(M, dtype=bool)
    keep[0, 1] = False
    assert np.isfinite(got[keep]).all()
    np.testing.assert_array_equal(got[keep], base[keep])  # mirror (1, 0) included


def test_an_inf_under_the_threshold_is_left_out_of_the_statistic(tmp_path, monkeypatch):
    d = _data()
    spi = SpectralGrangerCausality(**KWARGS)
    F, freq = spi._get_cache(d)
    spectrum = F[0, spi._band(freq), 1, 0]  # F[0, f, target, source]: 0 -> 1
    base = _baseline(tmp_path)

    _inject(monkeypatch, np.inf, n_bad=3)
    calc, key = _compute(tmp_path)
    got = calc.table[key].to_numpy(dtype=float)

    assert key not in calc.errors and not calc.pair_not_appropriate.get(key)
    assert np.isfinite(_off(got)).all()
    np.testing.assert_allclose(got[0, 1], np.mean(spectrum[3:]), rtol=1e-12)
    keep = ~np.eye(M, dtype=bool)
    keep[0, 1] = False
    np.testing.assert_array_equal(got[keep], base[keep])


def test_any_inf_with_ignore_nan_false_is_not_appropriate(tmp_path, monkeypatch):
    _inject(monkeypatch, np.inf, n_bad=1)
    calc, key = _compute(tmp_path, ignore_nan=False)
    got = calc.table[key].to_numpy(dtype=float)

    assert key not in calc.errors
    reasons = {(a, b): msg for a, b, msg in calc.pair_not_appropriate[key]}
    assert set(reasons) == {PAIR}
    assert "give non-finite values, and ignore_nan is False" in reasons[PAIR]
    keep = ~np.eye(M, dtype=bool)
    keep[0, 1] = False
    assert np.isnan(got[0, 1]) and np.isfinite(got[keep]).all()

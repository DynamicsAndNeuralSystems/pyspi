"""KSG tie-breaking on series with a large offset.

``_tie_break`` jitters repeated values by ``1e-10 * std``. For repeated values
far from zero relative to their spread (possible only without z-scoring) that
jitter spans few floating-point steps and is not random, so the series is
centred before it is jittered. The KSG estimators standardise each
coordinate, so the offset leaves the estimate within its spread across
jitter seeds. Centred, the data give their reference value, tie-free data
with any offset pass through the tie-break unchanged, and ties that survive
every jitter pass are not appropriate.
"""
import warnings

import numpy as np
import pytest
import yaml

import pyspi.statistics.infotheory as it
from pyspi.base import NotAppropriate
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.quality import Quality

SPIS = {
    "mi": lambda: it.MutualInfo(estimator="kraskov", prop_k=4),
    "te": lambda: it.TransferEntropy(estimator="kraskov", prop_k=4),
    "di": lambda: it.DirectedInfo(estimator="kraskov", prop_k=4, n=2),
}
# bivariate(0, 1) on `_quantised_pair()` with no offset, and on
# `_continuous_pair()` at an offset of 1e8 (reference values).
CENTRED = {"mi": 0.22682209247553153, "te": 0.07211902395082603,
           "di": 0.470721179561981}
TIE_FREE_OFFSET = {"mi": 0.21879575687994723, "te": 0.018973266662138055,
                   "di": 0.4712295897089631}


def _quantised_pair():
    rng = np.random.default_rng(0)
    x = np.round(rng.standard_normal(300) * 2)
    y = np.round(0.7 * x + rng.standard_normal(300) * 2)
    return np.vstack([x, y])


def _continuous_pair():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    return np.vstack([x, 0.7 * x + rng.standard_normal(300)])


def _value(name, X):
    return SPIS[name]().bivariate(Data(X, dim_order="ps", zscore=False), i=0, j=1)


def _seed_spread(name, X, monkeypatch):
    """Range of the estimate over eight other pairs of jitter seeds."""
    values = []
    for s in range(8):
        monkeypatch.setattr(it, "_TIE_SEED_SOURCE", 10 + 2 * s)
        monkeypatch.setattr(it, "_TIE_SEED_TARGET", 11 + 2 * s)
        values.append(_value(name, X))
    monkeypatch.undo()
    return np.ptp(values)


@pytest.mark.parametrize("name", SPIS)
def test_the_centred_series_gives_its_reference_value(name):
    assert _value(name, _quantised_pair()) == pytest.approx(CENTRED[name], rel=1e-9)


@pytest.mark.parametrize("name", SPIS)
def test_an_offset_leaves_the_estimate_within_its_spread_across_jitter_seeds(
        name, monkeypatch):
    """The offsets leave the jitter 3700 floating-point steps (500) down to
    a fraction of one (1e8); centring makes every one give the centred value
    to well within the spread that the choice of jitter seed itself causes."""
    X = _quantised_pair()
    centred = _value(name, X)
    spread = _seed_spread(name, X, monkeypatch)
    assert spread > 0
    for offset in (500.0, 1e3, 1e4, 1e6, 1e8):
        assert abs(_value(name, X + offset) - centred) < 0.5 * spread, offset


@pytest.mark.parametrize("name", SPIS)
def test_tie_free_data_with_a_large_offset_are_untouched(name):
    X = _continuous_pair() + 1e8
    assert it._tie_break(X[0], 0).tobytes() == X[0].tobytes()
    assert _value(name, X) == pytest.approx(TIE_FREE_OFFSET[name], rel=1e-9)


@pytest.mark.parametrize("name", SPIS)
def test_one_accidental_tie_at_a_large_offset_matches_the_centred_series(name):
    X = _continuous_pair()
    X[0, 17] = X[0, 230]
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)
    centred = _value(name, X)
    for offset in (260.0, 1e4, 1e6):
        assert _value(name, X + offset) == pytest.approx(centred, rel=1e-9), offset


def test_the_tie_break_centres_only_series_whose_jitter_would_be_unresolved():
    q = _quantised_pair()[0]                   # standard deviation about 2.1
    # The jitter, 1e-10 * 2.1, is about 3700 floating-point steps between 256
    # and 512 (spacing 5.7e-14), enough: the series keeps its offset.
    out = it._tie_break(q + 500.0, 0)
    assert len(np.unique(out)) == q.size
    assert np.allclose(out, q + 500.0, rtol=0, atol=1e-6)
    # From 512 (spacing 1.1e-13) it is about 1850 steps: the series is centred.
    out = it._tie_break(q + 512.0, 0)
    assert len(np.unique(out)) == q.size
    assert abs(out.mean()) < 1e-6


def test_centred_or_z_scored_quantised_data_keep_their_values():
    """A value repeated twice lies within sqrt((T - 1) / 2) standard deviations
    of the mean, so below T = 100,000 a centred series is never shifted: here a
    two-valued series with one rare level, the extreme case."""
    T = 20000
    y = np.zeros(T)
    y[:2] = 1.0                                # the rare level, repeated twice
    z = (y - y.mean()) / y.std(ddof=1)
    for series in (z, y - y.mean(), 1e6 * z):
        out = it._tie_break(series, 0)
        assert len(np.unique(out)) == T
        assert np.allclose(out, series, rtol=0, atol=1e-6 * series.std())


def test_ties_that_survive_every_jitter_pass_are_not_appropriate(monkeypatch):
    """With the jitter switched off no pass can separate a tie."""
    class NoJitter:
        def standard_normal(self, shape):
            return np.zeros(shape)

    q = _quantised_pair()[0]
    monkeypatch.setattr(it.np.random, "default_rng", lambda seed: NoJitter())
    with pytest.raises(NotAppropriate, match="after 100 jitter passes"):
        it._tie_break(q, 0)


def test_through_the_calculator_an_offset_pair_gives_a_good_value(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({".statistics.infotheory": {
        "MutualInfo": {"labels": ["test"], "configs": [
            {"estimator": "kraskov", "prop_k": 4}]}}}))
    data = Data(_quantised_pair() + 1e8, dim_order="ps", zscore=False)
    calc = Calculator(dataset=data, config=str(config), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    key = "mi_kraskov_NN-4"
    q = calc.quality[key].to_numpy()
    assert (q[~np.eye(2, dtype=bool)] == Quality.GOOD).all()
    assert key not in calc.errors

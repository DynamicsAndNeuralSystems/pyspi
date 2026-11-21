"""KSG ties that the tie-breaking jitter cannot break reliably are not appropriate.

``_tie_break`` jitters repeated values by ``1e-10 * std``. For repeated values
far from zero relative to their spread (possible only without z-scoring) that
jitter spans few floating-point steps, so it is not random: the values move
beyond their spread across jitter seeds, and at larger offsets the ties
survive. Below ``_TIE_JITTER_MIN_STEPS`` steps the KSG estimate is not
appropriate rather than a silently wrong number or ``inf``. Centred, the same
data give their reference value, and tie-free data with any offset pass
through the tie-break unchanged.
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


@pytest.mark.parametrize("name", SPIS)
@pytest.mark.parametrize("offset", [1e3, 1e4, 1e6, 1e8])
def test_ties_at_a_large_offset_are_not_appropriate(name, offset):
    data = Data(_quantised_pair() + offset, dim_order="ps", zscore=False)
    with pytest.raises(NotAppropriate, match="break the ties") as info:
        SPIS[name]().bivariate(data, i=0, j=1)
    assert "Centre or z-score" in str(info.value)


@pytest.mark.parametrize("name", SPIS)
def test_the_centred_series_gives_its_reference_value(name):
    data = Data(_quantised_pair(), dim_order="ps", zscore=False)
    assert SPIS[name]().bivariate(data, i=0, j=1) == pytest.approx(
        CENTRED[name], rel=1e-9)


@pytest.mark.parametrize("name", SPIS)
def test_tie_free_data_with_a_large_offset_are_untouched(name):
    X = _continuous_pair() + 1e8
    assert it._tie_break(X[0], 0).tobytes() == X[0].tobytes()
    data = Data(X, dim_order="ps", zscore=False)
    assert SPIS[name]().bivariate(data, i=0, j=1) == pytest.approx(
        TIE_FREE_OFFSET[name], rel=1e-9)


def test_tie_break_refuses_below_the_minimum_number_of_floating_point_steps():
    q = _quantised_pair()[0]                   # standard deviation about 2.1
    assert len(np.unique(it._tie_break(q, 0))) == q.size
    # The jitter, 1e-10 * 2.1, is about 3700 floating-point steps between 256
    # and 512 (spacing 5.7e-14) and about 1850 from 512 (spacing 1.1e-13).
    assert len(np.unique(it._tie_break(q + 500.0, 0))) == q.size
    with pytest.raises(NotAppropriate, match="offset is too large"):
        it._tie_break(q + 512.0, 0)
    with pytest.raises(NotAppropriate):
        it._tie_break(q + 1e8, 0)


@pytest.mark.parametrize("name", SPIS)
def test_just_inside_the_threshold_the_value_is_the_centred_one_within_the_seed_spread(
        name, monkeypatch):
    """At an offset of 500 (3700 steps) the jitter is coarse but still acts as
    random: the value differs from the centred one by much less than the
    centred value's own spread across jitter seeds."""
    X = _quantised_pair()
    centred = SPIS[name]().bivariate(Data(X, dim_order="ps", zscore=False), i=0, j=1)
    inside = SPIS[name]().bivariate(Data(X + 500.0, dim_order="ps", zscore=False),
                                    i=0, j=1)
    seeds = []
    for s in range(8):
        monkeypatch.setattr(it, "_TIE_SEED_SOURCE", 10 + 2 * s)
        monkeypatch.setattr(it, "_TIE_SEED_TARGET", 11 + 2 * s)
        seeds.append(SPIS[name]().bivariate(
            Data(X, dim_order="ps", zscore=False), i=0, j=1))
    assert abs(inside - centred) < 0.5 * np.ptp(seeds), (inside, centred, seeds)


def test_the_guard_never_fires_on_centred_or_z_scored_quantised_data():
    """A value repeated twice lies within sqrt((T - 1) / 2) standard deviations
    of the mean, so below T = 100,000 a centred series always has enough steps:
    here a two-valued series with one rare level, the extreme case."""
    T = 20000
    y = np.zeros(T)
    y[:2] = 1.0                                # the rare level, repeated twice
    z = (y - y.mean()) / y.std(ddof=1)
    for series in (z, y - y.mean(), 1e6 * z):
        out = it._tie_break(series, 0)
        assert len(np.unique(out)) == T


def test_through_the_calculator_the_cells_are_nan_with_the_reason(tmp_path):
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
    assert (calc.quality[key].to_numpy() == Quality.NAN).all()
    assert calc.errors[key].startswith("NotAppropriate: KSG cannot break the ties")

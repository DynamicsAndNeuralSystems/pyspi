"""KSG ties that the tie-breaking jitter cannot break are not appropriate.

``_tie_break`` jitters repeated values by ``1e-10 * std``. For values far from
zero relative to their spread (possible only without z-scoring) that jitter
is below the floating-point resolution, so the ties survive: the KSG estimate
is then not appropriate rather than a silently wrong number or ``inf``.
Centred, the same data give their reference value, and tie-free data with the
same offset pass through the tie-break unchanged.
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
@pytest.mark.parametrize("offset", [1e6, 1e8])
def test_ties_at_a_large_offset_are_not_appropriate(name, offset):
    data = Data(_quantised_pair() + offset, dim_order="ps", zscore=False)
    with pytest.raises(NotAppropriate, match="could not break the ties") as info:
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


def test_tie_break_raises_only_when_ties_survive():
    q = _quantised_pair()[0]
    assert len(np.unique(it._tie_break(q, 0))) == q.size
    assert len(np.unique(it._tie_break(q + 1e4, 0))) == q.size
    with pytest.raises(NotAppropriate):
        it._tie_break(q + 1e8, 0)


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
    assert calc.errors[key].startswith("NotAppropriate: KSG could not break the ties")

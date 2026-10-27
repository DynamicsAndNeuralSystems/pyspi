"""2.x keyword arguments still work in 3.x, with a DeprecationWarning."""
import warnings

import numpy as np
import pytest

from pyspi import Calculator
from pyspi.calculator import CalculatorFrame, CONFIG_DIR
from pyspi.data import Data

X = np.random.default_rng(0).standard_normal((3, 60))


def _deprecated(match):
    return pytest.warns(DeprecationWarning, match=match)


@pytest.mark.parametrize("old,new", [("all", "full"), ("fast", "fast"),
                                     ("sonnet", "sonnet"), ("fabfour", "fabfour")])
def test_subset_maps_to_config(old, new):
    with _deprecated("subset"):
        old_calc = Calculator(subset=old, verbose=False)
    new_calc = Calculator(config=new, verbose=False)
    assert list(old_calc.spis) == list(new_calc.spis)
    assert old_calc.run_spec["config"] == new_calc.run_spec["config"] == new


def test_subset_positional_slot_is_config():
    # 2.x: Calculator(dataset, name, labels, subset) -- 4th slot still selects SPIs.
    assert len(Calculator(X, None, None, "fabfour", verbose=False).spis) == 4


def test_configfile_maps_to_config_path():
    path = str(CONFIG_DIR / "fabfour.yaml")
    with _deprecated("configfile"):
        old_calc = Calculator(configfile=path, verbose=False)
    assert list(old_calc.spis) == list(Calculator(config=path, verbose=False).spis)


def test_normalise_on_calculator():
    with _deprecated("normalise"):
        calc = Calculator(X, config="fabfour", normalise=False, verbose=False)
    assert calc._zscore is False
    np.testing.assert_array_equal(calc.dataset.to_numpy()[..., 0], X)


def test_normalise_on_data():
    with _deprecated("normalise"):
        off = Data(X, normalise=False)
    assert off.zscore is False
    np.testing.assert_array_equal(off.to_numpy()[..., 0], X)
    with _deprecated("normalise"):
        on = Data(X, normalise=True)
    np.testing.assert_allclose(on.to_numpy(), Data(X, zscore=True).to_numpy())


def test_normalise_on_calculator_frame():
    with _deprecated("subset"):
        with pytest.warns(DeprecationWarning, match="normalise"):
            frame = CalculatorFrame(datasets=[X], names=["a"], labels=[None],
                                    subset="fabfour", normalise=False, verbose=False)
    calc = frame.calculators.iloc[0, 0]
    assert calc._zscore is False and len(calc.spis) == 4


def test_normalise_on_init_from_yaml(tmp_path):
    np.save(tmp_path / "x.npy", X)
    doc = tmp_path / "ds.yaml"
    doc.write_text(f"- file: {tmp_path / 'x.npy'}\n  dim_order: ps\n  name: a\n  labels: []\n")
    frame = CalculatorFrame()
    with _deprecated("normalise"):
        frame.init_from_yaml(str(doc), normalise=False, config="fabfour", verbose=False)
    assert frame.calculators.iloc[0, 0].dataset.zscore is False


def test_warning_points_at_user_code():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Calculator(subset="fabfour", verbose=False)
        Data(X, normalise=True)
    assert [w.filename for w in rec] == [__file__, __file__]


def test_new_api_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        Calculator(X, config="fabfour", zscore=False, verbose=False)
        Data(X, zscore=False)


@pytest.mark.parametrize("kwargs", [
    dict(subset="fast", config="fast"),
    dict(configfile="x.yaml", config="fast"),
    dict(subset="fast", configfile="x.yaml"),
    dict(normalise=True, zscore=True),
])
def test_old_and_new_together_raises(kwargs):
    with pytest.raises(TypeError, match="not both|mutually exclusive"):
        Calculator(verbose=False, **kwargs)


def test_data_old_and_new_together_raises():
    with pytest.raises(TypeError, match="not both"):
        Data(X, zscore=True, normalise=True)


def test_frame_old_and_new_together_raises():
    with pytest.raises(TypeError, match="not both"):
        CalculatorFrame(datasets=[X], names=["a"], labels=[None],
                        zscore=True, normalise=True, config="fabfour")

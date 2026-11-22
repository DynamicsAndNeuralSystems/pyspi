"""Information-theoretic refusals driven by the data are not appropriate.

The KSG estimators refuse a series too short for ``k`` neighbours (with the
Theiler window) and an auto-embedding search with no candidate they can
score; the kernel estimators refuse a Theiler window that leaves no
comparison points; the Kozachenko-Leonenko entropy refuses tied observations.
Each raises ``NotAppropriate``, so through the Calculator the cells are NaN
with ``Quality.NAN`` and the reason is kept. A bad parameter is still an
error, raised at construction.
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


def _compute(X, classes, tmp_path, **data_kwargs):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({".statistics.infotheory": {
        name: {"labels": ["test"], "configs": configs}
        for name, configs in classes.items()}}))
    calc = Calculator(dataset=Data(X, dim_order="ps", **data_kwargs),
                      config=str(config), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    return calc


@pytest.mark.parametrize("make,T,match", [
    (lambda: it.MutualInfo(estimator="kraskov", prop_k=4), 4, "usable neighbour"),
    (lambda: it.TransferEntropy(estimator="kraskov", prop_k=4), 5, "usable neighbour"),
    (lambda: it.DirectedInfo(estimator="kraskov", prop_k=4, n=2), 6, "usable neighbour"),
    (lambda: it.TransferEntropy(estimator="kraskov", auto_embed_method="MAX_CORR_AIS",
                                k_search_max=3, tau_search_max=2), 5, "can be scored"),
    (lambda: it.MutualInfo(estimator="kernel", dyn_corr_excl=5), 10,
     "leaves no comparison points"),
], ids=["mi_ksg", "te_ksg", "di_ksg", "te_ksg_auto_embedding", "mi_kernel_window"])
def test_a_series_too_short_for_the_estimator_is_not_appropriate(make, T, match, tmp_path):
    X = np.random.default_rng(T).standard_normal((3, T))
    data = Data(X, dim_order="ps")
    with pytest.raises(NotAppropriate, match=match):
        make().bivariate(data, i=0, j=1)

    spi = make()
    classes = {type(spi).__name__: [{
        k: v for k, v in {
            "estimator": spi._estimator,
            "prop_k": spi._prop_k if spi._estimator == "kraskov" else None,
            "dyn_corr_excl": spi._dyn_corr_excl,
            "n": getattr(spi, "_n", None) if isinstance(spi, it.DirectedInfo) else None,
            "auto_embed_method": getattr(spi, "_auto_embed_method", None),
            "k_search_max": (spi._k_search_max
                             if getattr(spi, "_auto_embed_method", None) else None),
            "tau_search_max": (spi._tau_search_max
                               if getattr(spi, "_auto_embed_method", None) else None),
        }.items() if v is not None}]}
    calc = _compute(X, classes, tmp_path)
    (key,) = calc.spis
    assert key == spi.identifier
    assert (calc.quality[key].to_numpy() == Quality.NAN).all()
    assert match in calc.not_appropriate[key]
    assert key not in calc.errors


def test_kozachenko_on_one_tied_process_is_nan_for_its_pairs_only(tmp_path):
    """``ce[i, j] = H(x_i, x_j) - H(x_i)``: the rows of the tied process are
    NAN with the reason, and every other cell is GOOD."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 200))
    X[0] = np.round(X[0], 1)
    calc = _compute(X, {"ConditionalEntropy": [{"estimator": "kozachenko"}]}, tmp_path)
    key = "ce_kozachenko"
    q = calc.quality[key].to_numpy()
    expected = np.full((3, 3), Quality.GOOD, dtype=np.uint8)
    np.fill_diagonal(expected, Quality.NAN)
    expected[0, 1] = expected[0, 2] = Quality.NAN
    np.testing.assert_array_equal(q, expected)
    assert key not in calc.errors and key not in calc.pair_errors
    assert key not in calc.not_appropriate
    reasons = calc.pair_not_appropriate[key]
    assert len(reasons) == 2
    assert all(msg.startswith("Kozachenko entropy is undefined "
                              "for tied observations") for _, _, msg in reasons)


def test_kozachenko_calculator_refuses_ties_directly():
    calc = it.KLEntropyCalculator()
    calc.initialise(1)
    calc.setObservations(np.array([0.0, 1.0, 1.0, 2.5]))
    with pytest.raises(NotAppropriate, match="use estimator='gaussian' or 'kernel'"):
        calc.computeAverageLocalOfObservations()


def test_a_bad_parameter_is_still_an_error_at_construction():
    with pytest.raises(ValueError) as info:
        it.MutualInfo(estimator="kraskov", dyn_corr_excl=-1)
    assert not isinstance(info.value, NotAppropriate)
    with pytest.raises(ValueError) as info:
        it._validate_ksg_sample(1, 4, -1)
    assert not isinstance(info.value, NotAppropriate)

"""Refusals driven by the data are not appropriate, in every module.

A series too short for a lag, order, embedding or window, repeated values
where an estimate needs distinct ones, collinear series, or a band with no
power: each raises ``pyspi.base.NotAppropriate`` with a reason naming the
numbers, so through the Calculator the cells are NaN with ``Quality.NAN``
and the reason is kept in ``calc.errors``. A bad parameter is still an
error, raised at construction.
"""
import importlib
import warnings

import numpy as np
import pytest
import yaml

from pyspi.base import NotAppropriate
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.quality import Quality


def _noise(T, M=3, seed=0):
    return np.random.default_rng(seed).standard_normal((M, T))


def _collinear(T):
    x = np.random.default_rng(1).standard_normal(T)
    z = np.random.default_rng(2).standard_normal(T)
    return np.vstack([x, 2.0 * x + 1.0, z])


def _quantised(T):
    return np.random.default_rng(3).integers(0, 2, (3, T)).astype(float)


def _sine_and_noise(T):
    """Process 0 has no power in a band near Nyquist; the others do."""
    t = np.arange(T)
    rng = np.random.default_rng(4)
    return np.vstack([np.sin(2 * np.pi * 0.02 * t),
                      rng.standard_normal(T), rng.standard_normal(T)])


# (module, class, kwargs, data, message fragment)
CASES = {
    "lagcorr_window": (".statistics.basic", "LagResolvedCorrelation",
                       {"statistic": "lag"}, lambda: _noise(3),
                       "lag window"),
    "lagcorr_tau": (".statistics.basic", "LagResolvedCorrelation",
                    {"statistic": "onesided", "tau": 5}, lambda: _noise(12),
                    "outside the lag window"),
    "ccm_E-None": (".statistics.causal", "ConvergentCrossMapping",
                   {"statistic": "mean"}, lambda: _noise(30),
                   "needs N >= 41"),
    "lccm_E-None": (".statistics.causal", "LaggedConvergentCrossMapping",
                    {"statistic": "tp"}, lambda: _noise(30),
                    "needs N >= 41"),
    "ccm_E-10": (".statistics.causal", "ConvergentCrossMapping",
                 {"statistic": "mean", "embedding_dimension": 10}, lambda: _noise(30),
                 "observations are too few"),
    "nli_short": (".statistics.causal", "NonlinearInterdependence",
                  {"statistic": "S"}, lambda: _noise(12),
                  "delay vectors"),
    "liang_short": (".statistics.causal", "LiangInformationFlow",
                    {"k": 5}, lambda: _noise(7), "observations; got 7"),
    "liang_conditional_collinear": (".statistics.causal", "LiangInformationFlow",
                                    {"conditional": True},
                                    lambda: _collinear(100), "nonsingular covariance"),
    "edge_short": (".statistics.misc", "EdgeTimeSeries", {},
                   lambda: _noise(2), "at least 3 observations"),
    "cgc_no_order": (".statistics.var", "ConditionalGrangerCausality", {},
                     lambda: _noise(7), "No feasible VAR order"),
    "cgc_fixed_order": (".statistics.var", "ConditionalGrangerCausality",
                        {"order": 5}, lambda: _noise(20), "is infeasible"),
    "cgc_collinear": (".statistics.var", "ConditionalGrangerCausality", {},
                      lambda: _collinear(100), "residual covariance is singular"),
    "pcohmag_collinear": (".statistics.var", "PartialCoherence", {"order": 1},
                          lambda: _collinear(100), "residual covariance is singular"),
    "pac_empty_bins": (".statistics.crossfreq", "PhaseAmplitudeCoupling",
                       {"statistic": "mi"}, lambda: _noise(40),
                       "phase bins are empty"),
}


def _instance(module, cls, kwargs):
    return getattr(importlib.import_module("pyspi" + module), cls)(**kwargs)


def _calculator(module, cls, kwargs, X, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({module: {cls: {
        "labels": ["test"], "configs": [kwargs]}}}))
    calc = Calculator(dataset=Data(X, dim_order="ps"), config=str(config),
                      verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    return calc


@pytest.mark.parametrize("name", CASES)
def test_a_direct_call_raises_not_appropriate(name):
    module, cls, kwargs, make, match = CASES[name]
    data = Data(make(), dim_order="ps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotAppropriate, match=match) as info:
            _instance(module, cls, kwargs).multivariate(data)
    assert isinstance(info.value, ValueError)


@pytest.mark.parametrize("name", CASES)
def test_through_the_calculator_the_cells_are_nan_with_the_reason(name, tmp_path):
    module, cls, kwargs, make, match = CASES[name]
    calc = _calculator(module, cls, kwargs, make(), tmp_path)
    (key,) = calc.spis
    q = calc.quality[key].to_numpy()
    assert (q == Quality.NAN).all(), q
    assert calc.errors[key].startswith("NotAppropriate: "), calc.errors
    assert match in calc.errors[key]


def test_liang_collinear_pair_is_nan_with_its_reason_and_the_others_good(tmp_path):
    """Bivariate Liang flow on one collinear pair: that pair alone is NaN."""
    calc = _calculator(".statistics.causal", "LiangInformationFlow", {},
                       _collinear(100), tmp_path)
    (key,) = calc.spis
    q = calc.quality[key].to_numpy()
    assert q[0, 1] == q[1, 0] == Quality.NAN
    assert (q[[0, 1, 2, 2], [2, 2, 0, 1]] == Quality.GOOD).all()
    from pyspi.statistics.causal import LiangInformationFlow
    with pytest.raises(NotAppropriate, match="pair \\(0, 1\\) is undefined"):
        LiangInformationFlow().bivariate(Data(_collinear(100), dim_order="ps"),
                                         i=0, j=1)


def test_nli_repeated_delay_vectors_make_only_their_pairs_nan(tmp_path):
    """Two identical quantised processes: each one's neighbourhoods, seen from
    the other, are at distance zero, so S is undefined for that pair only."""
    X = _quantised(200)
    X[1] = X[0]
    calc = _calculator(".statistics.causal", "NonlinearInterdependence",
                       {"statistic": "S"}, X, tmp_path)
    (key,) = calc.spis
    q = calc.quality[key].to_numpy()
    assert q[0, 1] == q[1, 0] == Quality.NAN
    assert key not in calc.errors
    reasons = {(a, b): msg for a, b, msg in calc.pair_errors[key]}
    procs = list(calc.dataset.procnames)
    assert reasons[(procs[0], procs[1])].startswith("NotAppropriate: ")
    assert "repeated delay vectors" in reasons[(procs[0], procs[1])]


def test_pac_target_without_amplitude_in_the_band_is_not_appropriate():
    from pyspi.statistics.crossfreq import PhaseAmplitudeCoupling
    spi = PhaseAmplitudeCoupling(statistic="mvl")
    zero = np.zeros(200)
    with pytest.raises(NotAppropriate, match="no amplitude"):
        spi._mean_vector_length(np.ones(200, dtype=complex), zero)


def test_pac_surrogates_without_spread_are_not_appropriate():
    from pyspi.statistics.crossfreq import PhaseAmplitudeCoupling
    spi = PhaseAmplitudeCoupling(statistic="mi_z")
    with pytest.raises(NotAppropriate, match="identical on every time-shifted"):
        spi._normalised(0.3, np.full(20, 0.5))


def test_a_bad_parameter_is_still_an_ordinary_error():
    from pyspi.statistics.var import ConditionalGrangerCausality
    with pytest.raises(ValueError) as info:
        ConditionalGrangerCausality(order="aic")
    assert not isinstance(info.value, NotAppropriate)

"""Refusals driven by the data are not appropriate, in every module.

A series too short for a lag, order, embedding or window, repeated values
where an estimate needs distinct ones, collinear series, or a band with no
power: each raises ``pyspi.base.NotAppropriate`` with a reason naming the
numbers, so through the Calculator the cells are NaN with ``Quality.NAN``
and the reason is kept in ``calc.not_appropriate``. A bad parameter is still an
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
    assert match in calc.not_appropriate[key], calc.errors
    assert key not in calc.errors


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
    assert key not in calc.errors and key not in calc.pair_errors
    reasons = {(a, b): msg for a, b, msg in calc.pair_not_appropriate[key]}
    procs = list(calc.dataset.procnames)
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


def test_a_onesided_lag_beyond_max_lag_is_an_error_at_construction():
    """tau > max_lag holds at every series length: a bad parameter, not data."""
    from pyspi.statistics.basic import LagResolvedCorrelation
    with pytest.raises(ValueError, match="exceeds max_lag") as info:
        LagResolvedCorrelation(statistic="onesided", tau=5, max_lag=3)
    assert not isinstance(info.value, NotAppropriate)
    LagResolvedCorrelation(statistic="onesided", tau="1e", max_lag=3)
    LagResolvedCorrelation(statistic="onesided", tau=5)


# ---------------------------------------------------------------------------
# Length thresholds: one sample short is refused, the threshold computes
# ---------------------------------------------------------------------------

def _ar(T, M=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((M, T))
    for t in range(1, T):
        X[:, t] += 0.5 * X[:, t - 1]
    return X


def _two_level(T, M=3, seed=0):
    """Each process half -1 and half +1: two bins of the cause, each T/2 deep."""
    rng = np.random.default_rng(seed)
    base = np.array([-1.0] * (T // 2) + [1.0] * (T - T // 2))
    return np.vstack([rng.permutation(base) for _ in range(M)])


def _two_peaks(T, M=3, seed=0):
    """Noise with two separated maxima per process: two events at a high quantile."""
    X = np.random.default_rng(seed).standard_normal((M, T))
    for m in range(M):
        X[m, 3 + m], X[m, T - 4 - m] = 10.0, 11.0
    return X


def _period_18(T, M=3, seed=0):
    """A cycle of 18 samples, in the 0.05-0.1 phase band, plus a faster one."""
    t = np.arange(T)
    rng = np.random.default_rng(seed)
    return np.vstack([np.sin(2 * np.pi * t / 18 + 0.17 * m)
                      + 0.3 * np.sin(2 * np.pi * 0.3 * t)
                      + 0.01 * rng.standard_normal(T) for m in range(M)])


# (module, class, kwargs, data, shortest accepted T, fragment of the refusal at T - 1)
THRESHOLDS = {
    "cds": (".statistics.causal", "ConditionalDistributionSimilarity", {},
            _two_level, 26, "so T >= 26; got T = 25"),
    "ccm_E-1_convergence": (".statistics.causal", "ConvergentCrossMapping",
                            {"statistic": "convergence", "embedding_dimension": 1},
                            _ar, 22, "needs N >= 22; got N = 21"),
    "ccm_E-10_convergence": (".statistics.causal", "ConvergentCrossMapping",
                             {"statistic": "convergence", "embedding_dimension": 10},
                             _ar, 41, "needs N >= 41; got N = 40"),
    "es_strength_q-0-95": (".statistics.events", "EventSynchronisation",
                           {"statistic": "strength", "quantile": 0.95},
                           _two_peaks, 22, "T = 21 samples has at most 1"),
    "es_asym_q-0-9": (".statistics.events", "EventSynchronisation",
                      {"statistic": "asym", "quantile": 0.9},
                      _two_peaks, 12, "so it needs T >= 12"),
    "tailasym_q-0-95": (".statistics.events", "TailDependenceAsymmetry",
                        {"quantile": 0.95}, _ar, 20, "T >= 20; got T = 19"),
    "tailasym_q-0-9": (".statistics.events", "TailDependenceAsymmetry",
                       {"quantile": 0.9}, _ar, 10, "T >= 10; got T = 9"),
    "cce_kozachenko_n-5": (".statistics.infotheory", "CausalEntropy",
                           {"estimator": "kozachenko"}, _ar, 6,
                           "T >= 6; got T = 5"),
    "di_kraskov_NN-4_n-5": (".statistics.infotheory", "DirectedInfo",
                            {"estimator": "kraskov", "prop_k": 4}, _ar, 10,
                            "KSG k=4 exceeds the 3 usable neighbour"),
    "di_kozachenko_n-5": (".statistics.infotheory", "DirectedInfo",
                          {"estimator": "kozachenko"}, _ar, 7,
                          "T >= 7; got T = 6"),
    "xme_kozachenko_k10": (".statistics.infotheory", "CrossmapEntropy",
                           {"estimator": "kozachenko", "history_length": 10},
                           _ar, 12, "T >= 12; got T = 11"),
    "te_symbolic_k-3": (".statistics.infotheory", "TransferEntropy",
                        {"estimator": "symbolic", "k_history": 3}, _ar, 4,
                        "T >= 4 samples"),
    "mgcx_maxlag-10": (".statistics.distance", "CrossMultiscaleGraphCorrelation",
                       {"max_lag": 10}, _ar, 15, "T >= 15; got T = 14"),
    "dcorrx_maxlag-10": (".statistics.distance", "CrossDistanceCorrelation",
                         {"max_lag": 10}, _ar, 14, "T >= 14; got T = 13"),
    "pac_mvl_excess": (".statistics.crossfreq", "PhaseAmplitudeCoupling",
                       {"phase_band": [0.01, 0.05], "amp_band": [0.15, 0.4],
                        "statistic": "mvl_excess"},
                       _ar, 28, "pad 27 samples"),
    "pac_mi_excess": (".statistics.crossfreq", "PhaseAmplitudeCoupling",
                      {"phase_band": [0.05, 0.1], "amp_band": [0.2, 0.45],
                       "statistic": "mi_excess"},
                      _period_18, 34, "T = 33 retains 17"),
    "mgc": (".statistics.distance", "MultiscaleGraphCorrelation", {}, _ar, 5,
            "T >= 5 samples; got T = 4"),
    "xi_tau-1": (".statistics.basic", "ChatterjeeXi", {"tau": 1}, _ar, 3,
                 "T >= 3; got T = 2"),
    "phi_star_t-1": (".statistics.infotheory", "IntegratedInformation",
                     {"phitype": "star"}, _ar, 6, "n = 4 samples for d = 4"),
    "phi_Geo_t-1": (".statistics.infotheory", "IntegratedInformation",
                    {"phitype": "Geo"}, _ar, 6, "n = 4 samples for d = 4"),
    "cohmag_multitaper": (".statistics.spectral", "CoherenceMagnitude",
                          {"statistic": "mean"}, _ar, 5, "T >= 5; got T = 4"),
    "dcoh_multitaper": (".statistics.spectral", "DirectedCoherence",
                        {"statistic": "mean"}, _ar, 5, "T >= 5; got T = 4"),
    "coint_aeg_maxlag-10": (".statistics.misc", "Cointegration",
                            {"method": "aeg", "statistic": "tstat", "autolag": "aic",
                             "maxlag": 10, "trend": "c"},
                            _ar, 22, "T // 2 - 1 = 9 lags"),
}


def _finite_off_diagonal(A):
    return int(np.isfinite(A[~np.eye(A.shape[0], dtype=bool)]).sum())


@pytest.mark.parametrize("name", THRESHOLDS)
def test_one_sample_below_the_threshold_is_refused_with_the_numbers(name):
    module, cls, kwargs, make, T, match = THRESHOLDS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotAppropriate, match=match):
            _instance(module, cls, kwargs).multivariate(
                Data(make(T - 1), dim_order="ps"))


@pytest.mark.parametrize("name", THRESHOLDS)
def test_at_the_threshold_a_value_is_computed(name):
    module, cls, kwargs, make, T, _ = THRESHOLDS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = _instance(module, cls, kwargs).multivariate(Data(make(T), dim_order="ps"))
    assert _finite_off_diagonal(np.asarray(A, float)) > 0


@pytest.mark.parametrize("name", THRESHOLDS)
def test_below_the_threshold_the_calculator_records_the_reason(name, tmp_path):
    module, cls, kwargs, make, T, match = THRESHOLDS[name]
    calc = _calculator(module, cls, kwargs, make(T - 1), tmp_path)
    (key,) = calc.spis
    assert (calc.quality[key].to_numpy() == Quality.NAN).all()
    assert match in calc.not_appropriate[key]
    assert key not in calc.errors and not calc.pair_errors.get(key)


def test_johansen_refuses_fewer_differences_than_lags(tmp_path):
    """k_ar_diff = 10 lags of the T - 1 differences need T >= 12."""
    kwargs = {"method": "johansen", "statistic": "trace_stat", "det_order": 0,
              "k_ar_diff": 10}
    with pytest.raises(NotAppropriate, match="T >= 12; got T = 11"):
        _instance(".statistics.misc", "Cointegration", kwargs).multivariate(
            Data(_ar(11), dim_order="ps"))
    calc = _calculator(".statistics.misc", "Cointegration", kwargs, _ar(11), tmp_path)
    (key,) = calc.spis
    assert "T >= 12" in calc.not_appropriate[key] and key not in calc.errors


@pytest.mark.parametrize("cls, kwargs, needed", [
    ("TailDependenceAsymmetry", {"quantile": 0.95}, 20),
    ("TailDependenceAsymmetry", {"quantile": 0.9}, 10),
    ("TailDependenceAsymmetry", {"quantile": 0.99}, 100),
    ("EventSynchronisation", {"quantile": 0.95}, 22),
    ("EventSynchronisation", {"quantile": 0.9}, 12),
    ("EventSynchronisation", {"quantile": 0.99}, 102),
])
def test_the_stated_threshold_is_the_first_length_accepted(cls, kwargs, needed):
    spi = _instance(".statistics.events", cls, kwargs)
    with pytest.raises(NotAppropriate, match=f"T >= {needed}"):
        spi._require_length(needed - 1)
    spi._require_length(needed)


# ---------------------------------------------------------------------------
# Pair-level refusals: one process is refused, the other pairs keep values
# ---------------------------------------------------------------------------

def _pair_reasons(calc, key):
    procs = list(calc.dataset.procnames)
    return {(procs.index(a), procs.index(b)): msg
            for a, b, msg in calc.pair_not_appropriate.get(key, [])}


def _one_tied_process(T=200):
    """Process 0 is integer-valued with heavy ties; 1 and 2 are continuous."""
    X = _ar(T)
    X[0] = np.random.default_rng(5).integers(0, 3, T)
    return X


@pytest.mark.parametrize("cls, kwargs, match", [
    ("EventSynchronisation", {"statistic": "strength"}, "ties at its 0.95 quantile"),
    ("EventSynchronisation", {"statistic": "asym", "quantile": 0.9}, "ties at its 0.9 quantile"),
    ("TailDependenceAsymmetry", {"quantile": 0.95}, "no sample in its upper tail"),
    ("TailDependenceAsymmetry", {"quantile": 0.9}, "no sample in its upper tail"),
])
def test_ties_refuse_the_pairs_of_the_tied_process(cls, kwargs, match, tmp_path):
    calc = _calculator(".statistics.events", cls, kwargs, _one_tied_process(), tmp_path)
    (key,) = calc.spis
    q = calc.quality[key].to_numpy()
    assert q[1, 2] == q[2, 1] == Quality.GOOD
    reasons = _pair_reasons(calc, key)
    for cell in ((0, 1), (1, 0), (0, 2), (2, 0)):
        assert q[cell] == Quality.NAN and match in reasons[cell], reasons
    assert key not in calc.errors and not calc.pair_errors.get(key)


def test_es_pair_with_a_single_run_is_refused():
    """At T = 22 and quantile 0.95 a series has two exceedances; adjacent, they are one event."""
    from pyspi.statistics.events import EventSynchronisation
    X = _two_peaks(22)
    X[1, 10], X[1, 11] = 20.0, 21.0  # the two largest are adjacent: one run
    with pytest.raises(NotAppropriate, match="process 1 has 1 event.*form 1 run"):
        EventSynchronisation().bivariate(Data(X, dim_order="ps"), i=0, j=1)
    A = EventSynchronisation().multivariate(Data(X, dim_order="ps"))
    assert np.isfinite(A[0, 2]) and np.isnan(A[0, 1]) and np.isnan(A[2, 1])


def test_cds_refuses_a_cause_with_fewer_than_two_deep_bins(tmp_path):
    """Process 0 splits 13/13 into two bins; continuous noise at T = 26 has no bin over 12."""
    X = _ar(26)
    X[0] = _two_level(26, M=1)[0]
    calc = _calculator(".statistics.causal", "ConditionalDistributionSimilarity",
                       {}, X, tmp_path)
    q = calc.quality["cds"].to_numpy()
    assert q[0, 1] == q[0, 2] == Quality.GOOD
    reasons = _pair_reasons(calc, "cds")
    for cell in ((1, 0), (1, 2), (2, 0), (2, 1)):
        assert "has 0 such bin(s)" in reasons[cell], reasons
    assert "cds" not in calc.errors


def _box_twins():
    """Processes 0 and 1 repeat their first 4 samples (joint neighbours); 2 is a spaced ramp."""
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([3.0, 0.0, 2.0, 1.0])
    return np.vstack([np.r_[a, a + 1e-3], np.r_[b, b + 1e-3],
                      np.arange(8.0)])


@pytest.mark.parametrize("cls, kwargs", [
    ("MutualInfo", {"estimator": "kernel", "kernel_width": 0.25}),
    ("MutualInfo", {"estimator": "kernel", "kernel_width": 0.25, "dyn_corr_excl": "AUTO"}),
])
def test_kernel_pairs_with_every_joint_box_empty_are_refused(cls, kwargs, tmp_path):
    calc = _calculator(".statistics.infotheory", cls, kwargs, _box_twins(), tmp_path)
    (key,) = calc.spis
    q = calc.quality[key].to_numpy()
    assert q[0, 1] == q[1, 0] == Quality.GOOD
    reasons = _pair_reasons(calc, key)
    for cell in ((0, 2), (2, 0), (1, 2), (2, 1)):
        assert "every joint count is zero" in reasons[cell], reasons
    assert key not in calc.errors and not calc.pair_errors.get(key)


def test_kernel_te_with_every_joint_box_empty_is_not_appropriate():
    from pyspi.statistics.infotheory import KernelTECalculator
    calc = KernelTECalculator()
    calc.setProperty("DYN_CORR_EXCL", "1")
    calc.setObservations(np.arange(6.0), np.arange(6.0) ** 2)
    with pytest.raises(NotAppropriate, match="every joint count is zero"):
        calc.computeAverageLocalOfObservations()
    calc.setProperty("k_HISTORY", "6")
    with pytest.raises(NotAppropriate, match="needs T > 6"):
        calc.computeAverageLocalOfObservations()


def test_kozachenko_entropy_of_one_observation_is_not_appropriate():
    from pyspi.statistics.infotheory import KLEntropyCalculator
    calc = KLEntropyCalculator()
    calc.initialise(2)
    calc.setObservations(np.ones((1, 2)))
    with pytest.raises(NotAppropriate, match="got N = 1"):
        calc.computeAverageLocalOfObservations()


def test_xi_refuses_the_pairs_of_a_target_constant_after_the_lag(tmp_path):
    """At tau = 1 the target is x_j[1:]; process 2 is tied there."""
    X = _ar(6)
    X[2, 1:] = 1.0
    X[2, 0] = 0.0
    calc = _calculator(".statistics.basic", "ChatterjeeXi", {"tau": 1}, X, tmp_path)
    (key,) = calc.spis
    q = calc.quality[key].to_numpy()
    assert (q[[0, 1, 2, 2], [1, 0, 0, 1]] == Quality.GOOD).all()
    reasons = _pair_reasons(calc, key)
    for cell in ((0, 2), (1, 2)):
        assert "process 2 is constant over its 5 target samples" in reasons[cell]
    assert key not in calc.errors


def test_liang_collinear_pair_is_recorded_per_pair(tmp_path):
    calc = _calculator(".statistics.causal", "LiangInformationFlow", {},
                       _collinear(100), tmp_path)
    (key,) = calc.spis
    reasons = _pair_reasons(calc, key)
    assert "pair (0, 1) is undefined" in reasons[(0, 1)]
    assert "pair (1, 0) is undefined" in reasons[(1, 0)]
    assert key not in calc.errors and not calc.pair_errors.get(key)

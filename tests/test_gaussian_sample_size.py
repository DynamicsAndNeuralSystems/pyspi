"""Gaussian information-theoretic SPIs need more samples than dimensions.

A sample covariance of ``n`` observations has rank at most ``n - 1``, so with
``n`` at or below the dimension ``d`` of the largest covariance block an
estimate needs, the log-determinant is set by the 1e-8 ridge, not by the data.
Each Gaussian estimate is then not appropriate: a direct call raises
``NotAppropriate`` naming ``n`` and ``d``, and through the Calculator the cells
are NaN with ``Quality.NAN`` and the reason in ``calc.errors``. ``n = d + 1``
is accepted, and the values computed there and above are unchanged.
"""
import re
import warnings

import numpy as np
import pytest
import yaml

import pyspi.statistics.infotheory as it
from pyspi.base import NotAppropriate
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.quality import Quality

# (factory, smallest accepted T, (n, d) at T - 1, values of bivariate(0, 1) at
# that T, the next T and T = 60, on `_pair(T)`; the values are those of the
# implementation without the sample-size check, so accepted input is
# unchanged).
CASES = {
    "je": (lambda: it.JointEntropy(estimator="gaussian"), 3, (2, 2),
           (2.5250926667871783, 2.7639908077467323, 2.837824153970789)),
    "ce": (lambda: it.ConditionalEntropy(estimator="gaussian"), 3, (2, 2),
           (1.1061541285825056, 1.3450522695420597, 1.4188856157661163)),
    "mi": (lambda: it.MutualInfo(estimator="gaussian"), 3, (2, 2),
           (0.31278440962216775, 0.07388626866261266, 5.2922438556429974e-05)),
    "tlmi": (lambda: it.TimeLaggedMutualInfo(estimator="gaussian"), 4, (2, 2),
             (0.915239432091216, 1.8233355134862042, 0.07270842229637522)),
    "gc_k1": (lambda: it.TransferEntropy(estimator="gaussian"), 5, (3, 3),
              (2.0238636046947205, 0.25038773922121044, 0.07282030592955398)),
    "gc_k3": (lambda: it.TransferEntropy(estimator="gaussian", k_history=3,
                                         k_tau=2, l_history=2), 12, (6, 6),
              (0.5940935977864721, 1.0747967419770983, 0.06270571549188758)),
    "xme_k1": (lambda: it.CrossmapEntropy(estimator="gaussian", history_length=1),
               4, (2, 2),
               (0.3007149347005471, -0.2610588220845711, 1.352789485474241)),
    "xme_k10": (lambda: it.CrossmapEntropy(estimator="gaussian", history_length=10),
                21, (10, 10),
                (-0.6771041516626211, -1.1791211733143978, 1.274874424939025)),
    "cce_n1": (lambda: it.CausalEntropy(estimator="gaussian", n=1), 3, (2, 2),
               (1.1061541285825056, 1.3450522695420597, 1.4188856157661163)),
    "cce_n5": (lambda: it.CausalEntropy(estimator="gaussian", n=5), 15, (10, 10),
               (4.316331398317587, 4.266976274739455, 6.752244556866522)),
    "di_n1": (lambda: it.DirectedInfo(estimator="gaussian", n=1), 4, (2, 2),
              (0.036673964506683365, 0.6196473247186527, 2.6598950753076167e-07)),
    "di_n5": (lambda: it.DirectedInfo(estimator="gaussian", n=5), 16, (10, 10),
              (2.267680917805441, 2.848511342687912, 0.33114020647158093)),
    "si_d1": (lambda: it.StochasticInteraction(estimator="gaussian", delay=1), 6,
              (4, 4),
              (0.8038016786159621, 1.0398278470960989, 0.08014274861542114)),
    "si_d2": (lambda: it.StochasticInteraction(estimator="gaussian", delay=2), 7,
              (4, 4),
              (1.1449675617661415, 0.9864052030964578, 0.0015006823357253118)),
}


def _pair(T):
    """A coupled pair of length T, seeded by T."""
    X = np.random.default_rng(T).standard_normal((2, T))
    X[1, 1:] += 0.5 * X[0, :-1]
    return Data(X, dim_order="ps")


@pytest.mark.parametrize("name", CASES)
def test_one_sample_too_few_is_not_appropriate_and_names_the_numbers(name):
    make, t_min, (n, d), _ = CASES[name]
    assert n == d  # one sample short of n = d + 1
    with pytest.raises(NotAppropriate) as info:
        make().bivariate(_pair(t_min - 1), i=0, j=1)
    assert isinstance(info.value, ValueError)
    assert re.search(rf"\bn = {n} samples for d = {d}\b", str(info.value)), info.value


@pytest.mark.parametrize("name", CASES)
def test_values_at_and_above_the_threshold_are_unchanged(name):
    make, t_min, _, expected = CASES[name]
    for T, value in zip((t_min, t_min + 1, 60), expected):
        assert make().bivariate(_pair(T), i=0, j=1) == value, (name, T)


@pytest.mark.parametrize("name", ["je", "mi", "tlmi"])
def test_the_vectorised_paths_refuse_as_the_pairwise_ones_do(name):
    make, t_min, _, _ = CASES[name]
    with pytest.raises(NotAppropriate):
        make().multivariate(_pair(t_min - 1))
    A = make().multivariate(_pair(t_min))
    assert np.isfinite(A[~np.eye(2, dtype=bool)]).all()


def test_the_documented_example_cross_map_entropy_needs_more_than_20_samples():
    spi = it.CrossmapEntropy(estimator="gaussian", history_length=10)
    for T in (12, 15, 20):
        with pytest.raises(NotAppropriate, match="history_length = 10"):
            spi.bivariate(_pair(T), i=0, j=1)
    assert np.isfinite(spi.bivariate(_pair(21), i=0, j=1))


def test_gaussian_auto_embedding_skips_candidates_with_too_few_samples():
    """The AIS of a candidate whose n aligned samples do not exceed 1 + k is
    not scored, as for kraskov, so it cannot be selected."""
    score = it._ais_scorer("gaussian")
    series = np.random.default_rng(0).standard_normal(12)
    # k = 5, delay = 2: n = 12 - 1 - 8 = 3 <= 6.
    assert score(series, 5, 2) == -np.inf
    assert np.isfinite(score(series, 2, 1))
    with pytest.raises(NotAppropriate, match="n = 3 samples for d = 6"):
        it._gaussian_ais(series, 5, 2)


def test_through_the_calculator_the_cells_are_nan_with_the_reason(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({".statistics.infotheory": {
        "CrossmapEntropy": {"labels": ["test"], "configs": [
            {"estimator": "gaussian", "history_length": 10},
            {"estimator": "gaussian", "history_length": 1}]},
        "DirectedInfo": {"labels": ["test"], "configs": [
            {"estimator": "gaussian", "n": 5}]}}}))
    X = np.random.default_rng(1).standard_normal((3, 15))
    calc = Calculator(dataset=X, config=str(config), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    off = ~np.eye(3, dtype=bool)
    for key in ("xme_gaussian_k10", "di_gaussian_n-5"):
        assert (calc.quality[key].to_numpy() == Quality.NAN).all(), key
        assert calc.errors[key].startswith("NotAppropriate: The Gaussian "), key
        assert "samples for d = 10" in calc.errors[key]
    assert (calc.quality["xme_gaussian_k1"].to_numpy()[off] == Quality.GOOD).all()
    assert "xme_gaussian_k1" not in calc.errors


def test_no_bundled_spi_on_the_drift_fixtures_is_refused():
    """T = 100 is far above every bundled embedding's threshold."""
    import yaml as _yaml
    from importlib import resources

    full = _yaml.safe_load(
        resources.files("pyspi").joinpath("configs/full.yaml").read_text())
    X = np.random.default_rng(0).standard_normal((3, 100))
    data = Data(X, dim_order="ps")
    for cls_name, spec in full[".statistics.infotheory"].items():
        if cls_name == "IntegratedInformation":
            continue
        for kwargs in spec.get("configs") or [{}]:
            kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
            if kwargs.get("estimator", "gaussian") != "gaussian":
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                A = getattr(it, cls_name)(**kwargs).multivariate(data)
            assert np.isfinite(A[~np.eye(3, dtype=bool)]).all(), (cls_name, kwargs)

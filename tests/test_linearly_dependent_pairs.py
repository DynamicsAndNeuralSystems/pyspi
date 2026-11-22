"""A pair of linearly dependent series is not appropriate, pair by pair.

Parametric spectral GC fits a VAR that inverts the pair's covariance, the
Wilson-derived SPIs (nonparametric spectral GC, directed coherence, ...)
factorise the pair's spectral matrix, and Johansen's test inverts the
residual covariance of the pair's differences. None exists for a series and
a scaled and shifted copy of it, so that pair is NaN with ``Quality.NAN`` and
its reason in ``calc.pair_not_appropriate``, and every other pair keeps the
value it has without the copy present. The backend's random start for
Wilson's iteration is never reached, so the values do not depend on NumPy's
global random state.
"""
import warnings

import numpy as np
import pytest
import yaml

from pyspi.base import NotAppropriate
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.quality import Quality
from pyspi.statistics.misc import Cointegration
from pyspi.statistics.spectral import DirectedCoherence, SpectralGrangerCausality
from pyspi.utils import DEPENDENT_PAIR_RCOND, dependent_pair_error

# One config of each affected class, as bundled in `full`.
CONFIG = {
    ".statistics.spectral": {
        "SpectralGrangerCausality": {"labels": ["directed"], "configs": [
            {"method": "parametric", "fmin": 0, "fmax": 0.5, "order": 1},
            {"method": "parametric", "fmin": 0, "fmax": 0.25, "order": None},
            {"method": "nonparametric", "fmin": 0, "fmax": 0.5},
        ]},
        "DirectedCoherence": {"labels": ["directed"], "configs": [
            {"fmin": 0, "fmax": 0.5},
        ]},
    },
    ".statistics.misc": {
        "Cointegration": {"labels": ["undirected"], "configs": [
            {"method": "johansen", "statistic": "max_eig_stat",
             "det_order": 0, "k_ar_diff": 1},
            {"method": "johansen", "statistic": "trace_stat",
             "det_order": 1, "k_ar_diff": 10},
        ]},
    },
}


def _ar1(seed, M=3, T=500):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((M, T + 100))
    x = np.zeros((M, T + 100))
    for t in range(1, T + 100):
        x[:, t] = 0.5 * x[:, t - 1] + e[:, t]
    return x[:, 100:]


def _with_copy(kind):
    """Four processes: 1 depends on 0 as `kind` says; 2 and 3 are independent."""
    x = _ar1(0, M=3)
    if kind == "exact":
        copy = x[0].copy()
    elif kind == "affine":
        copy = -2.5 * x[0] + 3.0
    else:  # a copy plus noise of relative size 1e-9
        copy = x[0] + 1e-9 * _ar1(1, M=1)[0]
    return np.vstack([x[0], copy, x[1], x[2]])


def _calculator(X, tmp_path):
    config = tmp_path / "dependent.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    calc = Calculator(dataset=Data(X, dim_order="ps"), config=str(config),
                      verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    return calc


@pytest.mark.parametrize("kind", ["exact", "affine", "near"])
def test_only_the_dependent_pair_is_not_appropriate(kind, tmp_path):
    calc = _calculator(_with_copy(kind), tmp_path)
    procs = list(calc.dataset.procnames)
    assert not calc.errors and not calc.pair_errors
    assert len(calc.spis) == 6
    off = ~np.eye(4, dtype=bool)
    for key in calc.spis:
        q = calc.quality[key].to_numpy()
        assert q[0, 1] == q[1, 0] == Quality.NAN, (key, q)
        good = off.copy()
        good[0, 1] = good[1, 0] = False
        assert (q[good] == Quality.GOOD).all(), (key, q)
        reasons = {(a, b): msg for a, b, msg in calc.pair_not_appropriate[key]}
        assert set(reasons) == {(procs[0], procs[1]), (procs[1], procs[0])}
        for msg in reasons.values():
            assert "are linearly dependent" in msg and "at most 1e-12" in msg


def test_the_other_pairs_keep_the_values_they_have_without_the_copy(tmp_path):
    X = _with_copy("affine")
    with_copy = _calculator(X, tmp_path)
    without = _calculator(X[[0, 2, 3]], tmp_path)
    keep = [0, 2, 3]
    for key in with_copy.spis:
        got = with_copy.table[key].to_numpy()[np.ix_(keep, keep)]
        want = without.table[key].to_numpy()
        np.testing.assert_array_equal(got, want, err_msg=key)


def test_a_strongly_correlated_pair_keeps_its_value(tmp_path):
    """r = 0.99 is far from the 1e-12 criterion: every pair is computed."""
    x = _ar1(2, M=3)
    rng = np.random.default_rng(3)
    x[1] = 0.99 * x[0] + np.sqrt(1 - 0.99 ** 2) * x[0].std() * rng.standard_normal(x.shape[1])
    assert np.corrcoef(x[0], x[1])[0, 1] > 0.98
    calc = _calculator(x, tmp_path)
    assert not calc.errors and not calc.pair_errors
    assert not calc.not_appropriate and not calc.pair_not_appropriate
    off = ~np.eye(3, dtype=bool)
    for key in calc.spis:
        assert (calc.quality[key].to_numpy()[off] == Quality.GOOD).all(), key


@pytest.mark.parametrize("make", [
    lambda: SpectralGrangerCausality(method="nonparametric", fmin=0, fmax=0.5),
    lambda: DirectedCoherence(fmin=0, fmax=0.5),
], ids=["sgc_nonparametric", "dcoh"])
def test_values_do_not_depend_on_the_global_random_state(make):
    """The backend draws a random Wilson start from the global stream when its
    Cholesky start fails, as it does for this nearly identical pair; the pair
    is refused first, so neither the values nor the caller's stream change."""
    X = _with_copy("near")
    results = []
    for seed in (1, 2, 3):
        np.random.seed(seed)
        before = np.random.get_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(make().multivariate(Data(X, dim_order="ps")))
        after = np.random.get_state()
        assert before[0] == after[0] and np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    for A in results[1:]:
        np.testing.assert_array_equal(A, results[0])
    assert np.isnan(results[0][0, 1]) and np.isnan(results[0][1, 0])
    off = ~np.eye(4, dtype=bool)
    off[0, 1] = off[1, 0] = False
    assert np.isfinite(results[0][off]).all()


def test_every_pair_dependent_makes_the_spi_not_appropriate():
    x = _ar1(4, M=1)[0]
    X = np.vstack([x, 2 * x + 1, -x])
    for spi in (SpectralGrangerCausality(method="parametric", order=1),
                SpectralGrangerCausality(),
                DirectedCoherence()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotAppropriate, match="linearly dependent"):
                spi.multivariate(Data(X, dim_order="ps"))


def test_johansen_refuses_the_pair_directly():
    X = _with_copy("exact")
    spi = Cointegration(method="johansen", statistic="trace_stat", det_order=0)
    with pytest.raises(NotAppropriate, match="first differences"):
        spi.bivariate(Data(X, dim_order="ps"), i=0, j=1)
    assert np.isfinite(spi.bivariate(Data(X, dim_order="ps"), i=0, j=2))


@pytest.mark.parametrize("one_minus_r, refused", [
    (0.0, True), (1e-13, True), (1.9e-12, True), (2.1e-12, False),
    (1e-10, False), (0.01, False),
])
def test_the_criterion_is_on_the_correlation(one_minus_r, refused):
    """(1 - |r|) / (1 + |r|) <= 1e-12, whatever the scales of the two series."""
    r = 1.0 - one_minus_r
    for scale in (1.0, 1e-6, 1e6):
        cov = np.array([[1.0, r * scale], [r * scale, scale ** 2]])
        err = dependent_pair_error("X", 0, 1, cov, "Y")
        assert (err is not None) == refused, (one_minus_r, scale)
        if refused:
            assert isinstance(err, NotAppropriate)
    assert DEPENDENT_PAIR_RCOND == 1e-12


def test_a_constant_series_counts_as_dependent():
    err = dependent_pair_error("X", 0, 1, np.array([[1.0, 0.0], [0.0, 0.0]]), "Y")
    assert isinstance(err, NotAppropriate)
    assert dependent_pair_error("X", 0, 1, np.full((2, 2), np.nan), "Y") is None

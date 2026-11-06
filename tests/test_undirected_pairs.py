"""``Undirected.multivariate`` evaluates each unordered pair once.

It visits ``i < j`` only, in the order ``Directed.multivariate`` reaches the
upper triangle, and mirrors each value (and each failure) onto ``(j, i)``.
These tests pin:

* the calls: one per unordered pair, as ``(i, j)`` with ``i < j``, in column
  order -- for a stub and for every undirected SPI in the bundled configs;
* the mirrored values and quality codes, the pair errors and the warning;
* the SPI raising when every pair fails;
* bit-identical values against a reference loop over both orders, including the SPIs
  whose ``bivariate`` caches a result for the opposite orientation.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

import pyspi
from pyspi import _parallel
from pyspi.base import Directed, Undirected, Unsigned, parse_bivariate
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.quality import Quality

GOOD, ERROR, NAN = Quality.GOOD, Quality.ERROR, Quality.NAN
FIXTURES = Path(__file__).parent / "data" / "fixtures"


def _upper_pairs(M):
    return [(i, j) for j in range(M) for i in range(j)]


def _both_orders(spi, data):
    """The algorithm ``Undirected.multivariate`` had: every ordered pair, then
    the upper triangle copied over the lower one."""
    A = Directed.multivariate(spi, data)
    lower = np.tril_indices(data.n_processes, -1)
    A[lower] = A.T[lower]
    return A


class _Stub(Undirected, Unsigned):
    """``bivariate`` records its calls and returns ``values.get((i, j), 10 i + j)``,
    raising when that is an exception."""

    identifier = "stub_undirected"
    labels = ["test"]

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        self.calls.append((i, j))
        v = self.values.get((i, j), float(10 * i + j))
        if isinstance(v, Exception):
            raise v
        return v


def _data(M=4, T=60, seed=0):
    return Data(np.random.default_rng(seed).standard_normal((M, T)),
                dim_order="ps", zscore=False)


def _execute(spi, M):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _parallel.execute_spi(spi, _data(M), "stub", M)
    return res, [str(w.message) for w in caught] + res.warns


# --------------------------------------------------------------------------
# Calls
# --------------------------------------------------------------------------

@pytest.mark.parametrize("M", [1, 2, 3, 6])
def test_each_unordered_pair_is_evaluated_once_in_column_order(M):
    spi = _Stub()
    A = spi.multivariate(_data(M))
    assert spi.calls == _upper_pairs(M)
    assert len(spi.calls) == M * (M - 1) // 2
    for i, j in spi.calls:
        assert A[i, j] == A[j, i] == 10 * i + j
    assert np.isnan(np.diag(A)).all()


def _bundled_undirected_spis():
    """One instance per distinct (class, identifier) of the undirected SPIs in
    every bundled config."""
    seen = {}
    for config in sorted(p.stem for p in
                         (Path(pyspi.__file__).parent / "configs").glob("*.yaml")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calc = Calculator(dataset=_data(3, 20), config=config, verbose=False)
        for key, spi in calc.spis.items():
            if isinstance(spi, Undirected):
                seen.setdefault(key, spi)
    return seen


_BUNDLED = None


def _bundled():
    global _BUNDLED
    if _BUNDLED is None:
        _BUNDLED = _bundled_undirected_spis()
    return _BUNDLED


def test_no_bundled_undirected_spi_evaluates_both_orders_of_a_pair(monkeypatch):
    """Count the ``bivariate`` calls of every undirected SPI in the bundled
    configs. Those that compute their matrix directly make none; the rest
    make one per unordered pair, as (i, j) with i < j."""
    M = 4
    data_z = np.random.default_rng(1).standard_normal((M, 100))
    calls = {}

    def counting(inner):
        def bivariate(self, data, data2=None, i=None, j=None, **kwargs):
            calls.setdefault(id(self), []).append((i, j))
            return inner(self, data, data2, i=i, j=j, **kwargs)
        return bivariate

    # Look every original up before patching any, so a subclass's wrapper
    # wraps the inherited original rather than its parent's wrapper.
    originals = {type(spi): type(spi).bivariate for spi in _bundled().values()}
    for cls, inner in originals.items():
        monkeypatch.setattr(cls, "bivariate", counting(inner))

    pair_looping = 0
    for key, spi in _bundled().items():
        data = Data(data_z.copy(), dim_order="ps", zscore=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                spi.multivariate(data)
            except Exception:
                pass  # an undefined estimate: only the calls matter here
        seen = calls.get(id(spi), [])
        if seen:
            pair_looping += 1
            assert seen == _upper_pairs(M), (key, seen)
    # Guard against the counting silently matching nothing.
    assert pair_looping >= 20, pair_looping


# --------------------------------------------------------------------------
# Values, codes, pair errors and the warning
# --------------------------------------------------------------------------

def test_values_and_codes_are_mirrored():
    """(0, 2) raises and (1, 2) returns NaN. Both are mirrored, so values and
    codes are symmetric, the failure is an ERROR in both cells with its
    message, and the warning counts one failed pair out of three."""
    M = 3
    spi = _Stub({(0, 2): ValueError("bad pair"), (1, 2): np.nan})
    res, msgs = _execute(spi, M)
    assert spi.calls == _upper_pairs(M)

    assert res.err is None
    assert res.values[0, 1] == res.values[1, 0] == 1.0
    assert np.isnan(res.values[[0, 2, 1, 2], [2, 0, 2, 1]]).all()
    assert np.array_equal(res.values, res.values.T, equal_nan=True)

    expected = np.full((M, M), GOOD, dtype=np.uint8)
    np.fill_diagonal(expected, NAN)
    expected[0, 2] = expected[2, 0] = ERROR
    expected[1, 2] = expected[2, 1] = NAN
    assert np.array_equal(res.quality, expected)

    assert res.pair_errors == [(0, 2, "ValueError: bad pair"),
                               (2, 0, "ValueError: bad pair")]
    warned = [m for m in msgs if "undefined and set to NaN" in m]
    assert len(warned) == 1 and "1 of 3 pairs" in warned[0], warned
    assert "first (0, 2)" in warned[0]


def test_failures_are_counted_per_unordered_pair():
    M = 4
    bad = {(0, 1): ValueError("a"), (1, 3): ValueError("b")}
    res, msgs = _execute(_Stub(bad), M)
    warned = [m for m in msgs if "undefined and set to NaN" in m]
    assert len(warned) == 1 and "2 of 6 pairs" in warned[0], warned
    assert sorted(res.pair_errors) == [(0, 1, "ValueError: a"), (1, 0, "ValueError: a"),
                                       (1, 3, "ValueError: b"), (3, 1, "ValueError: b")]
    assert np.array_equal(res.quality, res.quality.T)


def test_an_spi_whose_every_pair_fails_raises():
    M = 3
    spi = _Stub({p: ValueError(f"pair {p}") for p in _upper_pairs(M)})
    res, _ = _execute(spi, M)
    assert spi.calls == _upper_pairs(M)
    assert res.err == "ValueError: pair (0, 1)"
    assert res.pair_errors == []
    off = ~np.eye(M, dtype=bool)
    assert (res.quality[off] == ERROR).all()

    with pytest.raises(ValueError, match=r"pair \(0, 1\)"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _Stub({p: ValueError(f"pair {p}") for p in _upper_pairs(M)}).multivariate(_data(M))


def test_one_good_pair_keeps_the_spi():
    M = 3
    bad = {p: ValueError("x") for p in _upper_pairs(M) if p != (1, 2)}
    res, _ = _execute(_Stub(bad), M)
    assert res.err is None
    assert res.values[1, 2] == res.values[2, 1] == 12.0


# --------------------------------------------------------------------------
# Bit-identical to the old loop
# --------------------------------------------------------------------------

def _fixture(name):
    return Data(data=str(FIXTURES / f"{name}.npy"), dim_order="sp", name=name)


def _instances():
    from pyspi.statistics.basic import CrossCorrelation
    from pyspi.statistics.distance import (
        Barycenter, DistanceCorrelation, HilbertSchmidtIndependenceCriterion,
        LongestCommonSubsequence,
    )
    from pyspi.statistics.infotheory import (
        IntegratedInformation, JointEntropy, MutualInfo, StochasticInteraction,
    )
    return [
        CrossCorrelation(statistic="max", sigonly=True),
        CrossCorrelation(statistic="mean", sigonly=False),
        CrossCorrelation(squared=True, statistic="mean", sigonly=True),
        JointEntropy(estimator="kozachenko"),
        JointEntropy(estimator="kernel", kernel_width=0.5),
        MutualInfo(estimator="kernel", kernel_width=0.5),
        StochasticInteraction(estimator="kernel", kernel_width=0.5),
        Barycenter(mode="euclidean", statistic="max"),
        IntegratedInformation(phitype="star"),
        DistanceCorrelation(),
        HilbertSchmidtIndependenceCriterion(),
        LongestCommonSubsequence(),
    ]


@pytest.mark.parametrize("dataset", ["var1_M3_T100", "cml_M5_T100", "kuramoto_M7_T100"])
@pytest.mark.parametrize("spi", _instances(), ids=lambda s: s.identifier)
def test_values_are_bit_identical_to_the_both_orders_loop(spi, dataset):
    """On a fresh Data each, so a cache cannot hide a difference.
    ``xcorr_*`` and ``je_kozachenko`` cache the result for the opposite
    orientation; before their caches were made orientation-independent,
    evaluating only i < j changed them in the last bit (cml_M5_T100 for
    ``je_kozachenko``, every fixture for ``xcorr_*``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = spi.multivariate(_fixture(dataset))
        old = _both_orders(spi, _fixture(dataset))
    assert np.array_equal(np.asarray(new).view(np.uint64),
                          np.asarray(old).view(np.uint64))


def test_cross_correlation_computes_each_pair_once_and_in_one_orientation(monkeypatch):
    """One FFT per unordered pair, of (larger, smaller), whichever order is asked."""
    from pyspi.statistics import basic
    from pyspi.statistics.basic import CrossCorrelation

    seen = []
    inner = basic.signal.correlate

    def correlate(x, y, *args, **kwargs):
        seen.append((x.copy(), y.copy()))
        return inner(x, y, *args, **kwargs)
    monkeypatch.setattr(basic.signal, "correlate", correlate)

    data = _fixture("cml_M5_T100")
    z = data.to_numpy(squeeze=True)
    CrossCorrelation(statistic="mean", sigonly=False).multivariate(data)
    assert len(seen) == 10
    expected = [(z[j] - z[j].mean(), z[i] - z[i].mean()) for i, j in _upper_pairs(5)]
    for (x, y), (ex, ey) in zip(seen, expected):
        assert np.array_equal(x, ex) and np.array_equal(y, ey)

    # Asking for (1, 0) first gives (0, 1) the same value as asking for it alone.
    spi = CrossCorrelation(statistic="mean", sigonly=False)
    alone = spi.bivariate(_fixture("cml_M5_T100"), i=0, j=1)
    d = _fixture("cml_M5_T100")
    spi.bivariate(d, i=1, j=0)
    assert spi.bivariate(d, i=0, j=1) == alone


def test_joint_entropy_is_computed_in_one_orientation():
    """The cached joint entropy of a pair does not depend on which of (i, j)
    and (j, i) is asked for first."""
    from pyspi.statistics.infotheory import JointEntropy

    spi = JointEntropy(estimator="kozachenko")
    a, b = _fixture("cml_M5_T100"), _fixture("cml_M5_T100")
    for i, j in _upper_pairs(5):
        assert spi.bivariate(a, i=i, j=j) == spi.bivariate(b, i=j, j=i)

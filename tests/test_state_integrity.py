"""Data ownership, cache lifecycle, and process-name consistency.

These began as red tests. Most are now green: ``Data`` copies and freezes its
input, ``to_numpy()`` hands out a read-only view, every statistic cache listed
in ``Data._CACHE_ATTRS`` is dropped when the series change, the builder path
works, process names track add/remove, and ``dim_order``/non-finite inputs are
validated.

Nothing here is red or xfailed any more -- the last outstanding one, positional
``bivariate(data, i, j)`` binding ``i`` to ``data2``, is now rejected at the
decorator.

Do not relax an assertion to make one of these pass. Fix the code.
"""
import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.basic import Covariance


def _mts(seed=0, m=3, t=100):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, t))


# --------------------------------------------------------------------------
# Input ownership and read-only exposure
# --------------------------------------------------------------------------

def test_data_owns_its_input_array():
    """Mutating the caller's array after construction must not change the Data."""
    arr = _mts()
    data = Data(data=arr.copy(), dim_order="ps", zscore=False)
    before = data.to_numpy(squeeze=True).copy()

    # The caller mutates the array they passed in.
    passed = arr
    data2 = Data(data=passed, dim_order="ps", zscore=False)
    snapshot = data2.to_numpy(squeeze=True).copy()
    passed[0, 0] = 1e6

    assert np.allclose(data2.to_numpy(squeeze=True), snapshot), (
        "Data aliases the caller's array; mutating the input changed the dataset."
    )
    assert np.allclose(data.to_numpy(squeeze=True), before)


def test_to_numpy_does_not_expose_mutable_internals():
    """to_numpy() must not hand out a writable view of internal storage."""
    data = Data(data=_mts(), dim_order="ps", zscore=False)
    view = data.to_numpy()
    snapshot = np.array(view, copy=True)

    if view.flags.writeable:
        view[0, 0, 0] = 1e6

    assert np.allclose(data.to_numpy(), snapshot), (
        "Mutating the array returned by to_numpy() altered the Data's internal state."
    )


# --------------------------------------------------------------------------
# Cache invalidation on mutation
# --------------------------------------------------------------------------

def test_set_data_invalidates_caches():
    """Replacing the dataset must invalidate caches computed from the old one."""
    spi = Covariance()
    data = Data(data=_mts(seed=1), dim_order="ps", zscore=False)
    spi.multivariate(data)  # populates the cache on `data`

    fresh_arr = _mts(seed=2)
    data.set_data(fresh_arr, dim_order="ps")
    after_mutation = spi.multivariate(data)

    reference = spi.multivariate(Data(data=fresh_arr, dim_order="ps", zscore=False))
    assert np.allclose(after_mutation, reference, equal_nan=True), (
        "Stale cache survived set_data(): got the previous dataset's statistic."
    )


def test_add_and_remove_process_invalidate_caches():
    data = Data(data=_mts(seed=3), dim_order="ps", zscore=False)
    spi = Covariance()
    spi.multivariate(data)

    data.remove_process(2)
    after = spi.multivariate(data)

    reference = spi.multivariate(
        Data(data=_mts(seed=3)[:2], dim_order="ps", zscore=False)
    )
    assert after.shape == reference.shape, "Cache retained the pre-removal width."
    assert np.allclose(after, reference, equal_nan=True), (
        "Stale cache survived remove_process()."
    )


# --------------------------------------------------------------------------
# Builder path and raw-array API
# --------------------------------------------------------------------------

def test_builder_path_add_process_on_empty_data():
    """Data().add_process(x) is the documented builder entry point."""
    data = Data()
    x = _mts(m=1)[0]
    data.add_process(x)
    data.add_process(_mts(seed=9, m=1)[0])

    assert data.n_processes == 2
    assert data.n_observations == x.size


def test_bivariate_accepts_raw_arrays():
    """The two-array form routes through Data()+add_process(), i.e. the builder.

    Covariance is deliberately not used here: it is multivariate-only and
    raises NotImplementedError from bivariate() regardless of the builder, which
    would make this pass or fail for the wrong reason.
    """
    from pyspi.statistics.basic import SpearmanR

    x, y = _mts(m=2)
    val = SpearmanR().bivariate(x, y)
    assert np.isfinite(val)


def test_bivariate_rejects_indices_passed_positionally():
    """``bivariate(data, 0, 1)`` reads as (data, i, j) but binds data2=0, i=1.

    It used to produce an obscure dimension error from deep inside an estimator
    -- or, worse, no error at all: ``j=None`` reached ``z[None]``, which numpy
    reads as ``np.newaxis``, so the "pair" became the whole (1, M, T) block and
    the SPI computed something unrelated to what was asked for. Now rejected at
    the decorator, with a message that names the signature.
    """
    from pyspi.statistics import infotheory as it

    data = Data(data=_mts(m=2, t=200), dim_order="ps")
    with pytest.raises(ValueError, match="Both i and j must be given"):
        it.MutualInfo(estimator="kraskov").bivariate(data, 0, 1)


# --------------------------------------------------------------------------
# Process-name lifecycle
# --------------------------------------------------------------------------

def test_procnames_track_process_mutations():
    data = Data(data=_mts(), dim_order="ps", zscore=False,
                procnames=["a", "b", "c"])
    assert data.procnames == ["a", "b", "c"]

    data.remove_process(1)
    assert len(data.procnames) == data.n_processes, (
        "procnames desynchronised from n_processes after remove_process()."
    )
    assert data.procnames == ["a", "c"]

    data.add_process(_mts(seed=5, m=1)[0])
    assert len(data.procnames) == data.n_processes


# --------------------------------------------------------------------------
# dim_order validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["xx", "pp", "ss", "zz"])
def test_dim_order_rejects_invalid_symbols(bad):
    with pytest.raises((ValueError, RuntimeError)):
        Data(data=_mts(), dim_order=bad, zscore=False)


def test_non_finite_input_rejected_without_zscore():
    arr = _mts()
    arr[0, 0] = np.inf
    with pytest.raises(ValueError):
        Data(data=arr, dim_order="ps", zscore=False)

"""Parameterised statistic caches must key on every parameter.

These began as red tests and are now green. The spectral cache had two
defects that compounded:

1. ``NonparametricSpectral.key`` omitted ``fs``, so two SPIs differing only in
   sampling frequency collided in the cache.
2. ``_get_cache`` *wrote* the first result under ``self.measure`` (a plain
   string) but *read* under ``self.key`` (a tuple). The first write was
   therefore unreachable, and staleness only surfaced from the third call
   onward, once a tuple-keyed entry finally existed.

Defect 2 is why a naive two-call probe reports no problem. The sequence below
(fs=1 -> 4 -> 1 -> 4) is the minimum that exposes it, and every value is
compared against a freshly-constructed Data rather than against its
predecessor.

Keep these as regression tests: both defects were invisible to the obvious
probe, and the second would return silently wrong numbers if reintroduced.
"""
import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.spectral import CoherenceMagnitude


def _fixture_data():
    rng = np.random.default_rng(0)
    return Data(data=rng.standard_normal((3, 128)), dim_order="ps", zscore=False)


# An explicit band is required for fs to matter: with the default fmax=fs/2 the
# whole spectrum is selected whatever fs is, so both settings legitimately agree
# and the test would be vacuous. With a fixed [0, 0.25] band, fs changes which
# frequency bins fall inside it.
BAND = dict(fmin=0.0, fmax=0.25)


def _spi(fs):
    return CoherenceMagnitude(fs=fs, **BAND)


def _fresh_value(fs):
    """Ground truth: a brand-new Data can never serve a stale cache entry."""
    return _spi(fs).multivariate(_fixture_data())[0, 1]


def test_spectral_cache_distinguishes_sampling_frequency():
    """Alternating fs on one Data must match a fresh Data at every step."""
    data = _fixture_data()
    truth = {1: _fresh_value(1), 4: _fresh_value(4)}

    assert not np.isclose(truth[1], truth[4]), (
        "Test is vacuous: fs=1 and fs=4 give the same value on this fixture."
    )

    observed = []
    for step, fs in enumerate((1, 4, 1, 4), start=1):
        got = _spi(fs).multivariate(data)[0, 1]
        observed.append((step, fs, got, truth[fs]))

    bad = [o for o in observed if not np.isclose(o[2], o[3], equal_nan=True)]
    assert not bad, "Stale cache hits at " + ", ".join(
        f"call {s} (fs={f}): got {g:.6g}, fresh Data gives {t:.6g}" for s, f, g, t in bad
    )


def test_spectral_cache_uses_one_key_type():
    """The written and read cache keys must be the same type."""
    data = _fixture_data()
    # Two calls: the first creates the dict with a *string* key, the second
    # misses on the tuple lookup and inserts a *tuple* key alongside it.
    _spi(1).multivariate(data)
    _spi(1).multivariate(data)

    keys = list(data.spectral_mv.keys())
    stat_keys = [k for k in keys if k != "freq"]
    kinds = {type(k).__name__ for k in stat_keys}

    assert len(kinds) == 1, (
        f"Cache holds mixed key types {kinds} ({stat_keys!r}); the first write is "
        "unreachable by the reader."
    )


def test_cache_key_covers_every_identifier_parameter():
    """Any parameter that changes the identifier must change the cache key."""
    a, b = _spi(1), _spi(4)
    assert a.identifier != b.identifier, "Precondition: fs must reach the identifier."
    assert a.key != b.key, (
        f"fs changes the identifier ({a.identifier!r} vs {b.identifier!r}) but not "
        f"the cache key ({a.key!r} == {b.key!r})."
    )


@pytest.mark.slow
def test_cache_sharing_never_changes_a_value():
    """The automatic coverage check: caching must be an optimisation only.

    A per-SPI check that "different identifier implies different cache key" is
    the wrong invariant -- several classes cache a shared intermediate on
    purpose and apply the differing parameters *after* the lookup
    (`CoherenceMagnitude` caches one connectivity per (measure, fs) and takes
    the band statistic from it; `Cointegration` caches one Johansen fit and
    reads two statistics off it). What must hold is the consequence: computing
    the whole config against one Data, where every cache is shared, must give
    exactly what computing each SPI against its own Data gives.

    That is mechanical, needs no per-class knowledge, and fails precisely when
    a parameter that changes the cached value is missing from the key -- the
    second SPI would be served the first one's intermediate. `dyn_corr_excl`
    and the spectral `fs` were both of that shape.
    """
    import os

    from pyspi.calculator import Calculator, load_spis_from_yaml, resolve_config

    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "var1_M3_T100.npy")

    np.random.seed(42)
    shared = Calculator(dataset=Data(data=fixture, dim_order="sp"))
    shared.compute()

    # Soft-DTW barycentres are fitted by gradient descent seeded from the
    # global RNG, so isolating one moves the RNG position rather than the
    # cache; `tests/tools/measure_reproducibility.py` shows they reproduce
    # exactly when the config is computed twice in the same order.
    RNG_DEPENDENT = {"bary_sgddtw_mean", "bary_sgddtw_max",
                     "bary-sq_sgddtw_mean", "bary-sq_sgddtw_max"}

    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    mismatched = []
    for identifier, spi in spis.items():
        if getattr(type(spi), "_cache_namespace", None) is None:
            continue
        if identifier in RNG_DEPENDENT:
            continue
        np.random.seed(42)
        isolated = np.asarray(spi.multivariate(Data(data=fixture, dim_order="sp")),
                              dtype=float)
        got = shared.table[identifier].to_numpy(dtype=float)
        # Off-diagonal only: `_parallel.run_spi` NaNs the diagonal on the way
        # into the table, and several `multivariate` implementations do not.
        off = ~np.eye(got.shape[0], dtype=bool)
        if not np.allclose(isolated[off], got[off], rtol=1e-9, atol=1e-12,
                           equal_nan=True):
            mismatched.append(
                f"{identifier} (max|diff|="
                f"{np.nanmax(np.abs(isolated[off] - got[off])):.3g})")

    assert not mismatched, (
        "these SPIs differ depending on whether their cache was shared, so a "
        "parameter that changes the cached value is missing from the cache "
        "key:\n  " + "\n  ".join(sorted(mismatched))
    )

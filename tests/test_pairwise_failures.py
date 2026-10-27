"""One undefined pair must not void an SPI's whole table.

Before 3.0.0 an exception from any pair propagated out of ``multivariate``,
and ``_parallel.run_spi`` then recorded the SPI as failed with an all-NaN
table. On a battery of 60 synthetic datasets this hit the five
Kozachenko-based SPIs (a process on a period-6 orbit has tied values) and the
five ``DCE-AUTO`` SPIs (a near-periodic process has an autocorrelation time
longer than the series), wiping out the values of every other pair.
"""
import warnings

import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.infotheory import (
    ConditionalEntropy, MutualInfo, StochasticInteraction, TimeLaggedMutualInfo,
)


def _data_with_a_quantised_process(T=240, M=4):
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((M, T))
    # A period-6 orbit: six distinct values, so every point has a tie and the
    # Kozachenko-Leonenko nearest-neighbour distance is zero.
    Z[0] = np.tile(np.arange(6.0), T // 6)
    return Data(Z, dim_order="ps", zscore=True)


def _expected(spi):
    """Oracle: the pairs whose own `bivariate` call raises are the undefined
    ones; every other pair must hold its bivariate value."""
    M = 4
    undefined, defined = set(), {}
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            try:
                defined[(i, j)] = spi.bivariate(_data_with_a_quantised_process(), i=i, j=j)
            except ValueError:
                undefined.add((i, j))
    assert undefined and defined, (undefined, len(defined))
    return undefined, defined


@pytest.mark.parametrize("spi", [
    ConditionalEntropy(estimator="kozachenko"),   # Directed.multivariate loop
    StochasticInteraction(estimator="kozachenko"),  # the upper-triangle loop
], ids=lambda x: x.identifier)
def test_undefined_pairs_are_nan_and_the_rest_keep_their_values(spi):
    undefined, defined = _expected(spi)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        A = spi.multivariate(_data_with_a_quantised_process())
    for (i, j) in undefined:
        assert np.isnan(A[i, j]), (i, j)
    for (i, j), v in defined.items():
        assert A[i, j] == pytest.approx(v, rel=1e-12), (i, j)
    msgs = [str(w.message) for w in caught if "undefined and set to NaN" in str(w.message)]
    assert len(msgs) == 1, msgs
    assert spi.identifier in msgs[0]


def test_calculator_keeps_the_good_pairs_and_warns():
    """Through the Calculator the SPI is not recorded as failed: the table
    holds the good pairs and the pair failures arrive as a warning."""
    from pyspi.calculator import Calculator
    import tempfile, yaml

    cfg = {".statistics.infotheory": {"ConditionalEntropy": {
        "labels": ["undirected"], "configs": [{"estimator": "kozachenko"}]}}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    calc = Calculator(dataset=_data_with_a_quantised_process(), config=f.name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute()
    assert calc.errors == {}
    A = calc.table["ce_kozachenko"].to_numpy()
    assert np.isnan(A[0, 1:]).all()
    assert np.isfinite(A[1:, 1:][~np.eye(3, dtype=bool)]).all()
    assert any("undefined and set to NaN" in str(w.message) for w in caught)


# --------------------------------------------------------------------------
# The AUTO Theiler window is capped at N // 10 (hctsa's rule)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spi", [
    MutualInfo(estimator="kraskov", dyn_corr_excl="AUTO"),
    TimeLaggedMutualInfo(estimator="kraskov", dyn_corr_excl="AUTO"),
], ids=lambda x: x.identifier)
def test_auto_theiler_window_is_capped_at_a_tenth_of_the_series(spi):
    """Two sinusoids have autocorrelations that never decay, so the product
    window 2 <acf_i, acf_j> is 81 of 240 samples; capped it is 24, and the
    estimate exists. White-noise pairs keep their small window."""
    from pyspi import utils

    T = 240
    t = np.arange(T)
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((4, T))
    Z[0] = np.sin(2 * np.pi * t / 4)
    Z[1] = np.sin(2 * np.pi * t / 4 + 1.0)
    data = Data(Z, dim_order="ps", zscore=True)
    z = data.to_numpy(squeeze=True)
    uncapped = 2 * np.dot(utils.acf(z[0]), utils.acf(z[1]))
    assert uncapped > T // 10
    assert spi._resolve_theiler(data, 0, 1) == T // 10
    small = 2 * np.dot(utils.acf(z[2]), utils.acf(z[3]))
    assert spi._resolve_theiler(data, 2, 3) == int(small) < T // 10
    A = spi.multivariate(data)
    assert np.isfinite(A[~np.eye(4, dtype=bool)]).all()

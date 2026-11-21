"""Deliberately misbehaving SPIs, used to test failure semantics.

Kept as a top-level module (not a package submodule) so a YAML config can name
it directly and a spawned worker can import it, provided ``tests/`` is on
``PYTHONPATH`` -- see ``tests/test_execution_parity.py``, which sets that.

These exist to pin how the two execution paths report failure. They are not
part of the shipped SPI set.
"""
import numpy as np

from pyspi.base import Directed, Undirected, Unsigned, parse_bivariate


class AlwaysRaises(Undirected, Unsigned):
    """Raises a distinctive exception from every pair."""

    name = "AlwaysRaises"
    identifier = "always_raises"
    labels = ["test"]

    def __init__(self, message="deliberate test failure"):
        self._message = message

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        raise RuntimeError(self._message)


class AlwaysWarns(Undirected, Unsigned):
    """Emits a distinctive warning, then returns a finite value."""

    name = "AlwaysWarns"
    identifier = "always_warns"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        import warnings

        warnings.warn("deliberate test warning", UserWarning)
        return 0.5


class WrongShape(Undirected, Unsigned):
    """Returns a matrix of the wrong shape from multivariate()."""

    name = "WrongShape"
    identifier = "wrong_shape"
    labels = ["test"]

    def multivariate(self, data):
        return np.zeros((2, 5))


class NonFinite(Undirected, Unsigned):
    """Returns infinities rather than raising."""

    name = "NonFinite"
    identifier = "non_finite"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return np.inf


# --------------------------------------------------------------------------
# Partly misbehaving SPIs, for the per-cell quality codes
# (tests/test_quality.py, tests/quality_config.yaml). Each returns a matrix
# whose special cells sit at fixed positions, so a test can name them.
# --------------------------------------------------------------------------

def _finite(M):
    """A finite matrix with a distinct value in every cell."""
    return np.arange(M * M, dtype=float).reshape(M, M) + 1.0


class OneInfinite(Directed, Unsigned):
    """Finite everywhere except +inf at (0, 1) and -inf at (1, 0)."""

    name = "OneInfinite"
    identifier = "one_infinite"
    labels = ["test"]

    def multivariate(self, data):
        A = _finite(data.n_processes)
        A[0, 1] = np.inf
        A[1, 0] = -np.inf
        return A


class ComplexOutput(Directed, Unsigned):
    """Complex dtype; the imaginary part is non-zero only at (0, 2)."""

    name = "ComplexOutput"
    identifier = "complex_output"
    labels = ["test"]

    def multivariate(self, data):
        A = _finite(data.n_processes).astype(complex)
        A[0, 2] += 1j
        return A


class AllNaN(Directed, Unsigned):
    """Returns NaN everywhere without raising."""

    name = "AllNaN"
    identifier = "all_nan"
    labels = ["test"]

    def multivariate(self, data):
        return np.full((data.n_processes,) * 2, np.nan)


class OnePairRaises(Directed, Unsigned):
    """``bivariate`` raises for the ordered pair (0, 1) only."""

    name = "OnePairRaises"
    identifier = "one_pair_raises"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if (i, j) == (0, 1):
            raise ValueError("pair (0, 1) is undefined")
        return float(10 * i + j)


class OnePairRaisesUndirected(Undirected, Unsigned):
    """``bivariate`` raises for one ordered pair; ``Undirected`` mirrors i < j."""

    name = "OnePairRaisesUndirected"
    identifier = "one_pair_raises_undirected"
    labels = ["test"]

    def __init__(self, bad=(0, 1)):
        self._bad = tuple(bad)
        self.identifier = f"one_pair_raises_undirected_{self._bad[0]}{self._bad[1]}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if (i, j) == self._bad:
            raise ValueError(f"pair {self._bad} is undefined")
        return float(10 * min(i, j) + max(i, j))

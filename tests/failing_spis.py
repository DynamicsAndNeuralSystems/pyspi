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

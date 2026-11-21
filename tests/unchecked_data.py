"""A ``Data`` that holds an array as given, for tests of an SPI's own input handling."""
import numpy as np

from pyspi.data import Data


def unchecked_data(X):
    """
    A ``Data`` holding ``X`` exactly, with no preprocessing and no input checks.

    ``Data`` refuses a constant process, so a test of how an SPI itself
    treats one (NaN for its pairs, or a raise) builds its input here.
    ``X`` is ``(M, T)`` or ``(M, T, R)``.
    """
    X = np.array(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    return Data._from_prepared_array(np.ascontiguousarray(X))

# Refactored code for base.py for creating a unified public API for the PySPI library.

# Importing the libraries
import os
import pandas as pd
import numpy as np

from skbase import BaseObject
from pyspi.data import Data
import warnings, copy

# Base Class
class BaseSPI(BaseObject):
    """
    Base class for PySPI. This class provides a unified public API for the PySPI library.
    """
    _tags = {
        "capability-multivariate": True,
        "capability-univariate": True,
        "capability-bivariate": True,
        "python_dependencies": "sktime",
        "issigned": True,
        "identifier": "base",
        "name": "BasePySPI",
    }

    # so now using these tags, we dont need to separately handle the parsing of
    # univariate, bivariate and multivariate data
    # better add a deprecation warning here

    # defining the methods
    
    def _spi(self, data: Data, i: int = 0) -> float:
        raise NotImplementedError("Subclass must implement this methos")
    
    def spi(self, data, i=None):
        if not isinstance(data, Data):
            data = Data(data)
        if i is None:
            i = 0
        return self._spi(data, i)

    def _spi_mat(self, data: Data, data2: Data = None, i: int = None, j: int = None) -> np.ndarray:
        raise NotImplementedError("Subclass must implement this method")
    
    def spi_mat(self, data: Data, data2: Data = None, i=None, j=None):
        # logic yet to be implemented
        if not isinstance(data, Data):
            data = Data(data)
        if data2 is not None and not isinstance(data2, Data):
            data2 = Data(data2)
        return self._spi_mat(data, data2, i, j)

    def get_group(self, classes):
        warnings.warn(
            "The 'get_group' method is deprecated. Use skbase's tagging system instead.",
            DeprecationWarning,
        )
        labset = set(self.get_tags()["labels"])
        matches = [set(cls).issubset(labset) for cls in classes]

        if np.count_nonzero(matches) > 1:
            warnings.warn(f"More than one match for classes {classes}")
        elif np.count_nonzero(matches) == 1:
            try:
                idx = np.where(matches)[0][0]
                return idx, ", ".join(classes[idx])
            except (TypeError, IndexError):
                pass
        return None

class SignedSPI(BaseSPI):
    _tags = {
        "capability-signed": True
    }

class DirectedSPI(BaseSPI):
    _tags = {
        "capability-directed": True
    }

    def _spi_mat(self, data, data2 = None, i = None, j = None) -> np.ndarray:
        n_processes1 = data.n_processes
        n_processes2 = data2.n_processes if data2 is not None else n_processes1
        A = np.empty((n_processes1, n_processes2))
        A[:] = np.nan

        for col in range(n_processes2):
            for row in range(n_processes1):
                if row != col:  # Typically directed measures are off-diagonal
                    A[row, col] = self._compute_directed_pair(data, data2, row, col) # Placeholder
        if i is not None:
            A = A[i, :]
        if j is not None:
            A = A[:, j]
        return A
    
    def compute_directed_pair(self, data, data2, i, j):
        # Placeholder for the actual computation
        raise NotImplementedError("_compute_directed_pair must be implemented.")
    
class UndirectedSPI(BaseSPI):
    _tags = {
        "capability:multivariate": True,
    }

    def _spi_mat(self, data, data2 = None, i = None, j = None) -> np.ndarray:
        n_processes1 = data.n_processes
        n_processes2 = data2.n_processes if data2 is not None else n_processes1
        A = np.empty((n_processes1, n_processes2))
        A[:] = np.nan

        for col in range(n_processes2):
            for row in range(n_processes1):
                A[row, col] = self._compute_undirected_pair(data, data2, row, col) # Placeholder

        # Ensure symmetry for undirected measures
        li = np.tril_indices(n_processes1, -1)
        A[li] = A.T[li]
        if i is not None:
            A = A[i, :]
        if j is not None:
            A = A[:, j]
        return A
    
    def compute_undirected_pair(self, data, data2, i, j):
        # Placeholder for the actual computation
        raise NotImplementedError("_compute_undirected_pair must be implemented.")

    



import numpy as np
import copy
import warnings
from skbase import BaseObject
from pyspi.data import Data

# ------------------------------
# Decorators for input parsing
# ------------------------------
def parse_univariate(func):
    def wrapper(self, data, i=None, inplace=True):
        if not isinstance(data, Data):
            data = Data(data=data)
        elif not inplace:
            data = copy.deepcopy(data)
        if i is None:
            i = 0 if data.n_processes == 1 else ValueError("Please specify `i`.")
        return func(self, data, i=i)
    return wrapper

def parse_bivariate(func):
    def wrapper(self, data, data2=None, i=None, j=None, inplace=True):
        if not isinstance(data, Data):
            if data2 is None:
                raise TypeError("Provide either a Data object or two 1D arrays.")
            data = Data()
            data.add_process(data)
            data.add_process(data2)
        elif not inplace:
            data = copy.deepcopy(data)
        if i is None or j is None:
            if data.n_processes == 2:
                i, j = 0, 1
            else:
                raise ValueError("Indices i and j must be provided.")
        return func(self, data, i=i, j=j)
    return wrapper

def parse_multivariate(func):
    def wrapper(self, data, inplace=True):
        if not isinstance(data, Data):
            data = Data()
            for p in data:
                data.add_process(p)
        elif not inplace:
            data = copy.deepcopy(data)
        return func(self, data)
    return wrapper

# ------------------------------
# Base SPI class
# ------------------------------
class BaseSPI(BaseObject):
    _tags = {
        "capability-multivariate": True,
        "capability-bivariate": True,
        "capability-unequal_length": False,
        "python_dependencies": "sktime"
    }

    def _spi(self, data: Data, i: int = 0) -> float:
        raise NotImplementedError("Subclass must implement _spi.")

    @parse_univariate
    def spi(self, data, i=None):
        return self._spi(data, i)

    def _spi_mat(self, data: Data, data2: Data = None, i: int = None, j: int = None) -> np.ndarray:
        raise NotImplementedError("Subclass must implement _spi_mat.")

    def spi_mat(self, data, data2=None, i=None, j=None):
        if not isinstance(data, Data):
            data = Data(data)
        if data2 is not None and not isinstance(data2, Data):
            data2 = Data(data2)
        return self._spi_mat(data, data2, i, j)

# ------------------------------
# Directed / Undirected Interfaces
# ------------------------------
class Directed:
    name = "Bivariate Base"
    identifier = "bivariate_base"
    labels = ['signed']

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        raise NotImplementedError("bivariate method must be implemented.")

    @parse_multivariate
    def multivariate(self, data):
        n = data.n_processes
        A = np.full((n, n), np.nan)
        for j in range(n):
            for i in range(n):
                if i != j:
                    A[i, j] = self.bivariate(data, i=i, j=j)
        return A

class Undirected(Directed):
    name = "Undirected Base"
    identifier = "undirected_base"
    labels = ['unsigned']

    def ispositive(self):
        return False

    @parse_multivariate
    def multivariate(self, data):
        A = super().multivariate(data)
        li = np.tril_indices(data.n_processes, -1)
        A[li] = A.T[li]
        return A

# ------------------------------
# Signed / Unsigned Mixins
# ------------------------------
class Signed:
    def issigned(self): return True

class Unsigned:
    def issigned(self): return False

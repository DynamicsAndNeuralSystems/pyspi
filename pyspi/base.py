import numpy as np
from pyspi.data import Data
import warnings, copy

"""
Some parsing functions for decorating so that we can either input the time series directly or use the data structure
"""
def parse_univariate(function):
    def parsed_function(self,data,i=None,inplace=True):
        if not isinstance(data,Data):
            data1 = data
            data = Data(data=data1)
        elif not inplace:
            # Ensure we don't write over the original
            data = copy.deepcopy(data)

        if i is None:
            if data.n_processes == 1:
                i = 0
            else:
                raise ValueError('Require argument i to be set.')

        return function(self,data,i=i)

    return parsed_function

def parse_bivariate(function):
    def parsed_function(self,data,data2=None,i=None,j=None,inplace=True):
        if not isinstance(data,Data):
            if data2 is None:
                raise TypeError('Input must be either a pyspi.data object or two 1D-array inputs.'
                                    f' Received {type(data)} and {type(data2)}.')
            data1 = data
            data = Data()
            data.add_process(data1)
            data.add_process(data2)
        elif not inplace:
            # Ensure we don't write over the original
            data = copy.deepcopy(data)

        if i is None and j is None:
            if data.n_processes == 2:
                i, j = 0, 1
            else:
                raise ValueError(
                    f'Require arguments i and j to be set: the dataset has '
                    f'{data.n_processes} processes, so there is no default '
                    f'pair. (Use multivariate() for the whole matrix.)'
                )
        elif i is None or j is None:
            # One index alone was a warning-free path straight into
            # `z[None]`, which numpy reads as `np.newaxis`: the "pair" became
            # the whole (1, M, T) block and the SPI computed something with no
            # relation to what was asked for. The signature is
            # `(data, data2, i, j)`, so `bivariate(data, 0, 3)` -- the obvious
            # way to write it -- lands here with i=3 and j unset.
            raise ValueError(
                f'Both i and j must be given (got i={i!r}, j={j!r}). Note the '
                f'signature is bivariate(data, data2=None, i=None, j=None), so '
                f'positional indices must be passed as keywords.'
            )
        for name, idx in (("i", i), ("j", j)):
            if not (isinstance(idx, (int, np.integer)) and not isinstance(idx, bool)):
                raise TypeError(f'{name} must be an integer process index, '
                                f'got {idx!r}.')
            if not 0 <= idx < data.n_processes:
                raise IndexError(
                    f'{name}={idx} is out of range for {data.n_processes} '
                    f'process(es).')

        return function(self,data,i=i,j=j)

    return parsed_function

def parse_multivariate(function):
    def parsed_function(self,data,inplace=True):
        if not isinstance(data,Data):
            # Create a pyspi.Data object from iterable data object
            try:
                procs = data
                data = Data()
                for p in procs:
                    data.add_process(p)
            except IndexError:
                raise TypeError('Data must be either a pyspi.data.Data object or an and iterable of numpy.ndarray''s.')
        elif not inplace:
            # Ensure we don't write over the original
            data = copy.deepcopy(data)

        return function(self,data)

    return parsed_function

def fill_pairwise(identifier, A, pairs, compute, mirror=False):
    """Evaluate ``compute(i, j)`` into ``A[i, j]`` for each pair, isolating failures.

    A pair whose estimate is undefined (tied observations for a
    Kozachenko-Leonenko entropy, a Theiler window longer than the series, a
    singular fit) raises inside ``compute``. Before 3.0.0 that exception left
    the SPI's whole ``M x M`` table NaN, so one quantised or periodic process
    voided every other pair's value. Here the failing pair alone is NaN and
    the failures are reported once, as a warning the Calculator re-emits
    next to the SPI's name. If every pair fails the first exception is
    re-raised, so ``_parallel.run_spi`` records the SPI as failed with the
    estimator's own message, as before.

    ``bivariate`` itself still raises, so a caller asking for one pair sees
    the reason.
    """
    failures = []
    for i, j in pairs:
        try:
            value = compute(i, j)
        except Exception as err:  # the pair, not the table, is undefined
            value = np.nan
            failures.append(((i, j), err))
        A[i, j] = value
        if mirror:
            A[j, i] = value
    if failures and len(failures) == len(pairs):
        raise failures[0][1]
    if failures:
        (i, j), err = failures[0]
        warnings.warn(
            f'SPI "{identifier}": {len(failures)} of {len(pairs)} pairs are '
            f'undefined and set to NaN; first ({i}, {j}): '
            f'{type(err).__name__}: {err}'
        )
    return A


class Directed:
    """ Base class for directed statistics
    """

    name = 'Bivariate base class'
    identifier = 'bivariate_base'
    labels = ['signed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """ Overload method for getting the pairwise dependencies
        """
        raise NotImplementedError("Method not yet overloaded.")

    @parse_multivariate
    def multivariate(self,data):
        """ Compute the dependency statistics for the entire multivariate dataset
        """
        M = data.n_processes
        A = np.full((M, M), np.nan)
        pairs = [(i, j) for j in range(M) for i in range(M) if i != j]
        return fill_pairwise(self.identifier, A, pairs,
                             lambda i, j: self.bivariate(data, i=i, j=j))

    def get_group(self,classes):
        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(set(j_cls)), (f'Class {i_cls} is a subset of class {j_cls}.')

        self._group = None
        self._group_name = None

        labset = set(self.labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        if np.count_nonzero(matches) > 1:
            warnings.warn(f'More than one match for classes {classes}')
        else:
            try:
                id = np.where(matches)[0][0]
                self._group = id
                self._group_name = ', '.join(classes[id])
                return self._group, self._group_name
            except (TypeError,IndexError):
                pass
        return None

class Undirected(Directed):
    """ Base class for directed statistics
    """

    name = 'Base class'
    identifier = 'base'
    labels = ['unsigned']

    def ispositive(self):
        return False

    @parse_multivariate
    def multivariate(self,data):
        A = super(Undirected,self).multivariate(data)

        li = np.tril_indices(data.n_processes,-1)
        A[li] = A.T[li]
        return A

class Signed:
    """ Base class for signed SPIs
    """
    def issigned(self):
        return True

class Unsigned:
    """ Base class for unsigned SPIs
    """
    def issigned(self):
        return False
import numpy as np
from pyspi.data import Data
import warnings, copy
import contextlib
from contextvars import ContextVar

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
            # One index alone would reach `z[None]`, which numpy reads as
            # `np.newaxis`: the "pair" would become the whole (1, M, T) block.
            # The signature is `(data, data2, i, j)`, so `bivariate(data, 0, 3)`
            # -- the obvious way to write it -- lands here with i=3 and j unset.
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

# Pair-level failures of the SPI being computed. `_parallel.execute_spi` opens a
# collector around `SPI.multivariate` (`collect_pair_failures`), and
# `fill_pairwise` appends `(i, j, message)` to it for each cell whose `compute`
# raised, so the Calculator can label those cells `Quality.ERROR` instead of
# leaving them indistinguishable from a NaN the estimator returned. A ContextVar
# rather than a return value, because `multivariate` returns only the matrix
# and its signature is public. It is set per call in whichever process runs the
# SPI, so the serial path and the parallel workers behave the same. Outside a
# collector (an SPI called directly) nothing is recorded.
_PAIR_FAILURES: ContextVar = ContextVar("pyspi_pair_failures", default=None)


@contextlib.contextmanager
def collect_pair_failures():
    """Collect the pair failures that `fill_pairwise` reports inside the block.

    Yields a list that fills with ``(i, j, message)`` tuples, one per matrix
    cell whose value could not be computed (both cells of a mirrored pair).
    """
    failures = []
    token = _PAIR_FAILURES.set(failures)
    try:
        yield failures
    finally:
        _PAIR_FAILURES.reset(token)


def _report_pair_failure(i, j, err):
    failures = _PAIR_FAILURES.get()
    if failures is not None:
        failures.append((int(i), int(j), f"{type(err).__name__}: {err}"))


def fill_pairwise(identifier, A, pairs, compute, mirror=False):
    """Evaluate ``compute(i, j)`` into ``A[i, j]`` for each pair, isolating failures.

    A pair whose estimate is undefined (e.g., tied observations for a
    Kozachenko-Leonenko entropy, or a singular fit) raises inside ``compute``.
    That pair alone is set to NaN, and the failures are reported in one
    warning. If every pair fails, the first exception is re-raised so the SPI
    is recorded as failed with the estimator's own message.

    Each failed cell is also reported to the collector opened by
    :func:`collect_pair_failures`, if one is open, so the Calculator labels it
    ``Quality.ERROR`` and keeps its message in ``Calculator.pair_errors``.

    With ``mirror=True`` each value is also written to ``A[j, i]``, and a
    failure is reported for both cells but counted once, as one pair.
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
    for (i, j), err in failures:
        _report_pair_failure(i, j, err)
        if mirror:
            _report_pair_failure(j, i, err)
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
    """ Base class for undirected statistics
    """

    name = 'Base class'
    identifier = 'base'
    labels = ['unsigned']

    def ispositive(self):
        return False

    @parse_multivariate
    def multivariate(self,data):
        """``bivariate(i, j)`` for each pair ``i < j``, mirrored onto ``(j, i)``.

        Each unordered pair is evaluated once. A failed pair is NaN in both
        cells, with an error reported for both and counted once in the warning;
        the SPI raises if every pair fails.

        The pairs are visited column by column, ``(0, 1), (0, 2), (1, 2),
        (0, 3), ...``: the order in which ``Directed.multivariate`` reaches the
        upper triangle, which is the half this method used to keep after
        evaluating both orders of every pair. An SPI whose ``bivariate`` caches
        a result for the opposite orientation must compute it in an
        orientation that does not depend on which of ``(i, j)`` and ``(j, i)``
        comes first (see ``Barycenter``, ``CrossCorrelation`` and
        ``InfoTheoryBase._compute_joint_entropy``).
        """
        M = data.n_processes
        A = np.full((M, M), np.nan)
        pairs = [(i, j) for j in range(M) for i in range(j)]
        return fill_pairwise(self.identifier, A, pairs,
                             lambda i, j: self.bivariate(data, i=i, j=j),
                             mirror=True)

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

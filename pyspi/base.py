"""Base classes of the SPIs, and decorators that let an SPI take arrays or a ``Data``."""
import contextlib
import copy
from contextvars import ContextVar
import warnings

import numpy as np

from pyspi.data import Data


def parse_univariate(function):
    """Let a univariate method take an array or a ``Data``; ``i`` defaults to 0."""
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
    """
    Let ``bivariate`` take two arrays, or a ``Data`` and the indices ``i``, ``j``.

    With two processes, ``i`` and ``j`` default to 0 and 1; otherwise both must
    be given, as keywords. ``inplace=False`` works on a copy of the ``Data``.
    """
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
            # One index alone would reach `z[None]` (np.newaxis), and the
            # positional call `bivariate(data, 0, 3)` lands here with j unset.
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
    """Let ``multivariate`` take a ``Data`` or an iterable of 1-D arrays."""
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

class NotAppropriate(ValueError):
    """
    The quantity is undefined, or cannot be estimated, for this input.

    Raised by an SPI or estimator when the data cannot support the estimate
    (too few samples for the dimension, ties that cannot be broken), not when
    an argument is invalid. It is not a failure of the computation: the
    Calculator stores the cell, or the SPI, as NaN with ``Quality.NAN`` and
    keeps the message as the reason. A subclass of ``ValueError``.
    """


# Pair-level failures of the SPI being computed: `fill_pairwise` appends
# `(i, j, message)` for each cell whose computation raised, so the Calculator
# can label it (`Quality.ERROR`, or `Quality.NAN` for `NotAppropriate`) and
# keep its message. A ContextVar, not a return value, because the public
# `multivariate` returns only the matrix. Outside a collector (an SPI called
# directly) nothing is recorded.
_PAIR_FAILURES: ContextVar = ContextVar("pyspi_pair_failures", default=None)


@contextlib.contextmanager
def collect_pair_failures():
    """
    Collect the pair failures that `fill_pairwise` reports inside the block.

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
        # Every NotAppropriate, subclasses included, carries the one prefix
        # the Calculator reads as "not appropriate" rather than "failed".
        kind = "NotAppropriate" if isinstance(err, NotAppropriate) else type(err).__name__
        failures.append((int(i), int(j), f"{kind}: {err}"))


def fill_pairwise(identifier, A, pairs, compute, mirror=False):
    """
    Evaluate ``compute(i, j)`` into ``A[i, j]`` for each pair, isolating failures.

    A pair whose estimate is undefined (e.g., tied observations for a
    Kozachenko-Leonenko entropy, or a singular fit) raises inside ``compute``.
    That pair alone is set to NaN, and the failures are reported in one
    warning. If every pair fails, an exception is re-raised so the SPI is
    recorded with the estimator's own message: the first that is not
    :class:`NotAppropriate`, or the first of all when every pair is not
    appropriate.

    Each failed cell is also reported to the collector opened by
    :func:`collect_pair_failures`, if one is open. For a pair that raised
    :class:`NotAppropriate` the Calculator labels the cell ``Quality.NAN`` and
    keeps the reason in ``Calculator.pair_not_appropriate``; for any other
    exception it labels the cell ``Quality.ERROR`` and keeps the message in
    ``Calculator.pair_errors``.

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
        errors = [e for _, e in failures if not isinstance(e, NotAppropriate)]
        raise errors[0] if errors else failures[0][1]
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


# The four symmetry traits a matrix can declare, in the order `_merge_spi_labels`
# resolves them. Exactly one survives on every configured SPI.
DIRECTEDNESS_TRAITS = ("antisymmetric", "asymmetric", "undirected", "directed")
# Traits whose matrix holds one independent value per unordered pair: the lower
# triangle is the upper one copied (undirected) or negated (antisymmetric).
ONE_VALUE_PER_PAIR = ("undirected", "antisymmetric")


class Directed:
    """
    Base class of the SPIs: a directed statistic, ``A[i, j]`` with row = source.

    A subclass defines ``bivariate`` (one pair) or ``multivariate`` (the whole
    ``M x M`` matrix); the default ``multivariate`` evaluates ``bivariate`` for
    every ordered pair through ``fill_pairwise``.
    """

    name = 'Bivariate base class'
    identifier = 'bivariate_base'
    labels = ['signed']

    def directedness(self):
        """
        The one symmetry trait this SPI's matrix ``A`` obeys.

        One of ``"undirected"`` (``A[i, j] == A[j, i]``), ``"antisymmetric"``
        (``A[i, j] == -A[j, i]``), ``"directed"`` or ``"asymmetric"`` (no
        relation between the two orientations). The first two carry one
        independent value per pair, the last two carry two;
        :func:`pyspi.calculator.feature_vector` keys on this.

        Read from the instance's labels, which hold exactly one trait once
        ``Calculator`` has merged them. With no trait declared,
        :class:`Undirected` subclasses are ``"undirected"`` and anything else
        is ``"directed"``.

        Returns
        -------
        str
            ``'undirected'``, ``'antisymmetric'``, ``'directed'`` or
            ``'asymmetric'``.
        """
        own = [t for t in DIRECTEDNESS_TRAITS if t in self.labels]
        if own:
            return own[0]
        return "undirected" if isinstance(self, Undirected) else "directed"

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """The statistic of the pair ``(i, j)``; defined by each subclass."""
        raise NotImplementedError("Method not yet overloaded.")

    @parse_multivariate
    def multivariate(self,data):
        """The ``M x M`` matrix, from ``bivariate`` on every ordered pair."""
        M = data.n_processes
        A = np.full((M, M), np.nan)
        pairs = [(i, j) for j in range(M) for i in range(M) if i != j]
        return fill_pairwise(self.identifier, A, pairs,
                             lambda i, j: self.bivariate(data, i=i, j=j))

    def get_group(self,classes):
        """``(index, name)`` of the one class in ``classes`` this SPI's labels match."""
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
    """
    Base class of the undirected SPIs, with ``A[i, j] == A[j, i]``.

    Runs ``bivariate`` once per unordered pair and mirrors the value, so a
    subclass need only define ``bivariate``.
    """

    name = 'Base class'
    identifier = 'base'
    labels = []

    def ispositive(self):
        return False

    @parse_multivariate
    def multivariate(self,data):
        """
        ``bivariate(i, j)`` for each pair ``i < j``, mirrored onto ``(j, i)``.

        Each unordered pair is evaluated once. A failed pair is NaN in both
        cells, with an error reported for both and counted once in the warning;
        the SPI raises if every pair fails.

        The pairs are visited column by column, ``(0, 1), (0, 2), (1, 2),
        (0, 3), ...``: the order in which ``Directed.multivariate`` reaches the
        upper triangle. An SPI whose ``bivariate`` caches
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
    """
    Mixin for signed SPIs, which carry the ``signed`` label.

    Zero has a special meaning (no relationship, or no lead) and the sign says
    which of two opposite cases holds, so the absolute value is the size of
    the effect. Covariance (positive or negative coupling) is the model case.
    ``signed`` combines with the default one-ended strength and with the
    flags ``timing`` (a lead/lag whose sign says which process leads) and
    ``profile``.
    """
    def issigned(self):
        return True

class Unsigned:
    """
    Mixin for every SPI that is not signed, which carries no sign label.

    By default its value is a one-ended strength of dependence, higher meaning
    stronger (with the flag ``inverted``, lower meaning stronger, as for a
    distance or an entropy; with ``timing`` or ``profile``, an offset or a
    description of how the dependence is distributed), whatever values it
    takes: it can be negative through estimator bias, a floor that is not 0
    (soft-DTW, a differential entropy) or an inverted scale (the Engle-Granger
    t-statistic, more negative for stronger evidence).
    """
    def issigned(self):
        return False

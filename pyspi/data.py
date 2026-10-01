"""Provide data structures for multivariate analysis.

Code is adapted from Patricia Wollstadt's IDTxL (https://github.com/pwollstadt/IDTxl)
"""
import numpy as np
import pandas as pd
from pyspi import utils
from scipy.stats import zscore
from scipy.signal import detrend
import os

from ._logging import get_logger

logger = get_logger("pyspi.data")

VERBOSE = False


def _validate_procnames(procnames, n_processes):
    """Coerce process names to unique strings, or say why they cannot be.

    Names are the column and row labels of the results table, and
    ``Calculator.to_frame()`` stacks on them: duplicates raised pandas'
    "Columns with duplicate values are not supported in stack" from four frames
    away, with nothing pointing at the names. They are also written to the NPZ
    as a ``U`` array, so a non-string name came back as its ``str()`` and the
    file did not round-trip -- ``procnames=[1, 2]`` loaded as ``["1", "2"]``.
    Coercing here makes the object and the file agree from the start.
    """
    names = [str(p) for p in procnames]
    if len(names) != n_processes:
        raise ValueError(
            f"procnames length ({len(names)}) does not match "
            f"n_processes ({n_processes})."
        )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Process names must be unique; {duplicates} appear(s) more than "
            f"once. They label the rows and columns of the results table, and "
            f"`Calculator.to_frame()` cannot stack duplicated labels."
        )
    return names


class Data:
    """Store data for dependency analysis.

    Data takes a 2-dimensional array representing realisations of random
    variables in dimensions: processes and observations.
    Indicate the actual order of dimensions in the provided array in a two-character string, e.g. 'ps'
    for an array with realisations over (1) processes, and (2) observations in time.

    Example:
        >>> # Initialise empty data object
        >>> data = Data()
        >>>
        >>> # Load a prefilled financial dataset
        >>> data_forex = Data().load_dataset(forex)
        >>>
        >>> # Create data objects with data of various sizes
        >>> d = np.arange(3000).reshape((3, 1000))  # 3 procs.,
        >>> data_2 = Data(d, dim_order='ps')        # 1000 observations
        >>>
        >>> # Overwrite data in existing object with random data
        >>> d = np.arange(5000)
        >>> data_2.set_data(data_new, 's')

    Args:
        data (array_like, optional):
            2-dimensional array with raw data, default=None.
        dim_order (str, optional):
            Order of dimensions, accepts two combinations of the characters 'p', and 's' for processes and observations, default='ps'.
        detrend (bool, optional):
            If True, detrend each time series in the MTS dataset individually along the time axis, default=False.
        zscore (bool, optional):
            If True, z-score each time series in the MTS dataset individually
            along the time axis, default=True. Standardisation is per-process,
            not whole-dataset: it removes each process's arbitrary gain/units
            without letting the choice of the other processes in the dataset
            influence any pairwise statistic.
        name (str, optional):
            Name of the dataset
        procnames (list, optional):
            List of process names with length the number of processes, default=None.
        n_processes (int, optional):
            Truncates data to this many processes, default=None.
        n_observations (int, optional):
            Truncates data to this many observations, default=None.

    """

    # Every attribute a statistic may cache directly on a Data instance. This is
    # the authoritative list: anything that writes `data.<x> = ...` from
    # pyspi/statistics/ must appear here, or its cache will survive a mutation
    # of the underlying series and silently serve results from the old data.
    #
    # Derived mechanically from the statistics package:
    #   grep -rhoE "\bdata\.[a-z_][a-z0-9_]*\s*=" pyspi/statistics/*.py
    _CACHE_ATTRS = (
        "_spectral_bv_conn",
        "ais_embedding",
        "barycenter",
        "causal_entropy",
        "ccm",
        "coint",
        "covariance",
        "entropy",
        "joint_entropy",
        "mne",
        "mne_psi",
        "spectral_bv",
        "spectral_gc",
        "spectral_mv",
        "theiler",
        "xcorr",
    )

    def __init__(
        self,
        data=None,
        dim_order="ps",
        detrend=False,
        zscore=True,
        name=None,
        procnames=None,
        n_processes=None,
        n_observations=None,
    ):
        self.zscore = zscore
        self.detrend = detrend
        # Explicit empty state so attribute access is consistent before set_data
        # has run (Data() with no args is a legitimate builder-pattern entry point
        # used by add_process()).
        self._data = None
        self.n_processes = 0
        self.n_observations = 0
        self.n_replications = 0
        # _name and _procnames are only set when data is actually loaded — keeps
        # the existing contract that name="N/A" until data exists.

        if data is not None:
            dat = self.convert_to_numpy(data)
            self.set_data(
                dat,
                dim_order=dim_order,
                name=name,
                n_processes=n_processes,
                n_observations=n_observations,
            )
            if procnames is not None:
                self._procnames = _validate_procnames(procnames, self.n_processes)

    @classmethod
    def _from_prepared_array(cls, arr, procnames=None, name=None):
        """Build a Data around an already-preprocessed array, without copying.

        Internal constructor for the parallel workers, which attach to a
        shared-memory block holding the parent's already-detrended/z-scored
        series. Copying would defeat the point of sharing, so ownership is
        waived here and the array is exposed read-only instead — workers only
        ever read it.

        This replaces the previous ``Data.__new__`` + manual attribute
        assignment in ``pyspi._parallel._attach_data``, which had to be kept in
        sync with ``__init__`` by hand and silently skipped anything added
        there.
        """
        if arr.ndim != 3:
            raise ValueError(
                f"Prepared array must be (processes, observations, replications); "
                f"got shape {arr.shape}."
            )
        self = cls.__new__(cls)
        self.zscore = False
        self.detrend = False
        self._data = arr
        self.data_type = arr.dtype.type
        self._name = name or "N/A"
        self._reset_data_size()
        if procnames is not None:
            # Validated on the internal constructor too. The parallel workers
            # reach Data only through this path, so skipping the check here
            # would mean a name set that `Calculator.to_frame()` cannot stack
            # is caught in a serial run and not in a parallel one.
            self._procnames = _validate_procnames(procnames, self.n_processes)
        self._sync_procnames()
        return self

    @property
    def name(self):
        """Name of the data object."""
        if hasattr(self, "_name"):
            return self._name
        else:
            return "N/A"

    @name.setter
    def name(self, n):
        """Set the name of the data object."""
        if not isinstance(n, str):
            raise TypeError(f"Name should be a string, received {type(n)}.")
        self._name = n

    @property
    def procnames(self):
        """List of process names (a copy; mutating it does not affect the Data)."""
        if hasattr(self, "_procnames"):
            return list(self._procnames)
        else:
            return [f"proc-{i}" for i in range(self.n_processes)]

    def _apply_preprocessing(self, data, log=False):
        """Detrend and/or z-score along the time axis of a (p, s, r) array.

        The single preprocessing path. ``add_process`` previously appended its
        argument raw while ``set_data`` transformed it, so building a dataset
        with ``Data().add_process(x).add_process(y)`` z-scored the first process
        and left the second on its original scale -- every pairwise statistic
        then compared a standardised series against an unstandardised one.

        Both are per-process along time, so applying them to one appended
        process gives exactly the same result as applying them to the whole
        array at once.
        """
        if self.detrend:
            if log:
                logger.info("[1/2] Detrending time series in the dataset...")
            try:
                data = detrend(data, axis=1)
            except ValueError as err:
                logger.warning("Could not detrend data: %s", err)
        elif log:
            logger.info("[1/2] Skipping detrending of time series in the dataset.")

        if self.zscore:
            if log:
                logger.info("[2/2] Normalising (z-scoring) each time series in the dataset...")
            data = zscore(data, axis=1, nan_policy="omit", ddof=1)
        elif log:
            logger.info("[2/2] Skipping normalisation of time series in the dataset.")

        return data

    def _invalidate_caches(self):
        """Drop every statistic cache held on this instance.

        Called whenever the underlying series change. Statistics cache results
        keyed by parameters but *not* by the data, so a cache that outlives a
        mutation returns the previous dataset's numbers with no error and no
        warning — the most dangerous failure mode in the package.
        """
        for attr in self._CACHE_ATTRS:
            self.__dict__.pop(attr, None)

    def _set_internal(self, arr):
        """Install ``arr`` as the backing store, owned and frozen.

        Data takes ownership: the array is copied if it is not already private,
        then marked read-only so neither the caller nor a statistic can mutate
        the series behind the caches.
        """
        arr = np.array(arr, dtype=arr.dtype, copy=True, order="C")
        arr.setflags(write=False)
        self._data = arr
        self._invalidate_caches()

    def to_numpy(self, realisation=None, squeeze=False):
        """Return the numpy array.

        The result is a **read-only** view of the internal store. Copy it if you
        need to modify it; writing through it would desynchronise the statistic
        caches from the data they were computed on.
        """
        if realisation is not None:
            dat = self._data[:, :, realisation]
        else:
            dat = self._data

        if squeeze:
            dat = np.squeeze(dat)

        # Always freeze, never conditionally. Guarding on `owndata` was exactly
        # backwards: the shared-memory worker path is precisely where owndata is
        # False, so the one case that most needed protecting was the one case
        # left writable -- a worker could mutate the block every other worker
        # was reading.
        dat = dat.view()
        dat.setflags(write=False)
        return dat

    @staticmethod
    def convert_to_numpy(data):
        """Converts other data instances to default numpy format."""

        if isinstance(data, np.ndarray):
            npdat = data
        elif isinstance(data, pd.DataFrame):
            npdat = data.to_numpy()
        elif isinstance(data, str):
            ext = os.path.splitext(data)[1]
            if ext == ".npy":
                npdat = np.load(data)
            elif ext == ".txt":
                npdat = np.genfromtxt(data)
            elif ext == ".csv":
                npdat = np.genfromtxt(data, ",")
            elif ext == ".ts":
                try:
                    from aeon.datasets import load_from_ts_file
                except ImportError as e:
                    raise ImportError(
                        "Loading .ts files requires aeon. Install with "
                        "`pip install aeon` or `uv pip install aeon`."
                    ) from e
                npdat, _ = load_from_ts_file(data)
            else:
                raise TypeError(f"Unknown filename extension: {ext}")
        else:
            raise TypeError(f"Unknown data type: {type(data)}")

        return npdat

    def set_data(
        self,
        data,
        dim_order="ps",
        name=None,
        n_processes=None,
        n_observations=None,
        verbose=False,
    ):
        """Overwrite data in an existing instance.

        Args:
            data (np.ndarray):
                2-dimensional array of realisations
            dim_order (str, optional):
                order of dimensions, accepts a combination of the characters
                'p' and 's', for processes and observations;
                must have the same length as number of dimensions in data
        """
        if len(dim_order) > 3:
            raise RuntimeError("dim_order can not have more than two " "entries")
        if len(dim_order) != data.ndim:
            raise RuntimeError(
                "Data array dimension ({0}) and length of "
                "dim_order ({1}) are not equal.".format(data.ndim, len(dim_order))
            )
        # Unknown or repeated symbols previously slipped through and produced
        # 4-D/5-D internal states (e.g. 'xx', 'pp') that fail far from here.
        unknown = set(dim_order) - set("psr")
        if unknown:
            raise ValueError(
                f"dim_order contains unknown symbol(s) {sorted(unknown)}; "
                "valid symbols are 'p' (processes), 's' (observations), "
                "'r' (replications)."
            )
        if len(set(dim_order)) != len(dim_order):
            raise ValueError(f"dim_order has repeated symbols: {dim_order!r}.")

        # Bring data into the order processes x observations in a pandas dataframe.
        data = self._reorder_data(data, dim_order)

        if n_processes is not None:
            data = data[:n_processes]
        if n_observations is not None:
            data = data[:, :n_observations]

        data = self._apply_preprocessing(data, log=True)

        # Check all non-finite values, not just NaNs: with zscore=False an inf
        # passes straight through to the estimators, where it surfaces as an
        # unrelated failure much later.
        bad = ~np.isfinite(data)
        if bad.any():
            raise ValueError(
                f"Dataset {name} contains non-finite values (NaN/inf) in "
                f"processes: {np.unique(np.where(bad)[0])}."
            )

        self._set_internal(data)
        self.data_type = self._data.dtype.type

        self._reset_data_size()
        # Process names are positional, so any change in width invalidates them.
        self._sync_procnames()

        if name is not None:
            self._name = name

        if verbose:
            logger.info(
                'Dataset "%s" now has properties: %d processes, %d observations, %d replications',
                name, self.n_processes, self.n_observations, self.n_replications,
            )

    def add_process(self, proc, verbose=False):
        """Appends a univariate process to the dataset.

        Args:
            proc (ndarray):
                Univariate process to add, must be an array the same size as existing ones.
        """
        proc = np.squeeze(proc)
        if not isinstance(proc, np.ndarray) or proc.ndim != 1:
            raise TypeError("Process must be a 1D numpy array")

        # Guard on the value, not on hasattr: _data is now always present (set
        # to None in __init__), so hasattr is True even for an empty Data and
        # the builder path would reshape into a zero-width array.
        if self._data is None:
            self.set_data(proc, dim_order="s", verbose=verbose)
            return

        if proc.size != self.n_observations:
            raise ValueError(
                f"Process has {proc.size} observations but the dataset has "
                f"{self.n_observations}."
            )
        # Preprocess the incoming process the same way set_data would, so the
        # builder path produces a dataset with uniform preprocessing.
        block = self._apply_preprocessing(
            np.reshape(np.asarray(proc, dtype=float), (1, self.n_observations, 1))
        )
        appended = np.append(self._data, block, axis=0)
        self._set_internal(appended)
        self._reset_data_size()
        if hasattr(self, "_procnames"):
            # `proc-<index>` can already be taken: a caller who named their
            # processes ["a", "proc-2"] and then appended twice would produce a
            # second "proc-2", and `Calculator.to_frame()` cannot stack
            # duplicated labels. Fall past any name already in use rather than
            # silently creating a collision the constructor would have rejected.
            index = self.n_processes - 1
            taken = set(self._procnames)
            name = f"proc-{index}"
            while name in taken:
                index += 1
                name = f"proc-{index}"
            self._procnames.append(name)

    def remove_process(self, procs):
        try:
            reduced = np.delete(self._data, procs, axis=0)
        except IndexError:
            logger.error(
                "Process %s is out of bounds of multivariate time-series data "
                "with %d process(es)", procs, self.n_processes,
            )
            return

        keep = np.delete(np.arange(self.n_processes), procs)
        self._set_internal(reduced)
        self._reset_data_size()
        if hasattr(self, "_procnames"):
            self._procnames = [self._procnames[i] for i in keep]

    def _sync_procnames(self):
        """Drop stale process names after a change in width.

        Names are positional; once the number of processes changes under them
        they no longer identify anything, so falling back to the generated
        ``proc-i`` names is the only honest option.
        """
        names = self.__dict__.get("_procnames")
        if names is not None and len(names) != self.n_processes:
            del self._procnames

    def _reorder_data(self, data, dim_order):
        """Reorder data dimensions to processes x observations x realisations."""
        # add singletons for missing dimensions
        missing_dims = "psr"
        for dim in dim_order:
            missing_dims = missing_dims.replace(dim, "")
        for dim in missing_dims:
            data = np.expand_dims(data, data.ndim)
            dim_order += dim

        # reorder array dims if necessary
        if dim_order[0] != "p":
            ind_p = dim_order.index("p")
            data = data.swapaxes(0, ind_p)
            dim_order = utils.swap_chars(dim_order, 0, ind_p)
        if dim_order[1] != "s":
            data = data.swapaxes(1, dim_order.index("s"))

        return data

    def _reset_data_size(self):
        """Set the data size."""
        self.n_processes = self._data.shape[0]
        self.n_observations = self._data.shape[1]
        self.n_replications = self._data.shape[2]


# name -> (filename, dim_order, description). Every bundled .npy is stored
# (observations, processes), hence dim_order 'sp' throughout.
_DATASETS = {
    "forex":           ("forex.npy",           "sp", "Foreign-exchange rates (250 obs, 7 processes)."),
    "cml":             ("cml.npy",             "sp", "Coupled map lattice (500 obs, 10 processes)."),
    "standard_normal": ("standard_normal.npy", "sp", "i.i.d. standard normal null model (200 obs, 5 processes)."),
}


def available_datasets():
    """Return ``{name: description}`` for every bundled dataset."""
    return {name: desc for name, (_f, _d, desc) in _DATASETS.items()}


def load_dataset(name):
    """Load a bundled example dataset by name.

    See :func:`available_datasets` for the full list.
    """
    try:
        filename, dim_order, _desc = _DATASETS[name]
    except KeyError:
        raise NameError(
            f"Unknown dataset: {name}. Available: {', '.join(sorted(_DATASETS))}."
        ) from None
    basedir = os.path.join(os.path.dirname(__file__), "data")
    return Data(data=os.path.join(basedir, filename), dim_order=dim_order)

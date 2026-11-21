"""The ``Data`` container for multivariate time series, and the bundled datasets.

Adapted from the ``Data`` class of Patricia Wollstadt's IDTxl
(https://github.com/pwollstadt/IDTxl).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import zscore

from pyspi import utils
from ._compat import resolve_zscore
from ._logging import get_logger

logger = get_logger("pyspi.data")

VERBOSE = False


def _validate_procnames(procnames, n_processes):
    """
    Coerce process names to unique strings, or say why they cannot be.

    Names label the rows and columns of the results table, which
    ``Calculator.to_frame()`` cannot stack with duplicates, and are saved as
    strings, so they are made strings here for the object and file to agree.
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


def _frame_procnames(data, dim_order):
    """
    The process names a DataFrame carries, or None.

    Under ``"ps"`` processes are rows, named by the index; under ``"sp"`` they
    are columns. A RangeIndex holds positions, not names, so it gives None.
    """
    if not isinstance(data, pd.DataFrame) or dim_order not in ("ps", "sp"):
        return None
    labels = data.index if dim_order == "ps" else data.columns
    if isinstance(labels, pd.RangeIndex):
        return None
    return list(labels)


def _outside_pyspi_stacklevel():
    """``stacklevel`` for a warning raised by the caller, pointing at the first frame outside pyspi."""
    package = os.path.dirname(os.path.abspath(__file__))
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(
            frame.f_code.co_filename).startswith(package + os.sep):
        frame, level = frame.f_back, level + 1
    return level


class Data:
    """
    A multivariate time series: processes x observations x replications.

    Holds ``M`` processes of ``T`` observations (and ``R`` replications,
    usually 1) as a read-only array, each process optionally detrended and
    z-scored along time. Statistics cache intermediate results on the
    instance; changing the series with ``set_data``, ``add_process`` or
    ``remove_process`` drops every cache. A series with a non-finite value is
    refused.

    Parameters
    ----------
    data : array-like, str or None, optional
        The time series: a numpy array, a pandas DataFrame, or the path of a
        ``.npy``, ``.txt``, ``.csv`` or ``.ts`` file (``.ts`` needs aeon).
        Default: None, an empty ``Data`` to fill with ``add_process``.
    dim_order : str, optional
        The order of the dimensions of ``data``, one character per dimension:
        ``'p'`` for processes, ``'s'`` for observations (samples) and ``'r'``
        for replications. ``'ps'`` reads an ``(M, T)`` array, one row per
        process; ``'sp'`` reads a ``(T, M)`` array or a DataFrame with one
        column per process. A ``UserWarning`` says when the data are read as
        more processes than observations, which usually means the other
        order was meant. Default: ``'ps'``.
    detrend : bool, optional
        Remove the least-squares line from each process. Default: False.
    zscore : bool or None, optional
        z-score each process along time (``ddof=1``). This is per process, not
        over the whole dataset, so it removes each process's units without
        letting the other processes influence any pairwise statistic.
        Default: None, which means True.
    name : str or None, optional
        Name of the dataset. Default: None (``"N/A"``).
    procnames : list or None, optional
        One unique name per process, converted to strings. Default: None,
        which takes the names of a DataFrame's processes (its index under
        ``'ps'``, its columns under ``'sp'``, unless they are a
        ``RangeIndex``) and otherwise gives ``proc-0``, ``proc-1``, ...
    n_processes : int or None, optional
        Keep only the first ``n_processes`` processes. Default: None (all).
    n_observations : int or None, optional
        Keep only the first ``n_observations`` observations. Default: None
        (all).
    normalise : bool or None, optional
        Deprecated spelling of ``zscore``, removed in pyspi 4.0.
        Default: None.

    Examples
    --------
    >>> import numpy as np
    >>> from pyspi.data import Data, load_dataset
    >>> data = Data(np.random.randn(3, 1000), dim_order="ps")  # 3 processes
    >>> forex = load_dataset("forex")
    """

    # Every attribute a statistic may cache on a Data instance. Anything that
    # writes `data.<x> = ...` in pyspi/statistics must be listed, or its cache
    # would survive a change of the series. To list them:
    #   grep -rhoE "\bdata\.[a-z_][a-z0-9_]*\s*=" pyspi/statistics/*.py
    _CACHE_ATTRS = (
        "_spectral_bv_conn",
        "acf_1e",
        "ais_embedding",
        "barycenter",
        "causal_entropy",
        "ccm",
        "coint",
        "covariance",
        "crossfreq",
        "entropy",
        "events",
        "joint_entropy",
        "liang",
        "mne",
        "mne_psi",
        "nli",
        "spectral_bv",
        "spectral_gc",
        "spectral_mv",
        "theiler",
        "var",
        "wavelet_coherence",
        "xcorr",
    )

    def __init__(
        self,
        data=None,
        dim_order="ps",
        detrend=False,
        zscore=None,
        name=None,
        procnames=None,
        n_processes=None,
        n_observations=None,
        normalise=None,
    ):
        # `normalise` is the deprecated spelling of `zscore`.
        self.zscore = resolve_zscore(zscore, normalise)
        self.detrend = detrend
        # An empty Data() is filled with add_process().
        self._data = None
        self.n_processes = 0
        self.n_observations = 0
        self.n_replications = 0
        # _name and _procnames are set only once there is data.

        if data is not None:
            if procnames is None:
                procnames = _frame_procnames(data, dim_order)
                if procnames is not None and n_processes is not None:
                    procnames = procnames[:n_processes]
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
        """
        Build a Data around an already-preprocessed array, without copying.

        Used by the parallel workers, which attach to a shared-memory block
        holding the parent's preprocessed series and only read it.
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
            # Validated here too, so parallel runs refuse the same names.
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
        """
        Detrend and/or z-score along the time axis of a (p, s, r) array.

        Shared by ``set_data`` and ``add_process``. Both steps act on each
        process separately, so one appended process is treated exactly as it
        would be in the whole array.
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
        """
        Drop every statistic cache held on this instance.

        Called whenever the series change: caches are keyed by parameters, not
        by the data, so one that outlived a change would serve stale values.
        """
        for attr in self._CACHE_ATTRS:
            self.__dict__.pop(attr, None)

    def _set_internal(self, arr):
        """
        Install ``arr`` as the backing store, owned and frozen.

        Data takes ownership: the array is copied if it is not already private,
        then marked read-only so neither the caller nor a statistic can mutate
        the series behind the caches.
        """
        arr = np.array(arr, dtype=arr.dtype, copy=True, order="C")
        arr.setflags(write=False)
        self._data = arr
        self._invalidate_caches()

    def to_numpy(self, realisation=None, squeeze=False):
        """
        The series as a read-only numpy array.

        A view of the internal store, read-only so the statistic caches stay in
        step with the data; copy it to modify it.

        Parameters
        ----------
        realisation : int or None, optional
            Return only this replication. Default: None (all).
        squeeze : bool, optional
            Drop dimensions of length 1. Default: False.

        Returns
        -------
        numpy.ndarray
            Shape ``(M, T, R)``, or ``(M, T)`` for one ``realisation``, less
            any length-1 dimensions when ``squeeze``.
        """
        if realisation is not None:
            dat = self._data[:, :, realisation]
        else:
            dat = self._data

        if squeeze:
            dat = np.squeeze(dat)

        # Always freeze: in the workers the array is a shared-memory block.
        dat = dat.view()
        dat.setflags(write=False)
        return dat

    @staticmethod
    def convert_to_numpy(data):
        """
        Read ``data`` into a numpy array.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame or str
            An array, a DataFrame, or the path of a ``.npy``, ``.txt``,
            ``.csv`` (comma-separated) or ``.ts`` file (``.ts`` needs aeon).

        Returns
        -------
        numpy.ndarray
            The data, unchanged in shape.
        """

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
                npdat = np.genfromtxt(data, delimiter=",")
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
        """
        Replace the series held by this instance.

        The array is reordered to processes x observations x replications,
        truncated, preprocessed with this instance's ``detrend`` and ``zscore``
        settings and checked for non-finite values. Every statistic cache is
        dropped, and process names are kept only if the number of processes
        is unchanged. A ``UserWarning`` says when the result has more
        processes than observations.

        Parameters
        ----------
        data : numpy.ndarray
            Array of one to three dimensions.
        dim_order : str, optional
            The order of the dimensions of ``data``, one character per
            dimension: ``'p'`` (processes), ``'s'`` (observations) or ``'r'``
            (replications). Default: ``'ps'``.
        name : str or None, optional
            New name of the dataset; None keeps the current one.
            Default: None.
        n_processes : int or None, optional
            Keep only the first ``n_processes`` processes. Default: None (all).
        n_observations : int or None, optional
            Keep only the first ``n_observations`` observations.
            Default: None (all).
        verbose : bool, optional
            Log the new size. Default: False.
        """
        if len(dim_order) > 3:
            raise RuntimeError("dim_order can not have more than two " "entries")
        if len(dim_order) != data.ndim:
            raise RuntimeError(
                "Data array dimension ({0}) and length of "
                "dim_order ({1}) are not equal.".format(data.ndim, len(dim_order))
            )
        # Unknown or repeated symbols would give a 4-D or 5-D array.
        unknown = set(dim_order) - set("psr")
        if unknown:
            raise ValueError(
                f"dim_order contains unknown symbol(s) {sorted(unknown)}; "
                "valid symbols are 'p' (processes), 's' (observations), "
                "'r' (replications)."
            )
        if len(set(dim_order)) != len(dim_order):
            raise ValueError(f"dim_order has repeated symbols: {dim_order!r}.")

        shape = data.shape
        # Bring data into the order processes x observations x replications.
        data = self._reorder_data(data, dim_order)

        if n_processes is not None:
            data = data[:n_processes]
        if n_observations is not None:
            data = data[:, :n_observations]

        if data.shape[0] > data.shape[1]:
            swapped = dim_order.translate(str.maketrans("ps", "sp"))
            warnings.warn(
                f"Data of shape {shape} read with dim_order={dim_order!r} has "
                f"{data.shape[0]} processes of {data.shape[1]} observations. "
                f"If time runs along the other axis, pass "
                f"dim_order={swapped!r}.",
                UserWarning, stacklevel=_outside_pyspi_stacklevel(),
            )

        data = self._apply_preprocessing(data, log=True)

        # Inf as well as NaN: with zscore=False an Inf would reach the SPIs.
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
        """
        Append one process to the dataset.

        The process is preprocessed as the others were. If the processes are
        named, it is named ``proc-<index>``, or the next such name not already
        taken.

        Parameters
        ----------
        proc : numpy.ndarray
            A 1-D array (after squeezing) with as many observations as the
            dataset; the first process of an empty ``Data``.
        verbose : bool, optional
            Log the new size when this is the first process. Default: False.
        """
        proc = np.squeeze(proc)
        if not isinstance(proc, np.ndarray) or proc.ndim != 1:
            raise TypeError("Process must be a 1D numpy array")

        # _data is None, not absent, in an empty Data.
        if self._data is None:
            self.set_data(proc, dim_order="s", verbose=verbose)
            return

        if proc.size != self.n_observations:
            raise ValueError(
                f"Process has {proc.size} observations but the dataset has "
                f"{self.n_observations}."
            )
        # Preprocessed as set_data would.
        block = self._apply_preprocessing(
            np.reshape(np.asarray(proc, dtype=float), (1, self.n_observations, 1))
        )
        appended = np.append(self._data, block, axis=0)
        self._set_internal(appended)
        self._reset_data_size()
        if hasattr(self, "_procnames"):
            # `proc-<index>` may already be taken (names ["a", "proc-2"], then
            # two appends): skip to a free name, as names must be unique.
            index = self.n_processes - 1
            taken = set(self._procnames)
            name = f"proc-{index}"
            while name in taken:
                index += 1
                name = f"proc-{index}"
            self._procnames.append(name)

    def remove_process(self, procs):
        """
        Remove processes from the dataset.

        An index out of range is logged as an error and nothing is removed.

        Parameters
        ----------
        procs : int or list of int
            Index or indices of the processes to remove.
        """
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
        """
        Drop the process names if their number differs from the processes'.

        Names are positional; the generated ``proc-i`` names are used instead.
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


# name -> (filename, dim_order, description). Every bundled .npy is stored as
# (observations, processes).
_DATASETS = {
    "forex":           ("forex.npy",           "sp", "Foreign-exchange rates (250 obs, 7 processes)."),
    "cml":             ("cml.npy",             "sp", "Coupled map lattice (500 obs, 10 processes)."),
    "standard_normal": ("standard_normal.npy", "sp", "i.i.d. standard normal null model (200 obs, 5 processes)."),
}


def available_datasets():
    """
    The bundled example datasets.

    Returns
    -------
    dict
        ``{name: description}`` for every dataset :func:`load_dataset` takes.
    """
    return {name: desc for name, (_f, _d, desc) in _DATASETS.items()}


def load_dataset(name):
    """
    Load a bundled example dataset.

    Parameters
    ----------
    name : str
        ``'forex'``, ``'cml'`` or ``'standard_normal'`` (see
        :func:`available_datasets`).

    Returns
    -------
    Data
        The dataset, z-scored with the ``Data`` defaults.
    """
    try:
        filename, dim_order, _desc = _DATASETS[name]
    except KeyError:
        raise NameError(
            f"Unknown dataset: {name}. Available: {', '.join(sorted(_DATASETS))}."
        ) from None
    basedir = os.path.join(os.path.dirname(__file__), "data")
    return Data(data=os.path.join(basedir, filename), dim_order=dim_order)

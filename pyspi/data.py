"""Provide data structures for multivariate analysis.

Code is adapted from Patricia Wollstadt's IDTxL (https://github.com/pwollstadt/IDTxl)
"""
import numpy as np
import pandas as pd
from pyspi import utils
from scipy.stats import zscore
from scipy.signal import detrend
from colorama import init, Fore
import os

VERBOSE = False
init(autoreset=True) # automatically reset coloured outputs

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
            2-dimensional array with raw data, defaults to None.
        dim_order (str, optional):
            Order of dimensions, accepts two combinations of the characters 'p', and 's' for processes and observations, defaults to 'ps'.
        detrend (bool, optional):
            If True, detrend the dataset along the time axis before normalising (if enabled), defaults to True.
        normalise (bool, optional):
            If True, z-score normalise the dataset along the time axis before computing SPIs, defaults to True.
            Detrending (if enabled) is always applied before normalisation.
        name (str, optional):
            Name of the dataset
        procnames (list, optional):
            List of process names with length the number of processes, defaults to None.
        n_processes (int, optional):
            Truncates data to this many processes, defaults to None.
        n_observations (int, optional):
            Truncates data to this many observations, defaults to None.

    """

    def __init__(
        self,
        data=None,
        dim_order="ps",
        detrend=True,
        normalise=True,
        name=None,
        procnames=None,
        n_processes=None,
        n_observations=None,
    ):
        self.normalise = normalise
        self.detrend = detrend
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
            assert len(procnames) == self.n_processes

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
        """List of process names."""
        if hasattr(self, "_procnames"):
            return self._procnames
        else:
            return [f"proc-{i}" for i in range(self.n_processes)]

    def to_numpy(self, realisation=None, squeeze=False):
        """Return the numpy array."""
        if realisation is not None:
            dat = self._data[:, :, realisation]
        else:
            dat = self._data

        if squeeze:
            return np.squeeze(dat)
        else:
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
                from sktime.utils.data_io import load_from_tsfile_to_dataframe
                from sktime.datatypes._panel._convert import from_nested_to_3d_numpy

                tsdat, _ = load_from_tsfile_to_dataframe(data)
                npdat = from_nested_to_3d_numpy(tsdat)
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

        # Bring data into the order processes x observations in a pandas dataframe.
        data = self._reorder_data(data, dim_order)

        if n_processes is not None:
            data = data[:n_processes]
        if n_observations is not None:
            data = data[:, :n_observations]

        if self.detrend:
            print(Fore.GREEN + "[1/2] De-trending the dataset...")
            try:
                data = detrend(data, axis=1)
            except ValueError as err:
                print(f"Could not detrend data: {err}")
        else:
            print(Fore.RED + "[1/2] Skipping detrending of the dataset...")

        if self.normalise:
            print(Fore.GREEN + "[2/2] Normalising (z-scoring) the dataset...\n")
            data = zscore(data, axis=1, nan_policy="omit", ddof=1)
        else:
            print(Fore.RED + "[2/2] Skipping normalisation of the dataset...\n")

        nans = np.isnan(data)
        if nans.any():
            raise ValueError(
                f"Dataset {name} contains non-numerics (NaNs) in processes: {np.unique(np.where(nans)[0])}."
            )

        self._data = data
        self.data_type = type(data[0, 00, 0])

        self._reset_data_size()

        if name is not None:
            self._name = name

        if verbose:
            print(
                f'Dataset "{name}" now has properties: {self.n_processes} processes, {self.n_observations} observations, {self.n_replications} '
                "replications"
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

        if hasattr(self, "_data"):
            try:
                self._data = np.append(
                    self._data, np.reshape(proc, (1, self.n_observations, 1)), axis=0
                )
            except IndexError:
                raise IndexError()
        else:
            self.set_data(proc, dim_order="s", verbose=verbose)

        self._reset_data_size()

    def remove_process(self, procs):
        try:
            self._data = np.delete(self._data, procs, axis=0)
        except IndexError:
            print(
                f"Process {procs} is out of bounds of multivariate"
                f" time-series data with size {self.data.n_processes}"
            )

        self._reset_data_size()

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


def load_dataset(name):
    basedir = os.path.join(os.path.dirname(__file__), "data")
    if name == "forex":
        filename = "forex.npy"
        dim_order = "sp"
    elif name == "cml":
        filename = "cml.npy"
        dim_order = "sp"
    else:
        raise NameError(f"Unknown dataset: {name}.")
    return Data(data=os.path.join(basedir, filename), dim_order=dim_order)

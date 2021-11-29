"""Provide data structures for multivariate analysis.

Stolen mostly from IDTxL (for now...)
"""
import numpy as np
import pandas as pd
from pyspi import utils
from scipy.stats import zscore
import os
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy

VERBOSE = False

class Data():
    """Store data for dependency analysis.

    Data takes a 1- to 3-dimensional array representing realisations of random
    variables in dimensions: processes, observations (over time), and realisations.
    If necessary, data reshapes provided realisations to fit the format
    expected by _nats_, which is a 3-dimensional array with axes representing
    (process index, observation index, replication index). Indicate the actual order
    of dimensions in the provided array in a three-character string, e.g. 'spr'
    for an array with realisations over (1) observations in time, (2) processes, (3)
    realisations.

    Example:

        >>> data_mute = Data()              # initialise empty data object
        >>> data_mute.generate_mute_data()  # simulate data from MuTE paper
        >>>
        >>> # Create data objects with data of various sizes
        >>> d = np.arange(10000).reshape((2, 1000, 5))  # 2 procs.,
        >>> data_1 = Data(d, dim_order='psd')           # 1000 observations, 5 repl.
        >>>
        >>> d = np.arange(3000).reshape((3, 1000))  # 3 procs.,
        >>> data_2 = Data(d, dim_order='ps')        # 1000 observations
        >>>
        >>> # Overwrite data in existing object with random data
        >>> d = np.arange(5000)
        >>> data_2.set_data(data_new, 's')

    Note:
        Realisations are stored as attribute 'data'. This can only be set via
        the 'set_data()' method.

    Args:
        data : numpy array [optional]
            1/2/3-dimensional array with raw data
        dim_order : string [optional]
            order of dimensions, accepts any combination of the characters
            'd', 'p', and 's' for realisations, processes, and observations; must
            have the same length as the data dimensionality, e.g., 'ps' for a
            two-dimensional array of data from several processes over time
            (default='dps')
        normalise : bool [optional]
            if True, data gets normalised per time series (default=True)

    Attributes:
        data : numpy array
            realisations, can only be set via 'set_data' method
        n_processes : int
            number of processes
        n_observations : int
            number of observations in time
        n_realisations : int
            number of realisations
        normalise : bool
            if true, all data gets z-standardised per process

    """

    def __init__(self,data=None,dim_order='ps',normalise=True,name=None,n_processes=None,n_observations=None):
        self.normalise = normalise
        if data is not None:
            dat = self.convert_to_numpy(data)
            self.set_data(dat, dim_order=dim_order, name=name, n_processes=n_processes,n_observations=n_observations)

    @property
    def name(self):
        if hasattr(self,'_name'):
            return self._name 
        else:
            return ''

    def n_realisations(self, current_value=None):
        """Number of realisations over samples and replications.

        Args:
            current_value : tuple [optional]
                reference point for calculation of number of realisations
                (e.g. when using an embedding of length k, we count
                realisations from the k+1th sample because we loose the first k
                samples to the embedding); if no current_value is provided, the
                number of all samples is used
        """
        return (self.n_realisations_observations(current_value) *
                self.n_realisations_repl())

    def n_realisations_repl(self, current_value=None):
        """Number of realisations over observations and realisations.

        Args:
            current_value : tuple [optional]
                reference point for calculation of number of realisations
                (e.g. when using an embedding of length k, we count
                realisations from the k+1th observation because we loose the first k
                observations to the embedding); if no current_value is provided, the
                number of all observations is used
        """
        return (self.n_realisations_observations(current_value) * self.n_replications)

    def n_realisations_observations(self, current_value=None):
        """Number of realisations over observations.

        Args:
            current_value : tuple [optional]
                reference point for calculation of number of realisations
                (e.g. when using an embedding of length k, the current value is
                at observation k + 1; we thus count realisations from the k + 1st
                observation because we loose the first k observations to the embedding)
        """
        if current_value is None:
            return self.n_observations
        else:
            if current_value[1] >= self.n_observations:
                raise RuntimeError('The observation index of the current value '
                                   '({0}) is larger than the number of observations'
                                   ' in the data set ({1}).'.format(
                                              current_value, self.n_observations))
            return self.n_observations - current_value[1]

    def to_numpy(self,realisation=None,squeeze=False):
        """Return the numpy array."""
        if realisation is not None:
            dat = self._data[:,:,realisation]
        else:
            dat = self._data
            
        if squeeze:
            return np.squeeze(dat)
        else:
            return dat

    @staticmethod
    def convert_to_numpy(data):

        if isinstance(data, np.ndarray):
            npdat = data
        elif isinstance(data, pd.DataFrame):
            npdat = data.to_numpy()
        elif isinstance(data,str):
            ext = os.path.splitext(data)[1]
            if ext == '.npy':
                npdat = np.load(data)
            elif ext == '.txt':
                npdat = np.genfromtxt(data)
            elif ext == '.csv':
                npdat = np.genfromtxt(data,',')
            elif ext == '.ts':
                tsdat, tsclasses = load_from_tsfile_to_dataframe(data)
                npdat = from_nested_to_3d_numpy(tsdat)
            else:
                raise TypeError(f'Unknown filename extension: {ext}')
        else:
            raise TypeError(f'Unknown data type: {type(data)}')

        return npdat

    def set_data(self,data,dim_order='ps',name=None,n_processes=None,n_observations=None,verbose=False):
        """Overwrite data in an existing Data object.

        Args:
            data : numpy array
                1- to 3-dimensional array of realisations
            dim_order : string
                order of dimensions, accepts any combination of the characters
                'p', 's', and 'r' for processes, observations, and realisations;
                must have the same length as number of dimensions in data
        """
        if len(dim_order) > 3:
            raise RuntimeError('dim_order can not have more than two '
                            'entries')
        if len(dim_order) != data.ndim:
            raise RuntimeError('Data array dimension ({0}) and length of '
                            'dim_order ({1}) are not equal.'.format(
                                        data.ndim, len(dim_order)))

        # Bring data into the order processes x observations in a pandas dataframe.
        data = self._reorder_data(data, dim_order)

        if n_processes is not None:
            data = data[:n_processes]
        if n_observations is not None:
            data = data[:,:n_observations]

        if self.normalise:
            data = zscore(data,axis=1,nan_policy='omit',ddof=1)

        nans = np.isnan(data)
        if nans.any():
            raise ValueError(f'Dataset {name} contains non-numerics (NaNs) in processes: {np.unique(np.where(nans)[0])}.')

        self._data = data
        self.data_type = type(data[0, 00, 0])

        self._reset_data_size()

        if name is not None:
            self._name = name

        if verbose:
            print(f'Dataset "{name}" now has properties: {self.n_processes} processes, {self.n_observations} observations, {self.n_replications} '
                    'replications')

    def add_process(self,proc,verbose=False):
        proc = np.squeeze(proc)
        if not isinstance(proc,np.ndarray) or proc.ndim != 1:
            raise TypeError('Process must be a 1D numpy array')

        if hasattr(self,'_data'):
            try:
                self._data = np.append(self._data,np.reshape(proc,(1,self.n_observations,1)),axis=0)
            except IndexError:
                raise IndexError()
        else:
            self.set_data(proc, dim_order='s',verbose=verbose)
        
        self._reset_data_size()

    def remove_process(self, procs):
        try:
            self._data = np.delete(self._data,procs,axis=0)
        except IndexError:
            print(f'Process {procs} is out of bounds of multivariate'
                    f' time-series data with size {self.data.n_processes}')
        
        self._reset_data_size()

    def _reorder_data(self, data, dim_order):
        """Reorder data dimensions to processes x observations x realisations."""
        # add singletons for missing dimensions
        missing_dims = 'psr'
        for dim in dim_order:
            missing_dims = missing_dims.replace(dim, '')
        for dim in missing_dims:
            data = np.expand_dims(data, data.ndim)
            dim_order += dim

        # reorder array dims if necessary
        if dim_order[0] != 'p':
            ind_p = dim_order.index('p')
            data = data.swapaxes(0, ind_p)
            dim_order = utils.swap_chars(dim_order, 0, ind_p)
        if dim_order[1] != 's':
            data = data.swapaxes(1, dim_order.index('s'))
                
        return data

    def _reset_data_size(self):
        """Set the data size."""
        self.n_processes = self._data.shape[0]
        self.n_observations = self._data.shape[1]
        self.n_replications = self._data.shape[2]

def load_dataset(name):
    basedir = os.path.join(os.path.dirname(__file__),'data')
    if name == 'forex':
        filename = 'forex.npy'
        dim_order = 'sp'
    elif name == 'cml':
        filename = 'cml.npy'
        dim_order = 'sp'
    else:
        raise NameError(f'Unknown dataset: {name}.')
    return Data(data=os.path.join(basedir,filename),dim_order=dim_order)
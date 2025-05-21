import numpy as np
import warnings, copy

from skbase import BaseObject
from pyspi.data import Data

class BaseNewSPI(BaseObject):
    """
    Base class for PySPI. This class provides a unified public API for the PySPI library.

    This class serves as the base class for all the pairwaise statistical interaction
    measures (SPI) in the PySPI library. It uses the tagging system of skbase to indicate
    capabilities rather than using decorators.

    Notes
    -----
    This class follows a modern approach towards defining the public API by using
    the tagging system of skbase. The tags are used to indicate the capabilties of the
    handling the data. The old base class is still available for backward compatibility.
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _ensure_data_format(self, data: Data, data2: Data  = None) -> tuple:
        """
        Ensure the data is in the correct format.
        
        Parameters
        ----------
        data : Data
            The data to be checked.
        data2 : Data, optional
            The second data to be checked. Default is None.
    
        Returns
        -------
        tuple
            A tuple containing the data and the second data in the correct format.
        """

        if not isinstance(data, Data):
            data = Data(data)
        if data2 is not None and not isinstance(data2, Data):
            data2 = Data(data2)

        return data, data2
    
    ## Public API ##
    def spi(self, data, i=None, copy_data=False):
        """Compute the univariate SPI value
        
        Parameters
        ----------
        data : Data
            The data to be used for the computation.
        i : int, optional
            The index of the variable to be used. Default is None.
        copy_data : bool, optional
            Whether to copy the data or not. Default is False.

        Returns
        -------
        float
            The computed SPI value.
        """

        data, _ = self.ensure_data_format(data, None)
        if i is None:
            if data.n_processes == 1:
                i=0
            else:
                raise ValueError("i must be specified for multiple processes")
        
        if copy_data:
            data = copy.deepcopy(data)

        return self._spi(data, i)

    def spi_mat(self, data, data2=None, i=None, j=None, copy_data=False):
        """
        Compute the pairwise SPI matrix
        
        Parameters
        ----------
        data : Data
            The data to be used for the computation.
        data2 : Data, optional
            The second data to be used for the computation. Default is None.
        i : int, optional
            Row index filter of the output matrix. Default is None.
        j : int, optional
            Column index filter of the output matrix. Default is None.
        copy_data : bool, optional
            Whether to copy the data or not. Default is False.

        Returns
        -------
        np.ndarray
            The computed SPI matrix.
        """

        # get the correct formatted data
        data, data2 = self.ensure_data_format(data, data2)

        if copy_data:
            data = copy.deepcopy(data)
            if data2 is not None:
                data2 = copy.deepcopy(data2)

        if data2 is None:
            data2  = data
        
        n_processes1 = data.n_processes
        n_processes2 = data2.n_processes

        A = np.empty((n_processes1, n_processes2))
        A[:] = np.nan

        self._compute_matrix(A, data, data2)

        # Apply symmetry if the measure is undirected and comparing within the same data
        if not self.get_tag("capability-directed", False) and data is data2:
            li = np.tril_indices(n_processes1, -1)
            A[li] = A.T[li]
        
        # Apply filtering if required
        if i is not None:
            if isinstance(i, int):
                A = A[i:i+1, :]
            else:
                A = A[i, :]
        if j is not None:
            if isinstance(j, int):
                A = A[:, j:j+1]
            else:
                A = A[:, j]
        
        return A

    ## Private API ##
    def _spi(self, data, i=0):
        """
        Internal method to compute univariate SPI

        Parameters
        ----------
        data : Data
            The data to be used for the computation.
        i : int, optional
            The index of the variable to be used. Default is 0.
        
        Returns
        -------
        float
            The computed SPI value.
        
        Notes
        -----
        This method should be overridden by the subclasses to implement.
        """
        raise NotImplementedError("_spi must be implemented in the subclass")
    
    def _spi_mat(self, data, data2=None, i=None, j=None):
        """
        Internal method to compute the SPI matrix

        Parameters
        ----------
        data : Data
            The data to be used for the computation.
        data2 : Data, optional
            The second data to be used for the computation. Default is None.
        i : int, optional
            Row index filter of the output matrix. Default is None.
        j : int, optional
            Column index filter of the output matrix. Default is None.

        Returns
        -------
        np.ndarray
            The computed SPI matrix.

        Notes
        -----
        This method should be overridden by the subclasses to implement.
        """
        raise NotImplementedError("_spi_mat must be implemented in the subclass")
    
    def _compute_matrix(self, A, data, data2):
        """
        Internal method to compute the SPI matrix

        Parameters
        ----------
        A : np.ndarray
            The matrix to be filled with the computed SPI values.
        data : Data
            The data to be used for the computation.
        data2 : Data
            The second data to be used for the computation.
        
        Notes
        -----
        This method populates the A matrix with computed SPI values.
        Can be overridden by subclasses to implement specific logic.
        """
        n_processes1 = data.n_processes
        n_processes2 = data2.n_processes

        for row in range(n_processes1):
            for col in range(n_processes2):
                # skip diagonal elements
                if data is data2 and row == col and self.get_tag("capability-directed", False):
                    continue
                A[row, col] = self._compute_pair(data, data2, row, col)
        
    def _compute_pair(self, data, data2, i, j):
        """
        Internal method to compute the SPI value for a pair of processes
        
        Parameters
        ----------
        data : Data
            The data to be used for the computation.
        data2 : Data
            The second data to be used for the computation.
        i : int
            The index of the first process.
        j : int
            The index of the second process.

        Returns
        -------
        float
            The computed SPI value for the pair of processes.
        
        Notes
        -----
        This method should be overridden by the subclasses to implement.
        """
        raise NotImplementedError("_compute_pair must be implemented in the subclass")
    
class DirectedNewSPI(BaseNewSPI):
    """
    Base class for directed SPI measures.
    
    Captures asymmetric relationships between processes.

    Examples
    --------
    >>> from pyspi import DirectedNewSPI
    >>> class MyDirectedSPI(DirectedNewSPI):
    ...     def _compute_pair(self, data, data2, i, j):
    ...         # Implement the directed SPI computation here
    ...         return np.random.rand()
    >>> my_spi = MyDirectedSPI()
    >>> data = Data(np.random.rand(10, 5))
    >>> result = my_spi.spi(data, i=0) 
    >>> print(result)
    """

    _tags = {
        "capability-directed": True,
        "capability-signed": True,
        "identifier": "directed",
        "name": "DirectedPySPI",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class UndirectedNewSPI(BaseNewSPI):
    """
    Base class for undirected SPI measures.
    
    Captures symmetric relationships between processes.

    Examples
    --------
    >>> from pyspi import UndirectedNewSPI
    >>> class MyUndirectedSPI(UndirectedNewSPI):
    ...     def _compute_pair(self, data, data2, i, j):
    ...         # Implement the undirected SPI computation here
    ...         return np.random.rand()
    >>> my_spi = MyUndirectedSPI()
    >>> data = Data(np.random.rand(10, 5))
    >>> result = my_spi.spi(data, i=0) 
    >>> print(result)
    """

    _tags = {
        "capability-directed": False,
        "capability-signed": True,
        "identifier": "undirected",
        "name": "UndirectedPySPI",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SignedNewSPI(BaseNewSPI):
    """
    Base class for signed SPI measures.
    
    Captures both positive and negative relationships between processes.

    Examples
    --------
    >>> from pyspi import SignedNewSPI
    >>> class MySignedSPI(SignedNewSPI):
    ...     def _compute_pair(self, data, data2, i, j):
    ...         # Implement the signed SPI computation here
    ...         return np.random.rand()
    >>> my_spi = MySignedSPI()
    >>> data = Data(np.random.rand(10, 5))
    >>> result = my_spi.spi(data, i=0) 
    >>> print(result)
    """

    _tags = {
        "capability-signed": True,
        "identifier": "signed   ",
        "name": "SignedPySPI",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
### Example Usage to be added later after finalizing the API design ###

        

import numpy as np
import math
from pynats.data import Data

"""
Base class for pairwise dependency measurements.

The child classes should either overload the adjacency method (if it computes the full adjacency)
or the bivariate method if it computes only pairwise measurements
"""
def parse(measure):
    def parsed_measure(self,data,i=None,j=None):
        # Create a data
        if isinstance(data,np.ndarray):
            if data.ndim != 2:
                raise Exception('np.ndarray must be two-dimensional')
            data = Data(data=data,dim_order='ps')
            if data.n_processes == 2:
                i,j = 0,1

        if not isinstance(data,Data):
            raise TypeError('Data must be either numpy.ndarray or pynats.data.Data objects.')

        try:
            return measure(self,data,i=i,j=j)
        except TypeError:
            return measure(self,data)

    return parsed_measure

class directed:
    """ Directed measures
    """

    humanname = 'Bivariate base class'
    name = 'bivariate_base'

    @parse
    def bivariate(self,data,i=None,j=None):
        """ Overload method for getting the pairwise dependencies
        """
        raise Exception("You must overload this method.")

    def adjacency(self,data):
        """ Compute the dependency measures for the entire multivariate dataset
        """
        if not isinstance(data,Data):
            raise TypeError(f'Received data type {type(data)} but expected {type(Data)}.')

        A = np.empty((data.n_processes,data.n_processes))
        A[:] = np.NaN

        for j in range(data.n_processes):
            for i in [ii for ii in range(data.n_processes) if ii != j and math.isnan(A[ii,j])]:
                a, data = self.bivariate(data,i,j)
                try:
                    A[i,j] = a
                except (IndexError):
                    A[i,j] = a[0]
                    A[j,i] = a[1]
        
        return A, data

class undirected(directed):

    humanname = 'Base class'
    name = 'base'

    def ispositive(self):
        return False

    def adjacency(self,data):

        if not isinstance(data,Data):
            raise TypeError(f'Received data type {type(data)} but expected {type(Data)}.')

        A = np.empty((data.n_processes,data.n_processes))
        A[:] = np.NaN

        for j in range(data.n_processes):
            for i in [ii for ii in range(data.n_processes) if ii != j]:
                A[i,j], data = self.bivariate(data,i,j)
                A[j,i] = A[i,j]
        
        return A, data

# Maybe this would be more pythonic as decorators or something?
class positive:
    def ispositive(self):
        return True

class real:
    def ispositive(self):
        return False
import numpy as np
import math
from pynats.data import Data
import copy

"""
Base class for pairwise dependency measurements.

The child classes should either overload the adjacency method (if it computes the full adjacency)
or the bivariate method if it computes only pairwise measurements
"""
def parse_bivariate(measure):
    def parsed_measure(self,data,data2=None,i=None,j=None,inplace=True):
        if not isinstance(data,Data):
            if data2 is None:
                raise TypeError('Input must be either a pynats.data object or two 1D-array inputs.'
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
                i,j = 0,1
            else:
                Warning('i and j not set.')

        return measure(self,data,i=i,j=j)

    return parsed_measure

def parse_multivariate(measure):
    def parsed_measure(self,data,inplace=True):
        if not isinstance(data,Data):
            # Create a pynats.Data object from iterable data object
            try:
                procs = data
                data = Data()
                for p in procs:
                    data.add_process(p)
            except IndexError:
                raise TypeError('Data must be either a pynats.data.Data object or an and iterable of numpy.ndarray''s.')
        elif not inplace:
            # Ensure we don't write over the original
            data = copy.deepcopy(data)

        return measure(self,data)

    return parsed_measure

class directed:
    """ Directed measures
    """

    humanname = 'Bivariate base class'
    name = 'bivariate_base'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """ Overload method for getting the pairwise dependencies
        """
        raise NotImplementedError("Method not yet overloaded.")

    @parse_multivariate
    def adjacency(self,data,inplace=True):
        """ Compute the dependency measures for the entire multivariate dataset
        """
        A = np.empty((data.n_processes,data.n_processes))
        A[:] = np.NaN

        for j in range(data.n_processes):
            for i in [ii for ii in range(data.n_processes) if ii != j]:
                A[i,j] = self.bivariate(data,i=i,j=j)
        return A

class undirected(directed):

    humanname = 'Base class'
    name = 'base'

    def ispositive(self):
        return False

    def adjacency(self,data,inplace=True):
        A = super(undirected,self).adjacency(data,inplace)
        
        li = np.tril_indices(data.n_processes,-1)
        A[li] = A.T[li]
        return A

# Maybe this would be more pythonic as decorators or something?
class positive:
    def ispositive(self):
        return True

class real:
    def ispositive(self):
        return False
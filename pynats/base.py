import numpy as np
from pynats.data import Data
import warnings, copy

"""
Base class for pairwise statistics.

The child classes should either overload the adjacency method (if it computes the full adjacency)
or the bivariate method if it computes only pairwise statisticss
"""

"""
Some parsing functions for decorating so that we can either input the processes directly or use the data structure
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

        return function(self,data,i=i,j=j)

    return parsed_function

def parse_multivariate(function):
    def parsed_function(self,data,inplace=True):
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

        return function(self,data)

    return parsed_function

class directed:
    """ Directed statistics
    """

    humanname = 'Bivariate base class'
    name = 'bivariate_base'
    labels = ['signed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """ Overload method for getting the pairwise dependencies
        """
        raise NotImplementedError("Method not yet overloaded.")

    @parse_multivariate
    def adjacency(self,data):
        """ Compute the dependency statistics for the entire multivariate dataset
        """
        A = np.empty((data.n_processes,data.n_processes))
        A[:] = np.NaN

        for j in range(data.n_processes):
            for i in [ii for ii in range(data.n_processes) if ii != j]:
                A[i,j] = self.bivariate(data,i=i,j=j)
        return A

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

class undirected(directed):

    humanname = 'Base class'
    name = 'base'
    labels = ['unsigned']

    def ispositive(self):
        return False

    @parse_multivariate
    def adjacency(self,data):
        A = super(undirected,self).adjacency(data)
        
        li = np.tril_indices(data.n_processes,-1)
        A[li] = A.T[li]
        return A

# Maybe this would be more pythonic as decorators or something?
class signed:
    def issigned(self):
        return True
        
class unsigned:
    def issigned(self):
        return False
import numpy as np
import math

"""
Base class for pairwise dependency measurements.

The child classes should either overload the adjacency method (if it computes the full adjacency)
or the bivariate method if it computes only pairwise measurements
"""

class directed:
    """ Directed measures
    """

    humanname = 'Base class'
    name = 'base'

    def bivariate(self,x,y,i,j):
        """ Overload method for getting the pairwise dependencies
        """
        raise Exception("You must overload this method.")

    def adjacency(self,z):
        """ Compute the dependency measures for the entire multivariate dataset
        """
        nproc = z.shape[0]
        A = np.empty((nproc,nproc))
        A[:] = np.NaN

        for j in range(nproc):
            targ = z[j].flatten()
            for i in [ii for ii in range(nproc) if ii != j and math.isnan(A[ii,j])]:
                src = z[i].flatten()
                try:
                    a = self.bivariate(src,targ,i,j)
                    try:
                        A[i,j] = a
                    except (IndexError,TypeError):
                        A[i,j] = a[0]
                        A[j,i] = a[1]
                except:
                    pass
        
        return A

    @classmethod
    def preprocess(self,z):
        """ (Optional) implements any pre-processing at the class level that might be used by instances (e.g., in output parameters) or inheriting classes (e.g., in optimising input parameters)
        """
        pass

    def ispositive(self):
        return True

class undirected(directed):

    humanname = 'Base class'
    name = 'base'

    def ispositive(self):
        return False

    def adjacency(self,z):
        nproc = z.shape[0]
        A = np.empty((nproc,nproc))
        A[:] = np.NaN

        for j in range(nproc):
            targ = z[j].flatten()
            for i in [ii for ii in range(nproc) if ii != j]:
                src = z[i].flatten()
                try:
                    A[i,j] = self.bivariate(src,targ,i,j)
                    A[j,i] = A[i,j]
                except:
                    A[i,j] = np.NaN
                    A[j,i] = np.NaN
        
        return A
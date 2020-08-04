import numpy as np

"""
Base class for pairwise dependency measurements.

The child classes should either overload the getadj method (if it computes the full adjacency)
or the getpwd method if it computes only pairwise measurements
"""

class asymmeas:
    """ Asymmetric
    """

    humanname = 'Base class'
    name = 'base'

    def getpwd(self,x,y):
        """ Overload method for getting the pairwise dependencies
        """
        raise Exception("You must overload this method.")

    def ispositive(self):
        return False

    def getadj(self,z):
        """ Compute the dependency measures for the entire multivariate dataset
        """
        nproc = z.shape[0]
        adjacency = np.empty((nproc,nproc))
        adjacency[:] = np.NaN

        for target in range(nproc):
            target_proc = z[target]
            if target_proc.ndim > 1:
                target_proc = target_proc.flatten()

            for source in [x for x in range(nproc) if x != target]:
                source_proc = z[source]
                if source_proc.ndim > 1:
                    source_proc = source_proc.flatten()
                adjacency[source,target] = self.getpwd(source_proc,target_proc)
        
        return adjacency

class symmeas(asymmeas):

    humanname = 'Base class'
    name = 'base'

    def ispositive(self):
        return False

    def getadj(self,z):
        nproc = z.shape[0]
        adjacency = np.empty((nproc,nproc))
        adjacency[:] = np.NaN

        for target in range(nproc):
            target_proc = z[target]
            if target_proc.ndim > 1:
                    target_proc = target_proc.flatten()

            for source in [x for x in range(nproc) if x != target]:
                source_proc = z[source]
                if source_proc.ndim > 1:
                    source_proc = source_proc.flatten()
                adjacency[source,target] = self.getpwd(source_proc,target_proc)
                adjacency[target,source] = adjacency[source,target]
        
        return adjacency
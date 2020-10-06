from nilearn import connectome as fc
from sklearn import covariance as cov
from scipy import stats
from scipy import spatial
import numpy as np
from . import base
from hyppo.independence import MGC, Dcorr, HHG, Hsic
import warnings

class connectivity(base.undirected):
    """ Base class for (functional) connectivity-based measures 
    
    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    humanname = "Pearon's product-moment correlation coefficient"

    _ledoit_wolf = cov.LedoitWolf
    _empirical = cov.EmpiricalCovariance
    _shrunk = cov.ShrunkCovariance
    _oas = cov.OAS

    def __init__(self, kind,
                 cov_estimator='empirical'):
        self.name = 'fc' + '_' + kind + '_' + cov_estimator
        
        self._cov_estimator = eval('self._' + cov_estimator + '()')
        self._kind = kind

    def adjacency(self,z):
        fc_measure = fc.ConnectivityMeasure(cov_estimator=self._cov_estimator,
                                                     kind=self._kind)

        z = np.transpose(z)
        fc_matrix = fc_measure.fit_transform([z])[0]
        np.fill_diagonal(fc_matrix,np.nan)
        return np.square(fc_matrix)

class pearsonr(connectivity):

    humanname = "Pearson's product-moment correlation"

    def __init__(self,cov_estimator='empirical'):
        super(pearsonr,self).__init__(kind='correlation',cov_estimator=cov_estimator)
        self.name = 'pearsonr' + '_' + cov_estimator

class pcor(connectivity):

    humanname = "Partial correlation"

    def __init__(self,cov_estimator='empirical'):
        super(pcor,self).__init__(kind='partial correlation',cov_estimator=cov_estimator)
        self.name = 'pcor' + '_' + cov_estimator

class tangent(connectivity):

    humanname = "Tangent"

    def __init__(self,cov_estimator='empirical'):
        super(tangent,self).__init__('tangent',cov_estimator=cov_estimator)
        self.name = 'tangent' + '_' + cov_estimator

class covariance(connectivity):

    humanname = "Covariance"
    name = "cov"

    def __init__(self,cov_estimator='empirical'):
        super(covariance,self).__init__('covariance',cov_estimator=cov_estimator)
        self.name = 'cov' + '_' + cov_estimator

class precision(connectivity):

    humanname = "Precision"

    def __init__(self,cov_estimator='empirical'):
        super(precision,self).__init__('precision',cov_estimator=cov_estimator)
        self.name = 'prec' + '_' + cov_estimator

class spearmanr(base.undirected):

    humanname = "Spearman's correlation coefficient"
    name = "spearmanr"
    
    def bivariate(self,x,y,i,j):
        return stats.spearmanr(x,y).correlation ** 2

class kendalltau(base.undirected):

    humanname = "Kendall's tau"
    name = "kendalltau"

    def bivariate(self,x,y,i,j):
        return stats.kendalltau(x,y).correlation ** 2

""" TODO: include optional kernels in each method
"""
class hsic(base.undirected):
    """ Hilbert-Schmidt Independence Criterion (Hsic)
    """

    humanname = "Hilbert-Schmidt Independence Criterion"
    name = 'hsic'

    def bivariate(self,x,y,i,j):
        stat, _ = Hsic().test(x, y, auto=True )
        return stat

class hhg(base.undirected):
    """ Heller-Heller-Gorfine independence criterion
    """

    humanname = "Heller-Heller-Gorfine Independence Criterion"
    name = 'hhc'

    def bivariate(self,x,y,i,j):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _ = HHG().test(x, y, reps=0)
        return stat

class dcorr(base.undirected):
    """ Correlation of distances
    """

    humanname = "Distance correlation"
    name = 'dcorr'

    def bivariate(self,x,y,i,j):
        """
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _ = Dcorr().test(x, y, auto=True, reps=0 )
        return stat

class mgc(base.undirected):
    """ Multi-graph correlation
    """

    humanname = "Multi-scale graph correlation"
    name = "mgc"

    def bivariate(self,x,y,i,j):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = MGC().test(x, y, reps=0 )
        return stat
from nilearn import connectome as fc
from sklearn import covariance as cov
from scipy import stats as stats
from scipy import spatial
import numpy as np
from . import basedep as base

class connectivity(base.undirected):
    """ (Functional) Connectivity-based measures
    
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

    def getadj(self,z):
        fc_measure = fc.ConnectivityMeasure(cov_estimator=self._cov_estimator,
                                                     kind=self._kind)

        z = np.transpose(z)
        fc_matrix = fc_measure.fit_transform([z])[0]
        np.fill_diagonal(fc_matrix,np.nan)
        return fc_matrix

    def ispositive(self):
        return False

class pearsonr(connectivity):

    humanname = "Pearson's product-moment correlation"
    name = "pearsonr"

    def __init__(self,cov_estimator='empirical'):
        super(pearsonr,self).__init__(kind='correlation',cov_estimator=cov_estimator)
        self.name = 'pearsonr' + '_' + cov_estimator

class parcorr(connectivity):

    humanname = "Partial correlation"
    name = "parcorr"

    def __init__(self,cov_estimator='empirical'):
        super(parcorr,self).__init__(kind='partial correlation',cov_estimator=cov_estimator)
        self.name = 'parcorr' + '_' + cov_estimator

class tangent(connectivity):

    humanname = "Tangent"
    name = "tangent"

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
    name = "precision"

    def __init__(self,cov_estimator='empirical'):
        super(precision,self).__init__('precision',cov_estimator=cov_estimator)
        self.name = 'precision' + '_' + cov_estimator

class spearmanr(base.undirected):

    humanname = "Spearman's correlation coefficient"
    name = "spearmanr"
    
    def getpwd(self,x,y):
        return stats.spearmanr(x,y).correlation

class kendalltau(base.undirected):

    humanname = "Kendall's tau"
    name = "kendalltau"

    def getpwd(self,x,y):
        return stats.kendalltau(x,y).correlation

class distance(base.undirected):
    """ Correlation of distances
    
    Information on other covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    humanname = "Pairwise distance"

    def __init__(self,metric='euclidean'):
        self._metric = metric
        self.name = 'cdist' + '_' + metric

    def getadj(self,z):
        """ TODO: this needs to be a correlation, not a distance. see:
        https://arxiv.org/pdf/0803.4101.pdf
        or
        https://en.wikipedia.org/wiki/Distance_correlation#:~:text=In%20statistics%20and%20in%20probability,the%20random%20vectors%20are%20independent.
        """
        m = z.shape[0]
        try:
            if self._metric == 'mahalanobis':
                dist_matrix = spatial.distance.cdist(z,z,'mahalanobis',VI=None)
            else:
                dist_matrix = spatial.distance.cdist(z,z,metric=self._metric)
        except ValueError as err:
            print('cdist failed with metric={}: {}'.format(self._metric,err))
            dist_matrix = np.empty((m,m))
            dist_matrix[:] = np.NaN

        np.fill_diagonal(dist_matrix,np.nan)   
        return dist_matrix
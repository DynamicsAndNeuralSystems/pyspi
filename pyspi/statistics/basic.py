import sklearn.covariance as cov
from scipy import stats, spatial, signal
import numpy as np
from pyspi.base import undirected, directed, parse_bivariate, parse_multivariate, signed, unsigned
import warnings

class covariance_estimators(undirected,signed):
    """ Base class for (functional) connectivity-based statistics 
    
    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    humanname = "Covariance"
    labels = ['basic','unordered','linear','undirected']

    def __init__(self,kind,estimator='EmpiricalCovariance',squared=False):
        paramstr = f'_{estimator}'
        if squared:
            paramstr = '-sq' + paramstr
            self.labels = covariance_estimators.labels + ['unsigned']
            self.issigned = lambda : False
        else:
            self.labels = covariance_estimators.labels + ['signed']
        self.name = self.name + paramstr
        self._squared = squared
        self._estimator = estimator
        self._kind = kind

    def _from_cache(self,data):
        try:
            mycov = data.covariance[self._estimator]
        except (AttributeError,KeyError):
            z = data.to_numpy(squeeze=True).T
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mycov = getattr(cov,self._estimator)().fit(z)
            try:
                data.covariance[self._estimator] = mycov
            except AttributeError:
                data.covariance = {self._estimator: mycov}
        return mycov

    @parse_multivariate
    def adjacency(self,data):
        mycov = self._from_cache(data)
        matrix = getattr(mycov,self._kind+'_')
        np.fill_diagonal(matrix,np.nan)
        if self._squared:
            return np.square(matrix)
        else:
            return matrix

class covariance(covariance_estimators):

    humanname = "Covariance"
    name = 'cov'

    def __init__(self,estimator='EmpiricalCovariance',squared=False):
        super().__init__(kind='covariance',squared=squared,estimator=estimator)

class precision(covariance_estimators):

    humanname = "Precision"
    name = 'prec'

    def __init__(self,estimator='EmpiricalCovariance',squared=False):
        super().__init__(kind='precision',squared=squared,estimator=estimator)

class xcorr(undirected,signed):

    humanname = "Cross correlation"
    labels = ['basic','linear','undirected','temporal']

    def __init__(self,squared=False,statistic='max',sigonly=True):
        self.name = 'xcorr'
        self._squared = squared
        self._statistic = statistic
        self._sigonly = sigonly

        if self._squared:
            self.issigned = lambda : False
            self.name = self.name + '-sq'
            self.labels = xcorr.labels + ['unsigned']
        else:
            self.labels = xcorr.labels + ['signed']
        self.name += f'_{statistic}_sig-{sigonly}'
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):

        T = data.n_observations
        try: 
            r_ij = data.xcorr[(i,j)]
        except (KeyError,AttributeError):
            x, y = data.to_numpy()[[i,j]]

            r_ij = np.squeeze(signal.correlate(x,y,'full'))
            r_ij = r_ij / x.std() / y.std() / (T-1)

            # Truncate to T/4
            r_ij = r_ij[T-T//4:T+T//4]

            try:
                data.xcorr[(i,j)] = r_ij
            except AttributeError:
                data.xcorr = {(i,j): r_ij}
            data.xcorr[(j,i)] = data.xcorr[(i,j)]

        # Truncate at first statistically significant zero (i.e., |r| <= 1.96/sqrt(T))
        if self._sigonly:
            N = len(r_ij)//2
            fzf = np.where(np.abs(r_ij[len(r_ij)//2:]) <= 1.96/np.sqrt(N))[0][0]
            fzr = np.where(np.abs(r_ij[:len(r_ij)//2]) <= 1.96/np.sqrt(N))[0][-1]
            r_ij = r_ij[N-fzr:N+fzf]

        if self._statistic == 'max':
            if self._squared:
                return np.max(r_ij**2)
            else:
                return np.max(r_ij)
        elif self._statistic == 'mean':
            if self._squared:
                return np.mean(r_ij**2)
            else:
                return np.mean(r_ij)
        else:
            raise TypeError(f'Unknown statistic: {self._statistic}') 

class spearmanr(undirected,signed):

    humanname = "Spearman's correlation coefficient"
    name = "spearmanr"
    labels = ['basic','unordered','rank','linear','undirected']

    def __init__(self,squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda : False
            self.name = self.name + '-sq'
            self.labels += ['unsigned']
        else:
            self.labels += ['signed']
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        if self._squared:
            return stats.spearmanr(x,y).correlation ** 2
        else:
            return stats.spearmanr(x,y).correlation

class kendalltau(undirected,signed):

    humanname = "Kendall's tau"
    name = "kendalltau"
    labels = ['basic','unordered','rank','linear','undirected']

    def __init__(self,squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda : False
            self.name = self.name + '-sq'
            self.labels += ['unsigned']
        else:
            self.labels += ['signed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        if self._squared:
            return stats.kendalltau(x,y).correlation ** 2
        else:
            return stats.kendalltau(x,y).correlation
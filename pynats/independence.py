from hyppo.independence import MGC, Dcorr, HHG, Hsic
from cdt.causality.pairwise import ANM, CDS, GNN, IGCI, RECI
from pynats.base import directed, undirected, parse_bivariate, unsigned
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import warnings
import numpy as np

class linearmodel(directed,unsigned):
    humanname = 'Linear model regression'
    name = 'lmfit'
    labels = ['unsigned','model based','unordered','normal','linear','directed']

    def __init__(self,model):
        self.name += f'_{model}'
        self._model = getattr(linear_model,model)

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = self._model().fit(z[i], np.ravel(z[j]))
        y_predict = mdl.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))

class gpmodel(directed,unsigned):    
    humanname = 'Gaussian process regression'
    name = 'gpfit'
    labels = ['unsigned','model based','unordered','normal','nonlinear','directed']

    def __init__(self,kernel='RBF'):
        self.name += f'_{kernel}'
        self._kernel = kernels.ConstantKernel() + kernels.WhiteKernel()
        self._kernel += getattr(kernels,kernel)()

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=self._kernel).fit(z[i], np.ravel(z[j]))
        y_predict = gp.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))

""" TODO: include optional kernels in each method
"""
class hsic(undirected,unsigned):
    """ Hilbert-Schmidt Independence Criterion (Hsic)
    """

    humanname = "Hilbert-Schmidt Independence Criterion"
    name = 'hsic'
    labels = ['independence','unordered','nonlinear','undirected']

    def __init__(self,biased=False):
        self._biased = biased
        if biased:
            self.name += '_biased'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat = Hsic(bias=self._biased).statistic(x,y)
        return stat

class hhg(directed,unsigned):
    """ Heller-Heller-Gorfine independence criterion
    """

    humanname = "Heller-Heller-Gorfine Independence Criterion"
    name = 'hhg'
    labels = ['independence','unordered','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat = HHG().statistic(x,y)
        return stat

class dcorr(undirected,unsigned):
    """ Distance correlation
    """

    humanname = "Distance correlation"
    name = 'dcorr'
    labels = ['independence','unordered','nonlinear','undirected']

    def __init__(self,biased=False):
        self._biased = biased
        if biased:
            self.name += '_biased'
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """
        """
        x, y = data.to_numpy()[[i,j]]
        stat = Dcorr(bias=self._biased).statistic(x,y)
        return stat

class mgc(undirected,unsigned):
    """ Multiscale graph correlation
    """

    humanname = "Multiscale graph correlation"
    name = "mgc"
    labels = ['independence','unordered','nonlinear','undirected']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat = MGC().statistic(x,y)
        return stat

class anm(directed,unsigned):

    humanname = "Additive noise model"
    name = 'anm'
    labels = ['unsigned','model based','causal','unordered','linear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return ANM().anm_score(z[i], z[j])

class cds(directed,unsigned):
    
    humanname = 'Conditional distribution similarity statistic'
    name = 'cds'
    labels = ['unsigned','model based','causal','unordered','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return CDS().cds_score(z[i], z[j])

class reci(directed,unsigned):

    humanname = 'Regression error-based causal inference'
    name = 'reci'
    labels = ['unsigned','causal','unordered','neural network','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return RECI().b_fit_score(z[i], z[j])

class igci(directed,unsigned):

    humanname = 'Information-geometric conditional independence'
    name = 'igci'
    labels = ['unsigned','unordered','infotheory','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return IGCI().predict_proba((z[i],z[j]))
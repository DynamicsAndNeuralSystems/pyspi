from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from pyspi.base import directed, undirected, parse_bivariate, unsigned
import numpy as np
import warnings

class coint(undirected,unsigned):
    
    humanname = "Cointegration"
    name = "coint"
    labels = ['misc','unsigned','temporal','undirected','nonlinear']

    def __init__(self,method='johansen',statistic='trace_stat',
                    det_order=1,k_ar_diff=1,
                    autolag='aic',maxlag=10,trend='c'):
        self._method = method
        self._statistic = statistic
        if method == 'johansen':
            self.name += f'_{method}_{statistic}_order-{det_order}_ardiff-{k_ar_diff}'
            self._det_order = det_order
            self._k_ar_diff = k_ar_diff
        else:
            self._autolag = autolag
            self._maxlag = maxlag
            self._trend = trend
            self.name += f'_{method}_{statistic}_trend-{trend}_autolag-{autolag}_maxlag-{maxlag}'

    @property
    def key(self):
        key = (self._method,)
        if self._method == 'johansen':
            return key + (self._det_order,self._k_ar_diff)
        else:
            return key + (self._autolag,self._maxlag,self._trend)

    def _from_cache(self,data,i,j):
        idx = (i,j)
        try:
            ci = data.coint[self.key][idx]
        except (KeyError,AttributeError):
            z = data.to_numpy(squeeze=True)

            if self._method == 'aeg':
                stats = stattools.coint(z[i],z[j],autolag=self._autolag,maxlag=self._maxlag,trend=self._trend)
                ci = {'tstat': stats[0]}
            else:
                stats = coint_johansen(z[[i,j]].T,det_order=self._det_order,k_ar_diff=self._k_ar_diff)
                ci = {'max_eig_stat': stats.max_eig_stat[0], 'trace_stat': stats.trace_stat[0]}

            try:
                data.coint[self.key][idx] = ci
            except AttributeError:
                data.coint = {self.key: {idx: ci} }
            except KeyError:
                data.coint[self.key] = {idx: ci}
            data.coint[self.key][(j,i)] = ci

        return ci

    # Return the negative t-statistic (proxy for how co-integrated they are)
    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):
        ci = self._from_cache(data,i,j)
        return ci[self._statistic]

class linearmodel(directed,unsigned):
    humanname = 'Linear model regression'
    name = 'lmfit'
    labels = ['misc','unsigned','unordered','normal','linear','directed']

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
    labels = ['misc','unsigned','unordered','normal','nonlinear','directed']

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
import jpype as jp
import numpy as np
from pyspi import utils
from pyspi.base import directed, undirected, parse_univariate, parse_bivariate, unsigned

from oct2py import octave, Struct
import copy
import os
import logging

"""
Contains relevant dependence statistics from the information theory community.
"""
if not jp.isJVMStarted():
    jarloc = os.path.dirname(os.path.abspath(__file__)) + '/../lib/jidt/infodynamics.jar'
    # Change to debug info
    logging.debug(f'Starting JVM with java class {jarloc}.')
    jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=' + jarloc)

class jidt_base(unsigned):
    
    # List of (currently) modifiable parameters
    _NNK_PROP_NAME = 'k'
    _AUTO_EMBED_METHOD_PROP_NAME = 'AUTO_EMBED_METHOD'
    _DYN_CORR_EXCL_PROP_NAME = 'DYN_CORR_EXCL'
    _KERNEL_WIDTH_PROP_NAME = 'KERNEL_WIDTH'
    _K_HISTORY_PROP_NAME = 'k_HISTORY'
    _K_TAU_PROP_NAME = 'k_TAU'
    _L_HISTORY_PROP_NAME = 'l_HISTORY'
    _L_TAU_PROP_NAME = 'l_TAU'
    _K_SEARCH_MAX_PROP_NAME = 'AUTO_EMBED_K_SEARCH_MAX'
    _TAU_SEARCH_MAX_PROP_NAME = 'AUTO_EMBED_TAU_SEARCH_MAX'
    _BIAS_CORRECTION = 'BIAS_CORRECTION'
    _NORMALISE = 'NORMALISE'

    _base_class = jp.JPackage('infodynamics.measures.continuous')

    def __init__(self,estimator='gaussian',
                    kernel_width=0.5,
                    prop_k=4,
                    dyn_corr_excl=None):

        self._estimator = estimator
        self._kernel_width = kernel_width
        self._prop_k = prop_k
        self._dyn_corr_excl = dyn_corr_excl
        self._entropy_calc = self._getcalc('entropy')

        self.name = self.name + '_' + estimator
        if estimator == 'kraskov':
            self.name = self.name + '_NN-{}'.format(prop_k)
            self.labels = self.labels + ['nonlinear']
        elif estimator == 'kernel':
            self.name = self.name + '_W-{}'.format(kernel_width)
            self.labels = self.labels + ['nonlinear']
        elif estimator == 'symbolic':
            if not isinstance(self,transfer_entropy):
                raise NotImplementedError('Symbolic estimator is only available for transfer entropy.')
            self.labels = self.labels + ['symbolic']
            self._dyn_corr_excl = None
            return
        else:
            self.labels = self.labels + ['linear']
            self._dyn_corr_excl = None

        if self._dyn_corr_excl:
            self.name = self.name + '_DCE'

    def __getstate__(self):
        state = dict(self.__dict__)
        try:
            del state['_entropy_calc']
        except KeyError:
            pass
        try:
            del state['_calc']
        except KeyError:
            pass

        if '_entropy_calc' in state.keys() or '_calc' in state.keys():
            logging.info(f'{self.name} contains a calculator still')
        return state

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__.update(state)
        self._entropy_calc = self._getcalc('entropy')

    def __deepcopy__(self,memo):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        for attr in newone.__dict__:
            setattr(newone,attr,copy.deepcopy(getattr(self,attr),memo))
        return newone

    def _setup(self,calc):
        if self._estimator == 'kernel':
            calc.setProperty(self._KERNEL_WIDTH_PROP_NAME,str(self._kernel_width))
        elif self._estimator == 'kraskov':
            calc.setProperty(self._NNK_PROP_NAME,str(self._prop_k))

        calc.setProperty(self._BIAS_CORRECTION, 'false')

        return calc

    def _getkey(self):
        if self._estimator == 'kernel':
            return (self._estimator,self._kernel_width)
        elif self._estimator == 'kraskov':
            return (self._estimator,self._prop_k)
        else:
            return (self._estimator,)

    def _getcalc(self,measure):
        if measure == 'entropy':
            if self._estimator == 'kernel':
                calc = self._base_class.kernel.EntropyCalculatorMultiVariateKernel()
            elif self._estimator == 'kozachenko':
                calc = self._base_class.kozachenko.EntropyCalculatorMultiVariateKozachenko()
            else:
                calc = self._base_class.gaussian.EntropyCalculatorMultiVariateGaussian()
        elif measure == 'mutual_info':
            if self._estimator == 'kernel':
                calc = self._base_class.kernel.MutualInfoCalculatorMultiVariateKernel()
            elif self._estimator == 'kraskov':
                calc = self._base_class.kraskov.MutualInfoCalculatorMultiVariateKraskov1()
            else:
                calc = self._base_class.gaussian.MutualInfoCalculatorMultiVariateGaussian()
        elif measure == 'transfer_entropy':
            if self._estimator == 'kernel':
                calc = self._base_class.kernel.TransferEntropyCalculatorKernel()
            elif self._estimator == 'kraskov':
                calc = self._base_class.kraskov.TransferEntropyCalculatorKraskov()
            else:
                calc = self._base_class.gaussian.TransferEntropyCalculatorGaussian()
        else:
            raise TypeError(f'Unknown measure: {measure}')

        return self._setup(calc)

    # No Theiler window yet (can it be done?)
    @parse_univariate
    def _compute_entropy(self,data,i=None):
        if not hasattr(data,'entropy'):
            data.entropy = {}

        key = self._getkey()
        if key not in data.entropy:
            data.entropy[key] = np.full((data.n_processes,), -np.inf)

        if data.entropy[key][i] == -np.inf:
            x = np.squeeze(data.to_numpy()[i])
            self._entropy_calc.initialise(1)
            self._entropy_calc.setObservations(jp.JArray(jp.JDouble,1)(x))
            data.entropy[key][i] = self._entropy_calc.computeAverageLocalOfObservations()

        return data.entropy[key][i]

    # No Theiler window yet (can it be done?)
    @parse_bivariate
    def _compute_joint_entropy(self,data,i,j):
        if not hasattr(data,'joint_entropy'):
            data.joint_entropy = {}

        key = self._getkey()
        if key not in data.joint_entropy:
            data.joint_entropy[key] = np.full((data.n_processes,data.n_processes), -np.inf)

        if data.joint_entropy[key][i,j] == -np.inf:
            x,y = data.to_numpy()[[i,j]]

            self._entropy_calc.initialise(2)
            self._entropy_calc.setObservations(jp.JArray(jp.JDouble, 2)(np.concatenate([x,y],axis=1)))
            data.joint_entropy[key][i,j] = self._entropy_calc.computeAverageLocalOfObservations()
            data.joint_entropy[key][j,i] = data.joint_entropy[key][i,j]
        
        return data.joint_entropy[key][i,j]

    # No Theiler window yet (can it be done?)
    """
    TODO: match this function with previous ones (perhaps always allow multiple i's and j's)
    """
    def _compute_conditional_entropy(self,X,Y):
        XY = np.concatenate([X,Y],axis=1)

        self._entropy_calc.initialise(XY.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble,XY.ndim)(XY))
        H_XY = self._entropy_calc.computeAverageLocalOfObservations()

        self._entropy_calc.initialise(Y.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble,Y.ndim)(Y))
        H_Y = self._entropy_calc.computeAverageLocalOfObservations()

        return H_XY - H_Y

    def _set_theiler_window(self,data,i,j):
        if self._dyn_corr_excl == 'AUTO':
            if not hasattr(data,'theiler'):
                z = data.to_numpy()
                theiler_window = -np.ones((data.n_processes,data.n_processes))
                # Compute effective sample size for each pair
                for _i in range(data.n_processes):
                    targ = z[_i]
                    for _j in range(_i+1,data.n_processes):
                        src = z[_j]
                        # Init the Theiler window using Bartlett's formula
                        theiler_window[_i,_j] = 2*np.dot(utils.acf(src),utils.acf(targ))
                        theiler_window[_j,_i] = theiler_window[_i,_j]
                data.theiler = theiler_window

            self._calc.setProperty(self._DYN_CORR_EXCL_PROP_NAME,
                                    str(int(data.theiler[i,j])))
        elif self._dyn_corr_excl is not None:
            self._calc.setProperty(self._DYN_CORR_EXCL_PROP_NAME,
                                    str(int(self._dyn_corr_excl)))

class joint_entropy(jidt_base,undirected):

    humanname = 'Joint entropy'
    name = 'je'
    labels = ['unsigned','infotheory','unordered','undirected']

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        return self._compute_joint_entropy(data,i=i,j=j)

class conditional_entropy(jidt_base,directed):

    humanname = 'Conditional entropy'
    name = 'ce'
    labels = ['unsigned','infotheory','unordered','directed']

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        return self._compute_joint_entropy(data,i=i,j=j) - self._compute_entropy(data,i=i)
        
class mutual_info(jidt_base,undirected):
    humanname = "Mutual information"
    name = 'mi'
    labels = ['unsigned','infotheory','unordered','undirected']

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc('mutual_info')

    def __setstate__(self,state):
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc('mutual_info')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):
        """ Compute mutual information between Y and X
        """
        self._set_theiler_window(data,i,j)
        self._calc.initialise(1, 1)
        
        try:
            src, targ = data.to_numpy(squeeze=True)[[i,j]]
            self._calc.setObservations(jp.JArray(jp.JDouble)(src),jp.JArray(jp.JDouble)(targ))
            return self._calc.computeAverageLocalOfObservations()
        except:
            logging.warning('MI calcs failed. Maybe check input data for Cholesky factorisation?')
            return np.NaN

class time_lagged_mutual_info(mutual_info):
    humanname = "Time-lagged mutual information"
    name = 'tlmi'
    labels = ['unsigned','infotheory','temporal','undirected']

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc('mutual_info')

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc('mutual_info')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):
        self._set_theiler_window(data,i,j)
        self._calc.initialise(1, 1)
        try:
            src, targ = data.to_numpy(squeeze=True)[[i,j]]
            src = src[:-1]
            targ = targ[1:]
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(src), jp.JArray(jp.JDouble,1)(targ))
            return self._calc.computeAverageLocalOfObservations()
        except:
            logging.warning('Time-lagged MI calcs failed. Maybe check input data for Cholesky factorisation?')
            return np.NaN

class transfer_entropy(jidt_base,directed):

    humanname = "Transfer entropy"
    name = 'te'
    labels = ['unsigned','embedding','infotheory','temporal','directed']

    def __init__(self,auto_embed_method=None,k_search_max=None,tau_search_max=None,
                        k_history=1,k_tau=1,l_history=1,l_tau=1,**kwargs):

        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

        if 'estimator' not in kwargs.keys() or kwargs['estimator'] == 'gaussian':
            self.name = 'gc'
        super().__init__(**kwargs)
        self._calc = self._getcalc('transfer_entropy')

        # Auto-embedding
        if self._auto_embed_method is not None:
            self._calc.setProperty(self._AUTO_EMBED_METHOD_PROP_NAME,self._auto_embed_method)
            self._calc.setProperty(self._K_SEARCH_MAX_PROP_NAME,str(self._k_search_max))
            if self._estimator != 'kernel':
                self.name = self.name + '_k-max-{}_tau-max-{}'.format(k_search_max,tau_search_max)
                self._calc.setProperty(self._TAU_SEARCH_MAX_PROP_NAME,str(self._tau_search_max))
            else:
                self.name = self.name + '_k-max-{}'.format(k_search_max)
            # Set up calculator
        else:
            self._calc.setProperty(self._K_HISTORY_PROP_NAME,str(k_history))
            if self._estimator != 'kernel':
                self._calc.setProperty(self._K_TAU_PROP_NAME,str(k_tau))
                self._calc.setProperty(self._L_HISTORY_PROP_NAME,str(l_history))
                self._calc.setProperty(self._L_TAU_PROP_NAME,str(l_tau))
                self.name = self.name + '_k-{}_kt-{}_l-{}_lt-{}'.format(k_history,k_tau,l_history,l_tau)
            else:
                self.name = self.name + '_k-{}'.format(k_history)

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc('transfer_entropy')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):
        """
        Compute transfer entropy from i->j
        """
        self._set_theiler_window(data,i,j)
        self._calc.initialise()
        src, targ = data.to_numpy(squeeze=True)[[i,j]]
        try:
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(src), jp.JArray(jp.JDouble,1)(targ))
            return self._calc.computeAverageLocalOfObservations()
        except Exception as err:
            logging.warning(f'TE calcs failed: {err}.')
            return np.NaN

class crossmap_entropy(jidt_base,directed):

    humanname = 'Cross-map entropy'
    name = 'xme'
    labels = ['unsigned','infotheory','temporal','directed']

    def __init__(self,history_length=10,**kwargs):
        super().__init__(**kwargs)
        self.name += f'_k{history_length}'
        self._history_length = history_length

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        src, targ = data.to_numpy(squeeze=True)[[i,j]]
        k = self._history_length
        targ_future = targ[k:]
        src_past = np.expand_dims(src[k-1:-1],axis=1)
        for i in range(2,k):
            src_past = np.append(src_past,np.expand_dims(src[k-i:-i],axis=1),axis=1)

        joint = np.concatenate([src_past,np.expand_dims(targ_future,axis=1)], axis=1)

        self._entropy_calc.initialise(joint.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble, 2)(joint))
        H_xy = self._entropy_calc.computeAverageLocalOfObservations()
        
        self._entropy_calc.initialise(src_past.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble, 2)(src_past))
        H_y = self._entropy_calc.computeAverageLocalOfObservations()

        return H_xy - H_y

class causal_entropy(jidt_base,directed):

    humanname = 'Causally conditioned entropy'
    name = 'cce'
    labels = ['unsigned','infotheory','temporal','directed']

    def __init__(self,n=5,**kwargs):
        super().__init__(**kwargs)
        self._n = n

    def _compute_causal_entropy(self,src,targ):
        mUtils = jp.JPackage('infodynamics.utils').MatrixUtils
        H = 0
        src = np.squeeze(src)
        targ = np.squeeze(targ)
        for i in range(2,self._n):
            Yp = mUtils.makeDelayEmbeddingVector(jp.JArray(jp.JDouble,1)(targ), i-1)[1:]
            Xp = mUtils.makeDelayEmbeddingVector(jp.JArray(jp.JDouble,1)(src), i)
            XYp = np.concatenate([Yp,Xp],axis=1)

            Yf = np.expand_dims(targ[i-1:],1)
            H = H + self._compute_conditional_entropy(Yf,XYp)
        return H

    def _getkey(self):
        return super(causal_entropy,self)._getkey() + (self._n,)

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        if not hasattr(data,'causal_entropy'):
            data.causal_entropy = {}

        key = self._getkey()
        if key not in data.causal_entropy:
            data.causal_entropy[key] = np.full((data.n_processes,data.n_processes), -np.inf)

        if data.causal_entropy[key][i,j] == -np.inf:
            z = data.to_numpy(squeeze=True)
            data.causal_entropy[key][i,j] = self._compute_causal_entropy(z[i],z[j])

        return data.causal_entropy[key][i,j]

class directed_info(causal_entropy,directed):

    humanname = 'Directed information'
    name = 'di'
    labels = ['unsigned','infotheory','temporal','directed']

    def __init__(self,n=10,**kwargs):
        super().__init__(**kwargs)
        self._n = n

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """ Compute directed information from i to j
        """
        # Would prefer to match these two calls
        entropy = self._compute_entropy(data,j)
        causal_entropy = super(directed_info,self).bivariate(data,i=i,j=j)

        return entropy - causal_entropy

class stochastic_interaction(jidt_base,undirected):

    humanname = "Stochastic interaction"
    name = 'si'
    labels = ['unsigned','infotheory','temporal','undirected']

    def __init__(self,delay=1,**kwargs):
        super().__init__(**kwargs)
        self._delay = delay
        self.name += f'_k-{delay}'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):
        x,y = data.to_numpy()[[i,j]]
        xy = np.concatenate([x,y],axis=1)
        tau = self._delay

        H_joint = self._compute_conditional_entropy(xy[tau:],xy[:-tau])
        H_src = self._compute_conditional_entropy(x[tau:],x[:-tau])
        H_targ = self._compute_conditional_entropy(y[tau:],y[:-tau])

        return H_src + H_targ - H_joint

class integrated_information(undirected,unsigned):
    
    humanname = "Integrated information"
    name = "phi"
    labels = ['linear','unsigned','infotheory','temporal','undirected']

    def __init__(self,phitype='star',delay=1,normalization=0):
        self._params = Struct()
        self._params['tau'] = 1
        self._options = Struct()
        self._options['type_of_phi'] = phitype
        self._options['type_of_dist'] = 'Gauss'
        self._options['normalization'] = normalization
        self.name += f'_{phitype}_t-{delay}_norm-{normalization}'
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):
        
        if not octave.exist('phi_comp'):
            path = os.path.dirname(os.path.abspath(__file__)) + '/../lib/PhiToolbox/'
            octave.addpath(octave.genpath(path))

        P = [1, 2]
        Z = data.to_numpy(squeeze=True)[[i,j]]
        
        return octave.phi_comp(Z, P, self._params, self._options)
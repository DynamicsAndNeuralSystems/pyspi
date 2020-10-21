import jpype as jp
import numpy as np
from pynats import utils
from pynats.base import directed, undirected, parse, positive, real
from collections import namedtuple

import warnings

"""
Contains relevant dependence measures from the information theory community.
"""

def _getmiclass(jidt_base_class,estimator):
    if estimator == 'kernel':
        jidt_class = jidt_base_class.MutualInfoCalculatorMultiVariateKernel
    elif estimator == 'kraskov':
        jidt_class = jidt_base_class.MutualInfoCalculatorMultiVariateKraskov1            
    else:
        jidt_class = jidt_base_class.MutualInfoCalculatorMultiVariateGaussian

    return jidt_class()

def _getaisclass(jidt_base_class,estimator):
    if estimator == 'kernel':
        calc_class = jidt_base_class.ActiveInfoStorageCalculatorKernel
    elif estimator == 'kraskov':
        calc_class = jidt_base_class.ActiveInfoStorageCalculatorKraskov            
    else:
        calc_class = jidt_base_class.ActiveInfoStorageCalculatorGaussian
    return calc_class()

def _getteclass(jidt_base_class,estimator):
    if estimator == 'kernel':
        calc_class = jidt_base_class.TransferEntropyCalculatorKernel
    elif estimator == 'kraskov':
        calc_class = jidt_base_class.TransferEntropyCalculatorKraskov            
    else:
        calc_class = jidt_base_class.TransferEntropyCalculatorGaussian
    return calc_class()

def theiler(bivariate):

    @parse
    def compute_window(self,data,i=None,j=None):
        if i is None:
            i = 1
            warnings.warn('Source array not chosen - using first process.')
        if j is None:
            j = 1
            warnings.warn('Target array not chosen - using second process.')

        if self._dyn_corr_excl is not None:
            if not hasattr(data,'theiler'):
                z = data.to_numpy()
                theiler_window = -np.ones((data.n_processes,data.n_processes))
                # Compute effective sample size for each pair
                for i in range(data.n_processes):
                    targ = z[i]
                    for j in range(i+1,data.n_processes):
                        src = z[j]
                        # If needed, initialise the Theiler window for this pair
                        bartlett_var = 2*np.dot(utils.acf(src),utils.acf(targ))
                        theiler_window[i,j] = bartlett_var
                        theiler_window[j,i] = bartlett_var            
                data.theiler = theiler_window

            if self._dyn_corr_excl == 'AUTO':
                self._calc.setProperty(self._DYN_CORR_EXCL_PROP_NAME,
                                        str(int(data.theiler[i,j])))
            else:
                self._calc.setProperty(self._DYN_CORR_EXCL_PROP_NAME,
                                        str(int(self._dyn_corr_excl)))

        return bivariate(self,data,i,j)
    return compute_window

def takens(bivariate):

    @theiler
    def compute_embedding(self,data,i=None,j=None,k_search_max=1,tau_search_max=1,**kwargs):
        # TODO: Allow user to only embed source or target
        if self._auto_embed_method is not None:
            if i is None:
                i = 0
                warnings.warn('Source array not chosen - using first process.')
            if j is None:
                j = 1
                warnings.warn('Target array not chosen - using second process.')


            if not hasattr(data,'embeddings'):
                data.embeddings = []
            
            # If we've already got a calculator that's explored this k_search_max, we can use it
            if len(data.embeddings) > 0:
                ids = [calc >= self._ais_calc for calc in data.embeddings]
                if any(ids):
                    ais_calc = data.embeddings[ids]
                else:
                    ais_calc = self._ais_calc
                    ais_calc.compute_embeddings(data,i,j)
            else:
                ais_calc = self._ais_calc
                ais_calc.compute_embeddings(data,i,j)

            opt_k = np.min([ais_calc._optimal_history[i],k_search_max])
            opt_l = np.min([ais_calc._optimal_history[j],k_search_max])
            opt_ktau, opt_ltau = ais_calc._optimal_timedelay[[i,j]]

            self._calc.setProperty(self._K_HISTORY_PROP_NAME, str(int(opt_k)))
            if self._estimator != 'kernel':
                self._calc.setProperty(self._K_TAU_PROP_NAME, str(int(opt_ktau)))
                self._calc.setProperty(self._L_HISTORY_PROP_NAME, str(int(opt_l)))
                self._calc.setProperty(self._L_TAU_PROP_NAME, str(int(opt_ltau)))

        return bivariate(self,data,i,j)

    return compute_embedding

class jidt_base(positive):
    
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

    def __init__(self,estimator='gaussian',
                    kernel_width=0.5,
                    prop_k=4,
                    dyn_corr_excl=None,
                    getclass=_getmiclass):

        self._estimator = estimator
        self._kernel_width = kernel_width
        self._prop_k = prop_k
        self._dyn_corr_excl = dyn_corr_excl

        if not jp.isJVMStarted():
            jarloc = './pynats/lib/jidt/infodynamics.jar'
            jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=' + jarloc)
        
        jidt_base_class = jp.JPackage('infodynamics.measures.continuous.' + estimator)

        self._calc = getclass(jidt_base_class,estimator)
        self._calc.setProperty('NORMALISE', 'true')
        self._calc.setProperty('BIAS_CORRECTION', 'true')

        self.name = self.name + '_' + estimator
        if estimator == 'kraskov':
            self._calc.setProperty(self._NNK_PROP_NAME,str(prop_k))
            self.name = self.name + '_NN-{}'.format(prop_k)
        elif estimator == 'kernel':
            self._calc.setProperty(self._KERNEL_WIDTH_PROP_NAME,str(kernel_width))
            self.name = self.name + '_W-{}'.format(kernel_width)
        else:
            self._dyn_corr_excl = None

        if self._dyn_corr_excl:
            self.name = self.name + '_DCE'

class active_information_storage(jidt_base):

    _HISTORY_PROP_NAME = 'k_HISTORY'
    _TAU_PROP_NAME = 'TAU'
    name = 'ais'

    def __init__(self,auto_embed_method='MAX_CORR_AIS',k_search_max=1,tau_search_max=1,**kwargs):
        super(active_information_storage, self).__init__(**kwargs,getclass=_getaisclass)

        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max

        self._optimal_history = None
        self._optimal_timedelay = None

    # Overload dir() to get only attributes we care about
    def __dir__(self):
        atts = ['_auto_embed_method', '_estimator', '_dyn_corr_excl']
        if self._estimator == 'kraskov':
            atts = atts.append('_prop_k')
        elif self._estimator == 'kernel':
            atts = atts.append('_kernel_width')
        return atts

    def equivalent(self,other):
        # If the attribute list is different, not the same object
        if dir(other) != dir(self):
            return False
        
        # Otherwise, check each attribute for equality
        for att in dir(self):
            if getattr(self,att) != getattr(other,att):
                return False
        
        return True

    def __ge__(self,other):
        if self.equivalent(other):
            if self._k_search_max >= other._k_search_max and self._tau_search_max == other._tau_search_max:
                return True
        return False

    @theiler
    def compute_embeddings(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        if self._optimal_history is None:
            self._optimal_history = np.zeros((data.n_processes))
            self._optimal_timedelay = np.zeros((data.n_processes))

            for i in range(data.n_processes):
                src = z[i,:]
                self._calc.initialise(1, 1)
                self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()))
                try:
                    self._optimal_history[i] = int(self._calc.getProperty(self._HISTORY_PROP_NAME))
                    self._optimal_timedelay[i] = int(self._calc.getProperty(self._TAU_PROP_NAME))
                except TypeError: 
                    self._optimal_history[i] = int(self._calc.getProperty(self._HISTORY_PROP_NAME).toString())
                    self._optimal_timedelay[i] = int(self._calc.getProperty(self._TAU_PROP_NAME).toString())

class mutual_info(jidt_base,undirected):
    humanname = "Mutual information"
    name = 'mi'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __getstate__(self):
        """ The calculator class is unpickleable
        """
        state = self.__dict__.copy()
        del state['_calc']
        self.__dict__.update(state)

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__.update(state)
        self.__init__(self,estimator=self._estimator,
                        kernel_width=self._kernel_width,
                        prop_k=self._prop_k,
                        dyn_corr_excl=self._dyn_corr_excl)

    @theiler
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy(squeeze=True)
        src = z[i,:]
        targ = z[j,:]
        """ Compute mutual information between Y and X
        """

        self._calc.initialise(1, 1)
        try:
            self._calc.setObservations(jp.JArray(jp.JDouble)(src.tolist()), jp.JArray(jp.JDouble)(targ.tolist()))
            mi = self._calc.computeAverageLocalOfObservations()
        except:
            warnings.warn('MI calcs failed. Maybe check input data for Cholesky factorisation?')
            mi = np.NaN
        return mi, data

class time_lagged_mutual_info(mutual_info):
    humanname = "Time-lagged mutual information"
    name = 'tl_mi'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @theiler
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy(squeeze=True)
        src = z[i,:]
        targ = z[j,:]
        """ Compute mutual information between Y and X
        """

        src = src[:-1]
        targ = targ[1:]
        self._calc.initialise(1, 1)
        try:
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()), jp.JArray(jp.JDouble,1)(targ.tolist()))
            tl_mi = self._calc.computeAverageLocalOfObservations()
        except:
            warnings.warn('Time-lagged MI calcs failed. Maybe check input data for Cholesky factorisation?')
            tl_mi = np.NaN
        return tl_mi, data

class transfer_entropy(jidt_base,directed):

    humanname = "Transfer entropy"
    name = 'te'

    def __init__(self,auto_embed_method=None,k_search_max=None,tau_search_max=None,
                        k_history=1,k_tau=1,l_history=1,l_tau=1,**kwargs):

        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

        super().__init__(**kwargs,getclass=_getteclass)

        # Auto-embedding
        if auto_embed_method is not None:
            self.name = self.name + '_k-max-{}_tau-max-{}'.format(k_search_max,tau_search_max)
            # Set up calculator
            self._ais_calc = active_information_storage(k_search_max=k_search_max,tau_search_max=tau_search_max,**kwargs)
        else:
            self._calc.setProperty(self._K_HISTORY_PROP_NAME,str(k_history))
            if self._estimator != 'kernel':
                self._calc.setProperty(self._K_TAU_PROP_NAME,str(k_tau))
                self._calc.setProperty(self._L_HISTORY_PROP_NAME,str(l_history))
                self._calc.setProperty(self._L_TAU_PROP_NAME,str(l_tau))
                self.name = self.name + '_k-{}_kt-{}_l-{}_lt-{}'.format(k_history,k_tau,l_history,l_tau)
            else:
                self.name = self.name + '_k-{}'.format(k_history)

    def __getstate__(self):
        """ JVMs seem to be unpickleable, so lazy workaround is just to delete the _calc class
        """
        state = self.__dict__.copy()
        del state['_calc']
        self.__dict__.update(state)

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__.update(state)
        self.__init__(self,
                        auto_embed_method=self._auto_embed_method,
                        k_search_max=self._k_search_max,
                        tau_search_max=self._tau_search_max,
                        k_history=self._k_history,
                        k_tau=self._k_tau,
                        l_history=self._l_history,
                        l_tau=self._l_tau,
                        estimator=self._estimator,
                        kernel_width=self._kernel_width,
                        prop_k=self._prop_k,
                        dyn_corr_excl=self._dyn_corr_excl)

    @takens
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy(squeeze=True)
        src = z[i,:]
        targ = z[j,:]
        """ Compute transfer entropy from Y to X for all 
        """

        self._calc.initialise()
        try:
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()), jp.JArray(jp.JDouble,1)(targ.tolist()))
            te = self._calc.computeAverageLocalOfObservations()
        except:
            warnings.warn('TE calcs failed. Maybe check input time series for Cholesky decomposition?')
            te = np.NaN
        return te, data

class stochastic_interaction(jidt_base,undirected):

    humanname = "Stochastic interaction"
    name = 'si'

    def __init__(self,**kwargs):
        super(stochastic_interaction,self).__init__(**kwargs)

    @takens
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy(squeeze=True)
        src = z[i,:]
        targ = z[j,:]
        """ Compute mutual information between Y and X
        """

        self._calc.initialise(1, 1)
        self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()), jp.JArray(jp.JDouble,1)(targ.tolist()))
        return self._calc.computeAverageLocalOfObservations()
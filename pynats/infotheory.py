import jpype as jp
import numpy as np
from . import pynats_utils as utils
from . import base

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

class jidt_base():
    
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

    _theiler_window = None

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

    def _setup(self,i,j):
        if self._dyn_corr_excl is not None:
            if self._dyn_corr_excl == 'AUTO':
                self._calc.setProperty(self._DYN_CORR_EXCL_PROP_NAME,
                                        str(int(jidt_base._theiler_window[i,j])))
            else:
                self._calc.setProperty(self._DYN_CORR_EXCL_PROP_NAME,
                                        str(int(self._dyn_corr_excl)))

    @staticmethod
    def preprocess(z):
        M = z.shape[0]

        # Initialise to empty pairwise matrix if not already done
        if jidt_base._theiler_window is None:
            jidt_base._theiler_window = -np.ones((M,M))

        # Compute effective sample size for each pair
        for i in range(M):
            targ = z[i].flatten()

            for j in range(i+1,M):
                src = z[j]

                # If needed, initialise the Theiler window for this pair
                if jidt_base._theiler_window[i,j] < 0:
                    jidt_base._theiler_window[i,j] = 2*np.dot(utils.acf(src),utils.acf(targ))
                    jidt_base._theiler_window[j,i] = jidt_base._theiler_window[i,j]

    def ispositive(self):
        return True

class mutual_info(jidt_base,base.undirected):
    humanname = "Mutual information"

    def __init__(self,**kwargs):
        self.name = 'mi'
        super(mutual_info,self).__init__(**kwargs)

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

    def bivariate(self,src,targ,i,j):
        """ Compute mutual information between Y and X
        """
        super(mutual_info,self)._setup(i,j)

        self._calc.initialise(1, 1)
        self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()), jp.JArray(jp.JDouble,1)(targ.tolist()))
        return self._calc.computeAverageLocalOfObservations()

class time_lagged_mutual_info(mutual_info):
    humanname = "Time-lagged mutual information"

    def __init__(self,**kwargs):
        self.name = 'tl_mi'
        super(time_lagged_mutual_info,self).__init__(**kwargs)

    def bivariate(self,src,targ,i,j):
        """ Compute mutual information between Y and X
        """
        super(time_lagged_mutual_info,self)._setup(i,j)

        src = src[:-1]
        targ = targ[1:]
        self._calc.initialise(1, 1)
        self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()), jp.JArray(jp.JDouble,1)(targ.tolist()))
        return self._calc.computeAverageLocalOfObservations()

class stochastic_interaction(jidt_base,base.undirected):

    humanname = "Stochastic interaction"

    def __init__(self,**kwargs):
        self.name = 'si'
        super(stochastic_interaction,self).__init__(**kwargs)

    def bivariate(self,src,targ,i,j):
        """ Compute mutual information between Y and X
        """
        super(stochastic_interaction,self)._setup(i,j)

        self._calc.initialise(1, 1)
        self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()), jp.JArray(jp.JDouble,1)(targ.tolist()))
        return self._calc.computeAverageLocalOfObservations()

class active_information_storage(jidt_base):

    _HISTORY_PROP_NAME = 'k_HISTORY'
    _TAU_PROP_NAME = 'TAU'

    name = 'ais'

    def __init__(self,auto_embed_method='MAX_CORR_AIS',k_search_max=1,tau_search_max=1,**kwargs):
        super(active_information_storage, self).__init__(**kwargs,getclass=_getaisclass)

        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max

        self._calc.setProperty(self._AUTO_EMBED_METHOD_PROP_NAME,auto_embed_method)
        self._calc.setProperty(self._K_SEARCH_MAX_PROP_NAME,str(k_search_max))
        self._calc.setProperty(self._TAU_SEARCH_MAX_PROP_NAME,str(tau_search_max))

        self._optimal_history = None
        self._optimal_timedelay = None

    def __gt__(self, other):
        if not isinstance(other, active_information_storage):
            return NotImplemented

        if ( ( self._k_search_max >= other._k_search_max and
                self._tau_search_max >= other._tau_search_max ) and
                self._auto_embed_method == other._auto_embed_method and
                self._estimator == other._estimator and
                self._kernel_width == other._kernel_width and
                self._prop_k == other._prop_k and
                self._dyn_corr_excl == other._dyn_corr_excl ):
                print('Found previously valid AIS analysiser with k={} and tau={}'.format(self._k_search_max,self._tau_search_max))
                return True
        else:
            return False

    def compute_embeddings(self,z):
        if self._optimal_history is None:
            nproc = z.shape[0]
            self._optimal_history = np.zeros((nproc))
            self._optimal_timedelay = np.zeros((nproc))

            for i in range(nproc):
                proc = z[i]
                self._calc.initialise(1, 1)
                self._calc.setObservations(jp.JArray(jp.JDouble,1)(proc.tolist()))
                try:
                    self._optimal_history[i] = int(self._calc.getProperty(self._HISTORY_PROP_NAME))
                    self._optimal_timedelay[i] = int(self._calc.getProperty(self._TAU_PROP_NAME))
                except TypeError: 
                    self._optimal_history[i] = int(self._calc.getProperty(self._HISTORY_PROP_NAME).toString())
                    self._optimal_timedelay[i] = int(self._calc.getProperty(self._TAU_PROP_NAME).toString())


class transfer_entropy(jidt_base,base.directed):

    humanname = "Transfer entropy"
    
    _ais_calcs = []

    def __init__(self,auto_embed_method=None,k_search_max=None,tau_search_max=None,
                        k_history=1,k_tau=1,l_history=1,l_tau=1,**kwargs):

        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

        self.name = 'te'
        super(transfer_entropy, self).__init__(**kwargs,getclass=_getteclass)

        # Auto-embedding
        if auto_embed_method is not None:
            self.name = self.name + '_k-max-{}_tau-max-{}'.format(k_search_max,tau_search_max)

            if tau_search_max is None:
                tau_search_max = 1
            if k_search_max is None:
                k_search_max = 1

            ais_calc = active_information_storage(auto_embed_method=auto_embed_method,
                                                    k_search_max=k_search_max,
                                                    tau_search_max=tau_search_max,
                                                    **kwargs)

            # Check if we already have an equivalent AIS calculator
            mycalc = [calc for calc in transfer_entropy._ais_calcs if calc > ais_calc]

            if len(mycalc) == 0:
                transfer_entropy._ais_calcs.append(ais_calc)
                self._ais_calc = ais_calc
            else:
                self._ais_calc = mycalc[0]
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

    def _setup(self,i,j):
        # Sets up the Theiler windowing (if needed)
        super(transfer_entropy, self)._setup(i,j)
        
        # TODO: Allow user to only embed source or target
        if self._auto_embed_method is not None:
            self._calc.setProperty(self._K_HISTORY_PROP_NAME,
                                    str(min(int(self._ais_calc._optimal_history[j]),self._k_search_max)))
            if self._estimator != 'kernel':
                self._calc.setProperty(self._K_TAU_PROP_NAME,
                                        str(min(int(self._ais_calc._optimal_timedelay[j]),self._tau_search_max)))
                self._calc.setProperty(self._L_HISTORY_PROP_NAME,
                                        str(min(int(self._ais_calc._optimal_history[i]),self._k_search_max)))
                self._calc.setProperty(self._L_TAU_PROP_NAME,
                                        str(min(int(self._ais_calc._optimal_timedelay[i]),self._tau_search_max)))

    @staticmethod
    def preprocess(z):
        jidt_base.preprocess(z)

        for c in transfer_entropy._ais_calcs:
            c.compute_embeddings(z)
            
                

    def bivariate(self,src,targ,i,j,verbose=False):
        """ Compute transfer entropy from Y to X for all 
        """
        self._setup(i,j)

        self._calc.initialise()
        self._calc.setObservations(jp.JArray(jp.JDouble,1)(src.tolist()), jp.JArray(jp.JDouble,1)(targ.tolist()))

        if verbose is True:
            print('Inferred: k={}, ktau={}, l={}, ltau={}'.format(self._calc.getProperty(self._K_HISTORY_PROP_NAME),
                                                                self._calc.getProperty(self._K_TAU_PROP_NAME),
                                                                self._calc.getProperty(self._L_HISTORY_PROP_NAME),
                                                                self._calc.getProperty(self._L_TAU_PROP_NAME),))
        return self._calc.computeAverageLocalOfObservations()
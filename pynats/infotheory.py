from jpype import *
import numpy as np
from . import pynats_utils as utils
from . import basedep as base

"""
Contains relevant dependence measures from the information theory community.
"""

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

    _dyn_corr_excl = None

    def __init__(self,estimator='gaussian',
                    kernel_width=0.5,
                    prop_k=4,
                    dyn_corr_excl=None):

        self._estimator = estimator
        self._kernel_width = kernel_width
        self._prop_k = prop_k
        self._dyn_corr_excl = dyn_corr_excl

        if not isJVMStarted():
            jarloc = './pynats/lib/jidt/infodynamics.jar'
            startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=' + jarloc)
        
        base_class = JPackage('infodynamics.measures.continuous.' + estimator)

        self._calc = self.getclass(base_class,estimator)
        self._calc.setProperty('NORMALISE', 'true')
        self._calc.setProperty('BIAS_CORRECTION', 'true')

        self.name = estimator
        if estimator == 'kraskov':
            self._calc.setProperty(self._NNK_PROP_NAME,str(prop_k))
            self.name = self.name + '_NN-{}'.format(prop_k)
        elif estimator == 'kernel':
            self._calc.setProperty(self._KERNEL_WIDTH_PROP_NAME,str(kernel_width))
            self.name = self.name + '_W-{}'.format(kernel_width)
        else:
            self._dyn_corr_excl = None

    def getclass(self,base_class,estimator):
        if estimator == 'kernel':
            calc_class = base_class.MutualInfoCalculatorMultiVariateKernel
        elif estimator == 'kraskov':
            calc_class = base_class.MutualInfoCalculatorMultiVariateKraskov1            
        else:
            calc_class = base_class.MutualInfoCalculatorMultiVariateGaussian

        return calc_class()

    def preinit(self,src,targ):
        if self._dyn_corr_excl is not None:
            if self._dyn_corr_excl == 'AUTO':
                dce = int(2*np.dot(utils.acf(src),utils.acf(targ)))
            else:
                dce = self._dyn_corr_excl
            self._calc.setProperty(self._DYN_CORR_EXCL_PROP_NAME,str(dce))

    def ispositive(self):
        return True


class mutual_info(jidt_base,base.undirected):
    humanname = "Mutual information"

    def __init__(self,**kwargs):
        super(mutual_info,self).__init__(**kwargs)

        basename = self.name
        self.name = 'mi_{}'.format(basename)

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

    def getpwd(self,src,targ):
        """ Compute mutual information between Y and X
        """
        self.preinit(src,targ)
        self._calc.initialise(1, 1)
        self._calc.setObservations(JArray(JDouble,1)(src.tolist()), JArray(JDouble,1)(targ.tolist()))
        return self._calc.computeAverageLocalOfObservations()

class time_lagged_mutual_info(mutual_info):
    humanname = "Time-lagged mutual information"

    def __init__(self,**kwargs):
        super(time_lagged_mutual_info,self).__init__(**kwargs)

        basename = self.name
        self.name = 'tl_mi_{}'.format(basename)

    def getpwd(self,src,targ):
        """ Compute mutual information between Y and X
        """
        src = src[:-1]
        targ = targ[1:]
        self.preinit(src,targ)
        self._calc.initialise(1, 1)
        self._calc.setObservations(JArray(JDouble,1)(src.tolist()), JArray(JDouble,1)(targ.tolist()))
        return self._calc.computeAverageLocalOfObservations()

class transfer_entropy(jidt_base,base.directed):

    humanname = "Transfer entropy"

    def __init__(self,auto_embed_method=None,k_search_max=1,tau_search_max=1,
                        k_history=1,k_tau=1,l_history=1,l_tau=1,**kwargs):

        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

        super(transfer_entropy, self).__init__(**kwargs)

        basename = self.name
        self.name = 'te_' + basename

        # Auto-embedding
        if auto_embed_method is not None:
            self._calc.setProperty(self._AUTO_EMBED_METHOD_PROP_NAME,auto_embed_method)
            self._calc.setProperty(self._K_SEARCH_MAX_PROP_NAME,str(k_search_max))
            self._calc.setProperty(self._TAU_SEARCH_MAX_PROP_NAME,str(tau_search_max))
            self.name = self.name + '_k-max-{}_tau-max-{}'.format(k_search_max,tau_search_max)
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

    def getclass(self,base_class,estimator):
        if estimator == 'kernel':
            calc_class = base_class.TransferEntropyCalculatorKernel
        elif estimator == 'kraskov':
            calc_class = base_class.TransferEntropyCalculatorKraskov            
        else:
            calc_class = base_class.TransferEntropyCalculatorGaussian

        return calc_class()

    def getpwd(self,src,targ,verbose=False):
        """ Compute transfer entropy from Y to X for all 
        """
        self.preinit(src,targ)
        self._calc.initialise(1)
        self._calc.setObservations(JArray(JDouble,1)(src.tolist()), JArray(JDouble,1)(targ.tolist()))

        if verbose is True:
            print('Inferred: k={}, ktau={}, l={}, ltau={}'.format(self._calc.getProperty(self._K_HISTORY_PROP_NAME),
                                                                self._calc.getProperty(self._K_TAU_PROP_NAME),
                                                                self._calc.getProperty(self._L_HISTORY_PROP_NAME),
                                                                self._calc.getProperty(self._L_TAU_PROP_NAME),))
        return self._calc.computeAverageLocalOfObservations()
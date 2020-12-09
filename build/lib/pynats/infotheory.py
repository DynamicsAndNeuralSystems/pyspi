import jpype as jp
import numpy as np
from pynats import utils
from pynats.base import directed, undirected, parse, positive, real
from collections import namedtuple

import copy
import os
import warnings

if not jp.isJVMStarted():
    jarloc = os.path.dirname(os.path.abspath(__file__)) + '/lib/jidt/infodynamics.jar'
    print(f'Starting JVM with java class {jarloc}.')
    jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=' + jarloc)

"""
Contains relevant dependence measures from the information theory community.
"""

def computeConditionalEntropy(ent_calc,X,Y):
    XY = np.concatenate([X,Y],axis=1)

    ent_calc.initialise(XY.shape[1])
    ent_calc.setObservations(jp.JArray(jp.JDouble,XY.ndim)(XY))
    H_XY = ent_calc.computeAverageLocalOfObservations()

    ent_calc.initialise(Y.shape[1])
    ent_calc.setObservations(jp.JArray(jp.JDouble,Y.ndim)(Y))
    H_Y = ent_calc.computeAverageLocalOfObservations()

    return H_XY - H_Y

def theiler(bivariate):
    @parse
    def compute_window(self,data,i=None,j=None):
        if i is None:
            warnings.warn('Source array not chosen - using first process.')
            i = 0
        if j is None:
            warnings.warn('Target array not chosen - using second process.')
            j = 1

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
                warnings.warn('Source array not chosen - using first process.')
                i = 0
            if j is None:
                warnings.warn('Target array not chosen - using second process.')
                j = 1


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
            self._calc.setProperty(self._K_HISTORY_PROP_NAME, str(int(opt_k)))
            
            if self._estimator != 'kernel':
                opt_l = np.min([ais_calc._optimal_history[j],k_search_max])
                opt_ktau, opt_ltau = ais_calc._optimal_timedelay[[i,j]]
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

        self.name = self.name + '_' + estimator
        if estimator == 'kraskov':
            self.name = self.name + '_NN-{}'.format(prop_k)
        elif estimator == 'kernel':
            self.name = self.name + '_W-{}'.format(kernel_width)
        else:
            self._dyn_corr_excl = None

        if self._dyn_corr_excl:
            self.name = self.name + '_DCE'

    def __getstate__(self):
        state = dict(self.__dict__)
        try:
            del state['_calc'], state['_base_class']
        except KeyError:
            pass
        return state

    def __deepcopy__(self,memo):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        for attr in newone.__dict__:
            setattr(newone,attr,copy.deepcopy(getattr(self,attr),memo))
        return newone

    def initCalc(self,calc):
        if self._estimator == 'kernel':
            calc.setProperty(self._KERNEL_WIDTH_PROP_NAME,str(self._kernel_width))
        elif self._estimator == 'kraskov':
            calc.setProperty(self._NNK_PROP_NAME,str(self._prop_k))

        calc.setProperty(self._NORMALISE, 'true')
        calc.setProperty(self._BIAS_CORRECTION, 'true')

        return calc

    def getEntropyCalculator(self):
        if self._estimator == 'kernel':
            calc = self._base_class.kernel.EntropyCalculatorMultiVariateKernel()
        elif self._estimator == 'kozachenko':
            calc = self._base_class.kozachenko.EntropyCalculatorMultiVariateKozachenko()
        else:
            calc = self._base_class.gaussian.EntropyCalculatorMultiVariateGaussian()
        return self.initCalc(calc)

    def getMultivariateMutualInfoCalculator(self):
        if self._estimator == 'kernel':
            calc = self._base_class.kernel.ConditionalMutualInfoCalculatorMultiVariateKernel()
        elif self._estimator == 'kraskov':
            calc = self._base_class.kraskov.ConditionalMutualInfoCalculatorMultiVariateKraskov1()
        else:
            calc = self._base_class.gaussian.ConditionalMutualInfoCalculatorMultiVariateGaussian()
        return self.initCalc(calc)

    def getMutualInfoCalculator(self):
        if self._estimator == 'kernel':
            calc = self._base_class.kernel.MutualInfoCalculatorMultiVariateKernel()
        elif self._estimator == 'kraskov':
            calc = self._base_class.kraskov.MutualInfoCalculatorMultiVariateKraskov1()
        else:
            calc = self._base_class.gaussian.MutualInfoCalculatorMultiVariateGaussian()
        return self.initCalc(calc)

    def getActiveInfoStorageCalculator(self):
        if self._estimator == 'kernel':
            calc = self._base_class.kernel.ActiveInfoStorageCalculatorKernel()
        elif self._estimator == 'kraskov':
            calc = self._base_class.kraskov.ActiveInfoStorageCalculatorKraskov()
        else:
            calc = self._base_class.gaussian.ActiveInfoStorageCalculatorGaussian()
        return self.initCalc(calc)

    def getTransferEntropyCalculator(self):
        if self._estimator == 'kernel':
            calc = self._base_class.kernel.TransferEntropyCalculatorKernel()
        elif self._estimator == 'kraskov':
            calc = self._base_class.kraskov.TransferEntropyCalculatorKraskov()
        else:
            calc = self._base_class.gaussian.TransferEntropyCalculatorGaussian()
        return self.initCalc(calc)

class active_information_storage(jidt_base):

    _HISTORY_PROP_NAME = 'k_HISTORY'
    _TAU_PROP_NAME = 'TAU'
    name = 'ais'

    def __init__(self,auto_embed_method='MAX_CORR_AIS',k_search_max=1,tau_search_max=1,**kwargs):
        super(active_information_storage, self).__init__(**kwargs)

        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max

        self._optimal_history = None
        self._optimal_timedelay = None

        self._calc = self.getActiveInfoStorageCalculator()

    # Overload dir() to get only attributes we care about
    def __dir__(self):
        atts = ['_auto_embed_method', '_estimator', '_dyn_corr_excl']
        if self._estimator == 'kraskov':
            atts = atts.append('_prop_k')
        elif self._estimator == 'kernel':
            atts = atts.append('_kernel_width')
        return atts

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        self.__dict__.update(state)
        self._calc = self.getActiveInfoStorageCalculator()

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
                src = z[i]
                self._calc.initialise(1, 1)
                self._calc.setObservations(jp.JArray(jp.JDouble,1)(src))
                self._optimal_history[i] = int(str(self._calc.getProperty(self._HISTORY_PROP_NAME)))
                if self._estimator != 'kernel':
                    self._optimal_timedelay[i] = int(str(self._calc.getProperty(self._TAU_PROP_NAME)))

class mutual_info(jidt_base,undirected):
    humanname = "Mutual information"
    name = 'mi'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._calc = self.getMutualInfoCalculator()

    @theiler
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy(squeeze=True)
        src = z[i]
        targ = z[j]
        """ Compute mutual information between Y and X
        """

        self._calc.initialise(1, 1)
        try:
            self._calc.setObservations(jp.JArray(jp.JDouble)(src), jp.JArray(jp.JDouble)(targ))
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
        self._calc = self.getMutualInfoCalculator()

    @theiler
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy(squeeze=True)
        src = z[i]
        targ = z[j]
        """ Compute mutual information between Y and X
        """

        src = src[:-1]
        targ = targ[1:]
        self._calc.initialise(1, 1)
        try:
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(src), jp.JArray(jp.JDouble,1)(targ))
            tl_mi = self._calc.computeAverageLocalOfObservations()
        except:
            warnings.warn('Time-lagged MI calcs failed. Maybe check input data for Cholesky factorisation?')
            tl_mi = np.NaN
        return tl_mi, data

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        self.__dict__.update(state)
        self._calc = self.getMutualInfoCalculator()

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

        super().__init__(**kwargs)
        self._calc = self.getTransferEntropyCalculator()

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

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__.update(state)
        self._calc = self.getTransferEntropyCalculator()

    @takens
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy(squeeze=True)
        src = z[i]
        targ = z[j]
        """ Compute transfer entropy from Y to X for all 
        """

        self._calc.initialise()
        try:
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(src), jp.JArray(jp.JDouble,1)(targ))
            te = self._calc.computeAverageLocalOfObservations()
        except:
            warnings.warn('TE calcs failed. Maybe check input time series for Cholesky decomposition?')
            te = np.NaN
        return te, data

class conditional_entropy(jidt_base,directed):

    humanname = 'Conditional entropy'
    name = 'ce'

    def __init__(self,**kwargs):
        super(conditional_entropy,self).__init__(**kwargs)
        self._calc = self.getEntropyCalculator()

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__ = state
        self._calc = self.getEntropyCalculator()

    @theiler
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        src = z[i]
        targ = z[j]

        if not hasattr(data,'entropy'):
            data.entropy = np.ones((data.n_processes,1)) * -np.inf

        if data.entropy[j] == -np.inf:
            self._calc.initialise(1)
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(np.squeeze(targ)))
            data.entropy[j] = self._calc.computeAverageLocalOfObservations()

        if not hasattr(data,'joint_entropy'):
            data.joint_entropy = np.ones((data.n_processes,data.n_processes)) * -np.inf

        if data.joint_entropy[i,j] == -np.inf:
            self._calc.initialise(2)
            self._calc.setObservations(jp.JArray(jp.JDouble, 2)(np.concatenate([src,targ],axis=1)))
            data.joint_entropy[i,j] = self._calc.computeAverageLocalOfObservations()
            data.joint_entropy[j,i] = data.joint_entropy[i,j]

        return data.joint_entropy[i,j] - data.entropy[i], data

class causal_entropy(jidt_base,directed):

    humanname = 'Causally conditioned entropy'
    name = 'cce'

    def __init__(self,n=5,**kwargs):
        super(causal_entropy,self).__init__(**kwargs)
        self._n = n
        self._calc = self.getEntropyCalculator()

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__.update(state)
        self._calc = self.getEntropyCalculator()

    @staticmethod
    def computeCausalEntropy(calc,n,src,targ):
        mUtils = jp.JPackage('infodynamics.utils').MatrixUtils
        H = 0
        for i in range(2,n):
            Yp = mUtils.makeDelayEmbeddingVector(jp.JArray(jp.JDouble,1)(targ), i-1)[1:]
            Xp = mUtils.makeDelayEmbeddingVector(jp.JArray(jp.JDouble,1)(src), i)
            XYp = np.concatenate([Yp,Xp],axis=1)

            Yf = np.expand_dims(targ[i-1:],1)
            H = H + computeConditionalEntropy(calc,Yf,XYp)
        return H

    @theiler
    def bivariate(self,data,i=None,j=None):
        if not hasattr(data,'causal_entropy'):
            data.causal_entropy = np.ones((data.n_processes,data.n_processes)) * -np.inf

        if data.causal_entropy[i,j] == -np.inf:
            z = data.to_numpy(squeeze=True)
            src = z[i]
            targ = z[j]
            data.causal_entropy[i,j] = self.computeCausalEntropy(self._calc,self._n,src,targ)

        return data.causal_entropy[i,j], data

class directed_info(jidt_base,directed):

    humanname = 'Directed information'
    name = 'di'

    def __init__(self,n=5,**kwargs):
        super(directed_info,self).__init__(**kwargs)
        self._n = n
        self._calc = self.getEntropyCalculator()

    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__.update(state)
        self._calc = self.getEntropyCalculator()

    @theiler
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        src = z[i]
        targ = z[j]
        """ Compute mutual information between Y and X
        """

        if not hasattr(data,'causal_entropy'):
            data.causal_entropy = np.ones((data.n_processes,data.n_processes)) * -np.inf

        if data.causal_entropy[i,j] == -np.inf:
            data.causal_entropy[i,j] = causal_entropy.computeCausalEntropy(self._calc,self._n,src,targ)

        if not hasattr(data,'entropy'):
            data.entropy = np.ones((data.n_processes,1)) * -np.inf

        if data.entropy[i] == -np.inf:
            self._calc.initialise(1)
            self._calc.setObservations(jp.JArray(jp.JDouble,1)(targ))
            data.entropy[i] = self._calc.computeAverageLocalOfObservations()

        return data.entropy[i] - data.causal_entropy[i,j], data

class stochastic_interaction(jidt_base,undirected):

    humanname = "Stochastic interaction"
    name = 'si'

    def __init__(self,history=1,**kwargs):
        super(stochastic_interaction,self).__init__(**kwargs)
        self._history = history
        self._calc = self.getEntropyCalculator()
    
    def __setstate__(self,state):
        """ Re-initialise the calculator
        """
        # Re-initialise
        self.__dict__.update(state)
        self._calc = self.getEntropyCalculator()

    @theiler
    def bivariate(self,data,i=None,j=None,verbose=False):
        z = data.to_numpy()
        src = z[i]
        targ = z[j]
        """ Compute mutual information between Y and X
        """
        k = self._history

        joint = np.concatenate([src,targ],axis=1)

        H_joint = computeConditionalEntropy(self._calc,joint[k:],joint[:-k])
        H_src = computeConditionalEntropy(self._calc,src[k:],src[:-k])
        H_targ = computeConditionalEntropy(self._calc,targ[k:],targ[:-k])

        return H_src + H_targ - H_joint, data
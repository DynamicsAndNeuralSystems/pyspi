import jpype as jp
import numpy as np
from pyspi import utils
try:
    from oct2py import octave, Struct
except Exception:
    pass
import copy
import os
import logging

from pyspi.base import Undirected, Directed, Unsigned, parse_univariate, parse_bivariate

"""
Contains relevant dependence statistics from the information theory community.
"""
# if not jp.isJVMStarted():
#     jarloc = (
#         os.path.dirname(os.path.abspath(__file__)) + "/../lib/jidt/infodynamics.jar"
#     )
#     # Change to debug info
#     logging.debug(f"Starting JVM with java class {jarloc}.")
#     jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)


class JIDTBase(Unsigned):

    # List of (currently) modifiable parameters
    _NNK_PROP_NAME = "k"
    _AUTO_EMBED_METHOD_PROP_NAME = "AUTO_EMBED_METHOD"
    _DYN_CORR_EXCL_PROP_NAME = "DYN_CORR_EXCL"
    _KERNEL_WIDTH_PROP_NAME = "KERNEL_WIDTH"
    _K_HISTORY_PROP_NAME = "k_HISTORY"
    _K_TAU_PROP_NAME = "k_TAU"
    _L_HISTORY_PROP_NAME = "l_HISTORY"
    _L_TAU_PROP_NAME = "l_TAU"
    _K_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_K_SEARCH_MAX"
    _TAU_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_TAU_SEARCH_MAX"
    _BIAS_CORRECTION = "BIAS_CORRECTION"
    _NORMALISE = "NORMALISE"
    _SEED = "NOISE_SEED"

    _base_class = jp.JPackage("infodynamics.measures.continuous")

    def __init__(
        self, estimator="gaussian", kernel_width=0.5, prop_k=4, dyn_corr_excl=None
    ):

        self._estimator = estimator
        self._kernel_width = kernel_width
        self._prop_k = prop_k
        self._dyn_corr_excl = dyn_corr_excl
        self._entropy_calc = self._getcalc("entropy")

        self.identifier = self.identifier + "_" + estimator
        if estimator == "kraskov":
            self.identifier = self.identifier + "_NN-{}".format(prop_k)
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "kernel":
            self.identifier = self.identifier + "_W-{}".format(kernel_width)
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "symbolic":
            if not isinstance(self, TransferEntropy):
                raise NotImplementedError(
                    "Symbolic estimator is only available for transfer entropy."
                )
            self.labels = self.labels + ["symbolic"]
            self._dyn_corr_excl = None
            return
        else:
            self.labels = self.labels + ["linear"]
            self._dyn_corr_excl = None

        if self._dyn_corr_excl:
            self.identifier = self.identifier + "_DCE"

    def __getstate__(self):
        state = dict(self.__dict__)

        unserializable_objects = ["_entropy_calc", "_calc"]

        for k in unserializable_objects:
            if k in state.keys():
                del state[k]

        return state

    def __setstate__(self, state):
        """Re-initialise the calculator"""
        # Re-initialise
        self.__dict__.update(state)
        self._entropy_calc = self._getcalc("entropy")

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        for attr in newone.__dict__:
            setattr(newone, attr, copy.deepcopy(getattr(self, attr), memo))
        return newone

    def _setup(self, calc):
        if self._estimator == "kernel":
            calc.setProperty(self._KERNEL_WIDTH_PROP_NAME, str(self._kernel_width))
        elif self._estimator == "kraskov":
            calc.setProperty(self._NNK_PROP_NAME, str(self._prop_k))

        calc.setProperty(self._BIAS_CORRECTION, "false")
        calc.setProperty(self._SEED, "42")

        return calc

    def _getkey(self):
        if self._estimator == "kernel":
            return (self._estimator, self._kernel_width)
        elif self._estimator == "kraskov":
            return (self._estimator, self._prop_k)
        else:
            return (self._estimator,)

    def _getcalc(self, measure):
        if measure == "entropy":
            if self._estimator == "kernel":
                calc = self._base_class.kernel.EntropyCalculatorMultiVariateKernel()
            elif self._estimator == "kozachenko":
                calc = (
                    self._base_class.kozachenko.EntropyCalculatorMultiVariateKozachenko()
                )
            else:
                calc = self._base_class.gaussian.EntropyCalculatorMultiVariateGaussian()
        elif measure == "MutualInfo":
            if self._estimator == "kernel":
                calc = self._base_class.kernel.MutualInfoCalculatorMultiVariateKernel()
            elif self._estimator == "kraskov":
                calc = (
                    self._base_class.kraskov.MutualInfoCalculatorMultiVariateKraskov1()
                )
            else:
                calc = (
                    self._base_class.gaussian.MutualInfoCalculatorMultiVariateGaussian()
                )
        elif measure == "TransferEntropy":
            if self._estimator == "kernel":
                calc = self._base_class.kernel.TransferEntropyCalculatorKernel()
            elif self._estimator == "kraskov":
                calc = self._base_class.kraskov.TransferEntropyCalculatorKraskov()
            else:
                calc = self._base_class.gaussian.TransferEntropyCalculatorGaussian()
        else:
            raise TypeError(f"Unknown measure: {measure}")

        return self._setup(calc)

    # No Theiler window yet (can it be done?)
    @parse_univariate
    def _compute_entropy(self, data, i=None):
        if not hasattr(data, "entropy"):
            data.entropy = {}

        key = self._getkey()
        if key not in data.entropy:
            data.entropy[key] = np.full((data.n_processes,), -np.inf)

        if data.entropy[key][i] == -np.inf:
            x = np.squeeze(data.to_numpy()[i])

            self._entropy_calc.initialise(1)
            self._entropy_calc.setObservations(jp.JArray(jp.JDouble, 1)(x))

            data.entropy[key][
                i
            ] = self._entropy_calc.computeAverageLocalOfObservations()

        return data.entropy[key][i]

    # No Theiler window is available in the JIDT estimator
    @parse_bivariate
    def _compute_joint_entropy(self, data, i, j):
        if not hasattr(data, "joint_entropy"):
            data.joint_entropy = {}

        key = self._getkey()
        if key not in data.joint_entropy:
            data.joint_entropy[key] = np.full((data.n_processes, data.n_processes), -np.infty)

        if data.joint_entropy[key][i, j] == -np.inf:
            x, y = data.to_numpy()[[i, j]]

            self._entropy_calc.initialise(2)
            self._entropy_calc.setObservations(jp.JArray(jp.JDouble, 2)(np.concatenate([x, y], axis=1)))

            data.joint_entropy[key][i, j] = self._entropy_calc.computeAverageLocalOfObservations()
            data.joint_entropy[key][j, i] = data.joint_entropy[key][i, j]

        return data.joint_entropy[key][i, j]

    # No Theiler window is available in the JIDT estimator
    def _compute_conditional_entropy(self, X, Y):
        XY = np.concatenate([X, Y], axis=1)

        self._entropy_calc.initialise(XY.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble, XY.ndim)(XY))

        H_XY = self._entropy_calc.computeAverageLocalOfObservations()

        self._entropy_calc.initialise(Y.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble, Y.ndim)(Y))

        H_Y = self._entropy_calc.computeAverageLocalOfObservations()

        return H_XY - H_Y

    def _set_theiler_window(self, data, i, j):
        if self._dyn_corr_excl == "AUTO":
            if not hasattr(data, "theiler"):
                z = data.to_numpy()
                theiler_window = -np.ones((data.n_processes, data.n_processes))

                # Compute effective sample size for each pair
                for _i in range(data.n_processes):
                    targ = z[_i]
                    for _j in range(_i + 1, data.n_processes):
                        src = z[_j]

                        # Initialize the Theiler window using Bartlett's formula
                        theiler_window[_i, _j] = 2 * np.dot(
                            utils.acf(src), utils.acf(targ)
                        )
                        theiler_window[_j, _i] = theiler_window[_i, _j]
                data.theiler = theiler_window

            self._calc.setProperty(
                self._DYN_CORR_EXCL_PROP_NAME, str(int(data.theiler[i, j]))
            )
        elif self._dyn_corr_excl is not None:
            self._calc.setProperty(
                self._DYN_CORR_EXCL_PROP_NAME, str(int(self._dyn_corr_excl))
            )


class JointEntropy(JIDTBase, Undirected):

    name = "Joint entropy"
    identifier = "je"
    labels = ["unsigned", "infotheory", "unordered", "undirected"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j)


class ConditionalEntropy(JIDTBase, Directed):

    name = "Conditional entropy"
    identifier = "ce"
    labels = ["unsigned", "infotheory", "unordered", "directed"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j) - self._compute_entropy(
            data, i=i
        )


class MutualInfo(JIDTBase, Undirected):
    name = "Mutual information"
    identifier = "mi"
    labels = ["unsigned", "infotheory", "unordered", "undirected"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        """Compute mutual information between Y and X"""
        self._set_theiler_window(data, i, j)
        self._calc.initialise(1, 1)

        try:
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            self._calc.setObservations(
                jp.JArray(jp.JDouble)(src), jp.JArray(jp.JDouble)(targ)
            )
            return self._calc.computeAverageLocalOfObservations()
        except:
            logging.warning(
                "MI calcs failed. Maybe check input data for Cholesky factorisation?"
            )
            return np.nan


class TimeLaggedMutualInfo(JIDTBase, Directed):
    name = "Time-lagged mutual information"
    identifier = "tlmi"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        """Re-initialise the calculator"""
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        self._set_theiler_window(data, i, j)
        self._calc.initialise(1, 1)
        try:
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            src = src[:-1]
            targ = targ[1:]
            self._calc.setObservations(
                jp.JArray(jp.JDouble, 1)(src), jp.JArray(jp.JDouble, 1)(targ)
            )
            return self._calc.computeAverageLocalOfObservations()
        except:
            logging.warning(
                "Time-lagged MI calcs failed. Maybe check input data for Cholesky factorisation?"
            )
            return np.nan


class TransferEntropy(JIDTBase, Directed):

    name = "Transfer entropy"
    identifier = "te"
    labels = ["unsigned", "embedding", "infotheory", "temporal", "directed"]

    def __init__(
        self,
        auto_embed_method=None,
        k_search_max=None,
        tau_search_max=None,
        k_history=1,
        k_tau=1,
        l_history=1,
        l_tau=1,
        **kwargs,
    ):

        if "estimator" not in kwargs.keys() or kwargs["estimator"] == "gaussian":
            self.identifier = "gc"
        super().__init__(**kwargs)
        self._calc = self._getcalc("TransferEntropy")

        # Auto-embedding
        if auto_embed_method is not None:
            self._calc.setProperty(self._AUTO_EMBED_METHOD_PROP_NAME, auto_embed_method)
            self._calc.setProperty(self._K_SEARCH_MAX_PROP_NAME, str(k_search_max))
            if self._estimator != "kernel":
                self.identifier = self.identifier + "_k-max-{}_tau-max-{}".format(
                    k_search_max, tau_search_max
                )
                self._calc.setProperty(
                    self._TAU_SEARCH_MAX_PROP_NAME, str(tau_search_max)
                )
            else:
                self.identifier = self.identifier + "_k-max-{}".format(k_search_max)
            # Set up calculator
        else:
            self._calc.setProperty(self._K_HISTORY_PROP_NAME, str(k_history))
            if self._estimator != "kernel":
                self._calc.setProperty(self._K_TAU_PROP_NAME, str(k_tau))
                self._calc.setProperty(self._L_HISTORY_PROP_NAME, str(l_history))
                self._calc.setProperty(self._L_TAU_PROP_NAME, str(l_tau))
                self.identifier = self.identifier + "_k-{}_kt-{}_l-{}_lt-{}".format(
                    k_history, k_tau, l_history, l_tau
                )
            else:
                self.identifier = self.identifier + "_k-{}".format(k_history)

    def __setstate__(self, state):
        """Re-initialise the calculator"""
        # Re-initialise
        super().__setstate__(state)
        self.__dict__.update(state)
        self._calc = self._getcalc("TransferEntropy")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        """
        Compute transfer entropy from i->j
        """
        self._set_theiler_window(data, i, j)
        self._calc.initialise()
        src, targ = data.to_numpy(squeeze=True)[[i, j]]
        try:
            self._calc.setObservations(
                jp.JArray(jp.JDouble, 1)(src), jp.JArray(jp.JDouble, 1)(targ)
            )
            return self._calc.computeAverageLocalOfObservations()
        except Exception as err:
            logging.warning(f"TE calcs failed: {err}.")
            return np.nan


class CrossmapEntropy(JIDTBase, Directed):

    name = "Cross-map entropy"
    identifier = "xme"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, history_length=10, **kwargs):
        super().__init__(**kwargs)
        self.identifier += f"_k{history_length}"
        self._history_length = history_length

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        src, targ = data.to_numpy(squeeze=True)[[i, j]]
        k = self._history_length
        targ_future = targ[k:]
        src_past = np.expand_dims(src[k - 1 : -1], axis=1)
        for i in range(2, k):
            src_past = np.append(
                src_past, np.expand_dims(src[k - i : -i], axis=1), axis=1
            )

        joint = np.concatenate([src_past, np.expand_dims(targ_future, axis=1)], axis=1)

        self._entropy_calc.initialise(joint.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble, 2)(joint))
        H_xy = self._entropy_calc.computeAverageLocalOfObservations()

        self._entropy_calc.initialise(src_past.shape[1])
        self._entropy_calc.setObservations(jp.JArray(jp.JDouble, 2)(src_past))
        H_y = self._entropy_calc.computeAverageLocalOfObservations()

        return H_xy - H_y


class CausalEntropy(JIDTBase, Directed):

    name = "Causally conditioned entropy"
    identifier = "cce"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, n=5, **kwargs):
        super().__init__(**kwargs)
        self._n = n

    def _compute_causal_entropy(self, src, targ):

        src = np.squeeze(src)
        targ = np.squeeze(targ)

        m_utils = jp.JPackage("infodynamics.utils").MatrixUtils

        causal_entropy = 0
        for i in range(1, self._n + 1):
            Yp = m_utils.makeDelayEmbeddingVector(jp.JArray(jp.JDouble, 1)(targ), i - 1)[:-1]
            Xp = m_utils.makeDelayEmbeddingVector(jp.JArray(jp.JDouble, 1)(src), i)
            XYp = np.concatenate([Yp, Xp], axis=1)

            Yf = np.expand_dims(targ[i - 1 :], 1)
            causal_entropy = causal_entropy + self._compute_conditional_entropy(Yf, XYp)
        return causal_entropy

    def _getkey(self):
        return super(CausalEntropy, self)._getkey() + (self._n,)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if not hasattr(data, "causal_entropy"):
            data.causal_entropy = {}

        key = self._getkey()
        if key not in data.causal_entropy:
            data.causal_entropy[key] = np.full(
                (data.n_processes, data.n_processes), -np.inf
            )

        if data.causal_entropy[key][i, j] == -np.inf:
            z = data.to_numpy(squeeze=True)
            data.causal_entropy[key][i, j] = self._compute_causal_entropy(z[i], z[j])

        return data.causal_entropy[key][i, j]


class DirectedInfo(CausalEntropy, Directed):

    name = "Directed information"
    identifier = "di"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, n=5, **kwargs):
        super().__init__(**kwargs)
        self._n = n

    def _compute_entropy_rates(self, targ):

        targ = np.squeeze(targ)
        m_utils = jp.JPackage("infodynamics.utils").MatrixUtils

        entropy_rate_sum = 0
        for i in range(1, self._n + 1):
            # Compute entropy for an i-dimensional embedding
            self._entropy_calc.initialise(i)

            Yi = m_utils.makeDelayEmbeddingVector(jp.JArray(jp.JDouble, 1)(targ), i)
            self._entropy_calc.setObservations(Yi)
            entropy_rate_sum = entropy_rate_sum + self._entropy_calc.computeAverageLocalOfObservations() / i

        return entropy_rate_sum

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """Compute directed information from i to j"""

        entropy_rates = self._compute_entropy_rates(data.to_numpy(squeeze=True)[j])
        causal_entropy = super().bivariate(data, i=i, j=j)

        return entropy_rates - causal_entropy


class StochasticInteraction(JIDTBase, Undirected):

    name = "Stochastic interaction"
    identifier = "si"
    labels = ["unsigned", "infotheory", "temporal", "undirected"]

    def __init__(self, delay=1, **kwargs):
        super().__init__(**kwargs)
        self._delay = delay
        self.identifier += f"_k-{delay}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        x, y = data.to_numpy()[[i, j]]
        xy = np.concatenate([x, y], axis=1)
        tau = self._delay

        H_joint = self._compute_conditional_entropy(xy[tau:], xy[:-tau])
        H_src = self._compute_conditional_entropy(x[tau:], x[:-tau])
        H_targ = self._compute_conditional_entropy(y[tau:], y[:-tau])

        return H_src + H_targ - H_joint


class IntegratedInformation(Undirected, Unsigned):

    name = "Integrated information"
    identifier = "phi"
    labels = ["linear", "unsigned", "infotheory", "temporal", "undirected"]

    def __init__(self, phitype="star", delay=1, normalization=0):
        self._params = Struct()
        self._params["tau"] = 1
        self._options = Struct()
        self._options["type_of_phi"] = phitype
        self._options["type_of_dist"] = "Gauss"
        self._options["normalization"] = normalization
        self.identifier += f"_{phitype}_t-{delay}_norm-{normalization}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):

        if not octave.exist("phi_comp"):
            path = os.path.dirname(os.path.abspath(__file__)) + "/../lib/PhiToolbox/"
            octave.addpath(octave.genpath(path))

        P = [1, 2]
        Z = data.to_numpy(squeeze=True)[[i, j]]

        return octave.phi_comp(Z, P, self._params, self._options)

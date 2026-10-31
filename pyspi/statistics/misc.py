import warnings
import inspect
import numpy as np

from scipy.linalg import solve_triangular
from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from mne_connectivity import envelope_correlation
from pyspi.lib.ids.dependence import compute_IDS

from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)


class Cointegration(Directed, Unsigned):
    """Cointegration test statistics.

    ``johansen`` is symmetric by construction: it tests the rank of a VECM
    fitted to the pair, and swapping the columns leaves the trace and maximum
    eigenvalue statistics unchanged.

    ``aeg`` (augmented Engle-Granger) is *not* symmetric: it regresses the first
    series on the second and unit-root-tests the residuals, so swapping the
    arguments changes the residual series and hence the statistic.

    The class is ``Directed`` and reports each orientation as computed, with no
    symmetrisation rule (min, max or mean over orientations have no settled
    convention). ``johansen`` serves ``(j, i)`` from the ``(i, j)`` computation,
    since for it the two orientations are equal.

    For ``aeg``, ``table[i, j]`` is the Engle-Granger t-statistic of process
    *j* regressed on process *i* (regressor on the row, as for ``lmfit``).
    """

    name = "Cointegration"
    identifier = "coint"
    # Johansen fits a VECM and Engle-Granger an OLS regression.
    labels = ["misc", "unsigned", "temporal", "undirected", "linear"]
    _cache_namespace = "coint"

    @property
    def _cache_subkey(self):
        # Cache key matches self.key (per cache lookup in _from_cache).
        if self._method == "johansen":
            return (self._method, self._det_order, self._k_ar_diff)
        return (self._method, self._autolag, self._maxlag, self._trend)

    def __init__(
        self,
        method="johansen",
        statistic="trace_stat",
        det_order=1,
        k_ar_diff=1,
        autolag="aic",
        maxlag=10,
        trend="c",
    ):
        self._method = method
        self._statistic = statistic
        # Structural label follows the estimator, not the class. See the class
        # docstring: johansen is symmetric, aeg is not.
        if method == "aeg":
            self.labels = [l for l in self.labels if l != "undirected"] + ["directed"]
            if statistic == "tstat":
                # A *signed* Engle-Granger t-statistic: more negative is
                # stronger evidence of cointegration, and a positive value
                # means none at all. Signed, so `Calculator._rmmin` and
                # `set_group` do not treat the sign as noise.
                self.issigned = lambda: True
        if method == "johansen":
            self.identifier += (
                f"_{method}_{statistic}_order-{det_order}_ardiff-{k_ar_diff}"
            )
            self._det_order = det_order
            self._k_ar_diff = k_ar_diff
        else:
            self._autolag = autolag
            self._maxlag = maxlag
            self._trend = trend
            self.identifier += (
                f"_{method}_{statistic}_trend-{trend}_autolag-{autolag}_maxlag-{maxlag}"
            )

    @property
    def key(self):
        key = (self._method,)
        if self._method == "johansen":
            return key + (self._det_order, self._k_ar_diff)
        else:
            return key + (self._autolag, self._maxlag, self._trend)

    def _from_cache(self, data, i, j):
        idx = (i, j)
        try:
            ci = data.coint[self.key][idx]
        except (KeyError, AttributeError):
            z = data.to_numpy(squeeze=True)

            if self._method == "aeg":
                # statsmodels regresses its first argument on its second.
                stats = stattools.coint(
                    z[j],
                    z[i],
                    autolag=self._autolag,
                    maxlag=self._maxlag,
                    trend=self._trend,
                )

                ci = {"tstat": stats[0]}
            else:
                stats = coint_johansen(
                    z[[i, j]].T, det_order=self._det_order, k_ar_diff=self._k_ar_diff
                )

                ci = {
                    "max_eig_stat": stats.max_eig_stat[0],
                    "trace_stat": stats.trace_stat[0],
                }

            try:
                data.coint[self.key][idx] = ci
            except AttributeError:
                data.coint = {self.key: {idx: ci}}
            except KeyError:
                data.coint[self.key] = {idx: ci}
            if self._method == "johansen":
                # Provably orientation-independent, so serving (j, i) from the
                # same computation is an optimisation, not an assumption. aeg is
                # orientation-dependent and must not be aliased.
                data.coint[self.key][(j, i)] = ci

        return ci

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        ci = self._from_cache(data, i, j)
        return ci[self._statistic]


class LinearModel(Directed, Unsigned):
    """Mean squared error of a linear fit of the column on the row.

    ``table[i, j]`` regresses process *j* on process *i* and returns the
    in-sample MSE, so lower means a better prediction of the column from the
    row. On z-scored data an uninformative regressor gives ``(T - 1) / T``.
    """

    name = "Linear model regression"
    identifier = "lmfit"
    labels = ["misc", "unsigned", "unordered", "normal", "linear", "directed"]

    def __init__(self, model):
        self.identifier += f"_{model}"
        self._model = getattr(linear_model, model)
        # Cache whether model accepts random_state (avoids inspect.signature per pair)
        self._has_random_state = "random_state" in inspect.signature(self._model).parameters

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._has_random_state:
                mdl = self._model(random_state=42).fit(z[i], np.ravel(z[j]))
            else:
                mdl = self._model().fit(z[i], np.ravel(z[j]))
        y_predict = mdl.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))


class GPModel(Directed, Unsigned):
    """Gaussian process regression of process j on process i.

    ``table[i, j]`` is the leave-one-out mean squared error of a GP that
    predicts ``z[j]`` from ``z[i]`` (lower is better; about 1 for an unrelated
    pair of z-scored series). The kernel is ``ConstantKernel() + WhiteKernel()
    + {RBF, DotProduct}()``, with hyperparameters maximising the log marginal
    likelihood; ``DotProduct`` makes it Bayesian linear regression.

    The error is leave-one-out at the fitted hyperparameters (Rasmussen &
    Williams 2006, eq. 5.12). The in-sample error is not used because the RBF
    fit can interpolate the data, driving it to zero on unrelated pairs.
    """

    name = "Gaussian process regression"
    identifier = "gpfit"
    labels = ["misc", "unsigned", "unordered", "normal", "nonlinear", "directed"]

    def __init__(self, kernel="RBF"):
        self.identifier += f"_{kernel}"
        if kernel == "DotProduct":
            # Bayesian linear regression.
            self.labels = [l for l in self.labels if l != "nonlinear"] + ["linear"]
        self._kernel = kernels.ConstantKernel() + kernels.WhiteKernel()
        self._kernel += getattr(kernels, kernel)()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=self._kernel).fit(z[i], np.ravel(z[j]))
        # Leave-one-out residuals y_t - mu_{-t} = [K^-1 y]_t / [K^-1]_tt, from
        # the lower Cholesky factor of K (noise included) the fit already holds.
        L_inv = solve_triangular(gp.L_, np.eye(len(gp.alpha_)), lower=True,
                                 check_finite=False)
        resid = gp.alpha_ / np.einsum("ij,ij->j", L_inv, L_inv)
        return float(np.mean(resid ** 2))


class PowerEnvelopeCorrelation(Undirected, Unsigned):
    name = "Power envelope correlation"
    identifier = "pec"
    labels = ["unsigned", "misc", "undirected"]

    def __init__(self, orth=False, log=False, absolute=False):
        self._orth = False
        if orth:
            self._orth = "pairwise"
            self.identifier += "_orth"
        self._log = log
        if log:
            self.identifier += "_log"
        self._absolute = absolute
        if absolute:
            self.identifier += "_abs"
        # A correlation in [-1, 1]; only the orthogonalised variant with
        # `absolute` is non-negative (MNE ignores `absolute` otherwise).
        if not (self._orth and absolute):
            self.labels = [l for l in self.labels if l != "unsigned"] + ["signed"]
            self.issigned = lambda: True

    @parse_multivariate
    def multivariate(self, data):
        z = np.moveaxis(data.to_numpy(), 2, 0)
        ec = envelope_correlation(
            z, orthogonalize=self._orth, log=self._log, absolute=self._absolute
        )
        adj = np.squeeze(ec.get_data(output="dense"))
        np.fill_diagonal(adj, np.nan)
        return adj

class InterDependenceScore(Undirected, Unsigned):
    name = "Interdependence score"
    identifier = "ids"
    labels = ["unsigned", "misc", "undirected", "nonlinear"]

    def __init__(
            self,
            terms=6,
            pnorm='max',
            bandwidth=0.5
    ):
        self._num_terms = terms
        self._p_norm = pnorm
        self._bandwidth_term = bandwidth


    @parse_multivariate
    def multivariate(self, data):
        # reshape for the compute_IDS function which expects shape (obs, proc)
        z = np.squeeze(data.to_numpy(), axis=2).T
        ids = compute_IDS(z, num_terms=self._num_terms, p_norm=self._p_norm,
                           bandwidth_term=self._bandwidth_term)
        return ids

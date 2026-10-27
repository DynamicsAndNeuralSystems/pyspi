import warnings
import inspect
import numpy as np

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

    The two methods differ in a way the class previously hid.

    ``johansen`` is symmetric by construction: it tests the rank of a VECM
    fitted to the pair, and swapping the columns leaves the trace and maximum
    eigenvalue statistics unchanged (verified to ~3e-14).

    ``aeg`` (augmented Engle-Granger) is *not* symmetric: it regresses the first
    series on the second and unit-root-tests the residuals, so swapping the
    arguments changes the residual series and hence the statistic. Measured on
    random walks the two orientations differ by ~0.8 on average and up to ~1.6,
    against a statistic that typically ranges from -1 to -3. That is a genuine
    orientation dependence, not numerical noise.

    Previously the class declared itself ``Undirected`` and the cache wrote each
    computed value to both ``(i, j)`` and ``(j, i)``, so an asymmetric statistic
    was reported symmetrically and *which* of the two orientations you got
    depended on the order in which the pairs happened to be visited.

    The base is now ``Directed``: each orientation is reported as computed. No
    symmetrisation rule is invented here, because choosing one (min, max, or
    mean over orientations) is a scientific decision with no settled convention,
    and silently picking one is what caused the original problem. ``johansen``
    keeps the cache alias, since for it the two orientations are provably equal.
    """

    name = "Cointegration"
    identifier = "coint"
    labels = ["misc", "unsigned", "temporal", "undirected", "nonlinear"]
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
                # means none at all. Reporting it as unsigned made
                # `Calculator._rmmin` shift the whole column by its minimum and
                # `set_group` correlate it through `abs()`, both of which treat
                # the sign as noise when it is the entire finding. The
                # identifier already says `tstat`; the class now agrees.
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
                stats = stattools.coint(
                    z[i],
                    z[j],
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
                # same computation is an optimisation, not an assumption. For
                # aeg it would be exactly the aliasing bug this class had.
                data.coint[self.key][(j, i)] = ci

        return ci

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        ci = self._from_cache(data, i, j)
        return ci[self._statistic]


class LinearModel(Directed, Unsigned):
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
    name = "Gaussian process regression"
    identifier = "gpfit"
    labels = ["misc", "unsigned", "unordered", "normal", "nonlinear", "directed"]

    def __init__(self, kernel="RBF"):
        self.identifier += f"_{kernel}"
        self._kernel = kernels.ConstantKernel() + kernels.WhiteKernel()
        self._kernel += getattr(kernels, kernel)()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=self._kernel).fit(z[i], np.ravel(z[j]))
        y_predict = gp.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))


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

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

    Orientation of ``aeg``: ``table[i, j]`` is the Engle-Granger t-statistic of
    process *j* regressed on process *i*, i.e. the regressor sits on the row,
    as it does for ``LinearModel`` (``lmfit_*``) and as pyspi's source-by-target
    convention for directed SPIs reads. (Until 3.0.0 the call was
    ``coint(z[i], z[j])``, which regresses the row on the column.)
    """

    name = "Cointegration"
    identifier = "coint"
    # Linear: Johansen fits a VECM and Engle-Granger an OLS regression. The
    # class's linearity overrides the config's when labels merge.
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
                # statsmodels regresses its first argument on its second, so
                # the column j is regressed on the row i: regressor on the row,
                # as for LinearModel. See the class docstring.
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
                # same computation is an optimisation, not an assumption. For
                # aeg it would be exactly the aliasing bug this class had.
                data.coint[self.key][(j, i)] = ci

        return ci

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        ci = self._from_cache(data, i, j)
        return ci[self._statistic]


class LinearModel(Directed, Unsigned):
    """Mean squared error of a linear fit of the column on the row.

    ``table[i, j]`` regresses process *j* on process *i* (regressor on the
    row, as for ``coint_aeg``) and returns the in-sample MSE, so **lower means
    a better linear prediction of the column from the row**. On z-scored data
    the MSE of predicting the mean is ``(T - 1) / T``, the value a useless
    regressor returns (and why ``Lasso``/``ElasticNet`` at ``alpha = 1`` are
    not bundled).
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
    predicts ``z[j]`` from ``z[i]`` (row = predictor, column = target; lower
    is better prediction; about 1, the variance of ``z[j]``, for an unrelated
    pair of z-scored series). The kernel is ``ConstantKernel() + WhiteKernel()
    + {RBF, DotProduct}()`` at sklearn's defaults (bounds 1e-5 to 1e5, one
    optimizer start), its hyperparameters maximising the log marginal
    likelihood. ``DotProduct`` makes it Bayesian linear regression.

    The error is leave-one-out with the hyperparameters fixed at the fit
    (Rasmussen & Williams 2006, eq. 5.12), not in-sample. The RBF term has unit
    amplitude, so at a vanishing length scale it is white noise of variance 1
    and the likelihood cannot tell it from the WhiteKernel. On an unrelated
    pair that interpolating solution is usually the global optimum, and its
    in-sample prediction is y itself: with ten random restarts of the
    optimizer, the in-sample MSE returned before 3.0.0 fell below 0.05 on 41
    of the 68 RBF fixture pairs. The default start reaches it when x is in
    large units (``zscore=False``). Its leave-one-out prediction is ~0, so the
    score is right at either optimum.

    Hyperparameters at a bound are expected rather than a failure: on
    z-scored data the constant (and DotProduct's ``sigma_0``) sits at 1e-5,
    and the RBF length scale goes to 1e5 when there is no relation.
    """

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
        # A correlation coefficient carries its sign: negative means the two
        # envelopes move in antiphase. As unsigned, `Calculator._rmmin` would
        # shift the column by its minimum and `set_group` would correlate it
        # through `abs()`, folding anticorrelation onto correlation. Only the
        # orthogonalised variant with `absolute` is non-negative; MNE ignores
        # `absolute` without orthogonalisation.
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

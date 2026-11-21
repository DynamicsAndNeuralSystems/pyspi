import inspect
import warnings

from mne_connectivity import envelope_correlation
import numpy as np
from scipy import optimize, special
from scipy.linalg import solve_triangular
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import mean_squared_error
from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pyspi.base import (
    Directed,
    Undirected,
    Signed,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.lib.ids.dependence import compute_IDS
from pyspi.utils import fmt_param


class Cointegration(Directed, Unsigned):
    """
    Cointegration test statistic of a pair, Johansen [1]_ or Engle-Granger [2]_.

    ``method='johansen'`` fits a vector error-correction model (VECM) to the
    pair with statsmodels' ``coint_johansen`` and reports its trace or
    maximum-eigenvalue statistic for the null of no cointegrating relation
    (rank 0). Swapping the pair leaves both unchanged, so the result is
    symmetric (undirected) and non-negative, and ``(j, i)`` is served from the
    ``(i, j)`` fit. On stationary input these statistics mainly measure how
    quickly the two processes revert to their means: they fall steeply as the
    processes become more persistent, with or without coupling, so compare
    them only across pairs whose processes have similar autocorrelation.

    ``method='aeg'`` is statsmodels' augmented Engle-Granger test, ``coint``:
    ``A[i, j]`` regresses process ``j`` on process ``i`` (regressor on the row,
    as for ``lmfit``) and is the unit-root t-statistic of the residuals. It is
    not symmetric, and each orientation is reported as computed (directed).
    It is signed: more negative is stronger evidence of cointegration, and a
    positive value means none.

    References
    ----------
    .. [1] S. Johansen, "Estimation and hypothesis testing of cointegration
           vectors in Gaussian vector autoregressive models", Econometrica
           59(6), 1551-1580 (1991).
    .. [2] R. F. Engle, C. W. J. Granger, "Co-integration and error
           correction: representation, estimation, and testing", Econometrica
           55(2), 251-276 (1987).

    Parameters
    ----------
    method : str, optional
        ``'johansen'`` or ``'aeg'``. Default: ``'johansen'``.
    statistic : str, optional
        ``'trace_stat'`` or ``'max_eig_stat'`` for ``'johansen'``;
        ``'tstat'`` for ``'aeg'``. Default: ``'trace_stat'``.
    det_order : int, optional
        Deterministic terms of the Johansen VECM: -1 for none, 0 for a
        constant, 1 for a linear trend. Default: 1.
    k_ar_diff : int, optional
        Number of lagged differences in the Johansen VECM. Default: 1.
    autolag : str or None, optional
        How ``'aeg'`` chooses the lag of its unit-root test: ``'aic'``,
        ``'bic'``, ``'t-stat'``, or None to use ``maxlag`` lags. Default:
        ``'aic'``.
    maxlag : int, optional
        Largest lag ``'aeg'`` considers. Default: 10.
    trend : str, optional
        Deterministic terms of the ``'aeg'`` regression: ``'c'`` (constant),
        ``'ct'`` (constant and linear trend), ``'ctt'`` (and quadratic trend)
        or ``'n'`` (none). Default: ``'c'``.
    """

    name = "Cointegration"
    identifier = "coint"
    # Johansen fits a VECM and Engle-Granger an OLS regression.
    labels = ["misc", "unsigned", "temporal", "undirected", "linear"]
    _cache_namespace = "coint"

    @property
    def _cache_subkey(self):
        # The same key as self.key, which _from_cache looks up.
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
        # The structural label follows the method: johansen is symmetric,
        # aeg is not.
        if method == "aeg":
            self.labels = [l for l in self.labels if l != "undirected"] + ["directed"]
            if statistic == "tstat":
                # Signed, so `Calculator._rmmin` and the SPI-SPI correlations
                # keep the sign.
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
                # Johansen's statistics do not depend on the orientation; aeg's
                # do, so only johansen serves (j, i) from this fit.
                data.coint[self.key][(j, i)] = ci

        return ci

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        ci = self._from_cache(data, i, j)
        return ci[self._statistic]


class LinearModel(Directed, Unsigned):
    """
    Mean squared error of a linear regression of the column on the row.

    ``A[i, j]`` fits process ``j`` on process ``i`` with the scikit-learn
    linear model ``model`` and returns the in-sample mean squared error of the
    prediction of ``x_j``, so lower means the row predicts the column better.
    On z-scored data an uninformative regressor gives about ``(T - 1) / T``.
    Directed and unsigned. Models that take a ``random_state`` get a fixed
    seed.

    Parameters
    ----------
    model : str
        Name of a class in ``sklearn.linear_model``, such as
        ``'LinearRegression'``, ``'Ridge'`` or ``'Lasso'``. Required.
    """

    name = "Linear model regression"
    identifier = "lmfit"
    labels = ["misc", "unsigned", "unordered", "normal", "linear", "directed"]

    def __init__(self, model):
        self.identifier += f"_{model}"
        self._model = getattr(linear_model, model)
        # Checked once here rather than for every pair.
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
    """
    Gaussian process regression of the column on the row [1]_.

    ``A[i, j]`` is the leave-one-out mean squared error of a Gaussian process
    that predicts ``x_j`` from ``x_i`` (lower is better; about 1 for an
    unrelated pair of z-scored series). The kernel is
    ``ConstantKernel() + WhiteKernel() + K()``, with ``K`` named by ``kernel``
    and the hyperparameters maximising the log marginal likelihood;
    ``DotProduct`` makes it Bayesian linear regression. The error is
    leave-one-out at the fitted hyperparameters ([1]_, eq. 5.12), because the
    RBF fit can interpolate the data and drive the in-sample error to zero on
    unrelated pairs. Directed and unsigned.

    References
    ----------
    .. [1] C. E. Rasmussen, C. K. I. Williams, "Gaussian Processes for Machine
           Learning", MIT Press (2006).

    Parameters
    ----------
    kernel : str, optional
        Name of a kernel in ``sklearn.gaussian_process.kernels`` that takes no
        required arguments, such as ``'RBF'`` or ``'DotProduct'``.
        Default: ``'RBF'``.
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
    """
    Correlation of the amplitude envelopes of a pair [1]_.

    The envelope of each series is the modulus of its analytic signal
    (Hilbert transform), and ``A[i, j]`` is the Pearson correlation of the
    envelopes of ``x_i`` and ``x_j``, from mne-connectivity's
    ``envelope_correlation`` on the record as one epoch. With ``orth=True``
    each signal is first orthogonalised to the other, which removes
    zero-lag coupling such as field spread [1]_, and the two orientations are
    averaged. Undirected. Signed, in ``[-1, 1]``, except for the
    orthogonalised variant with ``absolute=True``, which is unsigned.

    References
    ----------
    .. [1] J. F. Hipp, D. J. Hawellek, M. Corbetta, M. Siegel, A. K. Engel,
           "Large-scale cortical correlation structure of spontaneous
           oscillatory activity", Nature Neuroscience 15(6), 884-890 (2012).

    Parameters
    ----------
    orth : bool, optional
        Orthogonalise each pair before taking envelopes. Default: False.
    log : bool, optional
        Correlate the logarithms of the squared envelopes (log power).
        Default: False.
    absolute : bool, optional
        Report ``|r|``. mne-connectivity applies it only with ``orth=True``;
        otherwise it changes the identifier but not the value.
        Default: False.
    """

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
        # Only the orthogonalised variant with `absolute` is non-negative
        # (MNE ignores `absolute` otherwise).
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
    """
    Interdependence score (IDS) of a pair, from correlations of kernel features.

    Each series is mapped to the first ``terms`` terms of the Taylor expansion
    of a Gaussian kernel, ``exp(-b x^2) x^k / sqrt(k!)`` for
    ``k = 0, ..., terms - 1`` with ``b = bandwidth``, and the
    ``terms x terms`` absolute Pearson correlations between the two sets of
    features are reduced to one number by ``pnorm``. Undirected and unsigned,
    in ``[0, 1]``, and sensitive to nonlinear as well as linear dependence. A
    constant feature contributes a correlation of 0. Adapted from
    A. Radhakrishnan's ``interdependence_scores`` package (see
    ``pyspi.lib.ids``).

    Parameters
    ----------
    terms : int, optional
        Number of terms of the expansion. Default: 6.
    pnorm : str or int, optional
        ``'max'`` takes the largest correlation (IDS-max), ``1`` their mean
        and ``2`` their root mean square. Default: ``'max'``.
    bandwidth : float, optional
        The constant ``b`` in the Gaussian kernel. Default: 0.5.
    """

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
        # Every parameter reaches the identifier, but only when it is not the
        # default, so the default SPI is `ids`.
        for tag, value, default in (("terms", terms, 6), ("pnorm", pnorm, "max"),
                                    ("bw", bandwidth, 0.5)):
            if value != default:
                self.identifier += f"_{tag}-{fmt_param(value)}".replace(".", "-")


    @parse_multivariate
    def multivariate(self, data):
        # compute_IDS takes (observations, processes).
        z = np.squeeze(data.to_numpy(), axis=2).T
        ids = compute_IDS(z, num_terms=self._num_terms, p_norm=self._p_norm,
                           bandwidth_term=self._bandwidth_term)
        return ids


# Edge time series: coupling nonstationarity.

# Quadrature grid over x ~ N(0, 1) for the Gaussian topshare null.
_EDGE_X = np.linspace(-9.0, 9.0, 4001)
_EDGE_W = np.exp(-0.5 * _EDGE_X**2) / np.sqrt(2 * np.pi) * (_EDGE_X[1] - _EDGE_X[0])
# frac -> CubicSpline of the null share over theta = arcsin(|r|).
_EDGE_TOPSHARE_NULL = {}


def _edge_gauss_topshare_point(rho, frac):
    """
    Gaussian topshare null at one correlation ``rho`` in [0, 1].

    For ``e = x y`` with ``(x, y)`` standard bivariate Gaussian of correlation
    ``rho``, returns ``E[e 1{|e| > q}] / E|e|`` where ``P(|e| > q) = frac``.
    Conditioning on ``x``, ``y | x ~ N(rho x, 1 - rho^2)``, so both the tail
    probability and the truncated first moment are one-dimensional integrals
    over ``x`` of normal cdfs and pdfs. ``E|e| = (2/pi)(sqrt(1-rho^2) + rho
    arcsin(rho))`` in closed form.
    """
    rho = float(rho)
    if rho >= 1.0:
        # y = x, e = x^2 ~ chi2(1); E[x^2 1{x^2 > q}] = P(chi2(3) > q); E|e| = 1.
        q = special.chdtri(1, frac)
        return float(special.chdtrc(3, q))
    s = np.sqrt(1.0 - rho**2)
    m = rho * _EDGE_X
    ax = np.maximum(np.abs(_EDGE_X), 1e-12)

    def bounds(q):
        a = np.minimum(q / ax, 1e6)
        return (a - m) / s, (-a - m) / s

    def tail(q):
        up, lo = bounds(q)
        return np.sum(_EDGE_W * (special.ndtr(-up) + special.ndtr(lo)))

    q = optimize.brentq(lambda q: tail(q) - frac, 1e-12, 500.0, xtol=1e-13)
    up, lo = bounds(q)
    phi = lambda u: np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    # E[y 1{|y| > a} | x] for y ~ N(m, s^2).
    ey = m * (special.ndtr(-up) + special.ndtr(lo)) + s * (phi(up) - phi(lo))
    num = np.sum(_EDGE_W * _EDGE_X * ey)
    den = 2.0 / np.pi * (s + rho * np.arcsin(rho))
    return float(num / den)


def _edge_gauss_topshare(r, frac):
    """
    Gaussian topshare null, vectorised over ``r`` (odd in ``r``).

    Tabulated once per process and ``frac`` on 129 nodes uniform in
    ``theta = arcsin|r|`` (smooth through ``|r| -> 1``, where the null has a
    ``sqrt(1 - r^2)`` term), then interpolated with a cubic spline.
    """
    from scipy.interpolate import CubicSpline

    key = float(np.round(frac, 12))
    spline = _EDGE_TOPSHARE_NULL.get(key)
    if spline is None:
        theta = np.linspace(0.0, np.pi / 2, 129)
        vals = [_edge_gauss_topshare_point(np.sin(t), key) for t in theta]
        spline = CubicSpline(theta, vals)
        _EDGE_TOPSHARE_NULL[key] = spline
    r = np.asarray(r, dtype=float)
    theta = np.arcsin(np.clip(np.abs(r), 0.0, 1.0))
    return np.sign(r) * spline(theta)


class EdgeTimeSeries(Undirected, Signed):
    r"""
    Coupling nonstationarity from the edge time series of a pair [1]_ [2]_.

    The edge time series of a pair is the frame-wise product of the z-scored
    series, ``e_t = z_x(t) z_y(t)``. Its mean is ``c r`` (below), so it
    resolves Pearson's ``r`` into per-frame contributions. Each statistic is
    referenced to its value for a serially independent Gaussian pair with the
    same ``r``, so it is ~0 in expectation for Gaussian iid data at any
    correlation. This separates nonstationary coupling from the sampling
    spread that windowed correlations show even when the coupling is
    stationary [3]_. For strongly autocorrelated series the finite-sample
    corrections act on the effective sample size rather than ``T``. The
    statistics describe the co-fluctuation structure of the pair's edge time
    series, not coupling strength.

    Conventions: each series is z-scored with ``ddof=1`` (a no-op on data
    z-scored by ``Data``), so ``mean(z^2) = c := (T-1)/T``,
    ``r = sum_t e_t / (T-1)`` and ``m := mean(e) = c r``. Variances are
    ``ddof=0`` over the ``T`` frames. All three statistics are symmetric in
    ``x`` and ``y`` (undirected) and signed; ``excessvar`` and ``topshare``
    are also invariant to permuting time.

    References
    ----------
    .. [1] J. Faskowitz, F. Z. Esfahlani, Y. Jo, O. Sporns, R. F. Betzel,
           "Edge-centric functional network representations of human cerebral
           cortex reveal overlapping system-level architecture", Nature
           Neuroscience 23(12), 1644-1654 (2020).
    .. [2] F. Z. Esfahlani, Y. Jo, J. Faskowitz, L. Byrge, D. P. Kennedy,
           O. Sporns, R. F. Betzel, "High-amplitude cofluctuations in cortical
           activity drive functional connectivity", Proceedings of the
           National Academy of Sciences 117(45), 28393-28401 (2020).
    .. [3] R. Hindriks, M. H. Adhikari, Y. Murayama, M. Ganzetti, D. Mantini,
           N. K. Logothetis, G. Deco, "Can sliding-window correlations reveal
           dynamic functional connectivity in resting-state fMRI?",
           NeuroImage 127, 242-256 (2016).

    Parameters
    ----------
    statistic : str, optional
        One of three, with identifiers in brackets. Default: ``'excessvar'``.

        ``excessvar`` (``edge_excessvar``)
            The excess variance ``var(e) - c^2 g(r)``,
            ``g(r) = 1 + r^2 - (2 + 4 r^2) / (T - 1)``. Isserlis' theorem gives
            ``var(xy) = 1 + rho^2`` for a Gaussian pair; the ``1/(T - 1)`` term
            reduces the plug-in's finite-sample bias to ``O(T^-2)``. Positive =
            burstier than Gaussian (heavy-tailed ``e``: the correlation comes and
            goes, or rides on shared high-amplitude events); negative = steadier
            (sub-Gaussian ``e``). It is the standardised cokurtosis
            ``cum(x, x, y, y)``, so a shared variance modulation with unchanged
            coupling also makes it positive.

        ``topshare`` (``edge_topshare_f-<frac>``)
            The excess share of the cofluctuation carried by the top ``frac`` of
            frames. With ``k = max(1, round(frac T))`` and ``top`` the ``k`` frames
            of largest ``|e_t|``, ``S = sum_{t in top} e_t / sum_t |e_t|`` and
            ``topshare = S - S0(r, k/T)``. ``S`` lies in [-1, 1]; it is a share of
            ``sum |e|`` because ``sum e`` vanishes as ``r -> 0``.
            ``S0(rho, f) = E[e 1{|e| > q}] / E|e|`` for ``e = x y``, ``(x, y)``
            standard bivariate Gaussian with correlation ``rho`` and ``q`` the
            ``(1 - f)`` quantile of ``|e|``, by one-dimensional quadrature
            (``y | x`` is normal). It responds to coupling carried by
            high-amplitude frames (the cofluctuation events of [2]_), not to
            coupling that switches on at ordinary amplitude. Odd under ``y -> -y``,
            like ``r``: for a positively correlated pair, positive = more
            event-driven than a Gaussian pair of the same ``r``, negative =
            coupling spread more evenly over frames; the signs swap for an
            anticorrelated pair.

        ``acf1`` (``edge_acf1``)
            The lag-1 autocorrelation of ``e`` beyond its Gaussian value,
            ``rho_e(1) - [rho_x(1) rho_y(1) + r_xy(1) r_yx(1)] / (1 + r^2)``, with
            ``rho_e(1) = sum_{t<T} (e_t - m)(e_{t+1} - m) / sum_t (e_t - m)^2``,
            ``rho_x(1) = sum_{t<T} z_x(t) z_x(t+1) / (T - 1)`` and the lagged
            cross-correlations ``r_xy(1) = sum_{t<T} z_x(t) z_y(t+1) / (T - 1)``
            and ``r_yx(1)`` (``x`` and ``y`` swapped). The bracket is
            ``corr(e_t, e_{t+1})`` for a stationary jointly Gaussian pair
            (Isserlis) at the sample values, so autocorrelated Gaussian series give
            ~0. Positive = coupling episodes persist longer than that implies;
            negative = they alternate faster.

    frac : float, optional
        Fraction of frames in the top set, ``0 < frac < 1``, for
        ``topshare`` only (identifier suffix ``_f-0-05``). Default: None,
        which ``topshare`` reads as 0.05.
    """

    name = "Edge time series (coupling nonstationarity)"
    identifier = "edge"
    labels = ["misc", "undirected", "nonlinear", "signed"]

    _STATISTICS = ("excessvar", "topshare", "acf1")

    def __init__(self, statistic="excessvar", frac=None):
        if statistic not in self._STATISTICS:
            raise ValueError(
                f"statistic must be one of {self._STATISTICS}, got {statistic!r}."
            )
        self._statistic = statistic
        self.identifier = f"{self.identifier}_{statistic}"
        if statistic == "topshare":
            frac = 0.05 if frac is None else float(frac)
            if not 0.0 < frac < 1.0:
                raise ValueError(f"frac must lie in (0, 1), got {frac!r}.")
            self._frac = frac
            self.identifier += "_f-" + f"{frac:g}".replace(".", "-")
        elif frac is not None:
            raise ValueError(
                f"frac applies only to statistic='topshare', not {statistic!r}."
            )

    @staticmethod
    def _zscore(Z):
        Z = np.asarray(Z, dtype=float)
        Z = Z - Z.mean(axis=1, keepdims=True)
        return Z / Z.std(axis=1, ddof=1, keepdims=True)

    def _excessvar(self, Z):
        T = Z.shape[1]
        c = (T - 1) / T
        m = Z @ Z.T / T
        Z2 = Z**2
        var = Z2 @ Z2.T / T - m**2
        ref = c**2 + m**2 - (2 * c**2 + 4 * m**2) / (T - 1)
        return var - ref

    def _topshare(self, Z):
        M, T = Z.shape
        k = max(1, int(round(self._frac * T)))
        A = np.full((M, M), np.nan)
        for i in range(M - 1):
            E = Z[i] * Z[i + 1:]
            absE = np.abs(E)
            if k < T:
                idx = np.argpartition(-absE, k - 1, axis=1)[:, :k]
                top = np.take_along_axis(E, idx, axis=1).sum(axis=1)
            else:
                top = E.sum(axis=1)
            share = top / absE.sum(axis=1)
            r = E.sum(axis=1) / (T - 1)
            A[i, i + 1:] = share - _edge_gauss_topshare(r, k / T)
        li = np.tril_indices(M, -1)
        A[li] = A.T[li]
        return A

    def _acf1(self, Z):
        T = Z.shape[1]
        m = Z @ Z.T / T
        Z2 = Z**2
        den = Z2 @ Z2.T - T * m**2
        a, b = Z[:, :-1], Z[:, 1:]
        P = a * b
        num = P @ P.T - m * (a @ a.T + b @ b.T) + (T - 1) * m**2
        acf_e = num / den
        C = a @ b.T / (T - 1)  # C[i, j] = sum_t z_i(t) z_j(t+1) / (T-1)
        rho = np.diag(C)
        r = Z @ Z.T / (T - 1)
        ref = (np.outer(rho, rho) + C * C.T) / (1 + r**2)
        return acf_e - ref

    def _matrix(self, Z):
        Z = self._zscore(Z)
        if Z.shape[1] < 3:
            raise ValueError("EdgeTimeSeries needs at least 3 observations.")
        with np.errstate(invalid="ignore", divide="ignore"):
            A = getattr(self, "_" + self._statistic)(Z)
        A = 0.5 * (A + A.T)  # symmetric by construction; remove BLAS round-off
        np.fill_diagonal(A, np.nan)
        return A

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        Z = data.to_numpy(squeeze=True)[[i, j]]
        return self._matrix(Z)[0, 1]

    @parse_multivariate
    def multivariate(self, data):
        return self._matrix(data.to_numpy(squeeze=True))

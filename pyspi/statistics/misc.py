import warnings
import inspect
import numpy as np

from scipy import optimize, special
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
    Signed,
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


# --------------------------------------------------------------------------
# Edge time series: coupling nonstationarity
# --------------------------------------------------------------------------

# Quadrature grid over x ~ N(0, 1) for the Gaussian topshare null.
_EDGE_X = np.linspace(-9.0, 9.0, 4001)
_EDGE_W = np.exp(-0.5 * _EDGE_X**2) / np.sqrt(2 * np.pi) * (_EDGE_X[1] - _EDGE_X[0])
# frac -> CubicSpline of the null share over theta = arcsin(|r|).
_EDGE_TOPSHARE_NULL = {}


def _edge_gauss_topshare_point(rho, frac):
    """Gaussian topshare null at one correlation ``rho`` in [0, 1].

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
    """Gaussian topshare null, vectorised over ``r`` (odd in ``r``).

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
    r"""Coupling nonstationarity from the edge time series.

    The edge time series of a pair is the frame-wise product of the z-scored
    series, ``e_t = z_x(t) z_y(t)`` (Faskowitz et al., 2020; Esfahlani et al.,
    2020). Its mean is ``c r`` (below), so it resolves Pearson's ``r`` into
    per-frame contributions. Each statistic is referenced to its value for a
    serially independent Gaussian pair with the same ``r``, so it is ~0 in
    expectation for Gaussian iid data at any correlation. This separates
    nonstationary coupling from the sampling spread that windowed correlations
    show even when the coupling is stationary (Hindriks et al., 2016). For
    strongly autocorrelated series the finite-sample corrections act on the
    effective sample size rather than ``T``.

    Conventions: each series is z-scored with ``ddof=1`` (a no-op on data
    z-scored by ``Data``), so ``mean(z^2) = c := (T-1)/T``,
    ``r = sum_t e_t / (T-1)`` and ``m := mean(e) = c r``. Variances are
    ``ddof=0`` over the ``T`` frames. All three statistics are symmetric in
    ``x`` and ``y`` (undirected) and signed; ``excessvar`` and ``topshare``
    are also invariant to permuting time.

    Parameters
    ----------
    statistic : {"excessvar", "topshare", "acf1"}
        ``"excessvar"`` (``edge_excessvar``): the excess variance
        ``var(e) - c^2 g(r)``, ``g(r) = 1 + r^2 - (2 + 4 r^2) / (T - 1)``.
        Isserlis' theorem gives ``var(xy) = 1 + rho^2`` for a Gaussian pair;
        the ``1/(T - 1)`` term reduces the plug-in's finite-sample bias to
        ``O(T^-2)``. Positive = burstier than Gaussian (heavy-tailed ``e``:
        the correlation comes and goes, or rides on shared high-amplitude
        events); negative = steadier (sub-Gaussian ``e``). It is the
        standardised cokurtosis ``cum(x, x, y, y)``, so a shared variance
        modulation with unchanged coupling also makes it positive.

        ``"topshare"`` (``edge_topshare_f-<frac>``): the excess share of the
        cofluctuation carried by the top ``frac`` of frames. With
        ``k = max(1, round(frac T))`` and ``top`` the ``k`` frames of largest
        ``|e_t|``, ``S = sum_{t in top} e_t / sum_t |e_t|`` and
        ``topshare = S - S0(r, k/T)``. ``S`` lies in [-1, 1]; it is a share of
        ``sum |e|`` because ``sum e`` vanishes as ``r -> 0``.
        ``S0(rho, f) = E[e 1{|e| > q}] / E|e|`` for ``e = x y``, ``(x, y)``
        standard bivariate Gaussian with correlation ``rho`` and ``q`` the
        ``(1 - f)`` quantile of ``|e|``, by one-dimensional quadrature
        (``y | x`` is normal). It responds to coupling carried by
        high-amplitude frames (Esfahlani et al.'s cofluctuation events), not
        to coupling that switches on at ordinary amplitude. Odd under
        ``y -> -y``, like ``r``: for a positively correlated pair, positive =
        more event-driven than a Gaussian pair of the same ``r``, negative =
        coupling spread more evenly over frames; the signs swap for an
        anticorrelated pair.

        ``"acf1"`` (``edge_acf1``): the lag-1 autocorrelation of ``e`` beyond
        its Gaussian value,
        ``rho_e(1) - [rho_x(1) rho_y(1) + r_xy(1) r_yx(1)] / (1 + r^2)``, with
        ``rho_e(1) = sum_{t<T} (e_t - m)(e_{t+1} - m) / sum_t (e_t - m)^2``,
        ``rho_x(1) = sum_{t<T} z_x(t) z_x(t+1) / (T - 1)`` and the lagged
        cross-correlations ``r_xy(1) = sum_{t<T} z_x(t) z_y(t+1) / (T - 1)``
        and ``r_yx(1)`` (``x`` and ``y`` swapped). The bracket is
        ``corr(e_t, e_{t+1})`` for a stationary jointly Gaussian pair
        (Isserlis) at the sample values, so autocorrelated Gaussian series
        give ~0. Positive = coupling episodes persist longer than that
        implies; negative = they alternate faster.
    frac : float, optional
        Fraction of frames in the top set, ``0 < frac < 1``, for
        ``"topshare"`` only (default 0.05; identifier suffix ``_f-0-05``).

    References
    ----------
    - Faskowitz, Esfahlani, Jo, Sporns & Betzel (2020). Edge-centric functional
      network representations of human cerebral cortex reveal overlapping
      system-level architecture. Nat. Neurosci. 23, 1644-1654.
    - Esfahlani, Jo, Faskowitz, Byrge, Kennedy, Sporns & Betzel (2020).
      High-amplitude cofluctuations in cortical activity drive functional
      connectivity. PNAS 117, 28393-28401.
    - Hindriks, Adhikari, Murayama, Ganzetti, Mantini, Logothetis & Deco
      (2016). Can sliding-window correlations reveal dynamic functional
      connectivity in resting-state fMRI? NeuroImage 127, 242-256.
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

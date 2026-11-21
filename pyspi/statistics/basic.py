import inspect
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats
import sklearn.covariance as cov

from pyspi.base import (
    Directed,
    Undirected,
    Signed,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import require_int


def _spearman_matrix(Z):
    """
    Spearman's rho between the rows of ``Z``: Pearson on average ranks.

    Same values as ``stats.spearmanr(Z, axis=1)``, but always a matrix, and a
    constant row gives NaN only in its own row and column.
    """
    return np.corrcoef(np.apply_along_axis(stats.rankdata, 1, Z))


class Estimators(Undirected, Signed):
    """
    Base class for the SPIs read off a scikit-learn covariance fit.

    The fit of ``estimator`` (a class in ``sklearn.covariance``) is cached as
    ``data.covariance[estimator]`` and shared by ``Covariance``, ``Precision``
    and ``PartialCorrelation``. Estimators that take a ``random_state``
    (``MinCovDet``, ``EllipticEnvelope``) get a fixed seed, and the graphical
    lasso is converged to ``tol=1e-8``, so the fit is reproducible. When
    cross-validation selects an ``alpha`` near 0, ``GraphicalLassoCV`` may not
    converge (silently, as fit warnings are suppressed); on data with a dense
    precision matrix, prefer ``GraphicalLasso`` or ``LedoitWolf``.
    """

    name = "Covariance"
    labels = ["basic", "unordered", "linear", "undirected"]
    _cache_namespace = "covariance"

    @property
    def _cache_subkey(self):
        # kind and squared are applied after the cache lookup.
        return (self._estimator,)

    def __init__(self, kind, estimator="EmpiricalCovariance", squared=False):
        paramstr = f"_{estimator}"
        if squared:
            paramstr = "-sq" + paramstr
            self.labels = Estimators.labels + ["unsigned"]
            self.issigned = lambda: False
        else:
            self.labels = Estimators.labels + ["signed"]
        self.identifier = self.identifier + paramstr
        self._squared = squared
        self._estimator = estimator
        self._kind = kind

    def _from_cache(self, data):
        try:
            mycov = data.covariance[self._estimator]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True).T

            estimator = getattr(cov, self._estimator)
            # MinCovDet and EllipticEnvelope search random subsets of the
            # observations: a fixed seed keeps the fit the same on every run.
            params = inspect.signature(estimator).parameters
            kwargs = {"random_state": 42} if "random_state" in params else {}
            # Converge GraphicalLasso(CV) tightly: at the default tol=1e-4 the
            # value depends on the solver path, which differs between versions.
            if "enet_tol" in params:
                kwargs.update(tol=1e-8, enet_tol=1e-8)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mycov = estimator(**kwargs).fit(z)
            try:
                data.covariance[self._estimator] = mycov
            except AttributeError:
                data.covariance = {self._estimator: mycov}
        return mycov

    def _matrix(self, mycov):
        """Copy of the cached fit's ``M x M`` matrix, so the cache is not modified."""
        return getattr(mycov, self._kind + "_").copy()

    def _pseudo_inverse_rank(self, mycov):
        """
        Rank of the fitted covariance if the precision is its pseudo-inverse, else None.

        scikit-learn sets ``precision_ = pinvh(covariance_)`` for every
        estimator except the graphical lasso, which solves for the precision
        directly. The rank uses the same cutoff as ``pinvh`` and
        ``np.linalg.matrix_rank``: eigenvalues of ``covariance_`` at most
        ``M * eps * max|eigenvalue|`` count as zero.
        """
        if isinstance(mycov, (cov.GraphicalLasso, cov.GraphicalLassoCV)):
            return None
        rank = int(np.linalg.matrix_rank(mycov.covariance_, hermitian=True))
        return rank if rank < mycov.covariance_.shape[0] else None

    @parse_multivariate
    def multivariate(self, data):
        mycov = self._from_cache(data)
        if self._kind != "covariance":
            # A singular covariance has no inverse: its pseudo-inverse gives
            # finite values that are not a precision or partial correlation.
            rank = self._pseudo_inverse_rank(mycov)
            if rank is not None:
                M = mycov.covariance_.shape[0]
                warnings.warn(
                    f"The {self._estimator} covariance estimate is singular "
                    f"(rank {rank} < M = {M}: no more samples than processes, or "
                    f"collinear processes), so {self.identifier} is NaN."
                )
                return np.full((M, M), np.nan)
        matrix = self._matrix(mycov)
        np.fill_diagonal(matrix, np.nan)
        if self._squared:
            return np.square(matrix)
        else:
            return matrix


class Covariance(Estimators):
    """
    Covariance matrix of the processes from a scikit-learn estimator.

    ``A[i, j]`` is the ``(i, j)`` entry of the estimator's ``covariance_``,
    the lag-0 covariance of ``x_i`` and ``x_j``. On data z-scored by ``Data``
    the ``EmpiricalCovariance`` entry is ``(T - 1) / T`` times Pearson's r,
    because the estimator divides by ``T``. Undirected and signed. The fit is
    shared with ``Precision`` and ``PartialCorrelation`` (see ``Estimators``).

    Parameters
    ----------
    estimator : str, optional
        Name of a class in ``sklearn.covariance``, such as
        ``'EmpiricalCovariance'``, ``'LedoitWolf'``, ``'OAS'``,
        ``'ShrunkCovariance'``, ``'GraphicalLasso'``, ``'GraphicalLassoCV'``
        or ``'MinCovDet'``. Default: ``'EmpiricalCovariance'``.
    squared : bool, optional
        Square the result, making it unsigned. Default: False.
    """

    name = "Covariance"
    identifier = "cov"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="covariance", squared=squared, estimator=estimator)


class Precision(Estimators):
    """
    Precision (inverse covariance) matrix of the ``M`` processes.

    ``A[i, j]`` is the ``(i, j)`` entry of ``P = C^-1``, with ``C`` the
    covariance from the scikit-learn ``estimator`` (the graphical lasso
    estimates ``P`` directly). Off the diagonal
    ``P_ij = -pcorr_ij sqrt(P_ii P_jj)``, with ``pcorr_ij`` the partial
    correlation of ``x_i`` and ``x_j`` given all other processes, so a
    positive entry means a negative conditional association. Undirected and
    signed. The fit is shared with ``Covariance`` and ``PartialCorrelation``.
    NaN, with a warning, when the covariance estimate is singular (for
    example with no more samples than processes).

    Parameters
    ----------
    estimator : str, optional
        Name of a class in ``sklearn.covariance``, such as
        ``'EmpiricalCovariance'``, ``'LedoitWolf'``, ``'OAS'``,
        ``'ShrunkCovariance'``, ``'GraphicalLasso'``, ``'GraphicalLassoCV'``
        or ``'MinCovDet'``. Default: ``'EmpiricalCovariance'``.
    squared : bool, optional
        Square the result, making it unsigned. Default: False.
    """

    name = "Precision"
    identifier = "prec"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="precision", squared=squared, estimator=estimator)


class PartialCorrelation(Estimators):
    """
    Partial correlation [1]_.

    With ``P`` the precision (inverse covariance) matrix of the ``M``
    processes from the scikit-learn ``estimator``,
    ``pcorr_ij = -P_ij / sqrt(P_ii P_jj)``: the correlation between ``x_i``
    and ``x_j`` conditional on (after linearly regressing out) all other
    ``M - 2`` processes, at lag 0. For ``EmpiricalCovariance`` this is the
    Pearson correlation of the OLS residuals of ``x_i`` and ``x_j`` on the
    remaining processes; the other estimators shrink or sparsify ``P`` first.

    Undirected and signed, in ``[-1, 1]``, with the opposite sign and the same
    zero pattern as the off-diagonal of ``prec_<estimator>``. The covariance
    fit is shared with ``Covariance`` and ``Precision``. NaN, with a warning,
    when the covariance estimate is singular (for example with no more
    samples than processes).

    References
    ----------
    .. [1] G. Marrelec, A. Krainik, H. Duffau, M. Pélégrini-Issac,
           S. Lehéricy, J. Doyon, H. Benali, "Partial correlation for
           functional brain interactivity investigation in functional MRI",
           NeuroImage 32(1), 228-237 (2006).

    Parameters
    ----------
    estimator : str, optional
        Name of a class in ``sklearn.covariance``, such as
        ``'EmpiricalCovariance'``, ``'LedoitWolf'``, ``'OAS'``,
        ``'ShrunkCovariance'``, ``'GraphicalLasso'`` or ``'MinCovDet'``.
        Default: ``'EmpiricalCovariance'``.
    squared : bool, optional
        Square the result, making it unsigned. Default: False.
    """

    name = "Partial correlation"
    identifier = "pcorr"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(
            kind="partial_correlation", squared=squared, estimator=estimator
        )

    def _matrix(self, mycov):
        prec = mycov.precision_
        with np.errstate(invalid="ignore", divide="ignore"):
            # A non-positive diagonal (a singular fit) gives NaN, not a value.
            d = np.sqrt(np.where(np.diag(prec) > 0, np.diag(prec), np.nan))
            return -prec / np.outer(d, d)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # Conditions on every other process, so needs the whole (cached) fit.
        return self.multivariate(data)[i, j]


class CrossCorrelation(Undirected, Signed):
    """
    Sample cross-correlation, reduced over the lags in ``[-T // 4, T // 4]``.

    ``r(l) = sum_t x[t+l] y[t] / (T sd_x sd_y)`` on the demeaned series. This
    is the biased estimator (divided by ``T``, not ``T - |l|``), as in
    statsmodels' ``ccf(adjusted=False)``, so every ``r(l)`` is in ``[-1, 1]``
    and ``r(0)`` is Pearson's r. Lags stop at a quarter of the record, beyond
    which each estimate rests on few overlapping samples. The window is
    centred on lag 0 and ``r_yx(l) = r_xy(-l)``, so the result does not depend
    on the order of the pair (undirected). Signed unless ``squared=True``.
    NaN when either series is constant.

    Parameters
    ----------
    statistic : str, optional
        How the retained lags are reduced: ``'max'``, the largest value, or
        ``'mean'``, their mean. Default: ``'max'``.
    squared : bool, optional
        Square ``r(l)`` before the reduction, making the result unsigned.
        Default: False.
    sigonly : bool, optional
        Keep only the lags with ``|r(l)| > 1.96 / sqrt(T)``; the result is 0
        when no lag clears it. Despite the name this is a pointwise amplitude
        threshold, not a significance test: the band is not widened for the
        series' autocorrelation or corrected for the number of lags tested.
        Default: True.
    """

    name = "Cross correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, squared=False, statistic="max", sigonly=True):
        self.identifier = "xcorr"
        self._squared = squared
        self._statistic = statistic
        self._sigonly = sigonly

        if self._squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = CrossCorrelation.labels + ["unsigned"]
        else:
            self.labels = CrossCorrelation.labels + ["signed"]
        self.identifier += f"_{statistic}_sig-{sigonly}"

    # Lags are examined out to +/- T // 4.
    _MAX_LAG_FRACTION = 4

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        T = data.n_observations
        try:
            r_ij = data.xcorr[(i, j)]
        except (KeyError, AttributeError):
            # Always computed as (larger, smaller) and reversed for the other
            # orientation: the FFTs of (x, y) and (y, x) differ in the last
            # bit, so the value must not depend on which pair comes first.
            hi, lo = max(i, j), min(i, j)
            x, y = data.to_numpy(squeeze=True)[[hi, lo]]
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                # Undefined; `sigonly` would otherwise turn it into 0.
                return np.nan

            # Divided by T (biased), not T - |l|, so `max` is not dominated by
            # the high-variance large lags; as statsmodels' ccf(adjusted=False).
            x = x - x.mean()
            y = y - y.mean()
            scale = T * np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))

            # Force FFT method: O(N log N) vs O(N^2) direct for short signals.
            r_full = signal.correlate(x, y, "full", method="fft") / scale

            # correlate(x, y, "full")[T - 1 + l] == sum_t x[t + l] * y[t]: the
            # window is centred on lag 0, at index T - 1.
            lag_max = T // self._MAX_LAG_FRACTION
            r_hilo = r_full[T - 1 - lag_max: T + lag_max]

            try:
                data.xcorr[(hi, lo)] = r_hilo
            except AttributeError:
                data.xcorr = {(hi, lo): r_hilo}
            # r_yx(l) == r_xy(-l): the opposite orientation is the reverse.
            data.xcorr[(lo, hi)] = r_hilo[::-1]
            r_ij = data.xcorr[(i, j)]

        # The two-sided 5% band for one correlation of two independent white
        # series, applied pointwise: "drop the small lags", not a test.
        if getattr(self, "_sigonly", False):
            threshold = 1.96 / np.sqrt(T)
            above = np.abs(r_ij) > threshold
            if not above.any():
                # Nothing clears the cut: the thresholded association is 0.
                return 0.0
            # The set of lags above the cut (not a contiguous run around lag
            # 0) is invariant under l -> -l, so both orientations agree.
            r_ij = r_ij[above]

        if self._squared:
            r_ij = r_ij ** 2
        if self._statistic == "max":
            return float(np.max(r_ij))
        elif self._statistic == "mean":
            return float(np.mean(r_ij))
        else:
            raise TypeError(f"Unknown statistic: {self._statistic}")


class SpearmanR(Undirected, Signed):
    """
    Spearman's rank correlation coefficient [1]_.

    Pearson's correlation of the ranks of ``x_i`` and ``x_j`` (average ranks
    for ties) at lag 0, as ``scipy.stats.spearmanr``. Undirected; signed, in
    ``[-1, 1]``, unless ``squared=True``. NaN when either series is constant.

    References
    ----------
    .. [1] C. Spearman, "The proof and measurement of association between two
           things", The American Journal of Psychology 15(1), 72-101 (1904).

    Parameters
    ----------
    squared : bool, optional
        Report ``rho ** 2``, which is unsigned. Default: False.
    """

    name = "Spearman's correlation coefficient"
    identifier = "spearmanr"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = self.labels + ["unsigned"]
        else:
            self.labels = self.labels + ["signed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        if self._squared:
            return stats.spearmanr(x, y).correlation ** 2
        else:
            return stats.spearmanr(x, y).correlation

    @parse_multivariate
    def multivariate(self, data):
        """Spearman's rho of every pair at once, from the ``(M, T)`` matrix."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        rho = _spearman_matrix(Z)
        if self._squared:
            rho = rho ** 2
        np.fill_diagonal(rho, np.nan)
        return rho


class KendallTau(Undirected, Signed):
    """
    Kendall's rank correlation coefficient tau [1]_.

    Kendall's tau-b [2]_ of ``x_i`` and ``x_j`` at lag 0, the default of
    ``scipy.stats.kendalltau``: the difference between the numbers of
    concordant and discordant pairs of samples, normalised with a correction
    for ties. Undirected; signed, in ``[-1, 1]``, unless ``squared=True``.
    NaN when either series is constant.

    References
    ----------
    .. [1] M. G. Kendall, "A new measure of rank correlation", Biometrika
           30(1-2), 81-93 (1938).
    .. [2] M. G. Kendall, "The treatment of ties in ranking problems",
           Biometrika 33(3), 239-251 (1945).

    Parameters
    ----------
    squared : bool, optional
        Report ``tau ** 2``, which is unsigned. Default: False.
    """

    name = "Kendall's tau"
    identifier = "kendalltau"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = self.labels + ["unsigned"]
        else:
            self.labels = self.labels + ["signed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        if self._squared:
            return stats.kendalltau(x, y).correlation ** 2
        else:
            return stats.kendalltau(x, y).correlation

    @parse_multivariate
    def multivariate(self, data):
        """Kendall's tau of every pair, from pandas ``corr(method='kendall')``."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        df = pd.DataFrame(Z.T)
        tau = np.asarray(df.corr(method="kendall").values, dtype=float).copy()
        if self._squared:
            tau = tau ** 2
        np.fill_diagonal(tau, np.nan)
        return tau


# Seed of the permutation that breaks ties in ChatterjeeXi's source reproducibly.
_XI_TIE_SEED = 0


def _xi_source_order(x):
    """Indices that sort ``x``, with ties broken by a fixed seeded permutation."""
    perm = np.random.default_rng(_XI_TIE_SEED).permutation(x.size)
    return perm[np.argsort(x[perm], kind="stable")]


def _xi_target_terms(y):
    """
    ``(r, denominator)`` for the target ``y``.

    ``r[k] = #{j : y_j <= y_k}`` and the denominator is
    ``2 * sum_k l_k (n - l_k) / n`` with ``l_k = #{j : y_j >= y_k}``.
    The denominator is 0 (xi undefined) when ``y`` is constant.
    """
    n = y.size
    r = stats.rankdata(y, method="max").astype(np.int64)
    l = n - stats.rankdata(y, method="min").astype(np.int64) + 1
    return r, 2 * np.sum(l * (n - l)) / n


class ChatterjeeXi(Directed, Unsigned):
    """
    Chatterjee's rank correlation xi [1]_.

    For a source ``X`` and a target ``Y`` of length ``n``, sort the pairs by
    ``X`` and let ``Y_(1), ..., Y_(n)`` be the targets in that order. With
    ``r_k = #{j : Y_j <= Y_(k)}`` and ``l_k = #{j : Y_j >= Y_(k)}``,
    ``xi_n(X, Y) = 1 - n sum_k |r_{k+1} - r_k| / (2 sum_k l_k (n - l_k))``,
    the first sum over ``k = 1..n-1`` and the second over ``k = 1..n``. This
    is the general form, valid with ties in ``Y``.

    It uses ranks only. As ``n -> infinity`` it tends to 0 if and only if
    ``Y`` is independent of ``X``, and to 1 if and only if ``Y`` is a
    measurable function of ``X``, so it also detects non-monotone dependence
    such as ``Y = X^2``. Under independence it can be slightly negative.

    Asymmetric and unsigned: ``A[i, j] = xi(X = x_i, Y = x_j)``, row = source,
    measures how nearly the column process is a function of the row process.

    Chatterjee breaks ties in ``X`` at random; here they are broken by a fixed
    seeded permutation, so values are reproducible. The value is NaN when the
    target is constant, when either series has a non-finite sample, or when
    fewer than two samples remain after the lag.

    References
    ----------
    .. [1] S. Chatterjee, "A new coefficient of correlation", Journal of the
           American Statistical Association 116(536), 2009-2022 (2021).

    Parameters
    ----------
    tau : int, optional
        Lag in samples, at least 0. For ``tau > 0`` the SPI is
        ``xi(x_i[:-tau], x_j[tau:])``: how much ``x_j[t]`` is a function of
        ``x_i[t - tau]``. Default: 0.
    """

    name = "Chatterjee's xi correlation"
    identifier = "xi"
    labels = ["basic", "rank", "nonlinear", "directed", "unsigned"]

    def __init__(self, tau=0):
        self._tau = require_int("tau", tau, minimum=0)
        self.identifier = f"xi_tau-{self._tau}"
        self.labels = ChatterjeeXi.labels + (
            ["unordered"] if self._tau == 0 else ["temporal"]
        )

    @staticmethod
    def _xi(order, r, denom):
        if not denom > 0:
            return np.nan
        steps = np.abs(np.diff(r[order])).sum()
        return float(1.0 - steps / denom)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        if z.shape[-1] - self._tau < 2:
            return np.nan
        tau = self._tau
        x, y = (z[i], z[j]) if tau == 0 else (z[i, :-tau], z[j, tau:])
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            return np.nan
        r, denom = _xi_target_terms(y)
        return self._xi(_xi_source_order(x), r, denom)

    @parse_multivariate
    def multivariate(self, data):
        """Sort each source once and rank each target once."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        A = np.full((M, M), np.nan)
        if T - self._tau < 2:
            return A
        tau = self._tau
        X, Y = (Z, Z) if tau == 0 else (Z[:, :-tau], Z[:, tau:])
        finite_x = np.all(np.isfinite(X), axis=1)
        finite_y = np.all(np.isfinite(Y), axis=1)
        orders = [_xi_source_order(X[i]) if finite_x[i] else None for i in range(M)]
        R = np.zeros(Y.shape, dtype=np.int64)
        denoms = np.full(M, np.nan)
        for j in range(M):
            if finite_y[j]:
                R[j], denoms[j] = _xi_target_terms(Y[j])
        valid_y = finite_y & (denoms > 0)
        for i in range(M):
            if orders[i] is None:
                continue
            # steps[j] = sum_k |R[j, order_k+1] - R[j, order_k]| for every target
            steps = np.abs(np.diff(R[:, orders[i]], axis=1)).sum(axis=1)
            A[i, valid_y] = 1.0 - steps[valid_y] / denoms[valid_y]
        np.fill_diagonal(A, np.nan)
        return A


class LaggedCorrelation(Undirected, Signed):
    """
    Symmetrised lagged correlation at lag ``tau``.

    ``0.5 * (corr(x_i[tau:], x_j[:-tau]) + corr(x_j[tau:], x_i[:-tau]))``: the
    mean of the correlations with each process leading the other by ``tau``
    samples, so the result is symmetric (undirected) and does not say which
    process leads. At ``tau = 0`` it is the plain correlation. ``corr`` is
    Pearson's r, Spearman's rho or Kendall's tau-b, as in ``scipy.stats``.
    Signed, in ``[-1, 1]``, unless ``squared=True``. NaN when ``tau >= T`` or
    when either of the two correlations is undefined (a constant segment).

    In a config, ``max_tau: K`` stands for the ``K`` SPIs with
    ``tau = 1, ..., K``.

    Parameters
    ----------
    estimator : str, optional
        ``'pearson'``, ``'spearman'`` or ``'kendall'`` (case-insensitive).
        Default: ``'pearson'``.
    tau : int
        Lag in samples, at least 0. Required: the default, None, raises.
    max_tau : None, optional
        Only expanded in configs; any value other than None raises here.
        Default: None.
    squared : bool, optional
        Square the symmetrised correlation, making it unsigned.
        Default: False.
    """
    name = "Lagged correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, estimator="pearson", tau=None, max_tau=None, squared=False):
        est = str(estimator).lower()
        if est not in {"pearson", "spearman", "kendall"}:
            raise ValueError(f"Unknown estimator: {estimator}")
        if max_tau is not None:
            raise ValueError("max_tau is only supported in config expansion; use tau.")
        if tau is None:
            raise ValueError("LaggedCorrelation requires tau.")

        self._estimator = est
        self._squared = bool(squared)
        if self._squared:
            self.issigned = lambda: False
            self.labels = LaggedCorrelation.labels + ["unsigned"]
            suffix = "-sq"
        else:
            self.labels = LaggedCorrelation.labels + ["signed"]
            suffix = ""
        self._tau = require_int("tau", tau, minimum=0)
        self.identifier = f"corr_{est}_tau-{self._tau}{suffix}"

    def _corr(self, x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.size < 2 or y.size < 2:
            return np.nan
        if self._estimator == "pearson":
            return stats.pearsonr(x, y).correlation
        if self._estimator == "spearman":
            return stats.spearmanr(x, y).correlation
        if self._estimator == "kendall":
            return stats.kendalltau(x, y).correlation
        raise ValueError(f"Unknown estimator: {self._estimator}")

    def _lagged_corr(self, x, y, tau):
        if tau == 0:
            return self._corr(x, y)
        if tau >= x.size:
            return np.nan
        return self._corr(x[tau:], y[:-tau])

    def _symmetric_lagged_corr(self, x, y, tau):
        # NaN if either direction is undefined, as in `multivariate`.
        forward = self._lagged_corr(x, y, tau)
        backward = self._lagged_corr(y, x, tau)
        return 0.5 * (forward + backward)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        value = self._symmetric_lagged_corr(x, y, self._tau)
        return value**2 if self._squared else value

    @parse_multivariate
    def multivariate(self, data):
        """Lagged correlation of every pair, from the stacked lead and lag blocks."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        tau = self._tau

        if tau == 0 or tau >= T:
            if tau >= T:
                return np.full((M, M), np.nan)
            if self._estimator == "pearson":
                C = np.corrcoef(Z)
            elif self._estimator == "spearman":
                C = _spearman_matrix(Z)
            elif self._estimator == "kendall":
                C = np.asarray(
                    pd.DataFrame(Z.T).corr(method="kendall").values, dtype=float
                ).copy()
            else:
                raise ValueError(f"Unknown estimator: {self._estimator}")
            if self._squared:
                C = C ** 2
            np.fill_diagonal(C, np.nan)
            return C

        Z_lead = Z[:, tau:]
        Z_lag = Z[:, :-tau]

        if self._estimator == "pearson":
            stacked = np.vstack([Z_lead, Z_lag])
            C_full = np.corrcoef(stacked)
            forward = C_full[:M, M:]
            backward = C_full[M:, :M]
        elif self._estimator == "spearman":
            stacked = np.vstack([Z_lead, Z_lag])
            rho = _spearman_matrix(stacked)
            forward = rho[:M, M:]
            backward = rho[M:, :M]
        elif self._estimator == "kendall":
            # Only the lead x lag block is needed; `backward` is its transpose.
            forward = np.full((M, M), np.nan)
            for a in range(M):
                for b in range(M):
                    if a != b:
                        forward[a, b] = stats.kendalltau(Z_lead[a], Z_lag[b])[0]
            backward = forward.T
        else:
            raise ValueError(f"Unknown estimator: {self._estimator}")

        C = 0.5 * (forward + backward)
        if self._squared:
            C = C ** 2
        np.fill_diagonal(C, np.nan)
        return C

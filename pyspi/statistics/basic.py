import inspect
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats
import sklearn.covariance as cov

from pyspi.base import (
    Directed,
    NotAppropriate,
    Undirected,
    Signed,
    Unsigned,
    fill_pairwise,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import acf, require_int


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
    scikit-learn's graphical lasso solver can fail on strongly correlated
    processes; every SPI of that fit is then an error (``FloatingPointError``).
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
            self.labels = list(Estimators.labels)
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
                try:
                    mycov = estimator(**kwargs).fit(z)
                except FloatingPointError as err:
                    # The graphical lasso's coordinate descent raises this
                    # when its iterate stops being positive definite.
                    if self._estimator not in ("GraphicalLasso",
                                               "GraphicalLassoCV"):
                        raise
                    raise FloatingPointError(
                        f"scikit-learn's graphical lasso solver failed on the "
                        f"{self._estimator} fit, as it can on strongly "
                        f"correlated processes, so {self.identifier} was not "
                        f"computed. This is a limitation of the solver: "
                        f"another estimator (e.g. LedoitWolf) may succeed."
                    ) from err
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
                raise NotAppropriate(
                    f"The {self._estimator} covariance estimate is singular "
                    f"(rank {rank} < M = {M}: no more samples than processes, or "
                    f"collinear processes), so {self.identifier} is undefined."
                )
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
    Not appropriate (``NotAppropriate``) when the covariance estimate is
    singular (for example with no more samples than processes).

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
    fit is shared with ``Covariance`` and ``Precision``. Not appropriate
    (``NotAppropriate``) when the covariance estimate is singular (for
    example with no more samples than processes).

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


# Lags are examined out to +/- T // 4: further out the sample
# cross-correlation rests on few overlapping samples.
_XCORR_MAX_LAG_FRACTION = 4


def _xcorr_profile(data, i, j):
    """
    The cached cross-correlation profile of the pair ``(i, j)``.

    ``P[T//4 + l] = sum_t x_i[t + l] x_j[t] / (T sd_i sd_j)`` on the demeaned
    series, for lags ``l = -T//4 .. +T//4`` (odd length ``2 (T//4) + 1``, the
    zero lag at the centre). A positive ``l`` pairs ``x_i`` with an *earlier*
    sample of ``x_j`` (``j`` leading), so the lead of ``i`` over ``j`` by
    ``tau`` sits on the negative side: ``P[T//4 - tau] = sum_t x_i[t]
    x_j[t + tau] / (T sd_i sd_j)``.

    Stored in ``data.xcorr`` under both orientations (``(j, i)`` is the reversed
    sequence) and shared by :class:`CrossCorrelation` and
    :class:`LagResolvedCorrelation`. All NaN when either series is constant.
    The returned array *is* the cache entry: read it, do not modify it in
    place.
    """
    try:
        return data.xcorr[(i, j)]
    except (KeyError, AttributeError):
        pass

    # Computed for (larger, smaller) and reversed for the other orientation,
    # so a value does not depend on the order the pairs are visited in.
    T = data.n_observations
    hi, lo = max(i, j), min(i, j)
    x, y = data.to_numpy(squeeze=True)[[hi, lo]]

    lag_max = T // _XCORR_MAX_LAG_FRACTION
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        # Undefined; rounding in the mean of a float constant would otherwise
        # give arbitrary finite values.
        r_hilo = np.full(2 * lag_max + 1, np.nan)
    else:
        # Biased estimator (demeaned, divided by T, as statsmodels'
        # `ccf(adjusted=False)`): |r| <= 1 and the zero lag is Pearson's r.
        x = x - x.mean()
        y = y - y.mean()
        scale = T * np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))

        # Force FFT method: O(N log N) vs O(N^2) direct for short signals.
        r_full = signal.correlate(x, y, "full", method="fft") / scale

        # correlate(x, y, "full")[T - 1 + l] == sum_t x[t + l] * y[t], so the
        # zero lag sits at T - 1.
        r_hilo = r_full[T - 1 - lag_max: T + lag_max]

    try:
        data.xcorr[(hi, lo)] = r_hilo
    except AttributeError:
        data.xcorr = {(hi, lo): r_hilo}
    # r_yx(l) == r_xy(-l): the opposite orientation is the reversed sequence.
    data.xcorr[(lo, hi)] = r_hilo[::-1]
    return data.xcorr[(i, j)]


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
    NaN when either series is constant. The lag profile is cached in
    ``data.xcorr`` and shared with ``LagResolvedCorrelation``.

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
    # data.xcorr has no parameters; shared with LagResolvedCorrelation.
    _cache_namespace = "xcorr"

    def __init__(self, squared=False, statistic="max", sigonly=True):
        self.identifier = "xcorr"
        self._squared = squared
        self._statistic = statistic
        self._sigonly = sigonly

        if self._squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = list(CrossCorrelation.labels)
        else:
            self.labels = CrossCorrelation.labels + ["signed"]
        self.identifier += f"_{statistic}_sig-{sigonly}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        T = data.n_observations
        r_ij = _xcorr_profile(data, i, j)
        if np.isnan(r_ij).all():
            # Undefined, and `sigonly` would otherwise turn it into 0.
            return np.nan

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


def _acf_1e_length(data, p):
    """
    First lag ``k >= 1`` at which the sample ACF of process ``p`` is below 1/e.

    ``pyspi.utils.acf`` is the biased (divide-by-T) sample autocorrelation of
    the demeaned series. The search runs out to ``T // 4``; if the ACF stays at
    or above 1/e that far the length is ``inf``. Cached per process in
    ``data.acf_1e``, so each series' ACF is computed once.
    """
    try:
        return data.acf_1e[p]
    except (KeyError, AttributeError):
        pass
    T = data.n_observations
    x = np.asarray(data.to_numpy(squeeze=True)[p], dtype=float)
    rho = acf(x)[1:T // _XCORR_MAX_LAG_FRACTION + 1]
    hits = np.flatnonzero(rho < np.exp(-1))
    length = float(hits[0] + 1) if hits.size else np.inf
    try:
        data.acf_1e[p] = length
    except AttributeError:
        data.acf_1e = {p: length}
    return length


def _round_to_lag(v, cap):
    """``v`` rounded half up to an integer lag in ``1 .. cap`` (``inf`` -> ``cap``)."""
    if not np.isfinite(v):
        return cap
    return int(min(max(np.floor(v + 0.5), 1), cap))


class LagResolvedCorrelation(Directed, Signed):
    """
    Directed, lag-resolved cross-correlation [1]_.

    Reads the lag profile that :class:`CrossCorrelation` caches in
    ``data.xcorr`` and reports where on the lag axis a pair's linear coupling
    sits and which way it runs.

    For a lag ``tau`` (in samples),
    ``r_{i->j}(tau) = sum_t (x_i[t] - m_i)(x_j[t + tau] - m_j) / (T s_i s_j)``,
    summed over the ``T - |tau|`` overlapping samples, with ``m`` the mean and
    ``s`` the divide-by-T standard deviation. ``tau > 0`` is ``i`` leading
    ``j``, ``r_{i->j}(-tau) = r_{j->i}(tau)``, and ``r_{i->j}(0)`` is
    Pearson's r. The denominator is ``T``, not ``T - |tau|`` (the biased
    estimator of [2]_, as in ``CrossCorrelation``), so ``|r| <= 1`` and a pure
    delay of ``d`` samples gives ``1 - d/T`` at its lag. Every statistic uses
    the lags ``-L .. L`` with ``L = min(max_lag, T // 4)``.

    The correlation length ``l_i`` is the first lag ``k >= 1`` at which the
    biased sample autocorrelation of ``x_i`` falls below ``1/e`` (hctsa's
    ``CO_f1ecac`` [3]_), searched out to ``T // 4``; it is unbounded if the
    autocorrelation stays at or above ``1/e`` that far. The adaptive
    statistics set one lag or window per pair from ``l_i`` and ``l_j`` and clip
    it to ``1 .. C`` with ``C = max(1, min(T // 10, L))``, so an unbounded
    length gives ``C``. In samples it follows the sampling rate, so in time it
    stays about the same.

    ``A[i, j]`` is row ``i`` (source) to column ``j`` (target):

    ``lag``
        ``l* = argmax_{l in -L..L} |r_{i->j}(l)|``, the signed lag of the
        strongest linear coupling in samples, positive when the row leads the
        column. Antisymmetric: ``A[i, j] = -A[j, i]``. Exact ties go to the
        smallest ``|l|``. The lag is on the sample grid, where [1]_
        interpolates between samples. Under independence ``l*`` is spread over
        the whole window, so on weakly coupled pairs prefer a short
        ``max_lag``.
    ``peak``
        ``r_{i->j}(l*)``, the signed correlation at that lag. Undirected and
        signed.
    ``onesided``
        ``A[i, j] = r_{i->j}(tau)`` and ``A[j, i] = r_{j->i}(tau)`` at one lag
        ``tau`` shared by both cells of the pair: directed and signed. With
        ``tau="1e"`` the lag is ``round(sqrt(l_i l_j))``, clipped to
        ``1 .. C``. The lag is shared because ``|r|`` decays with the lag, so
        lags that differ by direction would report a direction on correlated
        pairs with no directed coupling.
    ``onesidedpeak``
        ``max_{1 <= k <= w} |r_{i->j}(k)|``, the largest lead correlation from
        row to column within ``w = l_i + l_j`` lags (shared by both cells,
        clipped to ``1 .. C``): directed and unsigned. Lag 0 is excluded.
    ``asym``
        ``(S_{i->j} - S_{j->i}) / (S_{i->j} + S_{j->i})`` with
        ``S_{i->j} = sum_{l=1..L} r_{i->j}(l)^2``, a lead/lag asymmetry index
        in ``[-1, 1]``: antisymmetric, positive when the lagged coupling runs
        mostly from row to column, and 0 when both sums are 0. It does not
        condition on the target's own past, so a common driver that reaches
        ``i`` before ``j`` also makes it positive.

    The identifier is ``lagcorr_<statistic>[_tau-<tau>]_L-<max_lag>``, e.g.
    ``lagcorr_lag_L-None`` or ``lagcorr_onesided_tau-1e_L-None``. NaN for a
    pair with a constant process.

    References
    ----------
    .. [1] A. Mitra, A.Z. Snyder, C.D. Hacker, M.E. Raichle, "Lag structure in
           resting-state fMRI", Journal of Neurophysiology 111(11), 2374-2391
           (2014).
    .. [2] G.E.P. Box, G.M. Jenkins, G.C. Reinsel, G.M. Ljung, "Time Series
           Analysis: Forecasting and Control", 5th edition, Wiley (2015).
    .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
           time-series analysis: the empirical structure of time series and
           their methods", Journal of the Royal Society Interface 10(83),
           20130048 (2013).

    Parameters
    ----------
    statistic : str, optional
        ``'lag'``, ``'peak'``, ``'onesided'``, ``'onesidedpeak'`` or
        ``'asym'``, as above. Default: ``'lag'``.
    max_lag : int or None, optional
        Cap on the lag window, at least 1; None uses ``T // 4``.
        Default: None.
    tau : int, str or None, optional
        The lag of ``'onesided'``: an integer from 1 to ``L``, or ``'1e'`` for
        the pair's correlation lengths; None means 1. The other statistics
        refuse any value but None. Default: None.
    """

    name = "Lag-resolved cross-correlation"
    identifier = "lagcorr"
    labels = ["basic", "linear", "temporal"]
    # Shares CrossCorrelation's parameter-free cache, data.xcorr, so every
    # variant of both classes is one parallel task.
    _cache_namespace = "xcorr"

    # Structural trait of each statistic's matrix.
    _TRAITS = {
        "lag": "antisymmetric",
        "peak": "undirected",
        "onesided": "directed",
        "onesidedpeak": "directed",
        "asym": "antisymmetric",
    }
    _UNSIGNED = ("onesidedpeak",)

    def __init__(self, statistic="lag", max_lag=None, tau=None):
        if statistic not in self._TRAITS:
            raise ValueError(
                f"Unknown statistic {statistic!r}; expected one of "
                f"{sorted(self._TRAITS)}.")
        if statistic == "onesided":
            if tau is None:
                tau = 1
            if not (isinstance(tau, str) and tau == "1e"):
                tau = require_int("tau", tau, minimum=1)
        elif tau is not None:
            raise ValueError(
                f"tau applies only to statistic='onesided', got tau={tau!r} "
                f"with statistic={statistic!r}.")
        self._statistic = statistic
        self._tau = tau
        self._max_lag = (None if max_lag is None
                         else require_int("max_lag", max_lag, minimum=1))
        if (statistic == "onesided" and tau != "1e"
                and self._max_lag is not None and tau > self._max_lag):
            raise ValueError(f"tau = {tau} exceeds max_lag = {self._max_lag}.")

        if statistic in self._UNSIGNED:
            self.issigned = lambda: False
            sign = []
        else:
            sign = ["signed"]
        self.labels = LagResolvedCorrelation.labels + [
            self._TRAITS[statistic]] + sign
        # The lag of peak |r| is an offset, not a strength.
        if statistic == "lag":
            self.labels.append("timing")
        self.identifier = f"lagcorr_{statistic}"
        if statistic == "onesided":
            self.identifier += f"_tau-{tau}"
        self.identifier += f"_L-{self._max_lag}"

    def _window(self, T):
        L = T // _XCORR_MAX_LAG_FRACTION
        if self._max_lag is not None:
            L = min(L, self._max_lag)
        if L < 1:
            raise NotAppropriate(
                f"The lag window min(max_lag, T // 4) is {L} for T = {T}: "
                f"at least one lag is needed.")
        return L

    def _pair_lag(self, data, L):
        """
        The pair's ``onesided`` lag or ``onesidedpeak`` window, ``f(i, j)``.

        ``None`` for the statistics that read the whole window.
        """
        stat = self._statistic
        if stat == "onesided" and self._tau != "1e":
            if self._tau > L:
                raise NotAppropriate(
                    f"tau = {self._tau} is outside the lag window "
                    f"L = min(max_lag, T // 4) = {L} for T = "
                    f"{data.n_observations}.")
            return lambda i, j: self._tau
        if stat not in ("onesided", "onesidedpeak"):
            return None
        cap = max(1, min(data.n_observations // 10, L))
        if stat == "onesided":
            # Geometric mean of the two 1/e lengths.
            return lambda i, j: _round_to_lag(
                np.sqrt(_acf_1e_length(data, i) * _acf_1e_length(data, j)), cap)
        # Twice the mean of the two 1/e lengths.
        return lambda i, j: _round_to_lag(
            _acf_1e_length(data, i) + _acf_1e_length(data, j), cap)

    @staticmethod
    def _lead_profile(data, i, j, L):
        """
        ``d[L + tau] = r_{i->j}(tau)`` for ``tau = -L .. L``.

        The cached profile has ``P[c + l] = sum_t x_i[t + l] x_j[t] / (T s_i
        s_j)`` with ``c = T // 4``, so ``r_{i->j}(tau)``, which pairs ``x_i[t]``
        with ``x_j[t + tau]``, is ``P[c - tau]``: the window reversed.
        """
        P = _xcorr_profile(data, i, j)
        c = (len(P) - 1) // 2
        d = P[c - L: c + L + 1][::-1]
        if not np.isfinite(d).all():
            raise ValueError(
                f"The cross-correlation profile of ({i}, {j}) is not finite "
                f"(a constant or non-finite series).")
        return d

    def _pair(self, data, i, j, L, pair_lag):
        """``(A[i, j], A[j, i])``, both from the one profile of the pair."""
        d = self._lead_profile(data, i, j, L)
        stat = self._statistic
        if stat in ("lag", "peak"):
            # Lags in order of |l| (0, -1, +1, -2, ...), so argmax's first hit
            # breaks an exact tie towards the shortest lag.
            order = np.argsort(np.abs(np.arange(-L, L + 1)), kind="stable")
            k = int(order[np.argmax(np.abs(d[order]))])
            if stat == "lag":
                lag = k - L  # an int, so the reverse of lag 0 is not -0.0
                return float(lag), float(-lag)
            peak = float(d[k])
            return peak, peak
        if stat == "asym":
            forward = float(np.sum(d[L + 1:] ** 2))    # r_{i->j}(1..L)
            backward = float(np.sum(d[:L] ** 2))       # r_{j->i}(1..L)
            total = forward + backward
            if total == 0:
                return 0.0, 0.0
            # x - y == -(y - x) exactly in floating point: antisymmetric.
            return (forward - backward) / total, (backward - forward) / total
        if stat == "onesided":
            # r_{j->i}(tau) = r_{i->j}(-tau)
            tau = pair_lag(i, j)
            return float(d[L + tau]), float(d[L - tau])
        if stat == "onesidedpeak":
            w = pair_lag(i, j)
            return (float(np.max(np.abs(d[L + 1:L + w + 1]))),   # r_{i->j}(1..w)
                    float(np.max(np.abs(d[L - w:L]))))          # r_{j->i}(1..w)
        raise TypeError(f"Unknown statistic: {stat}")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        L = self._window(data.n_observations)
        # The lower index first, as in `multivariate`, so the two agree exactly
        # and the antisymmetric and symmetric statistics are exactly so.
        a, b = (i, j) if i <= j else (j, i)
        v_ab, v_ba = self._pair(data, a, b, L, self._pair_lag(data, L))
        return v_ab if i <= j else v_ba

    @parse_multivariate
    def multivariate(self, data):
        """Each unordered pair once: one profile fills both of its cells."""
        M = data.n_processes
        # A window or tau that cannot be met fails every pair alike: raise.
        L = self._window(data.n_observations)
        pair_lag = self._pair_lag(data, L)
        pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
        both = np.full((M, M, 2), np.nan)
        # A failing pair is NaN, and reported as a failure, in both of its
        # cells; only the upper triangle of `both` is read.
        fill_pairwise(self.identifier, both, pairs,
                      lambda i, j: self._pair(data, i, j, L, pair_lag),
                      mirror=True)
        A = np.full((M, M), np.nan)
        iu = np.triu_indices(M, 1)
        A[iu] = both[..., 0][iu]
        A[iu[1], iu[0]] = both[..., 1][iu]
        return A


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
    labels = ["basic", "unordered", "rank", "nonlinear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = list(self.labels)
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
    labels = ["basic", "unordered", "rank", "nonlinear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = list(self.labels)
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
    seeded permutation, so values are reproducible. The value is NaN when
    either series has a non-finite sample. It is not appropriate
    (``NotAppropriate``, NaN in a Calculator table) when fewer than two
    samples remain after the lag (``T < tau + 2``), and for a pair whose
    target is constant over ``x_j[tau:]``, as ties in a short series can make
    it.

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
    labels = ["basic", "rank", "nonlinear", "directed"]

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

    def _require_length(self, T):
        if T - self._tau < 2:
            raise NotAppropriate(
                f"Chatterjee's xi at tau = {self._tau} needs two samples after "
                f"the lag, so T >= {self._tau + 2}; got T = {T}.")

    def _constant_target(self, j, n):
        target = "x_j" if self._tau == 0 else f"x_j[{self._tau}:]"
        return (f"process {j} is constant over its {n} target samples "
                f"({target}), so xi is undefined.")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        self._require_length(z.shape[-1])
        tau = self._tau
        x, y = (z[i], z[j]) if tau == 0 else (z[i, :-tau], z[j, tau:])
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            return np.nan
        r, denom = _xi_target_terms(y)
        if not denom > 0:
            raise NotAppropriate(self._constant_target(j, y.size))
        return self._xi(_xi_source_order(x), r, denom)

    @parse_multivariate
    def multivariate(self, data):
        """Sort each source once and rank each target once."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        self._require_length(T)
        A = np.full((M, M), np.nan)
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
        constant = [j for j in range(M) if finite_y[j] and not denoms[j] > 0]
        if not constant:
            return A

        # Record a reason for each pair whose target is constant.
        def compute(i, j):
            if j in constant:
                raise NotAppropriate(self._constant_target(j, Y.shape[1]))
            return A[i, j]

        pairs = [(i, j) for j in range(M) for i in range(M) if i != j]
        return fill_pairwise(self.identifier, np.full((M, M), np.nan), pairs,
                             compute)


class LaggedCorrelation(Undirected, Signed):
    """
    Symmetrised lagged correlation at lag ``tau``.

    ``0.5 * (corr(x_i[tau:], x_j[:-tau]) + corr(x_j[tau:], x_i[:-tau]))``: the
    mean of the correlations with each process leading the other by ``tau``
    samples, so the result is symmetric (undirected) and does not say which
    process leads. At ``tau = 0`` it is the plain correlation. ``corr`` is
    Pearson's r, Spearman's rho or Kendall's tau-b, as in ``scipy.stats``.
    Signed, in ``[-1, 1]``, unless ``squared=True``. Linear with Pearson's r;
    nonlinear with Spearman's rho or Kendall's tau, which capture any
    monotone dependence. NaN when ``tau >= T`` or when either of the two
    correlations is undefined (a constant segment).

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
    labels = ["basic", "undirected", "temporal"]

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
        linearity = "linear" if est == "pearson" else "nonlinear"
        if self._squared:
            self.issigned = lambda: False
            self.labels = LaggedCorrelation.labels + [linearity]
            suffix = "-sq"
        else:
            self.labels = LaggedCorrelation.labels + [linearity, "signed"]
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

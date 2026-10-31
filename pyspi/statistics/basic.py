import inspect
import warnings
import sklearn.covariance as cov
from scipy import stats, signal
import numpy as np
import pandas as pd

from pyspi.base import (Directed, Undirected, Signed, fill_pairwise,
                        parse_bivariate, parse_multivariate)
from pyspi.utils import acf, require_int


def _spearman_matrix(Z):
    """Spearman's rho between the rows of ``Z``: Pearson on average ranks.

    Same values as ``stats.spearmanr(Z, axis=1)``, but always a matrix, and a
    constant row gives NaN only in its own row and column.
    """
    return np.corrcoef(np.apply_along_axis(stats.rankdata, 1, Z))


class Estimators(Undirected, Signed):
    """Base class for (functional) connectivity-based statistics

    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html

    When cross-validation selects an ``alpha`` near 0, ``GraphicalLassoCV`` may
    not converge (silently, as fit warnings are suppressed); on data with a
    dense precision matrix, prefer ``GraphicalLasso`` or ``LedoitWolf``.
    """

    name = "Covariance"
    labels = ["basic", "unordered", "linear", "undirected"]
    _cache_namespace = "covariance"

    @property
    def _cache_subkey(self):
        # Cache is data.covariance[estimator]; (kind, squared) are post-lookup.
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
            # MinCovDet and EllipticEnvelope search from random subsets of the
            # observations. Unpinned they drew those from the global NumPy
            # stream, so the fit -- and all four SPIs read from it -- changed
            # from run to run and with whatever had consumed the stream first.
            params = inspect.signature(estimator).parameters
            kwargs = {"random_state": 42} if "random_state" in params else {}
            # GraphicalLasso(CV) stop at the default tol=1e-4, so their values
            # depended on the solver path, which scikit-learn 1.8 changed (gap-
            # safe screening, new stopping rule): up to 1.5e-4 apart. Converging
            # tightly makes the result the solution, not the path (~1e-9 apart).
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

    @parse_multivariate
    def multivariate(self, data):
        mycov = self._from_cache(data)
        # Copy, so the cached fit is not modified.
        matrix = getattr(mycov, self._kind + "_").copy()
        np.fill_diagonal(matrix, np.nan)
        if self._squared:
            return np.square(matrix)
        else:
            return matrix


class Covariance(Estimators):

    name = "Covariance"
    identifier = "cov"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="covariance", squared=squared, estimator=estimator)


class Precision(Estimators):

    name = "Precision"
    identifier = "prec"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="precision", squared=squared, estimator=estimator)


# Lags are examined out to +/- T // 4: further out the sample
# cross-correlation rests on few overlapping samples.
_XCORR_MAX_LAG_FRACTION = 4


def _xcorr_profile(data, i, j):
    """The cached cross-correlation profile of the pair ``(i, j)``.

    ``P[T//4 + l] = sum_t x_i[t + l] x_j[t] / (T sd_i sd_j)`` on the demeaned
    series, for lags ``l = -T//4 .. +T//4`` (odd length ``2 (T//4) + 1``, the
    zero lag at the centre). A positive ``l`` pairs ``x_i`` with an *earlier*
    sample of ``x_j`` (``j`` leading), so the lead of ``i`` over ``j`` by
    ``tau`` sits on the negative side: ``P[T//4 - tau] = sum_t x_i[t]
    x_j[t + tau] / (T sd_i sd_j)``.

    Stored in ``data.xcorr`` under both orientations (``(j, i)`` is the reversed
    sequence) and shared by :class:`CrossCorrelation` and
    :class:`LagResolvedCorrelation`. The returned array *is* the cache entry:
    read it, do not modify it in place.
    """
    try:
        return data.xcorr[(i, j)]
    except (KeyError, AttributeError):
        pass

    T = data.n_observations
    x, y = data.to_numpy(squeeze=True)[[i, j]]

    # Biased estimator (demeaned, divided by T, as statsmodels'
    # `ccf(adjusted=False)`): |r| <= 1 and the zero lag is Pearson's r.
    x = x - x.mean()
    y = y - y.mean()
    scale = T * np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))

    # Force FFT method: O(N log N) vs O(N^2) direct for short signals.
    r_full = signal.correlate(x, y, "full", method="fft") / scale

    # correlate(x, y, "full")[T - 1 + l] == sum_t x[t + l] * y[t], so the
    # zero lag sits at T - 1.
    lag_max = T // _XCORR_MAX_LAG_FRACTION
    r_ij = r_full[T - 1 - lag_max: T + lag_max]

    try:
        data.xcorr[(i, j)] = r_ij
    except AttributeError:
        data.xcorr = {(i, j): r_ij}
    # r_yx(l) == r_xy(-l): the opposite orientation is the reversed sequence.
    data.xcorr[(j, i)] = r_ij[::-1]
    return r_ij


class CrossCorrelation(Undirected, Signed):
    """Sample cross-correlation over lags in [-T//4, +T//4].

    ``r(l) = sum_t x[t+l] y[t] / (T sd_x sd_y)`` on the demeaned series (the
    biased estimator, so ``r(0)`` is Pearson's r). ``statistic`` reduces the
    retained lags: ``max`` is the largest signed value, ``mean`` their mean.

    ``sigonly`` is a historical misnomer kept for compatibility. It is a
    pointwise amplitude threshold -- keep the lags with
    ``|r(l)| > 1.96/sqrt(T)`` -- and not a significance test: the band is not
    inflated for the series' own autocorrelation, and it is not corrected for
    being applied at every lag in the window. When no lag clears it the
    statistic is 0.
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
            self.labels = CrossCorrelation.labels + ["unsigned"]
        else:
            self.labels = CrossCorrelation.labels + ["signed"]
        self.identifier += f"_{statistic}_sig-{sigonly}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        T = data.n_observations
        r_ij = _xcorr_profile(data, i, j)

        # `sigonly` is a pointwise amplitude threshold, not an inferential
        # test. The name is historical and kept for compatibility: it keeps
        # only the lags whose sample cross-correlation exceeds
        # 1.96/sqrt(T) in magnitude. That cut is the two-sided 5% band for a
        # *single* correlation between two independent white series, and
        # neither of the two things that would make it a significance test is
        # done here -- the band is not inflated for the series'
        # autocorrelation, and it is
        # not corrected for having been applied at every one of the ~T/2 lags
        # in the window. Read it as "drop the small lags", not as "these lags
        # are significant".
        if getattr(self, "_sigonly", False):
            # The previous threshold was `1.96/sqrt(len(r_ij)//2)` -- the
            # half-width of the lag *window*, T//4 -- so it was twice too wide
            # and moved with the lag cut rather than with the sample size.
            threshold = 1.96 / np.sqrt(T)
            above = np.abs(r_ij) > threshold
            if not above.any():
                # Nothing clears the cut, so the thresholded association is
                # zero. Falling back to the unfiltered window instead reported
                # the largest of ~T/2 sample correlations under the null: on
                # two independent length-400 series that is about 0.135 for the
                # `max` statistic, which is the opposite of what a threshold is
                # supposed to do at the null.
                return 0.0
            # Selecting *the lags above the cut*, rather than the contiguous
            # run of them around lag zero. The previous code walked outwards
            # from the centre, which is only right when the peak is at lag 0:
            # for a pair where i leads j by one sample, r(0) is already below
            # the cut, so the lobe extended one way and not the other and the
            # two orientations of an SPI declared *undirected* disagreed --
            # measured 0.9957 against -0.0202 on a lag-1 pair. A set of lags is
            # invariant under the l -> -l reversal; a one-sided run is not.
            # (Its slice was independently wrong: `r_ij[N - fzr : N + fzf]`
            # mirrored `fzr`, which was already an absolute index.)
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
    """First lag ``k >= 1`` at which the sample ACF of process ``p`` is below 1/e.

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
    """Directed, lag-resolved Pearson cross-correlation.

    Reads the lag profile that :class:`CrossCorrelation` caches in
    ``data.xcorr`` -- the biased (divide-by-T) sample cross-correlation of the
    demeaned series over lags ``-T//4 .. +T//4`` -- and reports where on the
    lag axis a pair's linear coupling sits and which way it runs.

    **Lead correlation.** For a lag ``tau`` (in samples)::

        r_{i->j}(tau) = sum_t (x_i[t] - m_i)(x_j[t + tau] - m_j) / (T s_i s_j)

    summed over the ``T - |tau|`` overlapping samples, with ``m`` the mean and
    ``s`` the divide-by-T standard deviation. ``tau > 0`` is ``i`` leading ``j``
    by ``tau``, ``r_{i->j}(-tau) = r_{j->i}(tau)``, and ``r_{i->j}(0)`` is
    Pearson's r. The denominator is ``T``, not ``T - |tau|`` (Box & Jenkins'
    estimator, as in ``CrossCorrelation``), so ``|r| <= 1`` and a pure delay of
    ``d`` samples gives ``1 - d/T`` at its lag.

    **Window.** All statistics use the lags ``-L .. L`` with
    ``L = min(max_lag, T // 4)`` (``max_lag=None`` is the full cached window
    ``T // 4``).

    **Correlation length.** ``l_i`` is the first lag ``k >= 1`` at which the
    sample autocorrelation of ``x_i`` (``pyspi.utils.acf``, biased) falls below
    ``1/e`` -- hctsa's ``CO_f1ecac`` on the integer lags -- searched out to
    ``T // 4``. A process whose autocorrelation stays at or above ``1/e`` that
    far has an unbounded length. The adaptive statistics below set one lag or
    window per pair from ``l_i`` and ``l_j`` and clip it to ``1 .. C`` with
    ``C = max(1, min(T // 10, L))``, so an unbounded length gives ``C``. The
    lag follows the pair's correlation times, so it adapts to the sampling
    rate: sampling the same system more finely lengthens it in samples and
    leaves it about the same in time.

    ``A[i, j]`` is row ``i`` (source) to column ``j`` (target):

    ``lag``
        ``l* = argmax_{l in -L..L} |r_{i->j}(l)|``, the signed lag of the
        strongest linear coupling, an integer number of samples, positive when
        the row leads the column. Antisymmetric, ``A[i, j] = -A[j, i]``. Exact
        ties go to the smallest ``|l|``. The lag is on the sample grid (Mitra et
        al. 2014 interpolate between samples). Under independence ``l*`` is
        spread over the whole window, so on weakly coupled pairs prefer a short
        ``max_lag``.
    ``peak``
        ``r_{i->j}(l*)``: the signed correlation at that lag. Undirected (the
        same extremum is found from either orientation), signed.
    ``onesided``
        ``A[i, j] = r_{i->j}(tau)`` and ``A[j, i] = r_{j->i}(tau)`` at one lag
        ``tau`` shared by both cells of the pair: directed and signed. ``tau``
        is an integer in ``1 .. L`` (default 1), or ``"1e"`` for the pair lag
        ``tau_ij = round(sqrt(l_i l_j))``, the geometric mean of the two
        correlation lengths, clipped to ``1 .. C``. The lag is one per pair
        because lags that differ by direction report a direction on correlated
        pairs with no directed coupling: ``|r|`` decays with the lag, so the
        direction read at the shorter lag wins.
    ``onesidedpeak``
        ``max_{1 <= k <= w} |r_{i->j}(k)|``: the largest lead correlation from
        row to column within ``w`` lags, directed and unsigned. The window
        ``w = l_i + l_j`` (twice the pair's mean correlation length) is shared
        by both cells and clipped to ``1 .. C``. Lag 0 is excluded.
    ``asym``
        ``(S_{i->j} - S_{j->i}) / (S_{i->j} + S_{j->i})`` with
        ``S_{i->j} = sum_{l=1..L} r_{i->j}(l)^2``: a lead/lag asymmetry index in
        ``[-1, 1]``, antisymmetric, positive when the lagged coupling runs
        mostly from row to column (0 when both sums are 0). Lag 0 enters
        neither sum. It does not condition on the target's own past, so it is
        a lead/lag index, not a Granger-causality test: a common driver
        reaching ``i`` before ``j`` also makes it positive.

    Parameters
    ----------
    statistic : {"lag", "peak", "onesided", "onesidedpeak", "asym"}
    max_lag : int >= 1 or None
        Upper limit on the window; ``None`` uses ``T // 4``.
    tau : int >= 1 or "1e", only for ``statistic="onesided"`` (default 1).

    The identifier is ``lagcorr_<statistic>[_tau-<tau>]_L-<max_lag>``, e.g.
    ``lagcorr_lag_L-None``, ``lagcorr_onesided_tau-1e_L-None``,
    ``lagcorr_onesidedpeak_L-None``.

    References
    ----------
    Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). *Time
    Series Analysis: Forecasting and Control*, 5th ed. Wiley. -- the sample
    cross-correlation function.
    Fulcher, B. D., Little, M. A. & Jones, N. S. (2013). Highly comparative
    time-series analysis: the empirical structure of time series and their
    methods. *J. R. Soc. Interface* 10, 20130048. -- hctsa's ``CO_f1ecac``,
    the first 1/e crossing of the autocorrelation function.
    Mitra, A., Snyder, A. Z., Hacker, C. D. & Raichle, M. E. (2014). Lag
    structure in resting-state fMRI. *J. Neurophysiol.* 111, 2374-2391. -- the
    lag of the cross-covariance extremum, interpolated between samples.
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

        if statistic in self._UNSIGNED:
            self.issigned = lambda: False
            sign = "unsigned"
        else:
            sign = "signed"
        self.labels = LagResolvedCorrelation.labels + [
            self._TRAITS[statistic], sign]
        self.identifier = f"lagcorr_{statistic}"
        if statistic == "onesided":
            self.identifier += f"_tau-{tau}"
        self.identifier += f"_L-{self._max_lag}"

    def _window(self, T):
        L = T // _XCORR_MAX_LAG_FRACTION
        if self._max_lag is not None:
            L = min(L, self._max_lag)
        if L < 1:
            raise ValueError(
                f"The lag window min(max_lag, T // 4) is {L} for T = {T}: "
                f"at least one lag is needed.")
        return L

    def _pair_lag(self, data, L):
        """The pair's ``onesided`` lag or ``onesidedpeak`` window, ``f(i, j)``.

        ``None`` for the statistics that read the whole window.
        """
        stat = self._statistic
        if stat == "onesided" and self._tau != "1e":
            if self._tau > L:
                raise ValueError(
                    f"tau = {self._tau} is outside the lag window "
                    f"L = min(max_lag, T // 4) = {L}.")
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
        """``d[L + tau] = r_{i->j}(tau)`` for ``tau = -L .. L``.

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
        # One orientation per pair (the lower index first), as in
        # `multivariate`, so the two agree exactly and `lag`/`asym` are
        # exactly antisymmetric and `peak` exactly symmetric.
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
        # A failing pair is NaN in both of its cells (the scalar NaN that
        # fill_pairwise stores broadcasts over the last axis).
        fill_pairwise(self.identifier, both, pairs,
                      lambda i, j: self._pair(data, i, j, L, pair_lag))
        A = np.full((M, M), np.nan)
        iu = np.triu_indices(M, 1)
        A[iu] = both[..., 0][iu]
        A[iu[1], iu[0]] = both[..., 1][iu]
        return A


class SpearmanR(Undirected, Signed):

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
        """Vectorized: Spearman's rho on the full (M, T) matrix at once."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        rho = _spearman_matrix(Z)
        if self._squared:
            rho = rho ** 2
        np.fill_diagonal(rho, np.nan)
        return rho


class KendallTau(Undirected, Signed):

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
        """Vectorized: pandas .corr(method='kendall') on (T, M) DataFrame."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        df = pd.DataFrame(Z.T)
        tau = np.asarray(df.corr(method="kendall").values, dtype=float).copy()
        if self._squared:
            tau = tau ** 2
        np.fill_diagonal(tau, np.nan)
        return tau


class LaggedCorrelation(Undirected, Signed):
    """Lagged correlation SPI.

    Computes symmetric lagged correlation: 0.5 * (corr(x[tau:], y[:-tau]) + corr(y[tau:], x[:-tau])).
    Supports Pearson, Spearman, and Kendall estimators.
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
        """Vectorized multivariate lagged correlation."""
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

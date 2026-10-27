"""Coupling of extremes and events.

* :class:`EventSynchronisation` (``es_*``): how often the extreme events of two
  series coincide, and which series' events come first.
* :class:`TailDependenceAsymmetry` (``tailasym_*``): whether two series are
  jointly extreme more often at the top or at the bottom.

Both depend on each series only through its ranks, so they are invariant under
any strictly increasing transform of either series.
"""
import numpy as np
from scipy import stats

from pyspi.base import Directed, Undirected, Signed, parse_bivariate, parse_multivariate
from pyspi.utils import fmt_param, require_int


def _require_level(name, value, low, high):
    """Return ``value`` as a float strictly inside ``(low, high)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number in ({low}, {high}), got {value!r} "
                        f"({type(value).__name__}).")
    value = float(value)
    if not (low < value < high):
        raise ValueError(f"{name} must lie strictly between {low} and {high}, got {value!r}.")
    return value


def _level_token(value):
    """``0.95`` -> ``"0-95"``: the dash-for-dot convention of the identifiers."""
    return fmt_param(value).replace(".", "-")


def _event_times(x, quantile):
    """Onset times of the runs of ``x`` above its empirical ``quantile``.

    A sample is an exceedance when ``x[t] > np.quantile(x, quantile)``; a run of
    consecutive exceedances is one event, dated at its first sample. A series
    that is constant or not finite has no events.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return np.zeros(0, dtype=np.int64)
    above = x > np.quantile(x, quantile)
    onset = above.copy()
    onset[1:] &= ~above[:-1]
    return np.flatnonzero(onset)


def _half_gaps(t):
    """``min(t[k+1] - t[k], t[k] - t[k-1]) / 2`` for each event time ``t[k]``.

    The first and last events have one neighbour, and the missing gap is left
    out of the minimum (it is ``inf``). A lone event gets ``inf``.
    """
    gaps = np.diff(np.asarray(t, dtype=float))
    previous = np.concatenate(([np.inf], gaps))
    following = np.concatenate((gaps, [np.inf]))
    return np.minimum(previous, following) / 2.0


def _es_counts(t_src, h_src, t_tgt, h_tgt, max_delay=None):
    """Directed event-synchronisation counts ``(c(tgt|src), c(src|tgt))``.

    ``c(tgt|src)`` is Quian Quiroga et al.'s ``c(y|x)`` with ``x`` the source:
    the number of target events that *follow* a source event within the local
    tolerance, ``0 < t_tgt - t_src <= tau``, plus 1/2 for each pair at equal
    times. ``c(src|tgt)`` counts the reverse. ``h_*`` are the half gaps of
    :func:`_half_gaps`, so ``tau = min(h_src, h_tgt[, max_delay])``.

    Since ``tau`` is at most half of every adjacent gap, a source event can only
    coincide with its nearest target event on either side, so two
    ``searchsorted`` candidates per source event suffice. A coincidence in both
    directions (an event exactly midway between two others) counts 1/2
    (Odenweller & Donner 2020).
    """
    m_tgt = t_tgt.size
    k = np.searchsorted(t_tgt, t_src, side="left")  # t_tgt[k] >= t_src > t_tgt[k-1]
    cap = np.inf if max_delay is None else float(max_delay)

    # Candidate 1: the first target event at or after each source event.
    a1 = np.flatnonzero(k < m_tgt)
    b1 = k[a1]
    d1 = t_tgt[b1] - t_src[a1]
    tau1 = np.minimum(np.minimum(h_src[a1], h_tgt[b1]), cap)
    equal = d1 == 0
    fwd = (d1 > 0) & (d1 <= tau1)
    fa, fb = a1[fwd], b1[fwd]  # target follows source

    # Candidate 2: the last target event strictly before each source event.
    a2 = np.flatnonzero(k > 0)
    b2 = k[a2] - 1
    d2 = t_src[a2] - t_tgt[b2]
    tau2 = np.minimum(np.minimum(h_src[a2], h_tgt[b2]), cap)
    bwd = d2 <= tau2  # d2 > 0 by construction
    pa, pb = a2[bwd], b2[bwd]  # source follows target

    src_f = np.zeros(t_src.size, dtype=bool)
    src_f[fa] = True
    tgt_f = np.zeros(m_tgt, dtype=bool)
    tgt_f[fb] = True
    src_p = np.zeros(t_src.size, dtype=bool)
    src_p[pa] = True
    tgt_p = np.zeros(m_tgt, dtype=bool)
    tgt_p[pb] = True
    double_f = np.count_nonzero(src_p[fa] | tgt_p[fb])
    double_p = np.count_nonzero(src_f[pa] | tgt_f[pb])

    n_equal = np.count_nonzero(equal)
    follow = fa.size - 0.5 * double_f + 0.5 * n_equal
    precede = pa.size - 0.5 * double_p + 0.5 * n_equal
    return follow, precede


class EventSynchronisation(Directed, Signed):
    """Event synchronisation of extreme events (Quian Quiroga et al. 2002).

    **Events.** Each series ``x`` is reduced to the times at which it exceeds
    its own empirical ``quantile`` (``x[t] > np.quantile(x, quantile)``, upper
    tail only). A run of consecutive exceedances is one event, dated at its
    first sample, so a single excursion above the threshold is not counted as
    several synchronous events. The threshold is a quantile, so the event times
    (and both statistics) are unchanged by any strictly increasing transform
    of either series, z-scoring included. For the lower tail apply the SPI to
    ``-x``.

    **Coincidences.** With event times ``t^x_1 < ... < t^x_{m_x}`` and
    ``t^y_1 < ... < t^y_{m_y}``, the pair ``(a, b)`` has the local tolerance::

        tau_ab = min(t^x_{a+1} - t^x_a, t^x_a - t^x_{a-1},
                     t^y_{b+1} - t^y_b, t^y_b - t^y_{b-1}) / 2,

    capped at ``max_delay`` if given. A gap that does not exist (before the
    first event, after the last) is left out of the minimum. Then::

        c(y|x) = sum_ab J_ab,  J_ab = 1    if 0 < t^y_b - t^x_a <= tau_ab,
                                      1/2  if t^y_b == t^x_a,
                                      0    otherwise,

    the number of times an event of ``y`` follows an event of ``x`` within the
    tolerance, and ``c(x|y)`` the same with ``x`` and ``y`` exchanged (this is
    Quian Quiroga et al.'s notation: ``c(x|y)`` counts ``x`` events that
    follow ``y`` events). An event can coincide in both directions only when it
    lies exactly midway between two events of the other series, at distance
    ``tau`` from both; each such coincidence counts 1/2 rather than 1 (the
    double-counting correction of Odenweller & Donner 2020, as in pyunicorn),
    so ``Q`` cannot exceed 1. Such ties are common with integer sample times, so
    ``Q`` differs slightly from an uncorrected Quian Quiroga et al. (2002)
    implementation.

    **Statistics** (``A[i, j]`` with row ``i`` = ``x`` the source and column
    ``j`` = ``y`` the target, the package's directed convention)::

        Q[i, j] = (c(y|x) + c(x|y)) / sqrt(m_x m_y)        in [0, 1]
        q[i, j] = (c(y|x) - c(x|y)) / sqrt(m_x m_y)        in [-1, 1]

    ``Q`` (``statistic="Q"``) is the strength of synchronisation and is
    symmetric (``undirected``, unsigned); 1 means every event of the sparser
    series has a partner and the two series have equally many events. ``q``
    (``statistic="q"``) is the delay asymmetry and is exactly antisymmetric,
    ``q[j, i] = -q[i, j]``: **``q[i, j] > 0`` means the events of row ``i``
    precede those of column ``j``**, i.e. ``i`` leads ``j``. It is 1 when
    every event of ``i`` is followed by one of ``j`` and the counts are equal.
    Equal-time pairs add 1/2 to both counts and so to ``Q`` but not to ``q``.

    ``Q`` is not corrected for chance: because the tolerance grows with the
    local gaps, independent series score well above 0. Only upper extremes are
    matched, and a delay registers only when it is shorter than the tolerance.

    A pair is NaN when either series has fewer than two events (no gap, hence
    no tolerance), which includes a constant series.

    Parameters
    ----------
    statistic : {"Q", "q"}
        Synchronisation strength or delay asymmetry, as above.
    quantile : float in (0, 1)
        Exceedance level defining events (default 0.95, the top 5% of
        samples). Appears in the identifier as ``q-0-95``.
    max_delay : int >= 1 or None
        Global cap on the tolerance ``tau``, in samples (default ``None``, no
        cap: the tolerance is set by the local inter-event gaps alone). When
        set it is appended to the identifier as ``_maxdelay-<d>``.

    References
    ----------
    R. Quian Quiroga, T. Kreuz and P. Grassberger, "Event synchronization: a
    simple and fast method to measure synchronicity and time delay patterns",
    Phys. Rev. E 66, 041904 (2002).
    A. Odenweller and R. V. Donner, "Disentangling synchrony from serial
    dependency in paired-event time series", Phys. Rev. E 101, 052213 (2020).
    """

    name = "Event synchronisation"
    identifier = "es"
    labels = ["misc", "nonlinear", "temporal"]
    # Event times are cached per quantile, shared by Q, q and every max_delay.
    _cache_namespace = "events"

    @property
    def _cache_subkey(self):
        return (self._quantile,)

    def __init__(self, statistic="Q", quantile=0.95, max_delay=None):
        if statistic not in ("Q", "q"):
            raise ValueError(f"statistic must be 'Q' or 'q', got {statistic!r}.")
        self._statistic = statistic
        self._quantile = _require_level("quantile", quantile, 0.0, 1.0)
        self._max_delay = (
            None if max_delay is None
            else require_int("max_delay", max_delay, minimum=1)
        )
        trait = "undirected" if statistic == "Q" else "antisymmetric"
        self.labels = EventSynchronisation.labels + [trait]
        self.identifier = (
            f"{EventSynchronisation.identifier}_{statistic}"
            f"_q-{_level_token(self._quantile)}"
        )
        if self._max_delay is not None:
            self.identifier += f"_maxdelay-{self._max_delay}"

    def issigned(self):
        # Q is a normalised count in [0, 1]; q carries the lead/lag in its sign.
        return self._statistic == "q"

    def _events(self, data):
        """``[(event times, half gaps)]`` per process, cached on ``data``."""
        try:
            return data.events[self._quantile]
        except (AttributeError, KeyError):
            z = np.atleast_2d(data.to_numpy(squeeze=True))
            events = []
            for row in z:
                t = _event_times(row, self._quantile)
                events.append((t, _half_gaps(t)))
            try:
                data.events[self._quantile] = events
            except AttributeError:
                data.events = {self._quantile: events}
            return events

    def _pair(self, source, target):
        """``(Q, q)`` with ``source`` on the row, ``target`` on the column."""
        (t_x, h_x), (t_y, h_y) = source, target
        if t_x.size < 2 or t_y.size < 2:
            return np.nan, np.nan
        follow, precede = _es_counts(t_x, h_x, t_y, h_y, self._max_delay)
        norm = np.sqrt(float(t_x.size) * float(t_y.size))
        return (follow + precede) / norm, (follow - precede) / norm

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        events = self._events(data)
        Q, q = self._pair(events[i], events[j])
        return Q if self._statistic == "Q" else q

    @parse_multivariate
    def multivariate(self, data):
        events = self._events(data)
        M = len(events)
        A = np.full((M, M), np.nan)
        for i in range(M):
            for j in range(i + 1, M):
                Q, q = self._pair(events[i], events[j])
                if self._statistic == "Q":
                    A[i, j] = A[j, i] = Q
                else:
                    # Exact: c(i|j) and c(j|i) swap roles, the norm is shared.
                    A[i, j], A[j, i] = q, -q
        return A


class TailDependenceAsymmetry(Undirected, Signed):
    """Asymmetry between upper and lower empirical tail dependence.

    With ``R_x(t)`` the rank of ``x[t]`` among the ``T`` samples (average ranks
    for ties), level ``u = quantile`` and ``k = floor((1 - u) T)``, the upper
    and lower tails of ``x`` are the ``k`` samples with ``R_x > T - k`` and the
    ``k`` samples with ``R_x <= k``. The empirical tail-dependence coefficients
    (Schmidt & Stadtmueller 2006) are::

        lambda_U = #{t : R_x(t) > T - k and R_y(t) > T - k} / k
        lambda_L = #{t : R_x(t) <= k    and R_y(t) <= k}    / k

    each the fraction of one series' tail samples at which the other is in the
    same tail, the estimator of ``P(F_y(y) > u | F_x(x) > u)`` at a finite
    level. Without ties both tails of both series hold exactly ``k`` samples,
    so the coefficient is the same whichever series is conditioned on
    (undirected by construction). With ties (average ranks) a tail can hold a
    few more or fewer samples, and the denominator is the geometric mean of
    the two tail counts, which is ``k`` without ties and keeps the coefficient
    in ``[0, 1]``.

    ``statistic="asymmetry"`` (the bundled one) reports
    ``lambda_U - lambda_L`` in ``[-1, 1]``: positive when the two series are
    jointly extreme more often at the top than at the bottom, negative for the
    reverse. ``statistic="upper"`` and ``"lower"`` give the coefficients
    themselves (unsigned, in ``[0, 1]``). Reflecting both series
    (``x, y -> -x, -y``) swaps the two tails and negates the asymmetry
    exactly.

    Under any radially symmetric copula, the elliptical (e.g. Gaussian or
    Student-t) copulas included, ``lambda_U = lambda_L`` at every level, so the
    asymmetry is zero up to sampling error whatever the correlation. A Clayton
    copula (lower-tail dependent) gives a negative value, a Gumbel copula a
    positive one.

    A pair is NaN when either series is constant or not finite, or when
    ``k < 1``. Being rank-based, the statistic is unchanged by any strictly
    increasing transform of either series. It ignores time order
    (contemporaneous).

    Parameters
    ----------
    quantile : float in (0.5, 1)
        Tail level ``u`` (default 0.95: the top and bottom 5%). Appears in the
        identifier as ``q-0-95``.
    statistic : {"asymmetry", "upper", "lower"}
        ``lambda_U - lambda_L`` (identifier ``tailasym_q-<u>``), or one
        coefficient (``tailasym_upper_q-<u>``, ``tailasym_lower_q-<u>``).

    References
    ----------
    H. Joe, *Multivariate Models and Dependence Concepts*, Chapman & Hall
    (1997), Sec. 2.1.10 (tail dependence).
    R. Schmidt and U. Stadtmueller, "Non-parametric estimation of tail
    dependence", Scand. J. Stat. 33, 307-335 (2006).
    """

    name = "Tail-dependence asymmetry"
    identifier = "tailasym"
    labels = ["misc", "rank", "nonlinear", "unordered", "undirected"]

    _STATISTICS = ("asymmetry", "upper", "lower")

    def __init__(self, quantile=0.95, statistic="asymmetry"):
        if statistic not in self._STATISTICS:
            raise ValueError(f"statistic must be one of {self._STATISTICS}, got {statistic!r}.")
        self._statistic = statistic
        self._quantile = _require_level("quantile", quantile, 0.5, 1.0)
        self.labels = list(TailDependenceAsymmetry.labels)
        stat = "" if statistic == "asymmetry" else f"_{statistic}"
        self.identifier = (
            f"{TailDependenceAsymmetry.identifier}{stat}"
            f"_q-{_level_token(self._quantile)}"
        )

    def issigned(self):
        return self._statistic == "asymmetry"

    def _coefficients(self, Z):
        """``(lambda_U, lambda_L)`` as ``M x M`` matrices for the rows of ``Z``."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        M, T = Z.shape
        # floor((1 - u) T), robust to (1 - 0.9) * 1000 = 99.99999999999999.
        k = int(np.floor((1.0 - self._quantile) * T + 1e-9))
        valid = np.all(np.isfinite(Z), axis=1) & (np.ptp(Z, axis=1) > 0)
        if k < 1:
            valid[:] = False
        R = np.zeros_like(Z)
        if valid.any():
            R[valid] = stats.rankdata(Z[valid], axis=1)
        out = []
        for tail in (R > T - k, R <= k):
            tail = (tail & valid[:, None]).astype(float)
            both = tail @ tail.T
            n = np.diag(both).copy()
            with np.errstate(invalid="ignore", divide="ignore"):
                lam = both / np.sqrt(np.outer(n, n))
            bad = ~valid | (n == 0)
            lam[bad, :] = np.nan
            lam[:, bad] = np.nan
            out.append(lam)
        return out

    def _select(self, upper, lower):
        if self._statistic == "upper":
            return upper
        if self._statistic == "lower":
            return lower
        return upper - lower

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        upper, lower = self._coefficients(np.vstack([z[i], z[j]]))
        return float(self._select(upper, lower)[0, 1])

    @parse_multivariate
    def multivariate(self, data):
        upper, lower = self._coefficients(data.to_numpy(squeeze=True))
        A = self._select(upper, lower)
        np.fill_diagonal(A, np.nan)
        return A

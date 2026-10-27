"""Statistics read off one vector autoregressive (VAR) model of all processes.

A VAR fitted to all ``M`` processes at once conditions each link on the past
of every other process. The fit is computed once per dataset and cached on
``data.var``.

Conventions:

* ``Z`` is the ``(M, T)`` array of a :class:`~pyspi.data.Data` (z-scored by
  default); a VAR(p) predicts ``Z[:, t]`` from ``Z[:, t-1], ..., Z[:, t-p]``
  and an intercept, for ``t = p, ..., T-1``.
* ``coef[k - 1, j, i]`` is the coefficient of process ``i`` at lag ``k`` in the
  equation of process ``j`` (target ``j``, source ``i``): the layout of
  ``statsmodels``' ``VARResults.coefs``.
* The residual covariance divides by the number of fitted samples
  ``N = T - p`` (the maximum-likelihood estimate, ``statsmodels``'
  ``sigma_u_mle``), not by the residual degrees of freedom.
"""

import warnings

import numpy as np

from pyspi.base import Directed, Unsigned, parse_bivariate, parse_multivariate
from pyspi.utils import require_int

DEFAULT_MAX_ORDER = 10

# Above this condition number of the lagged design, restricted residual sums
# of squares are refitted by least squares rather than read off the full fit.
_PARTITIONED_MAX_COND = 1e6

# BIC needs ln det Sigma. A residual covariance whose smallest eigenvalue is
# below this fraction of its largest is treated as singular: the processes are
# linearly dependent and the criterion is rounding noise.
_SINGULAR_RCOND = 1e-12


def max_feasible_order(n_processes, n_observations):
    """The largest VAR order whose least-squares fit is well posed.

    A VAR(p) on ``M`` processes fits ``N = T - p`` samples with ``K = M p + 1``
    regressors per equation. The cap

        p_max = (T - 1) // (M + 1) - 1

    leaves ``N - K >= M + 1`` residual degrees of freedom, so the ``M x M``
    residual covariance can be nonsingular. Returns 0 when not even ``p = 1``
    qualifies.
    """
    return max((int(n_observations) - 1) // (int(n_processes) + 1) - 1, 0)


def lagged_design(Z, order, start=None):
    """Targets ``Y`` (``N x M``) and regressors ``X`` (``N x (1 + M p)``) of a VAR(p).

    Rows are the samples ``t = start, ..., T-1`` (``start`` defaults to ``p``;
    order selection passes a larger one so every candidate order is fitted on
    the same samples). Column 0 of ``X`` is the intercept and column
    ``1 + (k - 1) M + i`` is process ``i`` at lag ``k``.
    """
    Z = np.asarray(Z, dtype=float)
    M, T = Z.shape
    p = int(order)
    start = p if start is None else int(start)
    if start < p:
        raise ValueError(f"start={start} is before the first sample with {p} lags.")
    X = np.empty((T - start, 1 + M * p))
    X[:, 0] = 1.0
    for k in range(1, p + 1):
        X[:, 1 + (k - 1) * M:1 + k * M] = Z[:, start - k:T - k].T
    return Z[:, start:].T, X


def fit_var(Z, order, start=None):
    """Least-squares fit of a VAR(p) with intercept to the rows of ``Z``.

    Every equation shares the regressors, so the fit is one minimum-norm
    least-squares solve by SVD. Returns a dict with

    ``coef``
        ``(p, M, M)``; ``coef[k - 1, j, i]`` is the coefficient of source
        ``i`` at lag ``k`` in the equation of target ``j``.
    ``intercept``
        ``(M,)``.
    ``B``
        ``(1 + M p, M)``: intercept and lag coefficients stacked in the
        design's column order, one column per target equation.
    ``sigma``
        ``(M, M)`` residual covariance ``E'E / N``, ``N`` the number of fitted
        samples (``T - p`` by default): the maximum-likelihood estimate.
    ``rss``
        ``(M,)`` residual sum of squares of each equation, ``N diag(sigma)``.
    ``X``, ``Y``
        The lagged design and the targets, so that restricted models can be
        fitted from the same rows (:func:`restricted_rss_increase`).
    ``order``, ``n_samples``
        ``p`` and ``N``.
    """
    Y, X = lagged_design(Z, order, start)
    N, K = X.shape
    M = Y.shape[1]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # lstsq's default cutoff, so exactly collinear processes still give the
    # minimum-norm least-squares fit.
    tol = s[0] * max(N, K) * np.finfo(float).eps
    s_inv = np.divide(1.0, s, out=np.zeros_like(s), where=s > tol)
    B = Vt.T @ (s_inv[:, None] * (U.T @ Y))
    E = Y - X @ B
    p = int(order)
    return {
        "order": p,
        "n_samples": N,
        "coef": B[1:].reshape(p, M, M).transpose(0, 2, 1),
        "intercept": B[0],
        "B": B,
        "sigma": E.T @ E / N,
        "rss": np.einsum("nm,nm->m", E, E),
        "X": X,
        "Y": Y,
        "_svd": (s, Vt),
        "_full_rank": bool(np.all(s > tol)),
    }


def _source_columns(source, n_processes, order):
    """Columns of the lagged design holding lags 1..p of ``source``."""
    return 1 + source + n_processes * np.arange(order)


def restricted_rss_increase(fit, source):
    """``RSS_restricted - RSS_full`` for every target when ``source``'s lags are dropped.

    The restricted model of target ``j`` keeps the intercept and the ``p`` lags
    of every process except ``source``. For a well-conditioned design the
    increase is read off the full fit (the Wald identity for nested least
    squares),

        RSS_r - RSS_f = b_S' [(X'X)^{-1}]_SS^{-1} b_S,

    with ``S`` the source's ``p`` columns and ``b_S`` the target's full-model
    coefficients on them. Near collinearity the restricted model is refitted
    instead. Returns an ``(M,)`` array indexed by target (the entry for
    ``target == source`` is not a Granger causality).
    """
    M = fit["Y"].shape[1]
    p = fit["order"]
    cols = _source_columns(source, M, p)
    s, Vt = fit["_svd"]
    if fit["_full_rank"] and s[0] <= _PARTITIONED_MAX_COND * s[-1]:
        W = Vt.T[cols] / s                      # rows S of V diag(1/s)
        H_SS = W @ W.T                          # [(X'X)^{-1}]_SS
        b_S = fit["B"][cols]                    # (p, M)
        return np.einsum("km,km->m", b_S, np.linalg.solve(H_SS, b_S))
    keep = np.ones(fit["X"].shape[1], dtype=bool)
    keep[cols] = False
    X_r, Y = fit["X"][:, keep], fit["Y"]
    B_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
    E_r = Y - X_r @ B_r
    return np.einsum("nm,nm->m", E_r, E_r) - fit["rss"]


def select_order_bic(Z, max_order=DEFAULT_MAX_ORDER):
    """The VAR order in ``1..max_order`` that minimises BIC, and the criteria.

    ``BIC(p) = ln det Sigma_p + (M^2 p + M) ln(N) / N``, with ``Sigma_p`` the
    maximum-likelihood residual covariance of the full VAR(p) and
    ``M^2 p + M`` its free parameters (lag coefficients and intercepts). Every
    candidate is fitted on the same samples ``t = p_max, ..., T-1``
    (``N = T - p_max``) so the criteria are comparable (Lütkepohl 2005,
    Sec. 4.3).
    ``p_max = min(max_order, max_feasible_order(M, T))``. Ties go to the
    smaller order.

    Raises ``ValueError`` if no order is feasible, or if a residual covariance
    is singular (linearly dependent processes), where ``ln det`` is undefined.
    """
    Z = np.asarray(Z, dtype=float)
    M, T = Z.shape
    max_order = require_int("max_order", max_order, minimum=1)
    cap = max_feasible_order(M, T)
    p_max = min(max_order, cap)
    if p_max < 1:
        raise ValueError(
            f"No feasible VAR order for M={M} processes and T={T} observations: "
            f"a well-posed VAR(p) fit needs p <= (T - 1) // (M + 1) - 1 = {cap}."
        )
    N = T - p_max
    bic = np.empty(p_max)
    for p in range(1, p_max + 1):
        sigma = fit_var(Z, p, start=p_max)["sigma"]
        eig = np.linalg.eigvalsh(sigma)
        if eig[0] <= eig[-1] * _SINGULAR_RCOND:
            raise ValueError(
                f"The VAR({p}) residual covariance is singular (smallest/largest "
                f"eigenvalue {eig[0] / eig[-1]:.2g}): the processes are linearly "
                f"dependent, so ln det Sigma and BIC order selection are undefined. "
                f"Use a fixed order."
            )
        bic[p - 1] = np.sum(np.log(eig)) + (M * M * p + M) * np.log(N) / N
    return int(np.argmin(bic)) + 1, bic


class ConditionalGrangerCausality(Directed, Unsigned):
    """Conditional (multivariate) Granger causality from one VAR of all processes.

    A VAR(p) with intercept is fitted to all ``M`` processes by ordinary least
    squares (the full model). For ``i -> j``, the restricted model of target
    ``j`` drops all ``p`` lags of source ``i``. With ``var_f`` and ``var_r`` the
    residual variances of ``j`` under the two models (same ``N = T - p``
    samples, both divided by ``N``),

        F[i, j] = ln(var_r / var_f)                (statistic="F", default)

    in nats (Geweke 1984). It is zero when the source's past adds nothing to
    the prediction of the target beyond the past of all other processes, so an
    indirect path ``i -> k -> j`` or a common driver gives ``F[i, j]`` near zero.
    With no causal link, ``N F`` is asymptotically chi-squared with ``p`` degrees
    of freedom, so ``F`` is positively biased by about ``p / N``.
    This is twice the scale of pyspi's bivariate ``gc_gaussian_*``
    (``0.5 ln(var_r / var_f)``); with two processes
    ``cgc_p-1 = 2 * gc_gaussian_k-1_kt-1_l-1_lt-1``.

    ``statistic="coef"`` is the signed net lagged coefficient
    ``sum_k A[k, j, i]`` of source ``i`` in target ``j``'s equation. It is in
    units of the target per unit of the source, and lags of opposite sign can
    cancel, so a strong influence can have a small ``coef``.

    Direction: row = source, column = target, so ``A[i, j]`` is ``i -> j``.
    The diagonal is NaN.

    The fit needs ``p <= (T - 1) // (M + 1) - 1``; a fixed order above this
    raises ``ValueError``, and ``"bic"`` searches up to it. The order chosen by
    BIC depends on the data and is not in the identifier.

    Parameters:
        order (int or "bic"): VAR model order, or "bic" to choose it by the
            Bayesian information criterion (default).
        max_order (int, optional): largest order ``"bic"`` considers, default
            10. Only valid with ``order="bic"``; a non-default value is
            appended to the identifier as ``_pmax-<n>``.
        statistic ("F" or "coef"): the log variance ratio (unsigned, default)
            or the net lagged coefficient (signed, identifier suffix
            ``_coef``).

    References:
        Geweke, J. (1984). Measures of conditional linear dependence and
        feedback between time series. *J. Am. Stat. Assoc.* 79, 907-915.

        Barnett, L. & Seth, A. K. (2014). The MVGC multivariate Granger
        causality toolbox. *J. Neurosci. Methods* 223, 50-68.

        Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer.
    """

    name = "Conditional Granger causality"
    identifier = "cgc"
    labels = ["causal", "directed", "linear", "temporal", "unsigned", "multivariate"]
    # Variants with the same (order, max_order) share one fit in data.var.
    _cache_namespace = "var"

    @property
    def _cache_subkey(self):
        return (self._order, self._max_order)

    def __init__(self, order="bic", max_order=None, statistic="F"):
        if isinstance(order, str):
            if order != "bic":
                raise ValueError(f"order must be a positive int or 'bic', got {order!r}.")
            self._max_order = require_int(
                "max_order", DEFAULT_MAX_ORDER if max_order is None else max_order,
                minimum=1)
        else:
            order = require_int("order", order, minimum=1)
            if max_order is not None:
                raise ValueError(
                    f"max_order={max_order!r} is only used by order='bic'; with "
                    f"a fixed order={order} there is nothing to select.")
            self._max_order = None
        if statistic not in ("F", "coef"):
            raise ValueError(f"statistic must be 'F' or 'coef', got {statistic!r}.")
        self._order = order
        self._statistic = statistic

        self.identifier = f"{self.identifier}_p-{order}"
        if order == "bic" and self._max_order != DEFAULT_MAX_ORDER:
            self.identifier += f"_pmax-{self._max_order}"
        if statistic == "coef":
            self.identifier += "_coef"
            self.labels = [label if label != "unsigned" else "signed"
                           for label in self.labels]

    def issigned(self):
        return self._statistic == "coef"

    def _from_cache(self, data):
        cache = getattr(data, "var", None)
        if cache is None:
            cache = data.var = {}
        key = self._cache_subkey
        if key not in cache:
            Z = np.asarray(data.to_numpy(squeeze=True), dtype=float)
            M, T = Z.shape
            bic = None
            if self._order == "bic":
                p, bic = select_order_bic(Z, self._max_order)
            else:
                p = self._order
                cap = max_feasible_order(M, T)
                if p > cap:
                    raise ValueError(
                        f"VAR order p={p} is infeasible for M={M} processes and "
                        f"T={T} observations: a well-posed fit needs "
                        f"p <= (T - 1) // (M + 1) - 1 = {cap}."
                    )
            fit = fit_var(Z, p)
            fit["bic"] = bic
            fit["rss_increase"] = {}            # source -> (M,), filled lazily
            cache[key] = fit
        return cache[key]

    @staticmethod
    def _rss_increase(fit, source):
        rows = fit["rss_increase"]
        if source not in rows:
            rows[source] = restricted_rss_increase(fit, source)
        return rows[source]

    @staticmethod
    def _log_ratio(increase, rss_full):
        # Nested least squares: the increase is >= 0; clip rounding residue.
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.log1p(np.maximum(increase, 0.0) / rss_full)
        return np.where(np.isfinite(F), F, np.nan)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if i == j:
            return np.nan
        fit = self._from_cache(data)
        if self._statistic == "coef":
            return float(fit["coef"][:, j, i].sum())
        return float(self._log_ratio(self._rss_increase(fit, i)[j], fit["rss"][j]))

    @parse_multivariate
    def multivariate(self, data):
        fit = self._from_cache(data)
        M = fit["Y"].shape[1]
        if self._statistic == "coef":
            A = fit["coef"].sum(axis=0).T           # [j, i] -> [i, j]
        else:
            increase = np.vstack([self._rss_increase(fit, i) for i in range(M)])
            A = self._log_ratio(increase, fit["rss"][None, :])
        A = np.array(A, dtype=float)                # never the cached arrays
        np.fill_diagonal(A, np.nan)
        undefined = ~np.isfinite(A) & ~np.eye(M, dtype=bool)
        if undefined.any():
            warnings.warn(
                f'SPI "{self.identifier}": {int(undefined.sum())} of {M * (M - 1)} '
                f'pairs are undefined (a target with zero residual variance) '
                f'and set to NaN.'
            )
        return A

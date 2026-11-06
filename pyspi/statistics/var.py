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

Two families read the fit: :class:`ConditionalGrangerCausality` (``cgc_*``)
and the conditional spectral measures (:class:`ConditionalSpectral`:
``cpdc_*``, ``cgpdc_*``, ``cddtf_*``, ``pcohmag_*``), evaluated on a grid of
frequencies in cycles per sample.
"""

import warnings

import numpy as np

from pyspi.base import (Directed, Undirected, Unsigned, parse_bivariate,
                        parse_multivariate)
from pyspi.statistics.spectral import _closed_band
from pyspi.utils import fmt_param, require_int

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


# ---------------------------------------------------------------------------
# Conditional spectral measures: PDC, gPDC, dDTF and partial coherence of the
# same VAR of all processes.
# ---------------------------------------------------------------------------

DEFAULT_NFREQ = 256


def frequency_grid(nfreq=DEFAULT_NFREQ):
    """``nfreq`` evenly spaced frequencies over [0, 0.5] cycles per sample, both ends included."""
    return np.linspace(0.0, 0.5, require_int("nfreq", nfreq, minimum=2))


def var_frequency_response(coef, freq):
    """``A(f) = I - sum_k A_k exp(-2 pi i f k)`` at each frequency in ``freq``.

    ``coef`` is a fit's ``(p, M, M)`` lag tensor (``coef[k - 1, j, i]``:
    target ``j``, source ``i``). Returns an ``(F, M, M)`` complex array indexed
    ``[f, target, source]``: the Fourier transform of the VAR's lag
    polynomial. Its inverse is the transfer function ``H(f)``, and the model's
    spectral matrix is ``S(f) = H Sigma H^H``.
    """
    coef = np.asarray(coef, dtype=float)
    p, M, _ = coef.shape
    phase = np.exp(-2j * np.pi * np.outer(freq, np.arange(1, p + 1)))  # (F, p)
    return np.eye(M) - np.einsum("fk,kji->fji", phase, coef)


def full_frequency_dtf(H):
    """Full-frequency DTF of a transfer function ``H[f, target, source]`` (Korzeniewska et al. 2003).

    ``eta_ij(f) = |H_ij(f)| / sqrt(sum_f sum_k |H_ik(f)|^2)``, target ``i``,
    source ``j``: each target's inflow normalised over every source and every
    frequency of the grid, so ``sum_f sum_j eta_ij(f)^2 = 1``. Same layout as
    ``H``.
    """
    power = np.abs(H) ** 2
    return np.sqrt(power / power.sum(axis=(0, 2), keepdims=True))


def _innovation_precision(fit):
    """``Sigma^{-1}``, refusing a residual covariance that has none."""
    sigma = fit["sigma"]
    eig = np.linalg.eigvalsh(sigma)
    if not (eig[-1] > 0 and eig[0] > eig[-1] * _SINGULAR_RCOND):
        raise ValueError(
            f"The VAR({fit['order']}) residual covariance is singular "
            f"(eigenvalues {eig[0]:.3g} to {eig[-1]:.3g}): the processes are "
            f"linearly dependent, so the inverse spectral matrix, the partial "
            f"coherence and dDTF are undefined."
        )
    return np.linalg.inv(sigma)


def conditional_spectrum(fit, measure, nfreq=DEFAULT_NFREQ):
    """One conditional spectral measure of a VAR fit at every frequency of the grid.

    ``measure`` is ``"pdc"`` (``|pi|^2``), ``"gpdc"`` (generalised,
    ``|pi^g|^2``), ``"pcoh"`` (squared partial coherence ``|P|^2``) or
    ``"ddtf"`` (``eta |P|``); see :class:`ConditionalSpectral` for the
    definitions. Returns an ``(nfreq, M, M)`` read-only array indexed
    ``[f, source, target]`` (pyspi's row = source), diagonal included, on
    ``frequency_grid(nfreq)``. It is computed once and kept on the fit.
    """
    store = fit.setdefault("spectra", {})
    key = (measure, int(nfreq))
    if key in store:
        return store[key]
    freq = frequency_grid(nfreq)
    # Every array below is [f, target, source], the literature's layout, until
    # the last step.
    if measure in ("pdc", "gpdc"):
        power = np.abs(var_frequency_response(fit["coef"], freq)) ** 2
        if measure == "gpdc":
            variance = np.diag(fit["sigma"]).copy()
            if not np.all(variance > 0):
                raise ValueError(
                    "gPDC is undefined: a process has zero innovation variance "
                    "(it is predicted exactly by the VAR).")
            power = power / variance[:, None]       # 1 / sigma_k^2 on target k
        with np.errstate(divide="ignore", invalid="ignore"):
            spec = power / power.sum(axis=-2, keepdims=True)    # over targets
    elif measure == "pcoh":
        # S^{-1} = A^H Sigma^{-1} A, since S = H Sigma H^H and H = A^{-1}.
        # Made exactly Hermitian so that |P_ij| == |P_ji| exactly.
        A = var_frequency_response(fit["coef"], freq)
        Q = np.conj(np.swapaxes(A, -1, -2)) @ _innovation_precision(fit) @ A
        Q = 0.5 * (Q + np.conj(np.swapaxes(Q, -1, -2)))
        q = np.real(np.diagonal(Q, axis1=-2, axis2=-1))
        spec = np.abs(Q) ** 2 / (q[..., :, None] * q[..., None, :])
    elif measure == "ddtf":
        H = np.linalg.inv(var_frequency_response(fit["coef"], freq))
        pcoh = np.swapaxes(conditional_spectrum(fit, "pcoh", nfreq), -1, -2)
        spec = full_frequency_dtf(H) * np.sqrt(pcoh)
    else:
        raise ValueError(f"Unknown conditional spectral measure {measure!r}.")
    spec = np.ascontiguousarray(np.swapaxes(spec, -1, -2))   # [f, source, target]
    spec.setflags(write=False)
    store[key] = spec
    return spec


def _parse_order(order, max_order):
    """``(order, max_order)`` validated as :class:`ConditionalGrangerCausality` does."""
    if isinstance(order, str):
        if order != "bic":
            raise ValueError(f"order must be a positive int or 'bic', got {order!r}.")
        return order, require_int(
            "max_order", DEFAULT_MAX_ORDER if max_order is None else max_order,
            minimum=1)
    order = require_int("order", order, minimum=1)
    if max_order is not None:
        raise ValueError(
            f"max_order={max_order!r} is only used by order='bic'; with "
            f"a fixed order={order} there is nothing to select.")
    return order, None


def _band_edge(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer,
                                                          np.floating)):
        raise TypeError(f"{name} must be a number in [0, 0.5], got {value!r}.")
    # An int stays an int, so the identifier reads fmin-0 rather than fmin-0-0.
    value = int(value) if isinstance(value, (int, np.integer)) else float(value)
    if not 0 <= value <= 0.5:
        raise ValueError(
            f"{name}={value!r} is outside [0, 0.5] cycles per sample.")
    return value


class ConditionalSpectral(Unsigned):
    """Conditional (M-variate, parametric) spectral measures from one VAR of all processes.

    Base class of ``cpdc_*``, ``cgpdc_*``, ``cddtf_*`` and ``pcohmag_*``. Each
    is read off the same OLS VAR(p) with intercept that ``cgc_*`` fits to all
    ``M`` processes. With ``A_k`` the lag-``k`` coefficient matrix (row = target,
    column = source) and ``Sigma`` the residual covariance, on a grid of
    ``nfreq`` frequencies ``f`` evenly spaced over [0, 0.5] cycles per sample,

        A(f) = I - sum_k A_k exp(-2 pi i f k),   H(f) = A(f)^{-1},
        S(f) = H(f) Sigma H(f)^H,                S(f)^{-1} = A(f)^H Sigma^{-1} A(f).

    In the literature's ``[target i, source j]`` notation:

    * PDC (``cpdc``; Baccalá & Sameshima 2001):
      ``|pi_ij(f)|^2 = |A_ij(f)|^2 / sum_k |A_kj(f)|^2``, source ``j``'s outflow
      normalised over every target ``k`` (itself included), so
      ``sum_i |pi_ij(f)|^2 = 1`` at every frequency.
    * gPDC (``cgpdc``; Baccalá, Sameshima & Takahashi 2007): the same with each
      target weighted by its innovation variance ``sigma_k^2 = Sigma_kk``,
      ``|A_ij|^2 / sigma_i^2 / sum_k (|A_kj|^2 / sigma_k^2)``: unlike PDC,
      unchanged by rescaling any one process.
    * Partial coherence (``pcohmag``; Dahlhaus 2000):
      ``|P_ij(f)|^2 = |Q_ij|^2 / (Q_ii Q_jj)`` with ``Q = S(f)^{-1}``, the
      coherence of ``i`` and ``j`` once the linear influence of every other
      process, at all lags, is removed from both. Symmetric.
    * dDTF (``cddtf``; Korzeniewska et al. 2003):
      ``eta_ij(f) |P_ij(f)|``, the full-frequency DTF
      ``eta_ij(f) = |H_ij(f)| / sqrt(sum_f sum_k |H_ik(f)|^2)`` (target ``i``'s
      inflow normalised over every source and every grid frequency) gated by
      the partial coherence, so that only direct paths survive.

    Scale: ``cpdc``, ``cgpdc`` and ``pcohmag`` report squared magnitudes
    (``|pi|^2``, ``|P|^2``), as ``pdcoh_*``, ``gpdcoh_*`` and ``cohmag_*`` do;
    ``cddtf`` reports the magnitude ``eta |P|``, as ``ddtf_*`` does. All lie in
    [0, 1]. ``eta`` is normalised by a sum over the grid, so dDTF scales like
    ``1 / sqrt(nfreq)`` and is comparable only at a fixed ``nfreq``.

    Band statistic: ``statistic="mean"`` or ``"max"`` of the spectrum over the
    grid frequencies in the closed band ``fmin <= f <= fmax``.

    Direction: row = source, column = target, so ``A[i, j]`` is ``i -> j``
    (the transpose of the literature's ``[target, source]``). The diagonal is
    NaN.

    Unlike the bivariate ``pdcoh_*``, ``dtf_*`` and ``ddtf_*``, these condition
    on every other process: PDC is zero at all frequencies exactly when there
    is no direct link ``i -> j`` (the same null as ``cgc_*``), and dDTF and
    the partial coherence are near zero for a link only through other
    processes.

    Model order: as ``cgc_*`` (``order`` a positive int or ``"bic"``,
    ``max_order`` only with ``"bic"``), whose cached fit these share.

    Parameters:
        order (int or "bic"): VAR model order, default "bic".
        max_order (int, optional): largest order ``"bic"`` considers,
            default 10 (identifier ``_pmax-<n>`` when not the default).
        statistic ("mean" or "max"): band statistic, default "mean".
        fmin, fmax (float): band edges in cycles per sample,
            ``0 <= fmin <= fmax <= 0.5``; default the full band [0, 0.5]. The
            band must hold at least one grid frequency.
        nfreq (int): number of grid frequencies over [0, 0.5], default 256
            (identifier ``_nf-<n>`` when not the default).

    Identifiers: ``<stem>_p-<order>_<statistic>_fmin-<fmin>_fmax-<fmax>``
    with ``.`` written ``-`` (``cpdc_p-bic_mean_fmin-0_fmax-0-5``).

    References:
        Baccalá, L. A. & Sameshima, K. (2001). Partial directed coherence: a
        new concept in neural structure determination. *Biol. Cybern.* 84,
        463-474.

        Baccalá, L. A., Sameshima, K. & Takahashi, D. Y. (2007). Generalized
        partial directed coherence. *15th Int. Conf. Digital Signal
        Processing*, 163-166.

        Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J. &
        Kasicki, S. (2003). Determination of information flow direction among
        brain structures by a modified directed transfer function (dDTF)
        method. *J. Neurosci. Methods* 125, 195-207.

        Dahlhaus, R. (2000). Graphical interaction models for multivariate
        time series. *Metrika* 51, 157-172.
    """

    # One fit per (order, max_order), shared with cgc_* in data.var.
    _cache_namespace = "var"
    # Set by each subclass: the `conditional_spectrum` measure it reports.
    _measure = None
    _from_cache = ConditionalGrangerCausality._from_cache

    @property
    def _cache_subkey(self):
        return (self._order, self._max_order)

    def __init__(self, order="bic", max_order=None, statistic="mean", fmin=0,
                 fmax=0.5, nfreq=DEFAULT_NFREQ):
        self._order, self._max_order = _parse_order(order, max_order)
        if statistic not in ("mean", "max"):
            raise ValueError(f"statistic must be 'mean' or 'max', got {statistic!r}.")
        self._statistic = statistic
        self._statfn = np.mean if statistic == "mean" else np.max
        self._nfreq = require_int("nfreq", nfreq, minimum=2)
        fmin, fmax = _band_edge("fmin", fmin), _band_edge("fmax", fmax)
        if fmin > fmax:
            raise ValueError(f"fmin={fmin!r} is above fmax={fmax!r}.")
        self._fmin, self._fmax = fmin, fmax
        grid = frequency_grid(self._nfreq)
        self._band = np.flatnonzero(_closed_band(grid, fmin, fmax))
        if self._band.size == 0:
            raise ValueError(
                f"The band [{fmin}, {fmax}] holds no frequency of the "
                f"{self._nfreq}-point grid (step {grid[1]:.4g}); widen it or "
                f"raise nfreq.")

        identifier = f"{self.identifier}_p-{self._order}"
        if self._order == "bic" and self._max_order != DEFAULT_MAX_ORDER:
            identifier += f"_pmax-{self._max_order}"
        identifier += (f"_{statistic}_fmin-{fmt_param(fmin)}"
                       f"_fmax-{fmt_param(fmax)}").replace(".", "-")
        if self._nfreq != DEFAULT_NFREQ:
            identifier += f"_nf-{self._nfreq}"
        self.identifier = identifier

    @parse_multivariate
    def spectrum(self, data):
        """``(freq, S)``: the measure at every grid frequency, before the band statistic.

        ``S`` is ``(nfreq, M, M)``, indexed ``[f, source, target]`` with the
        diagonal included; it is the cached array, read-only.
        """
        fit = self._from_cache(data)
        return (frequency_grid(self._nfreq),
                conditional_spectrum(fit, self._measure, self._nfreq))

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if i == j:
            return np.nan
        _, spec = self.spectrum(data)
        return float(self._statfn(spec[self._band, i, j]))

    @parse_multivariate
    def multivariate(self, data):
        _, spec = self.spectrum(data)
        A = np.array(self._statfn(spec[self._band], axis=0), dtype=float)
        np.fill_diagonal(A, np.nan)
        return A


class ConditionalPartialDirectedCoherence(ConditionalSpectral, Directed):
    """Partial directed coherence of a VAR of all processes, ``cpdc_*`` (Baccalá & Sameshima 2001).

    ``cpdc[i, j]`` (source ``i``, target ``j``) is the band statistic of
    ``|A_ji(f)|^2 / sum_k |A_ki(f)|^2``, the share of source ``i``'s outflow
    ``A(f)[:, i]`` that goes directly to ``j``. Zero at every frequency
    exactly when ``j``'s equation has no lag of ``i``. See
    :class:`ConditionalSpectral` for the common definitions and parameters.
    """

    name = "Conditional partial directed coherence"
    identifier = "cpdc"
    labels = ["spectral", "directed", "linear", "unsigned", "multivariate"]
    _measure = "pdc"


class ConditionalGeneralizedPartialDirectedCoherence(ConditionalSpectral, Directed):
    """Generalised PDC of a VAR of all processes, ``cgpdc_*`` (Baccalá, Sameshima & Takahashi 2007).

    ``cgpdc[i, j]`` is the band statistic of
    ``(|A_ji(f)|^2 / sigma_j^2) / sum_k (|A_ki(f)|^2 / sigma_k^2)``, PDC with
    each target weighted by its innovation variance ``sigma_k^2 = Sigma_kk``,
    which makes it invariant to rescaling any one process. See
    :class:`ConditionalSpectral`.
    """

    name = "Conditional generalized partial directed coherence"
    identifier = "cgpdc"
    labels = ["spectral", "directed", "linear", "unsigned", "multivariate"]
    _measure = "gpdc"


class ConditionalDirectDirectedTransferFunction(ConditionalSpectral, Directed):
    """Direct directed transfer function of a VAR of all processes, ``cddtf_*`` (Korzeniewska et al. 2003).

    ``cddtf[i, j]`` is the band statistic of ``eta_ji(f) |P_ji(f)|``: the
    full-frequency DTF from ``i`` into ``j``, ``|H_ji(f)|`` normalised by
    ``sqrt(sum_f sum_k |H_jk(f)|^2)`` over the whole grid, times the partial
    coherence of ``i`` and ``j`` given every other process. See
    :class:`ConditionalSpectral` (and its note on the ``1 / sqrt(nfreq)``
    scale).
    """

    name = "Conditional direct directed transfer function"
    identifier = "cddtf"
    labels = ["spectral", "directed", "linear", "unsigned", "multivariate"]
    _measure = "ddtf"


class PartialCoherence(ConditionalSpectral, Undirected):
    """Squared partial coherence of a VAR of all processes, ``pcohmag_*`` (Dahlhaus 2000).

    ``pcohmag[i, j]`` is the band statistic of
    ``|Q_ij(f)|^2 / (Q_ii(f) Q_jj(f))``, ``Q = S(f)^{-1}`` the inverse of the
    VAR's spectral matrix: the squared coherence of ``i`` and ``j`` after the
    linear influence of all other processes (past, present and future) is
    removed from both. Symmetric and in [0, 1]. Zero at every frequency
    exactly when ``i`` and ``j`` are conditionally independent given the
    rest, an absent edge of Dahlhaus' partial correlation graph. See
    :class:`ConditionalSpectral`.
    """

    name = "Partial coherence"
    identifier = "pcohmag"
    labels = ["spectral", "undirected", "linear", "unsigned", "multivariate"]
    _measure = "pcoh"

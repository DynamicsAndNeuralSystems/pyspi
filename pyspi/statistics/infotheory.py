"""Information-theoretic statistics, implemented in NumPy.

Attribution. The Kraskov-Stoegbauer-Grassberger (KSG), box-kernel, Gaussian
and symbolic estimators of mutual information and transfer entropy in this
module, and the Kozachenko-Leonenko entropy estimator, follow the Java
Information Dynamics Toolkit (JIDT):

    Lizier, J. T. (2014). JIDT: An information-theoretic toolkit for studying
    the dynamics of complex systems. Frontiers in Robotics and AI 1:11.
    https://github.com/jlizier/jidt  (GPL-3.0-or-later; (C) 2012- J. T. Lizier
    and contributors)

The code is an independent NumPy implementation and needs no Java runtime,
but the estimator definitions, parameter names and default behaviours
(k_HISTORY, DYN_CORR_EXCL, KERNEL_WIDTH, AUTO_EMBED_METHOD, ...) derive from
JIDT, whose copyright notice is retained in ``pyspi/lib/LICENSE-jidt.txt``.
KSG MI and TE, Gaussian MI and TE, kernel MI and TE, and symbolic TE are
checked against values computed by JIDT 1.6.1; the Kozachenko-Leonenko
estimator is checked against the closed-form entropy and mutual information
of Gaussian variables. Known deviations from JIDT: kernel MI,
TLMI and TE exclude the point itself and a Theiler window w, that is JIDT's
DYN_CORR_EXCL = w + 1 (|j - i| <= w, as for KSG), where JIDT's default counts
the point itself; Gaussian estimators carry a deterministic 1e-8 relative
ridge; KSG breaks every exact tie with a small seeded jitter where JIDT adds
random noise by default; with a Theiler window, KSG MI takes its psi(N) term
over the reduced neighbour pool (see _ksg_pool_digamma).
"""

import copy
import logging
import math
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from pyspi import utils
from pyspi.base import (
    Directed, NotAppropriate, Undirected, Unsigned, fill_pairwise,
    parse_bivariate, parse_multivariate, parse_univariate)

# ---------------------------------------------------------------------------
# Entropy calculators (with the interface of JIDT's calculators)
# ---------------------------------------------------------------------------

# Every Gaussian quantity in this module treats each variable as observed with
# independent noise of variance GAUSSIAN_RIDGE times its own variance:
#
#     Sigma  ->  Sigma + GAUSSIAN_RIDGE * diag(diag(Sigma)).
#
# Unlike an isotropic ridge, this commutes with the entropy compositions
# (H(X) + H(Y) - H(X,Y) equals the direct MI formula below exactly) and is
# unaffected by rescaling any one variable.
GAUSSIAN_RIDGE = 1e-8


def _gaussian_mi_from_r(r, ridge_rel=GAUSSIAN_RIDGE):
    """
    I(X;Y) = -0.5*log(1 - r^2) for jointly Gaussian X, Y, ridged.

    The ridge is that of the entropy path on the correlation scale: noise of
    variance ridge*V on each variable attenuates the correlation by exactly
    (1 + ridge). The bound is therefore -0.5*log(1 - 1/(1+ridge)^2), about
    8.86 nats, the same bound the entropy composition reaches.
    """
    r2 = np.clip(np.asarray(r, dtype=np.float64) ** 2, 0.0, 1.0)
    return -0.5 * np.log(1.0 - r2 / (1.0 + ridge_rel) ** 2)


def _gaussian_pairwise_joint_entropy(Z, ridge_rel=GAUSSIAN_RIDGE):
    """
    Vectorised pairwise Gaussian joint entropy, matching the scalar path.

    Same ridge as _gaussian_log_det, so bivariate() and multivariate() cannot
    disagree.
    """
    R = np.corrcoef(Z)
    V = np.var(Z, axis=1, ddof=1)
    # Factored rather than (1+ridge)^2 - R^2: on a singular pair those two
    # terms agree to ~1e-8 and the subtraction loses most of the digits.
    det = (V[:, None] * V[None, :]) * (
        (1.0 + ridge_rel - np.abs(R)) * (1.0 + ridge_rel + np.abs(R)))
    with np.errstate(invalid="ignore", divide="ignore"):
        JE = np.log(2 * np.pi * np.e) + 0.5 * np.log(det)
    return np.where(det > 0, JE, np.nan)


def _gaussian_log_det(cov, ridge_rel=GAUSSIAN_RIDGE):
    """
    log|Σ + ridge*diag(diag(Σ))|, the shared Gaussian log-determinant.

    Every Gaussian entropy, joint entropy, conditional entropy, MI, TLMI, TE
    and AIS in this module goes through this or through `_gaussian_mi_from_r`,
    its closed form on a 2x2 block. Returns NaN, never -inf, on input that is
    still degenerate after the ridge (a variable with zero variance), so an
    undefined entropy does not propagate an infinity into a difference.

    For rank-deficient Σ the unregularised log|Σ| is -inf. JIDT adds 1e-8
    Gaussian noise to the observations (NOISE_LEVEL_TO_ADD) for the same
    reason; the ridge is that operation in expectation, made deterministic.
    """
    if np.ndim(cov) == 0:
        var = float(cov)
        if not np.isfinite(var) or var <= 0:
            return np.nan
        return float(np.log(var * (1.0 + ridge_rel)))
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    diag = np.diag(cov)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
        return np.nan
    try:
        sign, log_det = np.linalg.slogdet(cov + ridge_rel * np.diag(diag))
    except np.linalg.LinAlgError:
        return np.nan
    if sign <= 0 or not np.isfinite(log_det):
        return np.nan
    return float(log_det)


def _require_gaussian_samples(n, d, measure, setting):
    """
    Refuse unless ``n`` samples exceed the dimension ``d`` of a covariance.

    A sample covariance of ``n`` observations (``ddof=1``) has rank at most
    ``n - 1``, so with ``n <= d`` it is singular for any data and its
    log-determinant is set by the ridge, not by the data. ``n = d + 1`` is
    the smallest sample with a full-rank covariance and is accepted.
    ``setting`` names what sets ``d`` (a history length, say). Raises
    ``NotAppropriate``, so the Calculator stores NaN with the reason.
    """
    if n <= d:
        raise NotAppropriate(
            f"The Gaussian {measure} needs more aligned samples than the "
            f"dimension of its largest covariance block: n = {n} samples for "
            f"d = {d} ({setting}). With n <= d the sample covariance is "
            f"singular, so the estimate would be set by the ridge, not by the "
            f"data. Use a longer series or a smaller embedding."
        )


class GaussianEntropyCalculator:
    """
    Multivariate Gaussian entropy calculator.

    H = 0.5 * d * log(2*pi*e) + 0.5 * log|Sigma|, in nats. Same interface as
    JIDT's EntropyCalculatorMultiVariateGaussian.
    """
    def __init__(self):
        self._d = None
        self._obs = None

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def setProperty(self, key, value):
        pass

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        _require_gaussian_samples(N, d, "entropy", f"{d} variables")
        cov = np.cov(X, rowvar=False, ddof=1)
        log_det = _gaussian_log_det(cov)
        if not np.isfinite(log_det):
            return float('nan')
        return float(0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det)


class KLEntropyCalculator:
    """
    Kozachenko-Leonenko nearest-neighbour entropy calculator.

    H = psi(N) - psi(1) + log(c_d) + (d/N) * sum(log(eps_i)), in nats, with
    the L2 norm and k = 1 as in JIDT. Same interface as JIDT's
    EntropyCalculatorMultiVariateKozachenko.

    A repeated observation has a nearest-neighbour distance of zero, whose
    logarithm is -inf, so tied data raise ``NotAppropriate``. Unlike KSG,
    which uses neighbour counts and so can break ties with a tiny jitter,
    this estimator uses the distances themselves: a jitter (JIDT dithers)
    would set the estimate by its own scale.
    """
    def __init__(self):
        self._d = None
        self._obs = None

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def setProperty(self, key, value):
        pass

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        tree = cKDTree(X)
        dists, _ = tree.query(X, k=2, p=2)  # k=1 NN (index 0 = self)
        eps = dists[:, 1]
        # A duplicated observation puts a neighbour at distance 0 and sends the
        # entropy to -inf (common in quantised data), so fail with the cause.
        if not np.all(eps > 0):
            n_tied = int(np.sum(eps == 0))
            raise NotAppropriate(
                f"Kozachenko entropy is undefined for tied observations: "
                f"{n_tied} of {len(eps)} points have a duplicate (zero "
                f"nearest-neighbour distance). The data is quantised or has "
                f"repeated values, and a jitter would set the estimate by its "
                f"own size; use estimator='gaussian' or 'kernel'."
            )
        log_cd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)
        return float(
            digamma(N) - digamma(1) + log_cd + (d / N) * np.sum(np.log(eps))
        )


def _gaussian_entropy_from_data(data_2d):
    """
    Gaussian entropy of an (N, d) array, in nats.

    NaN (not -inf) when the covariance is singular after the ridge, so a
    difference such as H(X,Y) - H(Y) is NaN rather than infinite. Not
    appropriate when there are no more rows than columns
    (``_require_gaussian_samples``).
    """
    N, d = data_2d.shape
    _require_gaussian_samples(N, d, "entropy", f"{d} variables")
    cov = np.cov(data_2d, rowvar=False, ddof=1)
    log_det = _gaussian_log_det(cov)
    if not np.isfinite(log_det):
        return float('nan')
    return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det


# ---------------------------------------------------------------------------
# Box-kernel entropy, MI and TE calculators (after JIDT's kernel estimators)
# ---------------------------------------------------------------------------

class KernelEntropyCalculator:
    """
    Box-kernel entropy calculator.

    Box (Heaviside) kernel with the L-infinity norm, as in JIDT's
    EntropyCalculatorMultiVariateKernel, whose interface it shares:
    ``H = mean(log(N) - log(count_i)) + d * log(2 * kernel_width)`` in nats,
    where ``count_i = #{j : |x_j - x_i|_inf <= kernel_width}`` includes the
    point itself. With NORMALISE (the default, as in JIDT) each coordinate is
    divided by its standard deviation before counting, so ``kernel_width`` is
    in standard deviations, and the log standard deviations are added back.
    """

    def __init__(self):
        self._d = None
        self._obs = None
        self._kernel_width = 0.25
        self._normalise = True

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setProperty(self, key, value):
        if key == "KERNEL_WIDTH":
            self._kernel_width = float(value)
        elif key == "NORMALISE":
            self._normalise = str(value).lower() == "true"

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        w = self._kernel_width
        # JIDT scales the bandwidth by the std; the equivalent here is to
        # standardise X and add log(prod(std)), since H(X/std) = H(X) - log(std).
        log_std_total = 0.0
        if self._normalise:
            stds = np.std(X, axis=0, ddof=1)
            stds = np.where(stds > 0, stds, 1.0)
            X = X / stds[None, :]
            log_std_total = float(np.sum(np.log(stds)))

        tree = cKDTree(X)
        # JIDT half-width = kernel_width (not kernel_width/2)
        counts = tree.query_ball_point(X, r=w, p=np.inf,
                                        return_length=True)
        counts = np.asarray(counts, dtype=np.float64)
        # H = mean(log(N) - log(count)) + d*log(2*w) + sum_d log(std_d)  [nats]
        return float(np.mean(np.log(N) - np.log(counts))
                     + d * np.log(2.0 * w) + log_std_total)


def _unit_std(data, ref=None):
    """
    Columns divided by the sample std of ``ref`` (default: themselves), as
    JIDT's NORMALISE does; constant columns are kept.
    """
    stds = np.std(data if ref is None else ref, axis=0, ddof=1)
    stds = np.where(stds > 0, stds, 1.0)
    return data / stds[None, :]


def _kernel_counts(tree, pts, width, dce):
    """
    Box-kernel neighbour counts of each point of ``pts`` in ``tree``.

    ``dce`` is JIDT's kernel ``DYN_CORR_EXCL``: when positive, neighbours
    ``|j - i| < dce`` (the point itself included) do not count. That is the
    KSG window ``|j - i| <= dce - 1``, so `_ksg_counts` removes them.
    """
    if dce <= 0:
        return np.asarray(tree.query_ball_point(pts, r=width, p=np.inf,
                                                return_length=True),
                          dtype=np.float64)
    if len(pts) < 2 * dce:
        raise NotAppropriate(
            f"Kernel Theiler window {dce - 1} (excluding |j - i| <= "
            f"{dce - 1}) leaves no comparison points among N={len(pts)} "
            f"observations; it needs N >= {2 * dce}.")
    return _ksg_counts(tree, pts, np.full(len(pts), float(width)), dce - 1)


class KernelMICalculator:
    """
    Box-kernel mutual information calculator.

    ``MI = mean(log(n_xy * N / (n_x * n_y)))`` in nats, where ``n_x``, ``n_y``
    and ``n_xy`` are counts within an L-infinity ball of radius
    ``kernel_width``, as in JIDT's MutualInfoCalculatorMultiVariateKernel,
    whose interface it shares.

    With ``DYN_CORR_EXCL = dce > 0`` the counts exclude ``|j - i| < dce``
    (see `_kernel_counts`), ``N`` becomes the number of samples each point was
    compared with, and, as in JIDT, a sample with an empty joint box adds 0
    to the mean over all ``N`` samples. When every joint box is empty the
    result is NaN (JIDT gives 0).
    """

    def __init__(self):
        self._kernel_width = 0.25
        self._normalise = True
        self._dyn_corr_excl = 0
        self._d1 = 1
        self._d2 = 1
        self._obs1 = None
        self._obs2 = None

    def initialise(self, d1, d2):
        self._d1 = int(d1)
        self._d2 = int(d2)

    def setProperty(self, key, value):
        if key == "KERNEL_WIDTH":
            self._kernel_width = float(value)
        elif key == "NORMALISE":
            self._normalise = str(value).lower() == "true"
        elif key == "DYN_CORR_EXCL":
            self._dyn_corr_excl = int(value)

    def setObservations(self, src, targ):
        self._obs1 = np.asarray(src, dtype=np.float64)
        self._obs2 = np.asarray(targ, dtype=np.float64)
        if self._obs1.ndim == 1:
            self._obs1 = self._obs1.reshape(-1, 1)
        if self._obs2.ndim == 1:
            self._obs2 = self._obs2.reshape(-1, 1)

    def computeAverageLocalOfObservations(self):
        X = self._obs1
        Y = self._obs2
        N = X.shape[0]
        w = self._kernel_width

        if self._normalise:
            X = _unit_std(X)
            Y = _unit_std(Y)

        XY = np.column_stack([X, Y])
        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)
        tree_xy = cKDTree(XY)

        # JIDT half-width = kernel_width
        dce = self._dyn_corr_excl
        n_x = _kernel_counts(tree_x, X, w, dce)
        n_y = _kernel_counts(tree_y, Y, w, dce)
        n_xy = _kernel_counts(tree_xy, XY, w, dce)

        if dce <= 0:
            # MI = mean(log(n_xy * N / (n_x * n_y)))  [nats]
            mi = np.mean(np.log(n_xy) + np.log(N) - np.log(n_x) - np.log(n_y))
            return float(mi)

        # Samples compared with point i: N minus the excluded |j - i| < dce.
        i = np.arange(N)
        n_pool = (N - np.minimum(i, dce - 1) - np.minimum(N - 1 - i, dce - 1) - 1)
        # n_xy > 0 implies n_x, n_y > 0; JIDT adds 0 for an empty joint box.
        nz = n_xy > 0
        if not nz.any():
            # No sample has a neighbour, so there is nothing to estimate from.
            return float('nan')
        local = np.zeros(N)
        local[nz] = (np.log(n_xy[nz]) + np.log(n_pool[nz])
                     - np.log(n_x[nz]) - np.log(n_y[nz]))
        return float(np.mean(local))


class KernelTECalculator:
    """
    Box-kernel transfer entropy calculator.

    ``TE(X -> Y) = mean(log(n_yn_yp_x * n_yp / (n_yp_x * n_yn_yp)))`` in nats,
    with a box kernel of half-width ``kernel_width`` in the L-infinity norm,
    as in JIDT's TransferEntropyCalculatorKernel, whose interface it shares.
    The target past has ``k_HISTORY`` samples and the source one sample, at
    lag 1. ``DYN_CORR_EXCL`` excludes neighbours as in `KernelMICalculator`,
    and the result is NaN when every joint box is empty.
    """

    def __init__(self):
        self._kernel_width = 0.25
        self._normalise = True
        self._k_history = 1
        self._dyn_corr_excl = 0

    def initialise(self):
        pass

    def setProperty(self, key, value):
        if key == "KERNEL_WIDTH":
            self._kernel_width = float(value)
        elif key == "NORMALISE":
            self._normalise = str(value).lower() == "true"
        elif key == "k_HISTORY":
            self._k_history = int(value)
        elif key == "DYN_CORR_EXCL":
            self._dyn_corr_excl = int(value)

    def setObservations(self, src, targ):
        self._src = np.asarray(src, dtype=np.float64).ravel()
        self._targ = np.asarray(targ, dtype=np.float64).ravel()

    def computeAverageLocalOfObservations(self):
        src = self._src
        targ = self._targ
        k = self._k_history
        T = len(src)
        w = self._kernel_width

        if T <= k:
            return np.nan

        # Build vectors
        n_pts = T - k
        y_past = np.column_stack([targ[k - 1 - lag: T - 1 - lag] for lag in range(k)])
        y_next = targ[k:].reshape(-1, 1)
        x_t = src[k - 1: T - 1].reshape(-1, 1)

        if self._normalise:
            # As in JIDT, the next value is scaled like the most recent past
            # value (the same series), not by its own std.
            y_next = _unit_std(y_next, y_past[:, :1])
            y_past = _unit_std(y_past)
            x_t = _unit_std(x_t)

        # Joint spaces
        yn_yp = np.column_stack([y_next, y_past])
        yp_x = np.column_stack([y_past, x_t])
        yn_yp_x = np.column_stack([y_next, y_past, x_t])

        tree_yp = cKDTree(y_past)
        tree_yn_yp = cKDTree(yn_yp)
        tree_yp_x = cKDTree(yp_x)
        tree_yn_yp_x = cKDTree(yn_yp_x)

        # JIDT half-width = kernel_width. The Theiler window applies to the
        # embedded sample index, shared by all four spaces.
        dce = self._dyn_corr_excl
        n_yp = _kernel_counts(tree_yp, y_past, w, dce)
        n_yn_yp = _kernel_counts(tree_yn_yp, yn_yp, w, dce)
        n_yp_x = _kernel_counts(tree_yp_x, yp_x, w, dce)
        n_yn_yp_x = _kernel_counts(tree_yn_yp_x, yn_yp_x, w, dce)

        if dce <= 0:
            # Every count includes the point itself, so none is empty.
            # TE = mean(log(n_yn_yp_x * n_yp / (n_yp_x * n_yn_yp)))  [nats]
            te = np.mean(np.log(n_yn_yp_x) + np.log(n_yp) - np.log(n_yp_x) - np.log(n_yn_yp))
            return float(te)

        # n_yn_yp_x > 0 implies the other three are > 0. As in JIDT, a sample
        # with an empty joint box adds 0 to the mean over all samples.
        nz = n_yn_yp_x > 0
        if not nz.any():
            # No sample has a neighbour, so there is nothing to estimate from.
            return float('nan')
        local = np.zeros(n_pts)
        local[nz] = (np.log(n_yn_yp_x[nz]) + np.log(n_yp[nz])
                     - np.log(n_yp_x[nz]) - np.log(n_yn_yp[nz]))
        return float(np.mean(local))


# ---------------------------------------------------------------------------
# Symbolic transfer entropy (ordinal patterns)
# ---------------------------------------------------------------------------

def _ordinal_pattern_id(vec):
    """
    Convert a vector to its ordinal pattern ID.

    The ordinal pattern is the rank ordering. Maps to an integer in [0, d!).
    Uses the factorial number system (Lehmer code).
    """
    d = len(vec)
    # Get the rank order (argsort of argsort)
    order = np.argsort(vec)
    # Lehmer code
    code = 0
    remaining = list(range(d))
    factorial = 1
    for i in range(1, d):
        factorial *= i
    for i in range(d - 1):
        pos = remaining.index(order[i])
        code += pos * factorial
        remaining.pop(pos)
        if i < d - 2:
            factorial //= (d - 1 - i)
    return code


def _series_to_ordinal_symbols(x, k):
    """
    Convert a 1D time series to a sequence of ordinal pattern symbols.

    For each t, the embedding vector is [x[t], x[t-1], ..., x[t-k+1]].
    Returns integer array of symbol IDs, length T-k+1.
    """
    x = np.asarray(x).ravel()
    T = len(x)
    if T < k:
        return np.array([], dtype=int)

    n_pts = T - k + 1
    symbols = np.empty(n_pts, dtype=int)

    # Build embedding matrix
    embedding = np.column_stack([x[k - 1 - lag: T - lag] for lag in range(k)])

    for t in range(n_pts):
        symbols[t] = _ordinal_pattern_id(embedding[t])

    return symbols


class SymbolicTECalculator:
    """
    Symbolic (ordinal-pattern) transfer entropy calculator.

    Converts source and target to ordinal patterns of length k and computes
    TE from joint symbol histograms, as in JIDT's
    TransferEntropyCalculatorSymbolic, whose interface it shares:

    TE = H(Y_next | Y_past) - H(Y_next | Y_past, X)
       = H(Y_next, Y_past) - H(Y_past) - H(Y_next, Y_past, X) + H(Y_past, X)

    with discrete entropies in nats. One pattern length applies to source and
    destination alike, at unit delay, as Staniek & Lehnertz (2008) define it,
    so `TransferEntropy` refuses `k_tau`, `l_history` and `l_tau` under this
    estimator.
    """

    def __init__(self):
        self._k_history = 1
        self._props = {}

    def initialise(self):
        pass

    def setProperty(self, key, value):
        self._props[key] = value
        if key == "k_HISTORY":
            self._k_history = int(value)

    def setObservations(self, src, targ):
        self._src = np.asarray(src, dtype=np.float64).ravel()
        self._targ = np.asarray(targ, dtype=np.float64).ravel()

    def computeAverageLocalOfObservations(self):
        src = self._src
        targ = self._targ
        k = self._k_history
        T = len(src)

        if T <= k:
            return np.nan

        # Convert to ordinal patterns
        src_symbols = _series_to_ordinal_symbols(src, k)
        targ_symbols = _series_to_ordinal_symbols(targ, k)

        # targ_next is the target's next symbol; the source symbol is
        # concurrent with targ_past.
        n = min(len(src_symbols), len(targ_symbols)) - 1
        if n <= 0:
            return np.nan

        targ_next = targ_symbols[1:n + 1]
        targ_past = targ_symbols[:n]
        src_curr = src_symbols[:n]

        def _discrete_entropy(*arrs):
            """
            Joint entropy of integer-valued arrays from their histogram.

            Counts distinct rows directly: packing the symbols into one
            integer can overflow int64 for large alphabets (k >= 10).
            """
            if len(arrs) == 1:
                _, counts = np.unique(arrs[0], return_counts=True)
            else:
                stacked = np.column_stack(arrs)
                _, counts = np.unique(stacked, axis=0, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log(probs))

        # TE = H(yn, yp) - H(yp) - H(yn, yp, x) + H(yp, x)
        H_yn_yp = _discrete_entropy(targ_next, targ_past)
        H_yp = _discrete_entropy(targ_past)
        H_yn_yp_x = _discrete_entropy(targ_next, targ_past, src_curr)
        H_yp_x = _discrete_entropy(targ_past, src_curr)

        te = H_yn_yp - H_yp - H_yn_yp_x + H_yp_x
        return float(te)


def _numpy_delay_embedding(x, dim):
    """Delay embedding, as JIDT's MatrixUtils.makeDelayEmbeddingVector."""
    x = np.asarray(x).ravel()
    T = len(x)
    if dim == 0:
        return np.empty((T + 1, 0))
    return np.column_stack([x[dim - 1 - lag: T - lag] for lag in range(dim)])


# ---------------------------------------------------------------------------
# k-NN input conditioning
# ---------------------------------------------------------------------------

# JIDT adds random noise (1e-8) to every KSG input and normalises it. pyspi
# standardises each coordinate (below) but, as pyhctsa's KSG does, replaces
# the noise with deterministic jitter on repeated values only, since KSG
# assumes no ties. Unlike hctsa's BF_TieBreakNoise, which jitters every value
# of a series with fewer than 90% unique values, values that occur once are
# left alone, so tie-free data are returned bit-identically.

# Seeds of the per-variable jitter streams. Variables compared with one another
# must not share a stream (identical noise would create dependence).
_TIE_SEED_SOURCE = 0
_TIE_SEED_TARGET = 1
_TIE_SEED_SELECTION = 2


def _tie_break(y, seed=0):
    """
    Make every value of a raw series distinct with tiny fixed-seed jitter.

    Each entry whose value occurs more than once gets Gaussian jitter of
    standard deviation ``1e-10 * std(y, ddof=1)``: entry ``i`` gets the ``i``-th
    draw of a local generator seeded with ``seed`` (never the global NumPy
    RNG), so the same input always gives the same output. Jitter this small can
    land two entries of a large tie block on the same float again, so entries
    still repeated are jittered again with the next draws of the same generator,
    for at most 100 passes, until every value is distinct. Values that occur
    once are not touched, and a series with no repeated value, or with zero
    std, is returned unchanged. Apply to the raw series *before* delay
    embedding so that every embedded lag of a sample carries the same
    perturbation.

    Values far from zero relative to their spread (``|y| / std`` above about
    1e5, possible only without z-scoring) have a floating-point resolution
    coarser than the jitter, so the ties survive every pass; the estimate is
    then not appropriate (``NotAppropriate``), and centring or z-scoring the
    series resolves it.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2 or not np.all(np.isfinite(y)):
        return y        # non-finite input is rejected by _knn_condition
    sigma = y.std(ddof=1)
    if not sigma > 0:
        return y
    rng = None
    for _ in range(100):
        _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        repeated = counts[inverse.ravel()] > 1
        if not repeated.any():
            return y
        if rng is None:
            rng = np.random.default_rng(seed)
            y = y.copy()
        y[repeated] += 1e-10 * sigma * rng.standard_normal(y.shape)[repeated]
    if len(np.unique(y)) < y.size:
        raise NotAppropriate(
            f"KSG could not break the ties in a series with "
            f"{y.size - len(np.unique(y))} repeated value(s): the tie-breaking "
            f"jitter (1e-10 of the standard deviation, {1e-10 * sigma:.3g}) is "
            f"below the floating-point resolution of values of this magnitude "
            f"(largest |value| {np.max(np.abs(y)):.3g}). Centre or z-score the "
            f"series."
        )
    return y


def _knn_condition(X):
    """
    Validate KSG input and standardise each coordinate.

    Non-finite values and constant coordinates raise. Ties do not: callers
    apply ``_tie_break`` to raw series first. Deterministic: no rounding,
    dither or random state.
    """
    X = np.asarray(X, dtype=np.float64)
    reshaped = X.ndim == 1
    if reshaped:
        X = X.reshape(-1, 1)
    if not np.all(np.isfinite(X)):
        raise ValueError("KSG requires finite input: found NaN or infinite values.")

    out = np.empty_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]
        # Subtracting an observed origin first is algebraically identical and
        # avoids cancellation under a large affine offset.
        shifted = col - col[0]
        sd = shifted.std(ddof=1)      # JIDT MatrixUtils.stdDevs: sample std
        if not sd > 0:
            raise ValueError(
                f"KSG cannot estimate: coordinate {c} is constant, so all "
                f"neighbour distances are zero.")
        out[:, c] = (shifted - shifted.mean()) / sd
    return out.reshape(-1) if reshaped else out


# ---------------------------------------------------------------------------
# KSG MI estimator
# ---------------------------------------------------------------------------

def _validate_ksg_sample(N, k, w, context=""):
    """
    Reject KSG settings that cannot produce a meaningful estimate.

    The estimator needs k neighbours among the points that survive the
    Theiler exclusion; otherwise it returns a finite number that estimates
    nothing.
    """
    effective = N - (2 * w + 1) if w else N - 1
    where = f" ({context})" if context else ""
    if k < 1:
        raise ValueError(f"KSG needs k >= 1, got k={k}{where}.")
    if w < 0:
        raise ValueError(f"Theiler window must be >= 0, got w={w}{where}.")
    # The series cannot support the estimate: not appropriate, not an error.
    if N < 2:
        raise NotAppropriate(f"KSG needs at least 2 observations, got {N}{where}.")
    if k > effective:
        raise NotAppropriate(
            f"KSG k={k} exceeds the {effective} usable neighbour(s) for "
            f"N={N} observations with Theiler window w={w}{where}. Reduce k, "
            f"lengthen the series, or narrow the Theiler window."
        )


def _strict_radius(epsilon):
    """
    Largest representable radius strictly smaller than ``epsilon``.

    ``cKDTree.query_ball_point`` includes the radius boundary. KSG estimator 1
    requires marginal distances strictly below the joint kth-neighbour radius,
    so moving by one floating-point step implements that inequality without a
    scale-dependent tolerance that can discard genuine interior neighbours.
    """
    return np.nextafter(epsilon, -np.inf)


# Elements per temporary in the Theiler-window neighbour search below.
_KSG_CHUNK = 1 << 21


def _ksg_radius(tree, points, k, w):
    """
    Strict radius of each point's kth nearest neighbour in the joint space.

    With a Theiler window ``w`` the neighbours ``|j - i| <= w`` do not count.
    The nearest ``k + 2w + 2`` points always hold ``k`` that do (the point
    itself and at most ``2w`` window members are skipped), and when that many
    do not exist the query returns every point, so nothing further could be
    found: a row with fewer than ``k`` valid neighbours has radius ``inf``.
    Rows are searched in blocks so the ``N x (k + 2w + 2)`` neighbour table is
    never held at once.
    """
    N = points.shape[0]
    if w == 0:
        dists, _ = tree.query(points, k=k + 1, p=np.inf)
        return _strict_radius(dists[:, k])
    n_query = min(k + 2 * w + 2, N)
    eps = np.full(N, np.inf)
    step = max(1, _KSG_CHUNK // n_query)
    for start in range(0, N, step):
        rows = np.arange(start, min(start + step, N))
        dists, idx = tree.query(points[rows], k=n_query, p=np.inf)
        valid = np.abs(idx - rows[:, None]) > w
        valid[:, 0] = False
        rank = np.cumsum(valid, axis=1)
        found = rank[:, -1] >= k
        kth = np.argmax(rank[found] >= k, axis=1)
        eps[rows[found]] = dists[found, kth]
    return _strict_radius(eps)


def _ksg_counts(tree, points, radius, w):
    """
    Neighbours of each point within its ``radius``, boundary included.

    The point itself is not a neighbour, nor is any ``|j - i| <= w``. The tree
    counts every point in the ball; the excluded ones that fell inside it are
    then subtracted one time offset at a time, by the comparison the tree
    itself makes (largest coordinate difference ``<= radius``), which needs
    O(N) memory whatever the window.
    """
    n = tree.query_ball_point(points, radius, p=np.inf,
                              return_length=True).astype(np.float64)
    if w == 0:
        return n - 1
    n -= radius >= 0
    for d in range(1, min(w, points.shape[0] - 1) + 1):
        apart = np.abs(points[d:] - points[:-d]).max(axis=1)
        n[:-d] -= apart <= radius[:-d]
        n[d:] -= apart <= radius[d:]
    return n


def _ksg_pool_digamma(N, w):
    """
    KSG's ``psi(N)`` term for a neighbour pool reduced by a Theiler window.

    ``psi(N)`` is ``psi(m + 1)`` for a pool of ``m = N - 1`` candidate
    neighbours. The window removes ``|j - i| <= w`` from point i's pool,
    leaving ``m_i = N - 1 - min(i, w) - min(N - 1 - i, w)``, so the term is
    the mean of ``psi(m_i + 1)``; at ``w == 0`` it is ``digamma(N)``. (JIDT
    keeps ``psi(N)``, which biases MI upward by about ``ln(N / (N - 2w))``.)
    In the Frenzel-Pompe CMI the pool term cancels, so only MI needs it.
    """
    if w == 0:
        return digamma(N)
    i = np.arange(N)
    pool = N - 1 - np.minimum(i, w) - np.minimum(N - 1 - i, w)
    return float(np.mean(digamma(pool + 1)))


def _ksg_mi_pair(x, y, k, w):
    """KSG Estimator 1 MI for a single pair of scalar series."""
    N = len(x)
    _validate_ksg_sample(N, k, w, context="mutual information")
    # A constant marginal is almost always a broken input; checked here so the
    # error can name it.
    for name, v in (("x", x), ("y", y)):
        if np.ptp(v) == 0:
            raise ValueError(
                f"KSG cannot estimate mutual information: marginal {name} is "
                f"constant, so all neighbour distances are zero."
            )
    x = _tie_break(x, _TIE_SEED_SOURCE)
    y = _tie_break(y, _TIE_SEED_TARGET)
    x, y = _knn_condition(np.column_stack([x, y])).T
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    xy = np.column_stack([x, y])
    # Marginal neighbours are counted strictly within eps (< eps), as KSG
    # estimator 1 requires; counting the boundary would admit the kth neighbour.
    eps = _ksg_radius(cKDTree(xy), xy, k, w)
    n_x = _ksg_counts(tree_x, x, eps, w)
    n_y = _ksg_counts(tree_y, y, eps, w)

    # psi(pool size + 1); a Theiler window shrinks the pool (_ksg_pool_digamma).
    mi = (digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
          + _ksg_pool_digamma(N, w))
    return float(mi)


def _ksg_mi_general(A, B, k_nn, w=0, condition=True):
    """
    KSG estimator 1 MI(A; B) for A (N, dA) and B (N, dB), L-infinity norm.

    _ksg_mi_pair for any marginal dimensions. Marginal neighbours are counted
    strictly (< eps). A Theiler window w excludes |j - i| <= w from the
    neighbour set, and psi(N) becomes the mean psi of the reduced pool
    (_ksg_pool_digamma).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if condition:
        A, B = np.split(_knn_condition(np.column_stack([A, B])), [A.shape[1]], axis=1)
    N = A.shape[0]
    # The estimator's own precondition, checked on the aligned arrays: with
    # fewer usable neighbours than k the result would be finite but meaningless.
    _validate_ksg_sample(N, k_nn, w, context="mutual information")
    AB = np.column_stack([A, B])
    tree_ab = cKDTree(AB)
    tree_a = cKDTree(A)
    tree_b = cKDTree(B)
    eps = _ksg_radius(tree_ab, AB, k_nn, w)
    n_a = _ksg_counts(tree_a, A, eps, w)
    n_b = _ksg_counts(tree_b, B, eps, w)

    return float(digamma(k_nn) + _ksg_pool_digamma(N, w)
                 - np.mean(digamma(n_a + 1) + digamma(n_b + 1)))


# ---------------------------------------------------------------------------
# Transfer entropy helpers
# ---------------------------------------------------------------------------

class _DummyTECalculator:
    """Placeholder calculator: gaussian and kraskov TE are computed directly."""
    def __init__(self):
        self._props = {}
    def setProperty(self, key, value):
        self._props[key] = value
    def initialise(self):
        pass
    def setObservations(self, src, targ):
        pass
    def computeAverageLocalOfObservations(self):
        return np.nan


def _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau):
    """Build delay embeddings for transfer entropy computation."""
    T = len(src)
    max_lookback = max((k_history - 1) * k_tau, (l_history - 1) * l_tau)
    start = max_lookback
    end = T - 1

    if start >= end:
        return None, None, None

    Y_future = targ[start + 1: end + 1].reshape(-1, 1)

    Y_past_cols = []
    for lag_idx in range(k_history):
        lag = lag_idx * k_tau
        Y_past_cols.append(targ[start - lag: end - lag])
    Y_past = np.column_stack(Y_past_cols)

    X_past_cols = []
    for lag_idx in range(l_history):
        lag = lag_idx * l_tau
        X_past_cols.append(src[start - lag: end - lag])
    X_past = np.column_stack(X_past_cols)

    return Y_future, Y_past, X_past


def _gaussian_ais(targ, k, tau):
    """
    Bias-corrected Gaussian AIS, MI(Y_future; Y_past_embedding(k, tau)).

    The criterion for auto-embedding: (k, tau) = argmax AIS. The in-sample
    log-det estimate is biased upward by the mean of its chi-squared null,
    df / (2N) with df = k (the future is 1-D), so uncorrected it grows with k
    up to k_max. Subtracting k/(2N), as JIDT's
    ActiveInfoStorageCalculatorGaussian.computeAdditionalBiasToRemove does,
    gives an interior maximum.

    This is JIDT's maximum corrected AIS criterion
    (AUTO_EMBED_METHOD_MAX_CORR_AIS, after Garland et al. 2016), not the
    Ragwitz local-prediction criterion (Ragwitz & Kantz 2002), which is not
    implemented.
    """
    T = len(targ)
    start = (k - 1) * tau
    end = T - 1
    if start >= end:
        return -np.inf
    Y_f = targ[start + 1: end + 1].reshape(-1, 1)
    Y_p = np.column_stack([targ[start - i * tau: end - i * tau] for i in range(k)])
    YfYp = np.concatenate([Y_f, Y_p], axis=1)
    N = Y_f.shape[0]
    _require_gaussian_samples(N, k + 1, "active information storage",
                              f"1 + k with k = {k}")

    def _slogdet(arr):
        return _gaussian_log_det(np.cov(arr, rowvar=False, ddof=1))

    ais_raw = 0.5 * (_slogdet(Y_f) + _slogdet(Y_p) - _slogdet(YfYp))
    return ais_raw - k / (2.0 * N)


def _ksg_ais(targ, k, tau, k_nn, w=0):
    """
    KSG AIS, MI(Y_future; Y_past_embedding(k, tau)), for the kraskov TE.

    The counterpart of _gaussian_ais. KSG is approximately bias-free, so the
    maximum over k is interior without a bias term (JIDT's MAX_CORR_AIS also
    removes no extra bias for KSG), and the embedding is chosen by the
    estimator that computes the TE.
    """
    T = len(targ)
    start = (k - 1) * tau
    end = T - 1
    if start >= end:
        return -np.inf
    Y_f = targ[start + 1: end + 1].reshape(-1, 1)
    Y_p = np.column_stack([targ[start - i * tau: end - i * tau] for i in range(k)])
    return _ksg_mi_general(Y_f, Y_p, k_nn, w)


def _ais_scorer(estimator, k_nn=None, w=0):
    """
    ``f(series, dim, delay) -> score`` for the estimator that will run the TE.

    The source and destination embeddings are chosen by the same scorer, with
    the estimator that computes the TE: a Gaussian AIS would pick the
    embedding with the best linear predictability, which is not what a KSG
    transfer entropy measures.
    """
    if estimator == "gaussian":
        def score(series, dim, delay):
            # A candidate with no more aligned samples than its covariance
            # dimension (1 + dim) is skipped, as the kraskov scorer does.
            if len(series) - 1 - (dim - 1) * delay <= dim + 1:
                return -np.inf
            return _gaussian_ais(series, dim, delay)
        return score
    if estimator == "kraskov":
        def score(series, dim, delay):
            # `_ksg_ais` spends one sample on the one-step-ahead future as
            # well as (dim - 1) * delay on the lookback.
            n_eff = len(series) - 1 - (dim - 1) * delay
            try:
                _validate_ksg_sample(n_eff, k_nn, w)
            except NotAppropriate:
                # Unsupported candidates are skipped, so a valid smaller
                # embedding is chosen when one exists.
                return -np.inf
            return _ksg_ais(series, dim, delay, k_nn, w)
        return score
    raise NotImplementedError(
        f"No active-information-storage criterion for estimator {estimator!r}."
    )


def _select_embedding(series, scorer, dim_max, tau_max):
    """
    ``(dimension, delay, score)`` maximising the scorer.

    Ties go to the smaller dimension and then the smaller delay: a
    lower-dimensional embedding that stores as much information is the
    better model, and the AIS objective often plateaus.

    Raises when no candidate is scorable (a non-finite score counts as
    unscorable), rather than returning an embedding that was itself rejected.
    """
    best = (None, None, -np.inf)
    for dim in range(1, dim_max + 1):
        for delay in range(1, tau_max + 1):
            score = scorer(series, dim, delay)
            if np.isfinite(score) and score > best[2]:
                best = (dim, delay, score)
    if best[0] is None:
        raise NotAppropriate(
            f"No (dimension, delay) in 1..{dim_max} x 1..{tau_max} can be "
            f"scored on a series of {len(series)} observations: every "
            f"candidate leaves too few aligned samples for the estimator. "
            f"Lower the search bounds, or use a fixed embedding."
        )
    return best


def _gaussian_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau):
    """
    Gaussian TE via log-determinant ratio.

    Not appropriate when the aligned sample count, ``len(targ) - 1`` less
    the longer lookback, does not exceed the joint dimension
    ``1 + k_history + l_history``.
    """
    lookback = max((k_history - 1) * k_tau, (l_history - 1) * l_tau)
    _require_gaussian_samples(
        len(targ) - 1 - lookback, 1 + k_history + l_history,
        "transfer entropy",
        f"1 + k_history + l_history with k_history = {k_history}, "
        f"l_history = {l_history}")
    Y_f, Y_p, X_p = _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau)
    if Y_f is None:
        return np.nan

    def _slogdet(data):
        return _gaussian_log_det(np.cov(data, rowvar=False, ddof=1))

    YfYp = np.concatenate([Y_f, Y_p], axis=1)
    YfYpXp = np.concatenate([Y_f, Y_p, X_p], axis=1)
    YpXp = np.concatenate([Y_p, X_p], axis=1)

    te = 0.5 * (_slogdet(YfYp) - _slogdet(Y_p) - _slogdet(YfYpXp) + _slogdet(YpXp))
    return float(te)


def _ksg_cmi(A, B, C, k_nn, w=0, condition=True):
    """
    KSG conditional mutual information I(A; B | C), Frenzel-Pompe estimator.

    Estimates the CMI directly: the neighbour radius is fixed once in the
    joint space [A, B, C] and reused in every marginal count, so the
    dimension-dependent biases cancel, which they do not when the CMI is
    composed from separately estimated entropies.

    ``C`` may have zero columns, in which case this is the MI estimator.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    N = A.shape[0]
    # Constant columns have no continuous density. Other ties are rejected by
    # the shared conditioning primitive below.
    for name, arr in (("A", A), ("B", B)):
        if np.ptp(arr, axis=0).min() == 0:
            raise ValueError(f"KSG cannot estimate: {name} has a constant column.")
    has_C = C is not None and np.asarray(C).size and np.asarray(C).shape[1] > 0
    if condition:
        if has_C:
            C = np.atleast_2d(C)
            A, B, C = np.split(
                _knn_condition(np.column_stack([A, B, C])),
                [A.shape[1], A.shape[1] + B.shape[1]], axis=1,
            )
        else:
            A, B = np.split(
                _knn_condition(np.column_stack([A, B])), [A.shape[1]], axis=1)
    if not has_C:
        # With nothing to condition on this is mutual information: a constant
        # conditioning count of N - (2w + 1) would match it only at w = 0.
        return _ksg_mi_general(A, B, k_nn, w, condition=False)

    joint = np.concatenate([A, B, C], axis=1)
    AC = np.concatenate([A, C], axis=1)
    BC = np.concatenate([B, C], axis=1)

    tree_joint = cKDTree(joint)
    tree_AC = cKDTree(AC)
    tree_BC = cKDTree(BC)
    tree_C = cKDTree(C)

    eps_strict = _ksg_radius(tree_joint, joint, k_nn, w)
    n_AC = _ksg_counts(tree_AC, AC, eps_strict, w)
    n_BC = _ksg_counts(tree_BC, BC, eps_strict, w)
    n_C = _ksg_counts(tree_C, C, eps_strict, w)

    return float(digamma(k_nn) + np.mean(
        digamma(n_C + 1) - digamma(n_AC + 1) - digamma(n_BC + 1)
    ))


def _kraskov_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau, k_nn, w):
    """Kraskov TE via the Frenzel-Pompe CMI estimator: I(Y_f; X_p | Y_p)."""
    # Jitter the raw series (idempotent if already done), then embed.
    src = _tie_break(src, _TIE_SEED_SOURCE)
    targ = _tie_break(targ, _TIE_SEED_TARGET)
    Y_f, Y_p, X_p = _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau)
    if Y_f is None:
        return np.nan
    # Checked on the embedded sample count, which the lookback makes smaller
    # than len(targ).
    _validate_ksg_sample(Y_f.shape[0], k_nn, w, context="transfer entropy")
    return _ksg_cmi(Y_f, X_p, Y_p, k_nn, w)


# ---------------------------------------------------------------------------
# Information-theory base class, with estimator dispatch
# ---------------------------------------------------------------------------

_ESTIMATORS = frozenset({"gaussian", "kraskov", "kernel", "kozachenko", "symbolic"})


def _require_positive_int(name, value):
    """`utils.require_int(..., minimum=1)`."""
    return utils.require_int(name, value, minimum=1)

# Implemented auto-embedding criteria. Both maximise the bias-corrected active
# information storage of a series over (dimension, delay):
#   MAX_CORR_AIS            selects destination and source, independently;
#   MAX_CORR_AIS_DEST_ONLY  selects the destination only; the source embedding
#                           is the fixed l_history, l_tau.
_AUTO_EMBED_METHODS = frozenset({"MAX_CORR_AIS", "MAX_CORR_AIS_DEST_ONLY"})
_AUTO_EMBED_DEST_ONLY = "MAX_CORR_AIS_DEST_ONLY"


class InfoTheoryBase(Unsigned):
    """
    Base class for the information-theoretic SPIs, with estimator dispatch.

    Every measure is in nats, whatever the estimator (JIDT reports bits for
    its kernel and symbolic estimators; divide by ln 2 to compare). The
    ``estimator``, ``kernel_width``, ``prop_k`` and ``dyn_corr_excl``
    arguments are documented on each SPI. An argument the chosen estimator
    does not use, or an estimator the SPI does not implement, raises.
    """

    _AUTO_EMBED_METHOD_PROP_NAME = "AUTO_EMBED_METHOD"
    _K_HISTORY_PROP_NAME = "k_HISTORY"
    _K_TAU_PROP_NAME = "k_TAU"
    _L_HISTORY_PROP_NAME = "l_HISTORY"
    _L_TAU_PROP_NAME = "l_TAU"
    _K_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_K_SEARCH_MAX"
    _TAU_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_TAU_SEARCH_MAX"

    # The estimators each optional parameter applies to; supplying it to any
    # other estimator raises (kernel_width with estimator="gaussian", say).
    _PARAM_OWNER = {
        "kernel_width": ("kernel",),
        "prop_k": ("kraskov",),
        "dyn_corr_excl": ("kraskov", "kernel"),
    }

    def __init__(
        self, estimator="gaussian", kernel_width=None, prop_k=None, dyn_corr_excl=None
    ):
        if estimator not in _ESTIMATORS:
            raise ValueError(
                f"Unknown estimator {estimator!r}; expected one of "
                f"{sorted(_ESTIMATORS)}."
            )

        supplied = {
            "kernel_width": kernel_width,
            "prop_k": prop_k,
            "dyn_corr_excl": dyn_corr_excl,
        }
        for name, value in supplied.items():
            owners = self._PARAM_OWNER[name]
            if value is not None and estimator not in owners:
                raise ValueError(
                    f"{name}={value!r} is not used by estimator={estimator!r} "
                    f"(it applies to {'/'.join(owners)}). Remove it, or select "
                    f"the estimator it belongs to."
                )

        self._estimator = estimator
        # Defaults are applied after the check above, so "not supplied" stays
        # distinct from "supplied with the default". Values are validated, not
        # coerced: a cast would turn True or 2.7 into a plausible 1 or 2.
        self._kernel_width = (0.5 if kernel_width is None
                              else utils.require_positive_float("kernel_width",
                                                                kernel_width))
        self._prop_k = (4 if prop_k is None
                        else utils.require_int("prop_k", prop_k, minimum=1))
        if dyn_corr_excl is not None and not (
                isinstance(dyn_corr_excl, str) and dyn_corr_excl == "AUTO"):
            # None and "AUTO" aside, the window is an integer in samples.
            if isinstance(dyn_corr_excl, str):
                raise ValueError(
                    f"dyn_corr_excl must be an integer >= 0, None, or the "
                    f"string 'AUTO'; got {dyn_corr_excl!r}.")
            dyn_corr_excl = utils.require_int("dyn_corr_excl", dyn_corr_excl,
                                              minimum=0)
        self._dyn_corr_excl = dyn_corr_excl
        self._entropy_calc = self._getcalc("entropy")

        self.identifier = self.identifier + "_" + estimator
        if estimator == "kraskov":
            # Only the measures with a KSG implementation accept it; the others
            # are composed from marginal entropies, which KSG does not estimate.
            if not isinstance(self, (MutualInfo, TimeLaggedMutualInfo,
                                     TransferEntropy, DirectedInfo)):
                raise NotImplementedError(
                    f"The kraskov estimator is not implemented for "
                    f"{type(self).__name__}: it is composed from marginal "
                    f"entropies, and no KSG estimator exists for that "
                    f"composition. Use estimator='kozachenko' for a "
                    f"k-nearest-neighbour entropy, or 'gaussian'."
                )
            self.identifier = self.identifier + "_NN-{}".format(self._prop_k)
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "kernel":
            if self._dyn_corr_excl is not None and not isinstance(
                    self, (MutualInfo, TimeLaggedMutualInfo, TransferEntropy)):
                raise ValueError(
                    f"dyn_corr_excl is not implemented for the kernel "
                    f"{type(self).__name__}, which is composed from marginal "
                    f"entropies; it applies to kernel MI, TLMI and TE.")
            # Dots are kept here (``_W-0.5``), unlike most identifiers.
            self.identifier = (self.identifier + "_W-"
                               + utils.fmt_param(self._kernel_width))
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "symbolic":
            if not isinstance(self, TransferEntropy):
                raise NotImplementedError(
                    "Symbolic estimator is only available for transfer entropy."
                )
            self.labels = self.labels + ["symbolic"]
            self._dyn_corr_excl = None
            return
        elif estimator == "kozachenko":
            # Composing MI, TLMI or TE from separate Kozachenko-Leonenko
            # entropies is biased, since the per-space biases do not cancel;
            # KSG ("kraskov") couples its radii across spaces to avoid this.
            if isinstance(self, (MutualInfo, TimeLaggedMutualInfo, TransferEntropy)):
                raise NotImplementedError(
                    f"The kozachenko estimator is not available for "
                    f"{type(self).__name__}; use estimator='kraskov' for a "
                    f"k-nearest-neighbour estimate of this measure."
                )
            # Nearest-neighbour based, so nonlinear.
            self.labels = self.labels + ["nonlinear"]
            self._dyn_corr_excl = None
        else:
            self.labels = self.labels + ["linear"]
            self._dyn_corr_excl = None

        if self._dyn_corr_excl:
            # The window's value, since each window is a different SPI.
            self.identifier = self.identifier + f"_DCE-{self._dyn_corr_excl}"

    def __getstate__(self):
        state = dict(self.__dict__)
        unserializable_objects = ["_entropy_calc", "_calc"]
        for k in unserializable_objects:
            if k in state.keys():
                del state[k]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._entropy_calc = self._getcalc("entropy")

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        for attr in newone.__dict__:
            setattr(newone, attr, copy.deepcopy(getattr(self, attr), memo))
        return newone

    def _getkey(self):
        if self._estimator == "kernel":
            return (self._estimator, self._kernel_width, self._dyn_corr_excl)
        elif self._estimator == "kraskov":
            # Everything in the identifier is in the cache key, including
            # dyn_corr_excl, which the KSG paths do not cache on.
            return (self._estimator, self._prop_k, self._dyn_corr_excl)
        else:
            return (self._estimator,)

    def _getcalc(self, measure):
        est = self._estimator

        if measure == "entropy":
            if est == 'kozachenko':
                return KLEntropyCalculator()
            if est in ('gaussian', 'kraskov'):
                return GaussianEntropyCalculator()
            if est == 'kernel':
                calc = KernelEntropyCalculator()
                calc.setProperty("KERNEL_WIDTH", str(self._kernel_width))
                return calc
            if est == 'symbolic':
                return None  # symbolic TE doesn't use entropy calculator

        if measure == "MutualInfo":
            if est in ('gaussian', 'kraskov', 'kozachenko'):
                return GaussianEntropyCalculator()  # dummy; multivariate bypasses
            if est == 'kernel':
                calc = KernelMICalculator()
                calc.setProperty("KERNEL_WIDTH", str(self._kernel_width))
                return calc

        if measure == "TransferEntropy":
            if est in ('gaussian', 'kraskov', 'kozachenko'):
                return _DummyTECalculator()
            if est == 'kernel':
                calc = KernelTECalculator()
                calc.setProperty("KERNEL_WIDTH", str(self._kernel_width))
                return calc
            if est == 'symbolic':
                return SymbolicTECalculator()

        raise TypeError(f"Unknown measure/estimator: {measure}/{est}")

    @parse_univariate
    def _compute_entropy(self, data, i=None):
        if not hasattr(data, "entropy"):
            data.entropy = {}

        key = self._getkey()
        if key not in data.entropy:
            data.entropy[key] = np.full((data.n_processes,), -np.inf)

        if data.entropy[key][i] == -np.inf:
            x = np.squeeze(data.to_numpy()[i])
            est = self._estimator

            if est in ('gaussian', 'kraskov'):
                data.entropy[key][i] = _gaussian_entropy_from_data(x.reshape(-1, 1))
            else:
                # kozachenko or kernel
                self._entropy_calc.initialise(1)
                self._entropy_calc.setObservations(x)
                data.entropy[key][i] = self._entropy_calc.computeAverageLocalOfObservations()

        return data.entropy[key][i]

    @parse_bivariate
    def _compute_joint_entropy(self, data, i, j):
        if not hasattr(data, "joint_entropy"):
            data.joint_entropy = {}

        key = self._getkey()
        if key not in data.joint_entropy:
            data.joint_entropy[key] = np.full((data.n_processes, data.n_processes), -np.inf)

        if data.joint_entropy[key][i, j] == -np.inf:
            # Always computed as (larger, smaller), since the value is cached
            # for both orders and the Kozachenko-Leonenko estimates of [x, y]
            # and [y, x] can differ in the last bit.
            x, y = data.to_numpy()[[max(i, j), min(i, j)]]
            joint = np.concatenate([x, y], axis=1)
            est = self._estimator

            if est in ('gaussian', 'kraskov'):
                _require_gaussian_samples(joint.shape[0], 2, "joint entropy",
                                          "the two processes")
                val = _gaussian_entropy_from_data(joint)
            else:
                # kozachenko or kernel
                self._entropy_calc.initialise(2)
                self._entropy_calc.setObservations(joint)
                val = self._entropy_calc.computeAverageLocalOfObservations()

            data.joint_entropy[key][i, j] = val
            data.joint_entropy[key][j, i] = val

        return data.joint_entropy[key][i, j]

    def _compute_conditional_entropy(self, X, Y):
        est = self._estimator
        if est in ('gaussian', 'kraskov'):
            XY = np.concatenate([X, Y], axis=1)
            return _gaussian_entropy_from_data(XY) - _gaussian_entropy_from_data(Y)
        else:
            # kozachenko or kernel
            XY = np.concatenate([X, Y], axis=1)
            self._entropy_calc.initialise(XY.shape[1])
            self._entropy_calc.setObservations(XY)
            H_XY = self._entropy_calc.computeAverageLocalOfObservations()
            self._entropy_calc.initialise(Y.shape[1])
            self._entropy_calc.setObservations(Y)
            H_Y = self._entropy_calc.computeAverageLocalOfObservations()
            return H_XY - H_Y

    def _per_process(self, data, tag, i, compute):
        """
        ``compute()`` for process ``i``, cached on ``data.entropy``.

        For a quantity of one series that a pairwise statistic would otherwise
        recompute for every partner. ``tag`` names it and carries every
        parameter it depends on beyond ``_getkey()``. ``data.entropy`` is in
        ``Data._CACHE_ATTRS``, so the cache is dropped when the series change.
        """
        if not hasattr(data, "entropy"):
            data.entropy = {}
        key = self._getkey() + tag
        if key not in data.entropy:
            data.entropy[key] = np.full((data.n_processes,), -np.inf)
        if data.entropy[key][i] == -np.inf:
            data.entropy[key][i] = compute()
        return data.entropy[key][i]

    def _resolve_theiler(self, data, i, j):
        """
        Theiler (dynamic correlation exclusion) window for pair (i, j).

        None gives 0 (no window), an integer that window, and "AUTO" the
        autocorrelation time ``2 * <acf(x_i), acf(x_j)>``, capped at N // 10
        (hctsa's rule for autocorrelation-derived parameters, so the window
        cannot approach the series length) and cached per dataset. For TLMI
        and TE it applies to the lagged or embedded sample index.
        """
        raw_w = getattr(self, '_dyn_corr_excl', None)
        if raw_w is None:
            return 0
        if raw_w == "AUTO":
            if not hasattr(data, 'theiler'):
                z = data.to_numpy()
                M = data.n_processes
                cap = data.n_observations // 10
                theiler = -np.ones((M, M))
                for _i in range(M):
                    for _j in range(_i + 1, M):
                        w = 2 * np.dot(utils.acf(z[_i]), utils.acf(z[_j]))
                        theiler[_i, _j] = min(int(w), cap)
                        theiler[_j, _i] = theiler[_i, _j]
                data.theiler = theiler
            return int(data.theiler[i, j])
        return int(raw_w)

    def _kernel_window(self, data, i, j):
        """
        JIDT's kernel ``DYN_CORR_EXCL`` for pair (i, j).

        JIDT's kernel estimators exclude ``|j - i| < DYN_CORR_EXCL``, so
        ``w + 1`` excludes ``|j - i| <= w``, the kraskov window. With w = 0
        that is the point itself (a leave-one-out density), as for kraskov.
        """
        w = self._resolve_theiler(data, i, j)
        if w < 0:
            raise ValueError(f"Theiler window must be >= 0, got w={w}.")
        return w + 1


class JointEntropy(InfoTheoryBase, Undirected):
    """
    Joint entropy ``H(x_i, x_j)`` of two processes, in nats.

    The differential entropy of the pair's joint distribution at lag 0,
    treating the samples as independent draws. Undirected and unsigned,
    though like any differential entropy its value can be negative; for given
    marginals it is lower when the pair is more dependent.

    The ``gaussian`` estimate is the entropy of a Gaussian with the sample
    covariance (with a relative ridge of 1e-8 on its diagonal); it is not
    appropriate (``NotAppropriate``, NaN in a Calculator table) for two
    samples or fewer, where that covariance is singular.
    The ``kozachenko`` estimate is the Kozachenko-Leonenko nearest-neighbour
    estimator [1]_ (k = 1, Euclidean norm). The ``kernel`` estimate counts
    neighbours in a box kernel, as in JIDT [2]_.
    A ``kozachenko`` estimate is not appropriate (NaN) for a pair with any
    repeated value, as in quantised data: it averages the logarithms of
    nearest-neighbour distances, which are zero for a repeated value, so a
    jitter would set the answer by its own size (KSG, which counts
    neighbours, can break ties this way).

    References
    ----------
    .. [1] L.F. Kozachenko, N.N. Leonenko, "Sample estimate of the entropy of a
           random vector", Problems of Information Transmission 23(2),
           95-101 (1987).
    .. [2] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).

    Parameters
    ----------
    estimator : str, optional
        ``'gaussian'``, ``'kozachenko'`` (raises for repeated values, and the
        pair is then NaN) or ``'kernel'``. Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    """

    name = "Joint entropy"
    identifier = "je"
    labels = ["infotheory", "unordered", "undirected", "inverted"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j)

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            _require_gaussian_samples(data.n_observations, 2, "joint entropy",
                                      "the two processes")
            JE = _gaussian_pairwise_joint_entropy(data.to_numpy(squeeze=True))
            np.fill_diagonal(JE, np.nan)
            return JE
        return super().multivariate(data)


class ConditionalEntropy(InfoTheoryBase, Directed):
    """
    Conditional entropy ``H(x_j | x_i)`` of the column given the row, in nats.

    ``H(x_i, x_j) - H(x_i)`` at lag 0, treating the samples as independent
    draws: ``A[i, j]`` is the uncertainty left in the column process once the
    row process is known. Directed, since ``H(x_j | x_i)`` and
    ``H(x_i | x_j)`` differ unless the marginal entropies are equal (the
    ``gaussian`` estimate is symmetric for z-scored data). Unsigned, though
    a differential entropy can be negative; lower means the row tells more
    about the column.

    The ``gaussian`` estimate is the entropy of a Gaussian with the sample
    covariance (with a relative ridge of 1e-8 on its diagonal); it is not
    appropriate (``NotAppropriate``, NaN in a Calculator table) for two
    samples or fewer, where that covariance is singular.
    The ``kozachenko`` estimate is the Kozachenko-Leonenko nearest-neighbour
    estimator [1]_ (k = 1, Euclidean norm). The ``kernel`` estimate counts
    neighbours in a box kernel, as in JIDT [2]_.
    A ``kozachenko`` estimate is not appropriate (NaN) for a pair with any
    repeated value, as in quantised data: it averages the logarithms of
    nearest-neighbour distances, which are zero for a repeated value, so a
    jitter would set the answer by its own size (KSG, which counts
    neighbours, can break ties this way).

    References
    ----------
    .. [1] L.F. Kozachenko, N.N. Leonenko, "Sample estimate of the entropy of a
           random vector", Problems of Information Transmission 23(2),
           95-101 (1987).
    .. [2] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).

    Parameters
    ----------
    estimator : str, optional
        ``'gaussian'``, ``'kozachenko'`` (raises for repeated values, and the
        pair is then NaN) or ``'kernel'``. Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    """

    name = "Conditional entropy"
    identifier = "ce"
    # Directed because H(X|Y) != H(Y|X) in general, even though the Gaussian
    # form is symmetric for z-scored data.
    labels = ["infotheory", "unordered", "directed", "inverted"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j) - self._compute_entropy(
            data, i=i
        )

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            _require_gaussian_samples(Z.shape[1], 2, "conditional entropy",
                                      "the two processes")
            variances = np.var(Z, axis=1, ddof=1)
            # Marginal entropy uses the same ridge as the scalar path.
            H_marginal = 0.5 * np.log(2 * np.pi * np.e * (variances + 1e-8 * variances))
            JE = _gaussian_pairwise_joint_entropy(Z)
            CE = JE - H_marginal[:, None]
            np.fill_diagonal(CE, np.nan)
            return CE
        return super().multivariate(data)


class MutualInfo(InfoTheoryBase, Undirected):
    """
    Mutual information ``I(x_i; x_j)`` between two processes, in nats.

    The mutual information of the pair at lag 0, treating the samples as
    independent draws from their joint distribution. Undirected.

    The ``gaussian`` estimate is ``-0.5 ln(1 - r^2)`` of the Pearson
    correlation ``r`` (with a relative ridge of 1e-8, which bounds it at about
    8.86 nats); it is not appropriate (``NotAppropriate``, NaN in a
    Calculator table) for two samples or fewer, where the pair's sample
    covariance is singular. The ``kraskov`` estimate is the KSG
    nearest-neighbour estimator 1 [1]_ on standardised series, with repeated
    values first broken by tiny fixed-seed jitter, so it is deterministic.
    The ``kernel`` estimate counts neighbours in a box kernel, as in JIDT
    [2]_, excluding the point itself. The ``kraskov`` estimate, and the
    ``kernel`` estimate with a Theiler window, can fall slightly below zero
    for independent processes; a kernel estimate in which every joint box is
    empty is NaN.

    References
    ----------
    .. [1] A. Kraskov, H. Stögbauer, P. Grassberger, "Estimating mutual
           information", Phys. Rev. E 69(6), 066138 (2004).
    .. [2] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).

    Parameters
    ----------
    estimator : str, optional
        ``'gaussian'``, ``'kraskov'`` or ``'kernel'``. Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    prop_k : int, optional
        Number of nearest neighbours ``k`` (``'kraskov'`` only). Default: 4.
    dyn_corr_excl : int, 'AUTO' or None, optional
        Theiler window ``w`` in samples (``'kraskov'`` and ``'kernel'`` only):
        neighbours ``|t - s| <= w`` are not counted. ``'AUTO'`` sets ``w`` for
        each pair to ``2 * sum_k acf_i(k) acf_j(k)`` over non-negative lags,
        capped at ``N // 10`` (at least 1 for 10 or more samples). With None
        or 0 only the point itself is excluded. Default: None.
    """

    name = "Mutual information"
    identifier = "mi"
    labels = ["infotheory", "unordered", "undirected"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        super().__setstate__(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        """Compute mutual information between Y and X"""
        if self._estimator in ('gaussian', 'kraskov'):
            z = data.to_numpy(squeeze=True)
            if self._estimator == 'gaussian':
                _require_gaussian_samples(z.shape[1], 2, "mutual information",
                                          "the two processes")
                return float(_gaussian_mi_from_r(np.corrcoef(z[i], z[j])[0, 1]))
            else:
                # kraskov bivariate
                k = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _ksg_mi_pair(z[i], z[j], k, w)

        # kernel estimator
        if self._estimator == 'kernel':
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            self._calc.initialise(1, 1)
            self._calc.setProperty("DYN_CORR_EXCL", str(self._kernel_window(data, i, j)))
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        # Not reached: every accepted estimator is handled above.
        logging.warning(f"MI bivariate: unhandled estimator '{self._estimator}'")
        return np.nan

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            _require_gaussian_samples(Z.shape[1], 2, "mutual information",
                                      "the two processes")
            MI = _gaussian_mi_from_r(np.corrcoef(Z))
            np.fill_diagonal(MI, np.nan)
            return MI
        elif self._estimator == 'kraskov':
            Z = data.to_numpy(squeeze=True)
            M, N = Z.shape
            k = int(self._prop_k)
            result = np.full((M, M), np.nan)
            pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
            return fill_pairwise(
                self.identifier, result, pairs,
                lambda i, j: _ksg_mi_pair(Z[i], Z[j], k, self._resolve_theiler(data, i, j)),
                mirror=True)
        return super().multivariate(data)


class TimeLaggedMutualInfo(InfoTheoryBase, Directed):
    """
    Time-lagged mutual information from the row (source) to the column (target).

    ``I(x_i(t); x_j(t + 1))`` in nats: the source one sample ahead of the
    target, with any Theiler window applied to the lagged sample index. The
    lag is fixed at one sample, so coupling that acts only at a longer lag is
    seen only through the source's own autocorrelation.

    The ``gaussian`` estimate is ``-0.5 ln(1 - r^2)`` of the Pearson
    correlation ``r`` (with a relative ridge of 1e-8, which bounds it at about
    8.86 nats); it is not appropriate (``NotAppropriate``, NaN in a
    Calculator table) for three samples or fewer, which leave two or fewer
    lagged pairs and a singular sample covariance. The ``kraskov`` estimate
    is the KSG nearest-neighbour estimator 1 [1]_ on standardised series,
    with repeated values first broken by tiny fixed-seed jitter, so it is
    deterministic. The ``kernel`` estimate counts neighbours in a box kernel,
    as in JIDT [2]_, excluding the point itself. The ``kraskov`` estimate,
    and the ``kernel`` estimate with a Theiler window, can fall slightly
    below zero for independent processes; a kernel estimate in which every
    joint box is empty is NaN.

    References
    ----------
    .. [1] A. Kraskov, H. Stögbauer, P. Grassberger, "Estimating mutual
           information", Phys. Rev. E 69(6), 066138 (2004).
    .. [2] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).

    Parameters
    ----------
    estimator : str, optional
        ``'gaussian'``, ``'kraskov'`` or ``'kernel'``. Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    prop_k : int, optional
        Number of nearest neighbours ``k`` (``'kraskov'`` only). Default: 4.
    dyn_corr_excl : int, 'AUTO' or None, optional
        Theiler window ``w`` in samples (``'kraskov'`` and ``'kernel'`` only):
        neighbours ``|t - s| <= w`` are not counted. ``'AUTO'`` sets ``w`` for
        each pair to ``2 * sum_k acf_i(k) acf_j(k)`` over non-negative lags,
        capped at ``N // 10`` (at least 1 for 10 or more samples). With None
        or 0 only the point itself is excluded. Default: None.
    """

    name = "Time-lagged mutual information"
    identifier = "tlmi"
    labels = ["infotheory", "temporal", "directed"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        super().__setstate__(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        if self._estimator in ('gaussian', 'kraskov'):
            z = data.to_numpy(squeeze=True)
            src = z[i][:-1]
            tgt = z[j][1:]
            if self._estimator == 'gaussian':
                _require_gaussian_samples(
                    src.size, 2, "time-lagged mutual information",
                    "the source and the target one sample later")
                return float(_gaussian_mi_from_r(np.corrcoef(src, tgt)[0, 1]))
            else:
                k = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _ksg_mi_pair(src, tgt, k, w)

        # kernel estimator
        if self._estimator == 'kernel':
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            src = src[:-1]
            targ = targ[1:]
            self._calc.initialise(1, 1)
            self._calc.setProperty("DYN_CORR_EXCL", str(self._kernel_window(data, i, j)))
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        logging.warning(f"TLMI bivariate: unhandled estimator '{self._estimator}'")
        return np.nan

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            M, T = Z.shape
            _require_gaussian_samples(
                T - 1, 2, "time-lagged mutual information",
                "the source and the target one sample later")
            Z_src = Z[:, :-1]
            Z_tgt = Z[:, 1:]
            stacked = np.vstack([Z_src, Z_tgt])
            R = np.corrcoef(stacked)
            TLMI = _gaussian_mi_from_r(R[:M, M:])
            np.fill_diagonal(TLMI, np.nan)
            return TLMI
        elif self._estimator == 'kraskov':
            Z = data.to_numpy(squeeze=True)
            M, T = Z.shape
            k = int(self._prop_k)
            Z_src = Z[:, :-1]
            Z_tgt = Z[:, 1:]
            result = np.full((M, M), np.nan)
            pairs = [(i, j) for i in range(M) for j in range(M) if i != j]
            return fill_pairwise(
                self.identifier, result, pairs,
                lambda i, j: _ksg_mi_pair(Z_src[i], Z_tgt[j], k, self._resolve_theiler(data, i, j)))
        return super().multivariate(data)


class TransferEntropy(InfoTheoryBase, Directed):
    """
    Transfer entropy from the row (source) to the column (target) [1]_.

    ``TE(x_i -> x_j) = I(y_{t+1}; x_t^(l) | y_t^(k))`` in nats: the
    information that the source's past adds about the target's next sample,
    beyond what the target's own past holds. ``y_t^(k)`` is ``k_history``
    target samples spaced ``k_tau`` apart and ``x_t^(l)`` is ``l_history``
    source samples spaced ``l_tau`` apart, both ending at ``t``, so the
    source-target delay is one sample. ``A[i, j]`` is the transfer from
    ``x_i`` to ``x_j``.

    With the ``gaussian`` estimator (identifier prefix ``gc`` instead of
    ``te``) this is ``0.5 * ln(var_restricted / var_full)``, from the
    target's residual variances when it is predicted from its own past alone
    and together with the source's past: half of Geweke's linear feedback
    measure [2]_, since Granger causality and transfer entropy are equivalent
    for Gaussian variables [3]_. It is not appropriate (``NotAppropriate``,
    NaN in a Calculator table) when the aligned sample count
    ``T - 1 - max((k_history - 1) * k_tau, (l_history - 1) * l_tau)`` does
    not exceed ``1 + k_history + l_history``, where the sample covariance is
    singular. The ``kraskov`` estimator computes the
    conditional mutual information directly with the Frenzel-Pompe
    nearest-neighbour estimator [4]_ [5]_, after repeated values are broken
    by tiny fixed-seed jitter. The ``kernel`` estimator counts neighbours in
    a box kernel, as in JIDT [6]_, with a target past of ``k_history``
    samples at unit delay and the source at lag one; in short series it is
    biased upward, and it is NaN when every joint box is empty (JIDT returns
    0). The ``symbolic`` estimator [7]_ replaces source and target by ordinal
    patterns of length ``k_history`` at unit delay and computes the transfer
    entropy between the symbol sequences from their histograms.

    With ``auto_embed_method``, the target embedding (and with
    ``'MAX_CORR_AIS'`` also the source embedding, chosen independently) is
    the one that maximises the series' own bias-corrected active information
    storage [8]_ over dimensions ``1, ..., k_search_max`` and delays
    ``1, ..., tau_search_max``: JIDT's ``MAX_CORR_AIS`` criterion [6]_, which
    follows [9]_. Ties go to the smaller dimension, then the smaller delay.
    The ``gaussian`` criterion subtracts the AIS bias ``k / (2N)`` and skips
    a candidate whose ``N`` aligned samples do not exceed ``1 + k``, and the
    ``kraskov`` criterion uses the KSG estimator with the same ``prop_k`` and
    Theiler window. A pair is NaN when no candidate embedding leaves enough
    samples.

    With a source history of one sample the source enters only at lag one,
    so coupling at a longer lag from a weakly autocorrelated source is
    missed. The auto-embedded variants choose the source embedding from the
    source's own active information storage, which stays short for such a
    source, so they share this limit.

    References
    ----------
    .. [1] T. Schreiber, "Measuring information transfer", Phys. Rev. Lett.
           85(2), 461-464 (2000).
    .. [2] J. Geweke, "Measurement of linear dependence and feedback between
           multiple time series", J. Am. Stat. Assoc. 77(378), 304-313 (1982).
    .. [3] L. Barnett, A.B. Barrett, A.K. Seth, "Granger causality and
           transfer entropy are equivalent for Gaussian variables", Phys. Rev.
           Lett. 103(23), 238701 (2009).
    .. [4] A. Kraskov, H. Stögbauer, P. Grassberger, "Estimating mutual
           information", Phys. Rev. E 69(6), 066138 (2004).
    .. [5] S. Frenzel, B. Pompe, "Partial mutual information for coupling
           analysis of multivariate time series", Phys. Rev. Lett. 99(20),
           204101 (2007).
    .. [6] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).
    .. [7] M. Staniek, K. Lehnertz, "Symbolic transfer entropy", Phys. Rev.
           Lett. 100(15), 158101 (2008).
    .. [8] J.T. Lizier, M. Prokopenko, A.Y. Zomaya, "Local measures of
           information storage in complex distributed computation", Inf. Sci.
           208, 39-54 (2012).
    .. [9] J. Garland, R.G. James, E. Bradley, "Leveraging information storage
           to select forecast-optimal parameters for delay-coordinate
           reconstructions", Phys. Rev. E 93(2), 022221 (2016).

    Parameters
    ----------
    estimator : str, optional
        ``'gaussian'``, ``'kraskov'``, ``'kernel'`` or ``'symbolic'``.
        Default: ``'gaussian'``.
    k_history : int, optional
        Target history length (for ``'symbolic'``, the ordinal pattern length,
        which must be at least 2). Not with ``auto_embed_method``. Default: 1.
    k_tau : int, optional
        Delay between target history samples (``'gaussian'`` and
        ``'kraskov'`` only; not with ``auto_embed_method``). Default: 1.
    l_history : int, optional
        Source history length (``'gaussian'`` and ``'kraskov'`` only; with
        ``auto_embed_method``, only ``'MAX_CORR_AIS_DEST_ONLY'``). Default: 1.
    l_tau : int, optional
        Delay between source history samples (as for ``l_history``).
        Default: 1.
    auto_embed_method : str or None, optional
        ``'MAX_CORR_AIS'`` selects the target and source embeddings and
        ``'MAX_CORR_AIS_DEST_ONLY'`` the target embedding only, with the
        source embedding fixed by ``l_history`` and ``l_tau`` (``'gaussian'``
        and ``'kraskov'`` only). None uses the fixed embedding.
        Default: None.
    k_search_max : int, optional
        Largest embedding dimension searched (requires
        ``auto_embed_method``). Default: 10.
    tau_search_max : int, optional
        Largest embedding delay searched (requires ``auto_embed_method``).
        Default: 4.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    prop_k : int, optional
        Number of nearest neighbours ``k`` (``'kraskov'`` only). Default: 4.
    dyn_corr_excl : int, 'AUTO' or None, optional
        Theiler window ``w`` in samples (``'kraskov'`` and ``'kernel'`` only),
        on the embedded sample index: neighbours ``|t - s| <= w`` are not
        counted. ``'AUTO'`` sets ``w`` for each pair to
        ``2 * sum_k acf_i(k) acf_j(k)`` over non-negative lags, capped at
        ``N // 10``. With None or 0 only the point itself is excluded.
        Default: None.
    """

    name = "Transfer entropy"
    identifier = "te"
    labels = ["embedding", "infotheory", "temporal", "directed"]

    # Only the two estimators with a genuine embedding implement any of the
    # embedding and search arguments. The symbolic and kernel calculators read
    # one history length at unit delay and nothing else.
    _EMBEDDING_ONLY = ("gaussian", "kraskov")

    def __init__(
        self,
        auto_embed_method=None,
        k_search_max=None,
        tau_search_max=None,
        k_history=None,
        k_tau=None,
        l_history=None,
        l_tau=None,
        **kwargs,
    ):
        if "estimator" not in kwargs.keys() or kwargs["estimator"] == "gaussian":
            self.identifier = "gc"
        super().__init__(**kwargs)

        if auto_embed_method is not None and auto_embed_method not in _AUTO_EMBED_METHODS:
            # Validated, so a typo cannot fall through to MAX_CORR_AIS.
            raise ValueError(
                f"Unknown auto_embed_method {auto_embed_method!r}; implemented: "
                f"{sorted(_AUTO_EMBED_METHODS)}. Ragwitz local-prediction "
                f"selection is a different criterion and is not implemented."
            )

        # An embedding or search argument the estimator does not use raises.
        supplied = {"k_tau": k_tau, "l_history": l_history, "l_tau": l_tau}
        if self._estimator not in self._EMBEDDING_ONLY:
            for name, value in supplied.items():
                if value is not None:
                    raise ValueError(
                        f"{name}={value!r} is not used by "
                        f"estimator={self._estimator!r}: it computes a single "
                        f"history length at unit delay, applied to source and "
                        f"destination alike. Set k_history, or use "
                        f"estimator='gaussian'/'kraskov' for independently "
                        f"aligned source and destination embeddings."
                    )
            for name, value in (("auto_embed_method", auto_embed_method),
                                ("k_search_max", k_search_max),
                                ("tau_search_max", tau_search_max)):
                if value is not None:
                    raise ValueError(
                        f"{name}={value!r} is not implemented for "
                        f"estimator={self._estimator!r}: there is no active-"
                        f"information-storage criterion for it, so no embedding "
                        f"can be selected. Use a fixed k_history, or "
                        f"estimator='gaussian'/'kraskov'."
                    )

        if auto_embed_method is None:
            # Search bounds with nothing to search raise.
            for name, value in (("k_search_max", k_search_max),
                                ("tau_search_max", tau_search_max)):
                if value is not None:
                    raise ValueError(
                        f"{name}={value!r} requires auto_embed_method to be set; "
                        f"with a fixed embedding there is nothing to search."
                    )
        else:
            # A selected embedding and a supplied one cannot both be honoured.
            fixed = {"k_history": k_history, "k_tau": k_tau}
            if auto_embed_method != _AUTO_EMBED_DEST_ONLY:
                fixed.update({"l_history": l_history, "l_tau": l_tau})
            for name, value in fixed.items():
                if value is not None:
                    raise ValueError(
                        f"{name}={value!r} conflicts with "
                        f"auto_embed_method={auto_embed_method!r}, which selects "
                        f"it. Drop {name}, or drop auto_embed_method."
                        + ("" if auto_embed_method == _AUTO_EMBED_DEST_ONLY else
                           f" ({_AUTO_EMBED_DEST_ONLY} selects the destination "
                           f"embedding only and does accept a fixed source one.)")
                    )

        # Defaults resolved before anything is validated or named, so the
        # identifier reports what is computed rather than what was typed.
        k_history = 1 if k_history is None else k_history
        k_tau = 1 if k_tau is None else k_tau
        l_history = 1 if l_history is None else l_history
        l_tau = 1 if l_tau is None else l_tau
        k_search_max = 10 if k_search_max is None else k_search_max
        tau_search_max = 4 if tau_search_max is None else tau_search_max

        for name, value in (("k_history", k_history), ("k_tau", k_tau),
                            ("l_history", l_history), ("l_tau", l_tau),
                            ("k_search_max", k_search_max),
                            ("tau_search_max", tau_search_max)):
            _require_positive_int(name, value)

        if self._estimator == "symbolic" and k_history < 2:
            # A length-1 ordinal pattern has one symbol, so TE is identically 0.
            raise ValueError(
                "estimator='symbolic' requires k_history >= 2: a length-1 "
                "ordinal pattern has a single symbol, so the transfer entropy "
                "is identically zero."
            )

        # Embedding parameters
        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

        self._calc = self._new_te_calc()

        if auto_embed_method is not None:
            # The method changes what is computed, so it is in the identifier.
            self.identifier += "_{}_k-max-{}_tau-max-{}".format(
                auto_embed_method.replace("_", "-"), k_search_max, tau_search_max)
            if auto_embed_method == _AUTO_EMBED_DEST_ONLY:
                self.identifier += "_l-{}_lt-{}".format(l_history, l_tau)
        else:
            if self._estimator in self._EMBEDDING_ONLY:
                self.identifier = self.identifier + "_k-{}_kt-{}_l-{}_lt-{}".format(
                    k_history, k_tau, l_history, l_tau
                )
            else:
                # One history length at unit delay.
                self.identifier = self.identifier + "_k-{}".format(k_history)

    def __setstate__(self, state):
        super().__setstate__(state)
        self._calc = self._new_te_calc()

    def _new_te_calc(self):
        """
        A TE calculator configured with this instance's embedding.

        Used by ``__init__`` and by ``__setstate__``, since pickling drops the
        calculator and a fresh one has the default history length of 1.
        """
        calc = self._getcalc("TransferEntropy")
        if self._auto_embed_method is not None:
            calc.setProperty(self._AUTO_EMBED_METHOD_PROP_NAME, self._auto_embed_method)
            calc.setProperty(self._K_SEARCH_MAX_PROP_NAME, str(self._k_search_max))
            calc.setProperty(self._TAU_SEARCH_MAX_PROP_NAME, str(self._tau_search_max))
        else:
            calc.setProperty(self._K_HISTORY_PROP_NAME, str(self._k_history))
            if self._estimator in self._EMBEDDING_ONLY:
                calc.setProperty(self._K_TAU_PROP_NAME, str(self._k_tau))
                calc.setProperty(self._L_HISTORY_PROP_NAME, str(self._l_history))
                calc.setProperty(self._L_TAU_PROP_NAME, str(self._l_tau))
        return calc

    # A class attribute, so the embedding criterion can be substituted.
    _embedding_scorer = staticmethod(_ais_scorer)

    def _selected_embedding(self, data, process, series, scorer, w):
        """
        ``(dimension, delay)`` for one series, cached on the Data object.

        The criterion is the series' own active information storage, so the
        choice depends on the series, estimator, search bounds and Theiler
        window, not on the pair or the role. The window is in the key because
        ``dyn_corr_excl="AUTO"`` makes it pair-dependent.
        """
        key = (process, self._estimator, self._prop_k, w,
               self._k_search_max, self._tau_search_max)
        cache = getattr(data, "ais_embedding", None)
        if cache is None:
            cache = data.ais_embedding = {}
        if key not in cache:
            if self._estimator == "kraskov":
                # A role-independent seed, so the cached choice does not depend
                # on whether the series was first seen as source or target.
                series = _tie_break(series, _TIE_SEED_SELECTION)
            dim, delay, _ = _select_embedding(
                series, scorer, self._k_search_max, self._tau_search_max)
            cache[key] = (dim, delay)
        return cache[key]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        est = self._estimator
        auto = self._auto_embed_method

        src, targ = data.to_numpy(squeeze=True)[[i, j]]

        # gaussian or kraskov, fixed embedding
        if est in ('gaussian', 'kraskov') and auto is None:
            if est == 'gaussian':
                return _gaussian_te_bivariate(
                    src, targ, self._k_history, self._k_tau,
                    self._l_history, self._l_tau
                )
            else:
                k_nn = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _kraskov_te_bivariate(
                    src, targ, self._k_history, self._k_tau,
                    self._l_history, self._l_tau, k_nn, w
                )

        # Auto-embedding by the maximum corrected AIS criterion.
        if est in ('gaussian', 'kraskov') and auto is not None:
            w = self._resolve_theiler(data, i, j)
            k_nn = int(self._prop_k) if est == 'kraskov' else None
            scorer = self._embedding_scorer(est, k_nn, w)

            k_history, k_tau = self._selected_embedding(data, j, targ, scorer, w)
            if auto == _AUTO_EMBED_DEST_ONLY:
                l_history, l_tau = self._l_history, self._l_tau
            else:
                # Selected from the source, independently, by the same criterion.
                l_history, l_tau = self._selected_embedding(data, i, src, scorer, w)

            if est == 'gaussian':
                return _gaussian_te_bivariate(src, targ, k_history, k_tau,
                                              l_history, l_tau)
            return _kraskov_te_bivariate(src, targ, k_history, k_tau,
                                         l_history, l_tau, k_nn, w)

        # kernel or symbolic
        if est in ('kernel', 'symbolic'):
            self._calc.initialise()
            if est == 'kernel':
                self._calc.setProperty("DYN_CORR_EXCL", str(self._kernel_window(data, i, j)))
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        logging.warning(f"TE bivariate: unhandled estimator '{est}'")
        return np.nan


class CrossmapEntropy(InfoTheoryBase, Directed):
    """
    Cross-map entropy of the column given the row's past.

    ``H(y_t | x_{t-1}, ..., x_{t-k+1})`` in nats for ``k = history_length``,
    computed as ``H(x_past, y_t) - H(x_past)``: the entropy of the target's
    current sample given ``k - 1`` lagged source samples, so the joint space
    has ``k`` columns. ``history_length=1`` still conditions on the source at
    lag 1, so it gives the same quantity as ``history_length=2``. Cross-map
    entropy has no canonical published definition; this is the convention
    pyspi uses. ``A[i, j]`` conditions the column process on the row's past.
    Unsigned, though a differential entropy can be negative; lower means
    the row's past tells more about the column.

    The ``gaussian`` estimate uses the entropy of a Gaussian with the sample
    covariance (with a relative ridge of 1e-8 on its diagonal), the
    ``kozachenko`` estimate the Kozachenko-Leonenko nearest-neighbour
    estimator [1]_ (k = 1, Euclidean norm), and the ``kernel`` estimate a box
    kernel, as in JIDT [2]_. The ``gaussian`` estimate is not appropriate
    (``NotAppropriate``, NaN in a Calculator table) when the ``T - k``
    aligned samples do not exceed the joint dimension ``max(k, 2)``
    (``T <= 20`` for ``k = 10``), where the sample covariance is singular.
    A ``kozachenko`` estimate is not appropriate (NaN) for a pair with any
    repeated value, as in quantised data: it averages the logarithms of
    nearest-neighbour distances, which are zero for a repeated value, so a
    jitter would set the answer by its own size (KSG, which counts
    neighbours, can break ties this way).

    References
    ----------
    .. [1] L.F. Kozachenko, N.N. Leonenko, "Sample estimate of the entropy of a
           random vector", Problems of Information Transmission 23(2),
           95-101 (1987).
    .. [2] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).

    Parameters
    ----------
    history_length : int, optional
        Size ``k`` of the joint embedding (``k - 1`` source lags, at least
        one). Default: 10.
    estimator : str, optional
        ``'gaussian'``, ``'kozachenko'`` (raises for repeated values, and the
        pair is then NaN) or ``'kernel'``. Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    """

    name = "Cross-map entropy"
    identifier = "xme"
    labels = ["infotheory", "temporal", "directed", "inverted"]

    def __init__(self, history_length=10, **kwargs):
        super().__init__(**kwargs)
        history_length = utils.require_int("history_length", history_length,
                                           minimum=1)
        self.identifier += f"_k{history_length}"
        self._history_length = history_length

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        src, targ = data.to_numpy(squeeze=True)[[i, j]]
        k = self._history_length
        if self._estimator == "gaussian":
            # The joint block [source lags, target] has max(k, 2) columns.
            _require_gaussian_samples(
                targ.size - k, max(k, 2), "cross-map entropy",
                f"max(history_length, 2) with history_length = {k}")
        targ_future = targ[k:]
        src_past = np.expand_dims(src[k - 1 : -1], axis=1)
        for idx in range(2, k):
            src_past = np.append(
                src_past, np.expand_dims(src[k - idx : -idx], axis=1), axis=1
            )

        joint = np.concatenate([src_past, np.expand_dims(targ_future, axis=1)], axis=1)

        self._entropy_calc.initialise(joint.shape[1])
        self._entropy_calc.setObservations(joint)
        H_xy = self._entropy_calc.computeAverageLocalOfObservations()

        # H(source past) depends on the source alone: computed once per process.
        def source_past_entropy():
            self._entropy_calc.initialise(src_past.shape[1])
            self._entropy_calc.setObservations(src_past)
            return self._entropy_calc.computeAverageLocalOfObservations()

        H_y = self._per_process(data, ("xme_source_past", k), i,
                                source_past_entropy)

        return H_xy - H_y


class CausalEntropy(InfoTheoryBase, Directed):
    """
    Causally conditioned entropy of the column given the row [1]_.

    Kramer's ``H(Y^n || X^n) = sum_{m=1}^{n} H(y_t | y_{t-1}, ..., y_{t-m+1},
    x_t, ..., x_{t-m+1})`` in nats over a horizon of ``n`` samples, each
    term estimated over every time ``t`` at which its embedding exists, as a
    difference of joint entropies. ``A[i, j]`` conditions the column process
    on the row process up to and including the current sample. An entropy,
    not a dependence: it is also low for a column that is predictable from
    its own past (`DirectedInfo` measures the dependence). Unsigned, though a
    sum of differential entropies can be negative; lower means the column is
    more predictable.

    The ``gaussian`` estimate uses the entropy of a Gaussian with the sample
    covariance (with a relative ridge of 1e-8 on its diagonal), the
    ``kozachenko`` estimate the Kozachenko-Leonenko nearest-neighbour
    estimator [2]_ (k = 1, Euclidean norm), and the ``kernel`` estimate a box
    kernel, as in JIDT [3]_. The ``gaussian`` estimate is not appropriate
    (``NotAppropriate``, NaN in a Calculator table) when ``T < 3n``: its
    last term has ``2n`` dimensions over ``T - n + 1`` times, and with no
    more samples than dimensions the sample covariance is singular.
    A ``kozachenko`` estimate is not appropriate (NaN) for a pair with any
    repeated value, as in quantised data: it averages the logarithms of
    nearest-neighbour distances, which are zero for a repeated value, so a
    jitter would set the answer by its own size (KSG, which counts
    neighbours, can break ties this way).

    References
    ----------
    .. [1] G. Kramer, "Causal conditioning, directed information and the
           multiple-access channel with feedback", Proc. 1998 IEEE Int. Symp.
           Information Theory, 189 (1998).
    .. [2] L.F. Kozachenko, N.N. Leonenko, "Sample estimate of the entropy of a
           random vector", Problems of Information Transmission 23(2),
           95-101 (1987).
    .. [3] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).

    Parameters
    ----------
    n : int, optional
        Horizon in samples. Default: 5.
    estimator : str, optional
        ``'gaussian'``, ``'kozachenko'`` (raises for repeated values, and the
        pair is then NaN) or ``'kernel'``. Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    """

    name = "Causally conditioned entropy"
    identifier = "cce"
    labels = ["infotheory", "temporal", "directed", "inverted"]

    def __init__(self, n=5, **kwargs):
        super().__init__(**kwargs)
        self._n = utils.require_int("n", n, minimum=1)
        # n changes the measure, so it must reach the identifier.
        self.identifier += f"_n-{self._n}"

    def _compute_causal_entropy(self, src, targ):
        src = np.squeeze(src)
        targ = np.squeeze(targ)
        if self._estimator == "gaussian":
            # The last term (m = n) has the largest block, [y_t, n - 1 target
            # lags, n source samples], over the fewest times, T - n + 1.
            _require_gaussian_samples(
                targ.size - self._n + 1, 2 * self._n,
                "causally conditioned entropy", f"2 * n with n = {self._n}")

        causal_entropy = 0
        for i in range(1, self._n + 1):
            Yp = _numpy_delay_embedding(targ, i - 1)[:-1]
            Xp = _numpy_delay_embedding(src, i)
            XYp = np.concatenate([Yp, Xp], axis=1)
            Yf = np.expand_dims(targ[i - 1:], 1)
            causal_entropy += self._compute_conditional_entropy(Yf, XYp)
        return causal_entropy

    def _getkey(self):
        return super(CausalEntropy, self)._getkey() + (self._n,)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if not hasattr(data, "causal_entropy"):
            data.causal_entropy = {}

        key = self._getkey()
        if key not in data.causal_entropy:
            data.causal_entropy[key] = np.full(
                (data.n_processes, data.n_processes), -np.inf
            )

        if data.causal_entropy[key][i, j] == -np.inf:
            z = data.to_numpy(squeeze=True)
            data.causal_entropy[key][i, j] = self._compute_causal_entropy(z[i], z[j])

        return data.causal_entropy[key][i, j]


class DirectedInfo(CausalEntropy, Directed):
    """
    Directed information from the row (source) to the column (target) [1]_.

    Massey's ``I(X^n -> Y^n) = sum_{m=1}^{n} I(x_t, ..., x_{t-m+1}; y_t |
    y_{t-1}, ..., y_{t-m+1})`` in nats over a horizon of ``n`` samples.
    Every term is estimated over one common window of times
    ``t = n, ..., T - 1``, and the source terms include the current sample
    ``x_t``. The ``gaussian`` estimate is not appropriate
    (``NotAppropriate``, NaN in a Calculator table) when ``T <= 3n``, since
    its largest term has ``2n`` dimensions over ``T - n`` times and with no
    more samples than dimensions the sample covariance is singular; the
    other estimators give NaN when the series has ``n + 1`` samples or
    fewer.

    The ``kraskov`` estimator computes each conditional mutual information
    directly with the Frenzel-Pompe nearest-neighbour estimator [2]_ [3]_,
    after repeated values are broken by tiny fixed-seed jitter. The other
    estimators compose each term from four joint entropies, estimated as for
    `CausalEntropy`; for ``kozachenko`` and ``kernel`` the estimation biases
    of those entropies, each in its own dimension, do not cancel.
    A ``kozachenko`` estimate is not appropriate (NaN) for a pair with any
    repeated value, as in quantised data: it averages the logarithms of
    nearest-neighbour distances, which are zero for a repeated value, so a
    jitter would set the answer by its own size (KSG, which counts
    neighbours, can break ties this way).

    References
    ----------
    .. [1] J.L. Massey, "Causality, feedback and directed information", Proc.
           Int. Symp. Information Theory and its Applications (ISITA-90),
           Waikiki, Hawaii, 303-305 (1990).
    .. [2] A. Kraskov, H. Stögbauer, P. Grassberger, "Estimating mutual
           information", Phys. Rev. E 69(6), 066138 (2004).
    .. [3] S. Frenzel, B. Pompe, "Partial mutual information for coupling
           analysis of multivariate time series", Phys. Rev. Lett. 99(20),
           204101 (2007).

    Parameters
    ----------
    n : int, optional
        Horizon in samples. Default: 5.
    estimator : str, optional
        ``'gaussian'``, ``'kraskov'``, ``'kozachenko'`` (raises for repeated
        values, and the pair is then NaN) or ``'kernel'``.
        Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    prop_k : int, optional
        Number of nearest neighbours ``k`` (``'kraskov'`` only). Default: 4.
    dyn_corr_excl : int, 'AUTO' or None, optional
        Theiler window ``w`` in samples (``'kraskov'`` only): neighbours
        ``|t - s| <= w`` are not counted. ``'AUTO'`` sets ``w`` for each pair
        to ``2 * sum_k acf_i(k) acf_j(k)`` over non-negative lags, capped at
        ``N // 10``. With None or 0 only the point itself is excluded.
        Default: None.
    """

    name = "Directed information"
    identifier = "di"
    labels = ["infotheory", "temporal", "directed"]

    def __init__(self, n=5, **kwargs):
        # n is handled by CausalEntropy; re-appending here would double the suffix.
        super().__init__(n=n, **kwargs)

    def _entropy_of(self, M):
        """Joint entropy of the columns of ``M``; 0 for a zero-column matrix."""
        if M.shape[1] == 0:
            return 0.0
        if self._estimator == "gaussian":
            return _gaussian_entropy_from_data(M)
        self._entropy_calc.initialise(M.shape[1])
        self._entropy_calc.setObservations(M)
        return self._entropy_calc.computeAverageLocalOfObservations()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """
        Directed information from process ``i`` to process ``j``.

        Each term ``H(Y_i | Y^{i-1}) - H(Y_i | Y^{i-1}, X^i)`` is estimated on
        one common window of aligned samples.
        """
        z = data.to_numpy(squeeze=True)
        src, targ = np.asarray(z[i], float), np.asarray(z[j], float)
        if self._estimator == "kraskov":
            src = _tie_break(src, _TIE_SEED_SOURCE)
            targ = _tie_break(targ, _TIE_SEED_TARGET)
        n = self._n
        T = targ.size
        if self._estimator == "gaussian":
            # The largest block, [y_t, n - 1 target lags, n source samples],
            # over the common window of T - n times.
            _require_gaussian_samples(T - n, 2 * n, "directed information",
                                      f"2 * n with n = {n}")
        if T <= n + 1:
            return np.nan

        # One aligned window for every term: rows are t = n .. T-1.
        y_now = targ[n:].reshape(-1, 1)
        y_lags = [targ[n - k: T - k].reshape(-1, 1) for k in range(1, n + 1)]
        # X^i includes the current source sample (same time index as y_i).
        x_lags = [src[n - k: T - k].reshape(-1, 1) for k in range(0, n)]

        total = 0.0
        for order in range(1, n + 1):
            Ypast = np.hstack(y_lags[: order - 1]) if order > 1 else y_now[:, :0]
            Xpast = np.hstack(x_lags[:order])

            if self._estimator == "kraskov":
                # I(X^i; Y_i | Y^{i-1}) directly, so the dimension-dependent
                # biases cancel (see _ksg_cmi).
                w = self._resolve_theiler(data, i, j) if self._dyn_corr_excl else 0
                _validate_ksg_sample(y_now.shape[0], int(self._prop_k), w,
                                     context="directed information")
                total += _ksg_cmi(y_now, Xpast, Ypast, int(self._prop_k), w)
                continue

            # H(Y_i | Y^{i-1}) - H(Y_i | Y^{i-1}, X^i)
            h_y_given_ypast = (
                self._entropy_of(np.hstack([y_now, Ypast])) - self._entropy_of(Ypast)
            )
            h_y_given_ypast_x = (
                self._entropy_of(np.hstack([y_now, Ypast, Xpast]))
                - self._entropy_of(np.hstack([Ypast, Xpast]))
            )
            total += h_y_given_ypast - h_y_given_ypast_x

        return total


class StochasticInteraction(InfoTheoryBase, Undirected):
    """
    Stochastic interaction of two processes [1]_.

    ``SI = H(x_t | x_{t-d}) + H(y_t | y_{t-d}) - H(x_t, y_t | x_{t-d}, y_{t-d})``
    in nats with ``d = delay``: the processes' conditional entropies given
    their own pasts, less the pair's conditional entropy given the pair's
    past. Undirected. Each conditional entropy is a difference of joint
    entropies.

    The ``gaussian`` estimate uses the entropy of a Gaussian with the sample
    covariance (with a relative ridge of 1e-8 on its diagonal), the
    ``kozachenko`` estimate the Kozachenko-Leonenko nearest-neighbour
    estimator [2]_ (k = 1, Euclidean norm), and the ``kernel`` estimate a box
    kernel, as in JIDT [3]_. The ``gaussian`` estimate is not appropriate
    (``NotAppropriate``, NaN in a Calculator table) when the ``T - d``
    aligned samples do not exceed the four dimensions of the pair and its
    past, where the sample covariance is singular. The ``kernel`` estimate
    is positively biased in short series, so compare pairs within a dataset
    rather than reading the absolute level.
    A ``kozachenko`` estimate is not appropriate (NaN) for a pair with any
    repeated value, as in quantised data: it averages the logarithms of
    nearest-neighbour distances, which are zero for a repeated value, so a
    jitter would set the answer by its own size (KSG, which counts
    neighbours, can break ties this way).

    References
    ----------
    .. [1] N. Ay, "Information geometry on complexity and stochastic
           interaction", Entropy 17(4), 2432-2458 (2015).
    .. [2] L.F. Kozachenko, N.N. Leonenko, "Sample estimate of the entropy of a
           random vector", Problems of Information Transmission 23(2),
           95-101 (1987).
    .. [3] J.T. Lizier, "JIDT: An information-theoretic toolkit for studying the
           dynamics of complex systems", Front. Robot. AI 1, 11 (2014).

    Parameters
    ----------
    delay : int, optional
        Delay ``d`` between past and present, in samples. Default: 1.
    estimator : str, optional
        ``'gaussian'``, ``'kozachenko'`` (raises for repeated values, and the
        pair is then NaN) or ``'kernel'``. Default: ``'gaussian'``.
    kernel_width : float, optional
        Half-width of the box kernel, in standard deviations of each variable
        (``'kernel'`` only). Default: 0.5.
    """

    name = "Stochastic interaction"
    identifier = "si"
    labels = ["infotheory", "temporal", "undirected"]

    def __init__(self, delay=1, **kwargs):
        super().__init__(**kwargs)
        delay = utils.require_int("delay", delay, minimum=1)
        self._delay = delay
        self.identifier += f"_k-{delay}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        x, y = data.to_numpy()[[i, j]]
        xy = np.concatenate([x, y], axis=1)
        tau = self._delay
        if self._estimator == "gaussian":
            _require_gaussian_samples(
                xy.shape[0] - tau, 4, "stochastic interaction",
                f"both processes now and at delay = {tau}")

        H_joint = self._compute_conditional_entropy(xy[tau:], xy[:-tau])
        # H(X_t | X_{t-tau}) depends on one series: computed once per process.
        tag = ("si_conditional", tau)
        H_src = self._per_process(data, tag, i, lambda: (
            self._compute_conditional_entropy(x[tau:], x[:-tau])))
        H_targ = self._per_process(data, tag, j, lambda: (
            self._compute_conditional_entropy(y[tau:], y[:-tau])))

        return H_src + H_targ - H_joint


class IntegratedInformation(Undirected, Unsigned):
    """
    Integrated information of the pair as a two-element system [1]_ [2]_.

    The pair is split into its two processes, and the measure is computed
    from the system's covariances at lag ``delay``, so it is a Gaussian
    (linear) estimate. ``phitype='star'`` gives phi* [1]_, ``I - I*``: the
    mutual information between the system's past and present, less the
    mismatched-decoding information ``I*`` of the disconnected model,
    maximised over its inverse temperature ``beta``. ``phitype='Geo'`` gives
    geometric integrated information [2]_,
    ``0.5 * ln(|Sigma_E'| / |Sigma_E|)``: the smallest Kullback-Leibler
    divergence between the fitted first-order autoregressive model and one
    with no coupling between the processes, found by an augmented Lagrangian
    method.
    Undirected. The computation follows the PhiToolbox implementation
    (``pyspi.lib.phi_native``). NaN when the estimate fails, for example for
    a singular covariance.

    References
    ----------
    .. [1] M. Oizumi, S. Amari, T. Yanagawa, N. Fujii, N. Tsuchiya, "Measuring
           integrated information from the decoding perspective", PLoS Comput.
           Biol. 12(1), e1004654 (2016).
    .. [2] M. Oizumi, N. Tsuchiya, S. Amari, "Unified framework for information
           integration based on information geometry", Proc. Natl. Acad. Sci.
           USA 113(51), 14817-14822 (2016).

    Parameters
    ----------
    phitype : str, optional
        ``'star'`` (phi*) or ``'Geo'`` (geometric). Default: ``'star'``.
    delay : int, optional
        Lag between past and present states, in samples. Default: 1.
    normalization : int, optional
        0 for none, or 1 to divide by the smaller of the two processes'
        Gaussian entropies (of the past state). Default: 0.
    """

    name = "Integrated information"
    identifier = "phi"
    labels = ["linear", "infotheory", "temporal", "undirected"]

    def __init__(self, phitype="star", delay=1, normalization=0):
        if phitype not in ("star", "Geo"):
            raise ValueError(f"Unknown phitype {phitype!r}; use 'star' or 'Geo'.")
        if normalization not in (0, 1):
            raise ValueError(f"normalization must be 0 or 1, got {normalization!r}.")
        delay = utils.require_int("delay", delay, minimum=1)
        self._phitype = phitype
        self._delay = delay
        self._normalization = normalization
        self.identifier += (f"_{phitype}_t-{delay}"
                            f"_norm-{utils.fmt_param(normalization)}")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # Not caught here: `fill_pairwise` sets a failed pair to NaN with a
        # warning, and a failure on every pair reaches `Calculator.errors`.
        from pyspi.lib.phi_native import phi_comp

        P = np.array([1, 2])
        X = data.to_numpy(squeeze=True)[[i, j]]

        params = {"tau": self._delay}
        options = {
            "type_of_phi": self._phitype,
            "type_of_dist": "Gauss",
            "normalization": self._normalization
        }

        return phi_comp(X, P, params, options)

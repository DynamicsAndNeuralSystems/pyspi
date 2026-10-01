"""Information-theoretic statistics, implemented in pure NumPy.

The estimators here were ported from JIDT (Lizier, 2014, "JIDT: An
information-theoretic toolkit for studying the dynamics of complex systems",
Frontiers in Robotics and AI), which served as the reference implementation the
port was validated against; the version used for validation was JIDT 1.6.1.
"""

import math
import numpy as np
from pyspi import utils
import copy
import logging
import warnings

from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from pyspi.base import Undirected, Directed, Unsigned, parse_univariate, parse_bivariate, parse_multivariate

# ---------------------------------------------------------------------------
# Pure-numpy entropy calculators (drop-in replacements for JIDT)
# ---------------------------------------------------------------------------

# One singularity policy for every Gaussian quantity in this module: each
# variable is treated as observed with independent additive noise of variance
# GAUSSIAN_RIDGE times its own variance, so
#
#     Sigma  ->  Sigma + GAUSSIAN_RIDGE * diag(diag(Sigma)).
#
# Two properties make this the version worth having, as opposed to the
# isotropic eps = ridge * mean(diag(Sigma)) it replaces:
#
#   * It commutes with the compositions. log|.| of a 1-D block is
#     log(V(1+ridge)), of a 2-D block log(V_i V_j((1+ridge)^2 - r^2)), so
#     H(X) + H(Y) - H(X,Y) is exactly -0.5*log(1 - r^2/(1+ridge)^2) -- the
#     direct MI formula below, with no residual. An isotropic ridge does not
#     have this property, because eps then depends on which block it is in.
#   * It is equivariant to per-variable rescaling, so the regularisation does
#     not quietly depend on the units of the loudest process.
#
# The previous mismatch was not subtle. Gaussian MI clipped r^2 at 1 - 1e-15,
# giving -0.5*log(1e-15) = 17.2698 nats on a pair of identical N=100 series,
# while the entropy path's ridge gave 8.8638 for the same quantity on the same
# data. Both now return 8.8638.
GAUSSIAN_RIDGE = 1e-8


def _gaussian_mi_from_r(r, ridge_rel=GAUSSIAN_RIDGE):
    """I(X;Y) = -0.5*log(1 - r^2) for jointly Gaussian X, Y, ridged.

    The ridge is what the entropy path applies, expressed on the correlation
    scale: adding independent noise of variance ridge*V to each variable
    attenuates the correlation by exactly (1 + ridge). The bound is therefore
    -0.5*log(1 - 1/(1+ridge)^2) ~ 8.86 nats rather than an unreachable
    infinity, and it is the *same* bound the entropy composition reaches.
    """
    r2 = np.clip(np.asarray(r, dtype=np.float64) ** 2, 0.0, 1.0)
    return -0.5 * np.log(1.0 - r2 / (1.0 + ridge_rel) ** 2)


def _gaussian_pairwise_joint_entropy(Z, ridge_rel=GAUSSIAN_RIDGE):
    """Vectorised pairwise Gaussian joint entropy, matching the scalar path.

    Same ridge as _gaussian_log_det, so bivariate() and multivariate() cannot
    disagree; before either was regularised consistently they differed by 8.406
    nats on singular data.
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
    """log|Σ + ridge*diag(diag(Σ))| -- the shared Gaussian log-determinant.

    Every Gaussian entropy, joint entropy, conditional entropy, MI, TLMI, TE
    and AIS in this module goes through this or through `_gaussian_mi_from_r`,
    which is its closed form on a 2x2 block. Returns NaN, never -inf, on input
    that is still degenerate after the ridge -- a variable with zero variance
    has no scale for a *proportional* ridge to act on, and its differential
    entropy really is -inf, so NaN reports "undefined" rather than propagating
    an infinity into a difference of entropies.

    Rationale for regularising at all: for rank-deficient Σ the unregularised
    log|Σ| is -inf (the data lives on a lower-dimensional subspace). JIDT adds
    NOISE_LEVEL_TO_ADD=1e-8 Gaussian noise to the observations for the same
    reason; ridging Σ is that operation in expectation (cov(x + η) =
    Σ + cov(η)) and is deterministic.
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


class GaussianEntropyCalculator:
    """Drop-in for JIDT's EntropyCalculatorMultiVariateGaussian.

    H = 0.5 * d * log(2*pi*e) + 0.5 * log|Sigma|
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
        cov = np.cov(X, rowvar=False, ddof=1)
        log_det = _gaussian_log_det(cov)
        if not np.isfinite(log_det):
            return float('nan')
        return float(0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det)


class KLEntropyCalculator:
    """Drop-in for JIDT's EntropyCalculatorMultiVariateKozachenko.

    Uses L2 norm with k=1, matching JIDT.
    H = psi(N) - psi(1) + log(c_d) + (d/N) * sum(log(eps_i))
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
        # Duplicated observations put a neighbour at distance 0, and log(0)
        # sends the entropy to -inf. Quantised data does this readily: the
        # bundled `forex` series has one process with 24 distinct values in 250
        # samples. Fail with the cause named rather than emitting an infinity.
        if not np.all(eps > 0):
            n_tied = int(np.sum(eps == 0))
            raise ValueError(
                f"Kozachenko entropy is undefined for tied observations: "
                f"{n_tied} of {len(eps)} points have a duplicate (zero "
                f"nearest-neighbour distance). The data is quantised or has "
                f"repeated values; dither it or use estimator='gaussian'."
            )
        log_cd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)
        return float(
            digamma(N) - digamma(1) + log_cd + (d / N) * np.sum(np.log(eps))
        )


def _gaussian_entropy_from_data(data_2d):
    """Compute Gaussian entropy from (N, d) array.

    Returns NaN (not -inf) on singular covariance after ridge regularisation,
    so that downstream differences (e.g. H(X,Y)-H(Y)) never cascade to
    +/-inf and get clipped to float-max by np.nan_to_num.
    """
    N, d = data_2d.shape
    cov = np.cov(data_2d, rowvar=False, ddof=1)
    log_det = _gaussian_log_det(cov)
    if not np.isfinite(log_det):
        return float('nan')
    return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det


# ---------------------------------------------------------------------------
# Box-kernel KDE entropy/MI/TE calculators (replace JIDT kernel estimators)
# ---------------------------------------------------------------------------

class KernelEntropyCalculator:
    """Drop-in for JIDT's EntropyCalculatorMultiVariateKernel.

    Box kernel (Heaviside) with L-infinity norm, matching JIDT exactly:
    H = mean(log(N) - log(count_i))  [nats]
    where count_i = #{j : |x_j - x_i|_inf <= kernel_width} (includes self).

    When NORMALISE=true (JIDT default), data is normalised by std before
    counting; kernel_width is then in units of std dev.
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
        # NORMALISE: JIDT scales the bandwidth by std (kernelWidthsInUse = w*std)
        # rather than standardising data. Equivalent operation here is to
        # standardise X *and* add log(prod(std)) to the entropy — otherwise
        # we'd be reporting H(X/std) = H(X) - log(std), missing the scale
        # term and giving a constant downward bias of d*log(std) nats.
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


class KernelMICalculator:
    """Drop-in for JIDT's MutualInfoCalculatorMultiVariateKernel.

    MI = mean(log(n_xy * N / (n_x * n_y)))  [nats]
    where n_x, n_y, n_xy are counts within L∞ ball of radius kernel_width.
    Matches JIDT exactly.
    """

    def __init__(self):
        self._kernel_width = 0.25
        self._normalise = True
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
            def _norm(data):
                stds = np.std(data, axis=0, ddof=1)
                stds = np.where(stds > 0, stds, 1.0)
                return data / stds[None, :]
            X = _norm(X)
            Y = _norm(Y)

        XY = np.column_stack([X, Y])
        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)
        tree_xy = cKDTree(XY)

        # JIDT half-width = kernel_width
        n_x = np.asarray(tree_x.query_ball_point(X, r=w, p=np.inf,
                                                   return_length=True), dtype=np.float64)
        n_y = np.asarray(tree_y.query_ball_point(Y, r=w, p=np.inf,
                                                   return_length=True), dtype=np.float64)
        n_xy = np.asarray(tree_xy.query_ball_point(XY, r=w, p=np.inf,
                                                     return_length=True), dtype=np.float64)

        # MI = mean(log(n_xy * N / (n_x * n_y)))  [nats]
        mi = np.mean(np.log(n_xy) + np.log(N) - np.log(n_x) - np.log(n_y))
        return float(mi)


class KernelTECalculator:
    """Drop-in for JIDT's TransferEntropyCalculatorKernel.

    TE(X→Y) = mean(log(n_yn_yp_x * n_yp / (n_yp_x * n_yn_yp)))  [nats]
    Uses box kernel with L∞ norm, half-width = kernel_width.
    Matches JIDT exactly.
    """

    def __init__(self):
        self._kernel_width = 0.25
        self._normalise = True
        self._k_history = 1
        self._dyn_corr_excl = None
        self._props = {}

    def initialise(self):
        pass

    def setProperty(self, key, value):
        self._props[key] = value
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
            def _norm(data):
                stds = np.std(data, axis=0, ddof=1)
                stds = np.where(stds > 0, stds, 1.0)
                return data / stds[None, :]
            y_past = _norm(y_past)
            y_next = _norm(y_next)
            x_t = _norm(x_t)

        # Joint spaces
        yn_yp = np.column_stack([y_next, y_past])
        yp_x = np.column_stack([y_past, x_t])
        yn_yp_x = np.column_stack([y_next, y_past, x_t])

        tree_yp = cKDTree(y_past)
        tree_yn_yp = cKDTree(yn_yp)
        tree_yp_x = cKDTree(yp_x)
        tree_yn_yp_x = cKDTree(yn_yp_x)

        # JIDT half-width = kernel_width
        dce = self._dyn_corr_excl
        if dce is None or dce <= 0:
            n_yp = np.asarray(tree_yp.query_ball_point(y_past, r=w, p=np.inf,
                                                         return_length=True), dtype=np.float64)
            n_yn_yp = np.asarray(tree_yn_yp.query_ball_point(yn_yp, r=w, p=np.inf,
                                                               return_length=True), dtype=np.float64)
            n_yp_x = np.asarray(tree_yp_x.query_ball_point(yp_x, r=w, p=np.inf,
                                                             return_length=True), dtype=np.float64)
            n_yn_yp_x = np.asarray(tree_yn_yp_x.query_ball_point(yn_yp_x, r=w, p=np.inf,
                                                                   return_length=True), dtype=np.float64)
        else:
            # Theiler window: exclude neighbours j with |j - i| <= dce, matching
            # JIDT's DYN_CORR_EXCL convention (excludes 2*dce+1 points centered on i,
            # including self). We query index lists then filter by temporal separation.
            def _counts_with_theiler(tree, pts):
                idx_lists = tree.query_ball_point(pts, r=w, p=np.inf)
                out = np.empty(len(idx_lists), dtype=np.float64)
                for i, nbrs in enumerate(idx_lists):
                    nbrs_arr = np.asarray(nbrs, dtype=np.int64)
                    out[i] = np.sum(np.abs(nbrs_arr - i) > dce)
                return out
            n_yp = _counts_with_theiler(tree_yp, y_past)
            n_yn_yp = _counts_with_theiler(tree_yn_yp, yn_yp)
            n_yp_x = _counts_with_theiler(tree_yp_x, yp_x)
            n_yn_yp_x = _counts_with_theiler(tree_yn_yp_x, yn_yp_x)

        # Drop samples where any bin is empty (log(0) = -inf; JIDT skips these).
        valid = (n_yp > 0) & (n_yn_yp > 0) & (n_yp_x > 0) & (n_yn_yp_x > 0)
        if not np.any(valid):
            return float('nan')
        n_yp, n_yn_yp = n_yp[valid], n_yn_yp[valid]
        n_yp_x, n_yn_yp_x = n_yp_x[valid], n_yn_yp_x[valid]

        # TE = mean(log(n_yn_yp_x * n_yp / (n_yp_x * n_yn_yp)))  [nats]
        te = np.mean(np.log(n_yn_yp_x) + np.log(n_yp) - np.log(n_yp_x) - np.log(n_yn_yp))
        return float(te)


# ---------------------------------------------------------------------------
# Symbolic Transfer Entropy (ordinal patterns)
# ---------------------------------------------------------------------------

def _ordinal_pattern_id(vec):
    """Convert a vector to its ordinal pattern ID.

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
    """Convert a 1D time series to a sequence of ordinal pattern symbols.

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
    """Drop-in for JIDT's TransferEntropyCalculatorSymbolic.

    Converts source and target to ordinal patterns of length k,
    then computes TE from joint symbol histograms.

    One pattern length, applied to source and destination alike, at unit
    delay -- Staniek & Lehnertz (2008) define it this way and JIDT's
    TransferEntropyCalculatorSymbolic does the same. There is no separate
    source history or embedding delay here, so `TransferEntropy` refuses
    `k_tau`, `l_history` and `l_tau` under this estimator rather than
    accepting them, writing them into the identifier and ignoring them.

    TE = H(Y_next | Y_past) - H(Y_next | Y_past, X)
       = H(Y_next, Y_past) - H(Y_past) - H(Y_next, Y_past, X) + H(Y_past, X)
    where all entropies are discrete (histogram-based).
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

        # Align: targ_next starts at index 1 of the symbol sequence
        # targ_past = targ_symbols[:-1], targ_next = targ_symbols[1:]
        # src_current = src_symbols[:-1] (concurrent with targ_past)
        n = min(len(src_symbols), len(targ_symbols)) - 1
        if n <= 0:
            return np.nan

        targ_next = targ_symbols[1:n + 1]
        targ_past = targ_symbols[:n]
        src_curr = src_symbols[:n]

        def _discrete_entropy(*arrs):
            """Joint entropy of integer-valued arrays using histograms.

            Counts distinct rows directly rather than packing the symbols into
            a single integer. The previous encoding multiplied by a multiplier
            that squared at each step, so the packed value reached (k!)^3 and
            exceeded int64 for k >= 10, wrapping silently. Wrapping is not the
            same as colliding -- no collisions occur on the shipped fixtures --
            but the encoding gave no guarantee, and correctness should not rest
            on the arithmetic happening to stay injective.
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
    """Numpy equivalent of JIDT's MatrixUtils.makeDelayEmbeddingVector."""
    x = np.asarray(x).ravel()
    T = len(x)
    if dim == 0:
        return np.empty((T + 1, 0))
    return np.column_stack([x[dim - 1 - lag: T - lag] for lag in range(dim)])


# ---------------------------------------------------------------------------
# k-NN input conditioning
# ---------------------------------------------------------------------------

# JIDT switches both of these on by default for every KSG-family calculator:
# MutualInfoCalculatorMultiVariateKraskov, ConditionalMutualInfoCalculator-
# MultiVariateKraskov and EntropyCalculatorMultiVariateKozachenko all set
# `addNoise = true; noiseLevel = 1e-8` in their constructors ("to match the
# noise order in MILCA toolkit"); `normalise = true` is inherited from their
# common base. pyspi retains normalisation but intentionally does not copy the
# random-noise policy.
# KSG estimator 1 is a continuous-density estimator. Exact coordinate ties
# make the kth-neighbour radius/counting convention ambiguous; deterministic
# jitter merely replaces that ambiguity with dependence on an arbitrary noise
# realisation, and content-derived jitter additionally depends on sample order.
# pyspi therefore refuses tied coordinates instead of pretending to recover a
# discrete information measure. Per-coordinate standardisation remains: the
# joint L-infinity geometry is then covariant under nonzero affine marginal
# changes, including reflections, without rounding or random state.


def _knn_condition(X):
    """Validate continuous KSG input and standardise each coordinate.

    Every coordinate must be tie-free. Quantised/discrete data requires an
    external discrete plug-in information estimator; pyspi has no general
    discrete MI/TLMI/DI estimator. Adding deterministic jitter does not turn
    KSG into one and can make the result depend on sample order.

    Standardisation uses an observed origin before mean/std calculation to
    reduce cancellation under affine offsets. No rounding, dither, seed, or
    global RNG state participates in the estimator.
    """
    X = np.asarray(X, dtype=np.float64)
    reshaped = X.ndim == 1
    if reshaped:
        X = X.reshape(-1, 1)

    out = np.empty_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]
        n_unique = np.unique(col).size
        if n_unique != col.size:
            raise ValueError(
                f"KSG requires continuous, tie-free coordinates: coordinate "
                f"{c} has {n_unique} unique value(s) among {col.size} "
                f"observations. Deterministic jitter would make the estimate "
                f"depend on arbitrary noise. Use a discrete plug-in "
                f"information estimator outside pyspi for quantised/discrete "
                f"data, or "
                f"remove measurement rounding only when scientifically "
                f"justified."
            )
        # Subtract an observed origin before taking the mean/std. This is
        # algebraically identical, but avoids needless cancellation when an
        # affine transform adds an offset large relative to the variation.
        shifted = col - col[0]
        sd = shifted.std(ddof=1)      # JIDT MatrixUtils.stdDevs: sample std
        out[:, c] = (shifted - shifted.mean()) / sd
    return out.reshape(-1) if reshaped else out


# ---------------------------------------------------------------------------
# KSG MI estimator
# ---------------------------------------------------------------------------

def _validate_ksg_sample(N, k, w, context=""):
    """Reject KSG settings that cannot produce a meaningful estimate.

    The estimator needs k neighbours drawn from the points that survive the
    Theiler exclusion. Without this check, k >= N silently returned a finite
    number that grows with k: k=30 on N=20 gave 0.414 and k=100 gave 1.63,
    neither of which is an estimate of anything.
    """
    effective = N - (2 * w + 1) if w else N - 1
    where = f" ({context})" if context else ""
    if N < 2:
        raise ValueError(f"KSG needs at least 2 observations, got {N}{where}.")
    if k < 1:
        raise ValueError(f"KSG needs k >= 1, got k={k}{where}.")
    if w < 0:
        raise ValueError(f"Theiler window must be >= 0, got w={w}{where}.")
    if k > effective:
        raise ValueError(
            f"KSG k={k} exceeds the {effective} usable neighbour(s) for "
            f"N={N} observations with Theiler window w={w}{where}. Reduce k, "
            f"lengthen the series, or narrow the Theiler window."
        )


def _strict_radius(epsilon):
    """Largest representable radius strictly smaller than ``epsilon``.

    ``cKDTree.query_ball_point`` includes the radius boundary. KSG estimator 1
    requires marginal distances strictly below the joint kth-neighbour radius,
    so moving by one floating-point step implements that inequality without a
    scale-dependent tolerance that can discard genuine interior neighbours.
    """
    return np.nextafter(epsilon, -np.inf)


def _ksg_mi_pair(x, y, k, w):
    """KSG Estimator 1 MI for a single pair of scalar series."""
    N = len(x)
    _validate_ksg_sample(N, k, w, context="mutual information")
    # A constant marginal carries no information, but it is almost always a
    # broken input rather than a result the caller wants. Checked before
    # conditioning so the error can name the marginal directly.
    for name, v in (("x", x), ("y", y)):
        if np.ptp(v) == 0:
            raise ValueError(
                f"KSG cannot estimate mutual information: marginal {name} is "
                f"constant, so all neighbour distances are zero."
            )
    x, y = _knn_condition(np.column_stack([x, y])).T
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))
    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)

    if w == 0:
        dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
        eps = dists[:, k]
        eps_strict = _strict_radius(eps)
        nx_lists = tree_x.query_ball_point(x.reshape(-1, 1), eps_strict, p=np.inf)
        ny_lists = tree_y.query_ball_point(y.reshape(-1, 1), eps_strict, p=np.inf)
        n_x = np.array([len(lst) - 1 for lst in nx_lists], dtype=np.float64)
        n_y = np.array([len(lst) - 1 for lst in ny_lists], dtype=np.float64)
    else:
        n_query = min(k + 2 * w + 2, N)
        dists_all, idx_all = tree_xy.query(xy, k=n_query, p=np.inf)

        eps = np.empty(N)
        n_x = np.empty(N)
        n_y = np.empty(N)

        for i in range(N):
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False
            d_valid = dists_all[i][valid]

            if len(d_valid) < k:
                all_dists = np.max(np.abs(xy - xy[i]), axis=1)
                all_dists[max(0, i - w): i + w + 1] = np.inf
                all_dists[i] = np.inf
                d_valid = np.sort(all_dists)
                d_valid = d_valid[np.isfinite(d_valid)]

            e = d_valid[k - 1] if len(d_valid) >= k else np.inf
            eps[i] = e

            # Count marginal neighbours STRICTLY within eps (< eps), matching
            # KSG1 and the w==0 branch above. Inclusive (<= eps) counting wrongly
            # admits the k-th neighbour at the boundary, inflating n_x/n_y and
            # flipping the sign of the Theiler-window effect on MI.
            e_strict = _strict_radius(e)
            ix = tree_x.query_ball_point([[x[i]]], e_strict, p=np.inf)[0]
            iy = tree_y.query_ball_point([[y[i]]], e_strict, p=np.inf)[0]
            n_x[i] = sum(1 for j in ix if abs(j - i) > w and j != i)
            n_y[i] = sum(1 for j in iy if abs(j - i) > w and j != i)

    # psi(N) uses the full, unreduced N (KSG1 convention; only the neighbour
    # set is window-restricted, not the normalisation). See JIDT
    # MutualInfoCalculatorMultiVariateKraskov.
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)
    return float(mi)


def _ksg_mi_general(A, B, k_nn, w=0, condition=True):
    """KSG Estimator 1 MI(A; B) for multivariate A (N,dA), B (N,dB), L-inf norm.

    Generalisation of _ksg_mi_pair to arbitrary marginal dimensions.
    Marginal neighbours are counted strictly (< eps); psi(N) uses full N. An
    optional Theiler window w excludes |j-i| <= w from the neighbour set.
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
    # Validated here, on the aligned arrays, not only by whatever built them.
    # This is the estimator's own precondition: with fewer usable neighbours
    # than k, `tree.query(..., k=k_nn+1)` pads with infinities and the digamma
    # assembly returns a finite number that is not an estimate of anything.
    _validate_ksg_sample(N, k_nn, w, context="mutual information")
    AB = np.column_stack([A, B])
    tree_ab = cKDTree(AB)
    tree_a = cKDTree(A)
    tree_b = cKDTree(B)

    if w == 0:
        dists, _ = tree_ab.query(AB, k=k_nn + 1, p=np.inf)
        eps = _strict_radius(dists[:, k_nn])
        n_a = np.array([len(lst) - 1 for lst in
                        tree_a.query_ball_point(A, eps, p=np.inf)], dtype=np.float64)
        n_b = np.array([len(lst) - 1 for lst in
                        tree_b.query_ball_point(B, eps, p=np.inf)], dtype=np.float64)
    else:
        n_query = min(k_nn + 2 * w + 2, N)
        dists_all, idx_all = tree_ab.query(AB, k=n_query, p=np.inf)
        n_a = np.empty(N)
        n_b = np.empty(N)
        for i in range(N):
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False
            d_valid = dists_all[i][valid]
            if len(d_valid) < k_nn:
                all_d = np.max(np.abs(AB - AB[i]), axis=1)
                all_d[max(0, i - w): i + w + 1] = np.inf
                all_d[i] = np.inf
                d_valid = np.sort(all_d)
                d_valid = d_valid[np.isfinite(d_valid)]
            e = d_valid[k_nn - 1] if len(d_valid) >= k_nn else np.inf
            e_strict = _strict_radius(e)
            ia = tree_a.query_ball_point(A[i], e_strict, p=np.inf)
            ib = tree_b.query_ball_point(B[i], e_strict, p=np.inf)
            n_a[i] = sum(1 for j in ia if abs(j - i) > w and j != i)
            n_b[i] = sum(1 for j in ib if abs(j - i) > w and j != i)

    return float(digamma(k_nn) + digamma(N)
                 - np.mean(digamma(n_a + 1) + digamma(n_b + 1)))


# ---------------------------------------------------------------------------
# Transfer Entropy helpers
# ---------------------------------------------------------------------------

class _DummyTECalculator:
    """Dummy TE calculator for gaussian/kraskov fixed-embedding."""
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
    """Bias-corrected Gaussian AIS criterion: MI(Y_future; Y_past_embedding(k, tau)).

    Used for AIS-criterion auto-embedding: select (k, tau) = argmax AIS.

    The raw in-sample log-det multiinformation is biased upward by the mean of
    its chi-squared null, E[MI_null] = df / (2N) with df = dim(Y_f)*k = k (the
    future is 1-D). Without this correction the criterion increases monotonically
    in k and saturates at k_max. Subtracting k/(2N) gives an interior maximum and
    matches JIDT's ActiveInfoStorageCalculatorGaussian.computeAdditionalBiasToRemove.

    This is the *maximum corrected AIS* criterion (Wibral et al. 2014; JIDT's
    AUTO_EMBED_METHOD_MAX_CORR_AIS). It is not the Ragwitz criterion, which
    selects by local-prediction error (Ragwitz & Kantz 2002) and is not
    implemented here.
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

    def _slogdet(arr):
        return _gaussian_log_det(np.cov(arr, rowvar=False, ddof=1))

    ais_raw = 0.5 * (_slogdet(Y_f) + _slogdet(Y_p) - _slogdet(YfYp))
    return ais_raw - k / (2.0 * N)


def _ksg_ais(targ, k, tau, k_nn, w=0):
    """KSG (Kraskov) AIS = MI(Y_future; Y_past_embedding(k, tau)), for embedding
    selection of the *kraskov* TE estimator.

    Estimator-consistent counterpart of _gaussian_ais: the KSG estimator is
    approximately bias-free, so max-KSG-AIS over k has a genuine interior peak
    without an explicit bias term (cf. JIDT MAX_CORR_AIS, which returns 0 extra
    bias for KSG; Wibral et al. 2014). Selecting the embedding with the same
    estimator used for the final TE avoids the linear/nonlinear mismatch of
    using Gaussian AIS to embed a nonlinear estimator.
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
    """``f(series, dim, delay) -> score`` for the estimator that will run the TE.

    Factored out so the source and the destination embeddings are chosen by the
    same code, with the same estimator, and so a test can substitute a scorer
    that deliberately picks different values for the two and check that all four
    reach the final estimate.

    Selecting with the estimator that will do the work is the point: scoring a
    KSG transfer entropy's embedding by a Gaussian AIS picks the embedding with
    the best *linear* predictability, which is not what the final estimator
    measures.
    """
    if estimator == "gaussian":
        return lambda series, dim, delay: _gaussian_ais(series, dim, delay)
    if estimator == "kraskov":
        def score(series, dim, delay):
            # `T - 1 - (dim-1)*delay`, not `T - (dim-1)*delay`. `_ksg_ais`
            # aligns a one-step-ahead future against the embedding, so it
            # spends a sample on the shift as well as on the lookback. The
            # off-by-one let a candidate that the estimator cannot support be
            # scored: at T=5, k_nn=4, dim=1 the guard saw N=5 (usable
            # neighbours 4, exactly k) and passed, while the aligned arrays
            # have N=4 (usable 3) -- and `_ksg_mi_general` had no guard of its
            # own, so it returned a finite 0.0 for a candidate with fewer
            # neighbours than k.
            n_eff = len(series) - 1 - (dim - 1) * delay
            try:
                _validate_ksg_sample(n_eff, k_nn, w)
            except ValueError:
                # Skip candidates the estimator cannot support rather than
                # ranking them and failing on the winner: on kuramoto M7/T100
                # an invalid (k=10, tau=4, w=33) candidate won and the whole
                # SPI then failed, though valid smaller embeddings existed.
                return -np.inf
            return _ksg_ais(series, dim, delay, k_nn, w)
        return score
    raise NotImplementedError(
        f"No active-information-storage criterion for estimator {estimator!r}."
    )


def _select_embedding(series, scorer, dim_max, tau_max):
    """``(dimension, delay, score)`` maximising the scorer.

    Ties go to the smaller dimension and then the smaller delay: a
    lower-dimensional embedding that stores as much information is the better
    model, and the finite-sample AIS objective plateaus often enough that ties
    are not rare.

    Raises when *no* candidate is scorable. Returning ``(1, 1)`` there would
    hand the estimator an embedding that had itself been rejected, and the
    caller would get a number rather than the reason there isn't one -- which
    on a short series is the difference between "transfer entropy at the
    selected embedding" and "there is not enough data to select an embedding".
    A non-finite score counts as unscorable, so a candidate whose covariance is
    degenerate cannot win by default either.
    """
    best = (None, None, -np.inf)
    for dim in range(1, dim_max + 1):
        for delay in range(1, tau_max + 1):
            score = scorer(series, dim, delay)
            if np.isfinite(score) and score > best[2]:
                best = (dim, delay, score)
    if best[0] is None:
        raise ValueError(
            f"No (dimension, delay) in 1..{dim_max} x 1..{tau_max} can be "
            f"scored on a series of {len(series)} observations: every "
            f"candidate leaves too few aligned samples for the estimator. "
            f"Lower the search bounds, or use a fixed embedding."
        )
    return best


def _gaussian_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau):
    """Gaussian TE via log-determinant ratio."""
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
    """KSG conditional mutual information I(A; B | C), Frenzel-Pompe estimator.

    Estimates the CMI *directly* rather than as a sum of four separately
    estimated entropies. That distinction is the whole point: the neighbour
    radius is fixed once in the joint space [A,B,C] and reused in every
    marginal count, so the dimension-dependent biases cancel by construction.
    Composing the same quantity from marginal entropies leaves each one with
    its own bias in its own dimensionality, and those do not cancel.

    ``C`` may have zero columns, in which case this delegates to the MI
    estimator -- conditioning on nothing is mutual information.
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
        # With nothing to condition on this *is* mutual information, so use the
        # MI estimator rather than emulating it. Faking the conditioning count
        # as a constant N-(2w+1) matched _ksg_mi_general only at w=0 and drifted
        # with the Theiler window (0.005 at w=1, 0.051 at w=10 on a probe).
        return _ksg_mi_general(A, B, k_nn, w, condition=False)

    joint = np.concatenate([A, B, C], axis=1)
    AC = np.concatenate([A, C], axis=1)
    BC = np.concatenate([B, C], axis=1)

    tree_joint = cKDTree(joint)
    tree_AC = cKDTree(AC)
    tree_BC = cKDTree(BC)
    tree_C = cKDTree(C)

    if w == 0:
        dists, _ = tree_joint.query(joint, k=k_nn + 1, p=np.inf)
        eps = dists[:, k_nn]
        eps_strict = _strict_radius(eps)

        n_AC = np.array([len(l) - 1 for l in
                         tree_AC.query_ball_point(AC, eps_strict, p=np.inf)], dtype=np.float64)
        n_BC = np.array([len(l) - 1 for l in
                         tree_BC.query_ball_point(BC, eps_strict, p=np.inf)], dtype=np.float64)
        n_C = np.array([len(l) - 1 for l in
                        tree_C.query_ball_point(C, eps_strict, p=np.inf)], dtype=np.float64)
    else:
        n_query = min(k_nn + 2 * w + 2, N)
        dists_all, idx_all = tree_joint.query(joint, k=n_query, p=np.inf)
        n_AC = np.empty(N); n_BC = np.empty(N); n_C = np.empty(N)
        for i in range(N):
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False
            d_valid = dists_all[i][valid]
            e = d_valid[k_nn - 1] if len(d_valid) >= k_nn else np.inf
            e_strict = _strict_radius(e)
            n_AC[i] = sum(1 for j in tree_AC.query_ball_point(AC[i], e_strict, p=np.inf)
                          if abs(j - i) > w and j != i)
            n_BC[i] = sum(1 for j in tree_BC.query_ball_point(BC[i], e_strict, p=np.inf)
                          if abs(j - i) > w and j != i)
            n_C[i] = sum(1 for j in tree_C.query_ball_point(C[i], e_strict, p=np.inf)
                         if abs(j - i) > w and j != i)

    return float(digamma(k_nn) + np.mean(
        digamma(n_C + 1) - digamma(n_AC + 1) - digamma(n_BC + 1)
    ))


def _kraskov_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau, k_nn, w):
    """Kraskov TE via the Frenzel-Pompe CMI estimator: I(Y_f; X_p | Y_p)."""
    Y_f, Y_p, X_p = _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau)
    if Y_f is None:
        return np.nan
    # Same precondition as the MI path, applied to the *embedded* sample count
    # rather than the raw series length: embedding consumes the lookback, so
    # the usable N here is smaller than len(targ).
    _validate_ksg_sample(Y_f.shape[0], k_nn, w, context="transfer entropy")
    return _ksg_cmi(Y_f, X_p, Y_p, k_nn, w)


# ---------------------------------------------------------------------------
# Information-theory base class — with estimator dispatch
# ---------------------------------------------------------------------------

_ESTIMATORS = frozenset({"gaussian", "kraskov", "kernel", "kozachenko", "symbolic"})


def _require_positive_int(name, value):
    """Thin alias for `utils.require_int(..., minimum=1)`; see it for the rules."""
    return utils.require_int(name, value, minimum=1)

# Auto-embedding selection criteria that are actually implemented. The search
# maximises active information storage under the destination's own estimator.
# Auto-embedding selection criteria that are actually implemented, and what
# each one selects. Both maximise the bias-corrected active information storage
# of a series over (dimension, delay); they differ only in which series.
#   MAX_CORR_AIS            -- destination *and* source, chosen independently.
#   MAX_CORR_AIS_DEST_ONLY  -- destination only; the source embedding is the
#                              caller's fixed l_history/l_tau.
# Ragwitz local-prediction selection is a different criterion and is not
# implemented.
_AUTO_EMBED_METHODS = frozenset({"MAX_CORR_AIS", "MAX_CORR_AIS_DEST_ONLY"})
_AUTO_EMBED_DEST_ONLY = "MAX_CORR_AIS_DEST_ONLY"


class InfoTheoryBase(Unsigned):
    """Base for the information-theoretic SPIs.

    **Every measure in this module is reported in nats.** The kernel and
    symbolic calculators used to report bits, inherited from JIDT, which uses
    base 2 for its box-kernel and discrete estimators and base e for its
    Gaussian and k-nearest-neighbour ones. Carrying that split into a single
    results table means `mi_kernel_W-0-5` and `mi_gaussian` are on axes
    differing by a factor of ln 2 with nothing in the identifier to say so, so
    anything comparing *magnitudes* across that boundary -- one threshold over
    several estimators, a difference or ratio of two columns, a "which found
    the most information" ranking across estimators -- is off by that factor.
    (A Pearson or Spearman correlation between two columns is invariant to a
    positive rescaling and was never affected.) Divide by ln 2 to recover the
    JIDT-comparable value.
    """

    _AUTO_EMBED_METHOD_PROP_NAME = "AUTO_EMBED_METHOD"
    _K_HISTORY_PROP_NAME = "k_HISTORY"
    _K_TAU_PROP_NAME = "k_TAU"
    _L_HISTORY_PROP_NAME = "l_HISTORY"
    _L_TAU_PROP_NAME = "l_TAU"
    _K_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_K_SEARCH_MAX"
    _TAU_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_TAU_SEARCH_MAX"

    # Which estimator each optional parameter belongs to. A parameter supplied
    # to an estimator that ignores it is rejected rather than silently dropped:
    # accepting kernel_width under estimator="gaussian" told the caller a
    # kernel width had been applied when nothing used it.
    _PARAM_OWNER = {
        "kernel_width": ("kernel",),
        "prop_k": ("kraskov",),
        "dyn_corr_excl": ("kraskov",),
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
        # Defaults applied after validation so "not supplied" stays
        # distinguishable from "supplied with the default value".
        # Validated, not coerced. `float(True)` is 1.0 and `int(2.7)` is 2, so
        # a permissive cast turns a plainly wrong argument into a plausible one:
        # a non-positive box-kernel half-width counts only the point itself, so
        # every log ratio is log(N) and the "estimate" is a constant, and k < 1
        # has no kth neighbour at all.
        self._kernel_width = (0.5 if kernel_width is None
                              else utils.require_positive_float("kernel_width",
                                                                kernel_width))
        self._prop_k = (4 if prop_k is None
                        else utils.require_int("prop_k", prop_k, minimum=1))
        if dyn_corr_excl is not None and not (
                isinstance(dyn_corr_excl, str) and dyn_corr_excl == "AUTO"):
            # `None` and the exact string "AUTO" are the only non-integer
            # values; everything else is a Theiler window in samples.
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
            # Only the measures with a genuine KSG implementation may accept it.
            # The composed measures (joint/conditional/crossmap/causal entropy,
            # directed info, stochastic interaction) are built from marginal
            # entropies, and _getcalc hands them GaussianEntropyCalculator for
            # "kraskov" -- so they returned exactly the Gaussian result while
            # advertising kraskov_NN-<k> in the identifier. Reporting a k-NN
            # estimate that was never computed is worse than refusing.
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
            self.identifier = self.identifier + "_W-{}".format(self._kernel_width)
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
            # Kozachenko-Leonenko estimates *entropy* from k-NN distances. The
            # measures below are computed directly rather than as a sum of
            # marginal entropies, and there is no KL path for them: composing
            # them from separate KL entropies is biased, since the per-space
            # biases do not cancel (avoiding exactly that is why KSG couples
            # its radii across spaces -- use estimator="kraskov" instead).
            # Fail here rather than returning NaN at compute time.
            if isinstance(self, (MutualInfo, TimeLaggedMutualInfo, TransferEntropy)):
                raise NotImplementedError(
                    f"The kozachenko estimator is not available for "
                    f"{type(self).__name__}; use estimator='kraskov' for a "
                    f"k-nearest-neighbour estimate of this measure."
                )
            # k-NN based, so nonlinear -- not "linear" as gaussian is.
            self.labels = self.labels + ["nonlinear"]
            self._dyn_corr_excl = None
        else:
            self.labels = self.labels + ["linear"]
            self._dyn_corr_excl = None

        if self._dyn_corr_excl:
            # The *value*, not just the flag. `_DCE` alone gave
            # dyn_corr_excl=5, =10 and ="AUTO" one identifier between them --
            # three different Theiler windows, three different numbers, one
            # name -- so a config setting two of them collided silently.
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
            return (self._estimator, self._kernel_width)
        elif self._estimator == "kraskov":
            # dyn_corr_excl is in the key even though the entropy caches this
            # keys are not consumed by the KSG paths today: the rule is that
            # everything reaching the identifier reaches the cache key, and an
            # exception maintained by argument is an exception that stops being
            # true.
            return (self._estimator, self._prop_k, self._dyn_corr_excl)
        else:
            return (self._estimator,)

    def _getcalc(self, measure):
        est = self._estimator

        # --- Pure-numpy calculators (no JIDT/JVM needed) ---

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
                # kozachenko, kernel — all have numpy calculators
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
            x, y = data.to_numpy()[[i, j]]
            joint = np.concatenate([x, y], axis=1)
            est = self._estimator

            if est in ('gaussian', 'kraskov'):
                val = _gaussian_entropy_from_data(joint)
            else:
                # kozachenko, kernel — all have numpy calculators
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
            # kozachenko, kernel — all have numpy calculators
            XY = np.concatenate([X, Y], axis=1)
            self._entropy_calc.initialise(XY.shape[1])
            self._entropy_calc.setObservations(XY)
            H_XY = self._entropy_calc.computeAverageLocalOfObservations()
            self._entropy_calc.initialise(Y.shape[1])
            self._entropy_calc.setObservations(Y)
            H_Y = self._entropy_calc.computeAverageLocalOfObservations()
            return H_XY - H_Y

    def _resolve_theiler(self, data, i, j):
        """Theiler/dynamic-correlation-exclusion window for pair (i, j).

        None -> 0 (no window); an integer -> that window; "AUTO" -> the
        autocorrelation time 2*<acf(x_i), acf(x_j)>, cached per dataset.
        Shared by MI/TLMI/TE; only the kNN (kraskov) paths consume it.
        """
        raw_w = getattr(self, '_dyn_corr_excl', None)
        if raw_w is None:
            return 0
        if raw_w == "AUTO":
            if not hasattr(data, 'theiler'):
                z = data.to_numpy()
                M = data.n_processes
                theiler = -np.ones((M, M))
                for _i in range(M):
                    for _j in range(_i + 1, M):
                        theiler[_i, _j] = 2 * np.dot(
                            utils.acf(z[_i]), utils.acf(z[_j])
                        )
                        theiler[_j, _i] = theiler[_i, _j]
                data.theiler = theiler
            return int(data.theiler[i, j])
        return int(raw_w)


class JointEntropy(InfoTheoryBase, Undirected):

    name = "Joint entropy"
    identifier = "je"
    labels = ["unsigned", "infotheory", "unordered", "undirected"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j)

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            JE = _gaussian_pairwise_joint_entropy(data.to_numpy(squeeze=True))
            np.fill_diagonal(JE, np.nan)
            return JE
        return super().multivariate(data)


class ConditionalEntropy(InfoTheoryBase, Directed):

    name = "Conditional entropy"
    identifier = "ce"
    # Directed: H(X|Y) != H(Y|X). Measured on var1_M3_T100 under the default
    # z-scoring, max|A - A.T| is 0.115 for kozachenko and 0.039 for kernel; the
    # Gaussian form is symmetric there only because equal marginal variances
    # make it so, and it becomes asymmetric (0.73) with zscore=False. A
    # structural label describes the measure, not one estimator under one
    # preprocessing default.
    labels = ["unsigned", "infotheory", "unordered", "directed"]

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
            variances = np.var(Z, axis=1, ddof=1)
            # Marginal entropy uses the same ridge as the scalar path.
            H_marginal = 0.5 * np.log(2 * np.pi * np.e * (variances + 1e-8 * variances))
            JE = _gaussian_pairwise_joint_entropy(Z)
            CE = JE - H_marginal[:, None]
            np.fill_diagonal(CE, np.nan)
            return CE
        return super().multivariate(data)


class MutualInfo(InfoTheoryBase, Undirected):
    name = "Mutual information"
    identifier = "mi"
    labels = ["unsigned", "infotheory", "unordered", "undirected"]

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
            # Handled by multivariate; bivariate fallback
            z = data.to_numpy(squeeze=True)
            if self._estimator == 'gaussian':
                return float(_gaussian_mi_from_r(np.corrcoef(z[i], z[j])[0, 1]))
            else:
                # kraskov bivariate
                k = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _ksg_mi_pair(z[i], z[j], k, w)

        # kernel estimator: use numpy KernelMICalculator
        if self._estimator == 'kernel':
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            self._calc.initialise(1, 1)
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        # Fallback should not be reached (all estimators handled above)
        logging.warning(f"MI bivariate: unhandled estimator '{self._estimator}'")
        return np.nan

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            MI = _gaussian_mi_from_r(np.corrcoef(Z))
            np.fill_diagonal(MI, np.nan)
            return MI
        elif self._estimator == 'kraskov':
            Z = data.to_numpy(squeeze=True)
            M, N = Z.shape
            k = int(self._prop_k)
            result = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    w = self._resolve_theiler(data, i, j)
                    mi = _ksg_mi_pair(Z[i], Z[j], k, w)
                    result[i, j] = result[j, i] = mi
            return result
        return super().multivariate(data)


class TimeLaggedMutualInfo(InfoTheoryBase, Directed):
    name = "Time-lagged mutual information"
    identifier = "tlmi"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

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
                return float(_gaussian_mi_from_r(np.corrcoef(src, tgt)[0, 1]))
            else:
                k = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _ksg_mi_pair(src, tgt, k, w)

        # kernel estimator: use numpy KernelMICalculator
        if self._estimator == 'kernel':
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            src = src[:-1]
            targ = targ[1:]
            self._calc.initialise(1, 1)
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        logging.warning(f"TLMI bivariate: unhandled estimator '{self._estimator}'")
        return np.nan

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            M, T = Z.shape
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
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    w = self._resolve_theiler(data, i, j)
                    mi = _ksg_mi_pair(Z_src[i], Z_tgt[j], k, w)
                    result[i, j] = mi
            return result
        return super().multivariate(data)


class TransferEntropy(InfoTheoryBase, Directed):

    name = "Transfer entropy"
    identifier = "te"
    labels = ["unsigned", "embedding", "infotheory", "temporal", "directed"]

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
            # The value was previously never inspected: any non-None string
            # took the auto-embed branch, so a typo silently ran MAX_CORR_AIS.
            raise ValueError(
                f"Unknown auto_embed_method {auto_embed_method!r}; implemented: "
                f"{sorted(_AUTO_EMBED_METHODS)}. Ragwitz local-prediction "
                f"selection is a different criterion and is not implemented."
            )

        # Only the two estimators with a genuine embedding implement any of
        # this. The symbolic and kernel calculators read one history length at
        # unit delay and nothing else, so every embedding and search argument
        # is refused rather than accepted and dropped.
        _EMBEDDING_ONLY = ("gaussian", "kraskov")
        supplied = {"k_tau": k_tau, "l_history": l_history, "l_tau": l_tau}
        if self._estimator not in _EMBEDDING_ONLY:
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
            # Search bounds with nothing to search are not a harmless default:
            # they reached the identifier under the auto branch only, so here
            # they were accepted and silently discarded.
            for name, value in (("k_search_max", k_search_max),
                                ("tau_search_max", tau_search_max)):
                if value is not None:
                    raise ValueError(
                        f"{name}={value!r} requires auto_embed_method to be set; "
                        f"with a fixed embedding there is nothing to search."
                    )
        else:
            # A selected embedding and a supplied one cannot both be honoured.
            # Silently ignoring the supplied value is how the identifier came to
            # advertise parameters the estimator never used.
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
            # An ordinal pattern of length 1 has exactly one possible symbol, so
            # every entropy term is zero and TE is identically zero. It is not a
            # degenerate edge case, it is a guaranteed-null statistic.
            raise ValueError(
                "estimator='symbolic' requires k_history >= 2: a length-1 "
                "ordinal pattern has a single symbol, so the transfer entropy "
                "is identically zero."
            )

        self._calc = self._getcalc("TransferEntropy")

        # Store embedding params for numpy path
        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max
        self._tau_search_max = tau_search_max
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

        if auto_embed_method is not None:
            self._calc.setProperty(self._AUTO_EMBED_METHOD_PROP_NAME, auto_embed_method)
            self._calc.setProperty(self._K_SEARCH_MAX_PROP_NAME, str(k_search_max))
            self._calc.setProperty(self._TAU_SEARCH_MAX_PROP_NAME, str(tau_search_max))
            # The method is in the identifier because it changes what is
            # computed: MAX_CORR_AIS selects the source embedding too, and used
            # to name a search that only ever touched the destination.
            self.identifier += "_{}_k-max-{}_tau-max-{}".format(
                auto_embed_method.replace("_", "-"), k_search_max, tau_search_max)
            if auto_embed_method == _AUTO_EMBED_DEST_ONLY:
                self.identifier += "_l-{}_lt-{}".format(l_history, l_tau)
        else:
            self._calc.setProperty(self._K_HISTORY_PROP_NAME, str(k_history))
            if self._estimator in _EMBEDDING_ONLY:
                self._calc.setProperty(self._K_TAU_PROP_NAME, str(k_tau))
                self._calc.setProperty(self._L_HISTORY_PROP_NAME, str(l_history))
                self._calc.setProperty(self._L_TAU_PROP_NAME, str(l_tau))
                self.identifier = self.identifier + "_k-{}_kt-{}_l-{}_lt-{}".format(
                    k_history, k_tau, l_history, l_tau
                )
            else:
                # One history length, unit delay: the identifier says only what
                # is computed.
                self.identifier = self.identifier + "_k-{}".format(k_history)

    def __setstate__(self, state):
        super().__setstate__(state)
        self._calc = self._getcalc("TransferEntropy")

    # Overridable so a test can inject a scorer that deliberately picks
    # different source and destination embeddings and check that all four
    # selected values reach the final estimator.
    _embedding_scorer = staticmethod(_ais_scorer)

    def _selected_embedding(self, data, process, series, scorer, w):
        """``(dimension, delay)`` for one series, cached on the Data object.

        Selection depends on the series, the estimator, the search bounds and
        the Theiler window -- not on which pair the series appears in, nor on
        whether it is the source or the destination there: the criterion is the
        series' own active information storage either way. With
        ``dyn_corr_excl="AUTO"`` the window *is* pair-dependent, so it is part
        of the key; otherwise one search per process serves every pair in both
        roles, instead of 2*(M-1) identical searches.
        """
        key = (process, self._estimator, self._prop_k, w,
               self._k_search_max, self._tau_search_max)
        cache = getattr(data, "ais_embedding", None)
        if cache is None:
            cache = data.ais_embedding = {}
        if key not in cache:
            dim, delay, _ = _select_embedding(
                series, scorer, self._k_search_max, self._tau_search_max)
            cache[key] = (dim, delay)
        return cache[key]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        est = self._estimator
        auto = self._auto_embed_method

        src, targ = data.to_numpy(squeeze=True)[[i, j]]

        # Pure-numpy path for gaussian/kraskov with fixed embedding
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
                # Selected from the *source*, independently and by the same
                # criterion. The previous implementation hard-coded (1, 1) here
                # while the method name claimed selection for both.
                l_history, l_tau = self._selected_embedding(data, i, src, scorer, w)

            if est == 'gaussian':
                return _gaussian_te_bivariate(src, targ, k_history, k_tau,
                                              l_history, l_tau)
            return _kraskov_te_bivariate(src, targ, k_history, k_tau,
                                         l_history, l_tau, k_nn, w)

        # kernel/symbolic path: numpy calculators
        if est in ('kernel', 'symbolic'):
            self._calc.initialise()
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        logging.warning(f"TE bivariate: unhandled estimator '{est}'")
        return np.nan


class CrossmapEntropy(InfoTheoryBase, Directed):
    """H(Y_t | X_{t-1}, ..., X_{t-k+1}) -- a *k*-dimensional joint embedding.

    ``history_length=k`` gives ``k - 1`` source lags, not ``k``: the loop runs
    ``range(2, k)``, so the joint space [source past, target future] has
    exactly ``k`` columns. Both readings of the parameter are internally
    consistent -- "k source lags" would need ``range(2, k + 1)`` -- and the
    implementation has always been the second one.

    It is documented here rather than changed. Cross-map entropy has no
    canonical published definition to arbitrate between the two conventions,
    and silently re-picking one would change every ``xme_*`` value on a guess
    about intent. The parameter name is the misleading part; the arithmetic is
    self-consistent, and `test_crossmap_entropy_embedding_dimension` pins it so
    a future change has to be deliberate.
    """

    name = "Cross-map entropy"
    identifier = "xme"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, history_length=10, **kwargs):
        super().__init__(**kwargs)
        self.identifier += f"_k{history_length}"
        self._history_length = history_length

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        src, targ = data.to_numpy(squeeze=True)[[i, j]]
        k = self._history_length
        targ_future = targ[k:]
        src_past = np.expand_dims(src[k - 1 : -1], axis=1)
        for idx in range(2, k):
            src_past = np.append(
                src_past, np.expand_dims(src[k - idx : -idx], axis=1), axis=1
            )

        joint = np.concatenate([src_past, np.expand_dims(targ_future, axis=1)], axis=1)

        # All estimators now have numpy entropy calculators
        self._entropy_calc.initialise(joint.shape[1])
        self._entropy_calc.setObservations(joint)
        H_xy = self._entropy_calc.computeAverageLocalOfObservations()

        self._entropy_calc.initialise(src_past.shape[1])
        self._entropy_calc.setObservations(src_past)
        H_y = self._entropy_calc.computeAverageLocalOfObservations()

        return H_xy - H_y


class CausalEntropy(InfoTheoryBase, Directed):

    name = "Causally conditioned entropy"
    identifier = "cce"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, n=5, **kwargs):
        super().__init__(**kwargs)
        self._n = utils.require_int("n", n, minimum=1)
        # n changes the measure, so it must reach the identifier.
        self.identifier += f"_n-{self._n}"

    def _compute_causal_entropy(self, src, targ):
        src = np.squeeze(src)
        targ = np.squeeze(targ)

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

    name = "Directed information"
    identifier = "di"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, n=5, **kwargs):
        # n is handled by CausalEntropy; re-appending here doubled the suffix.
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
        r"""Directed information from process ``i`` to process ``j``.

        Massey's finite-horizon definition:

        .. math::
            I(X^n \to Y^n) = \sum_{i=1}^{n} I(X^i; Y_i \mid Y^{i-1})
                           = \sum_{i=1}^{n} \left[ H(Y_i \mid Y^{i-1})
                             - H(Y_i \mid Y^{i-1}, X^i) \right]

        The previous implementation summed :math:`H(Y^i)/i` and subtracted the
        causal entropy. That is not the above and is not a dependence measure:
        with a source statistically independent of the target it returned 0.007
        at target autocorrelation 0, rising to 1.53 at 0.95 -- it grew with how
        predictable the *target* was from its own past, with no source coupling
        present at all.

        Each conditional entropy is expanded as a difference of joint entropies
        over one common row window, so the terms telescope correctly and every
        entropy is estimated on identically aligned samples.
        """
        z = data.to_numpy(squeeze=True)
        src, targ = np.asarray(z[i], float), np.asarray(z[j], float)
        n = self._n
        T = targ.size
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
                # Estimate I(X^i; Y_i | Y^{i-1}) directly. The KSG/Frenzel-Pompe
                # estimator fixes one neighbour radius in the joint space and
                # reuses it in every marginal count, so the dimension-dependent
                # biases cancel. Composing the same term from four separate
                # entropies does not: each is biased in its own dimensionality,
                # which is why the kernel and kozachenko compositions are unusable
                # here (kernel sat near +4 on independent data at every T).
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

    name = "Stochastic interaction"
    identifier = "si"
    labels = ["unsigned", "infotheory", "temporal", "undirected"]

    def __init__(self, delay=1, **kwargs):
        super().__init__(**kwargs)
        self._delay = delay
        self.identifier += f"_k-{delay}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        x, y = data.to_numpy()[[i, j]]
        xy = np.concatenate([x, y], axis=1)
        tau = self._delay

        H_joint = self._compute_conditional_entropy(xy[tau:], xy[:-tau])
        H_src = self._compute_conditional_entropy(x[tau:], x[:-tau])
        H_targ = self._compute_conditional_entropy(y[tau:], y[:-tau])

        return H_src + H_targ - H_joint


class IntegratedInformation(Undirected, Unsigned):

    name = "Integrated information"
    identifier = "phi"
    labels = ["linear", "unsigned", "infotheory", "temporal", "undirected"]

    def __init__(self, phitype="star", delay=1, normalization=0):
        self._phitype = phitype
        self._delay = delay
        self._normalization = normalization
        self.identifier += f"_{phitype}_t-{delay}_norm-{normalization}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        try:
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

        except Exception as e:
            logging.error(f"Integrated information computation failed: {e}")
            return np.nan

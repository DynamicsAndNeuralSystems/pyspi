import os
import warnings
import numpy as np
from sklearn.metrics import pairwise_distances
import tslearn.metrics
from tslearn.barycenters import (
    euclidean_barycenter,
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    softdtw_barycenter,
)
from hyppo.independence import (
    MGC,
    Dcorr,
    HHG,
    Hsic,
)
from hyppo.time_series import MGCX, DcorrX

from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    Signed,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import fmt_param, require_int, require_positive_float


# ---------------------------------------------------------------------------
# DTW Sakoe-Chiba auto-radius configuration (env-var tunable)
# ---------------------------------------------------------------------------
_DTW_SAKOE_LINEAR_FRAC = float(os.getenv("PYSPI_DTW_SAKOE_LINEAR_FRAC", "0.10"))
_DTW_SAKOE_SQRT_COEFF = float(os.getenv("PYSPI_DTW_SAKOE_SQRT_COEFF", "1.5"))
_DTW_SAKOE_MIN_RADIUS = int(os.getenv("PYSPI_DTW_SAKOE_MIN_RADIUS", "10"))


def _auto_sakoe_radius(length):
    if length <= 1:
        return 1
    linear = int(np.ceil(max(0.0, _DTW_SAKOE_LINEAR_FRAC) * length))
    sqrt_scaled = int(np.ceil(max(0.0, _DTW_SAKOE_SQRT_COEFF) * np.sqrt(length)))
    radius = min(linear, sqrt_scaled)
    radius = max(1, _DTW_SAKOE_MIN_RADIUS, radius)
    return min(radius, length - 1)


class PairwiseDistance(Undirected, Unsigned):
    """``metric`` distance between each pair of processes, optionally / sqrt(T).

    ``normalise=True`` divides by ``sqrt(T)`` and adds ``_rmse`` to the
    identifier. For ``metric="euclidean"`` that is exactly the root mean square
    difference, ``sqrt(mean((x - y)**2))``, and the name is literal. For the
    other metrics -- cityblock, cosine, chebyshev, canberra, braycurtis -- the
    suffix is pyspi's house label for "divided by sqrt(T)" and nothing more.
    Those quantities are not root *mean* squares, and dividing them by sqrt(T)
    does **not** make them comparable across record lengths either -- only the
    Euclidean norm grows as sqrt(T) under i.i.d. differences, so only for
    `euclidean` is the division the right power. Read `_rmse` as the name of the
    normalisation that was applied, not as a claim about the metric.
    """

    name = "Pairwise distance"
    identifier = "pdist"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, metric="euclidean", normalise=False, **kwargs):
        self._metric = metric
        self._normalise = normalise
        self.identifier += f"_{metric}"
        if normalise:
            self.identifier += "_rmse"

    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)
        D = pairwise_distances(Z, metric=self._metric)
        if self._normalise:
            T = Z.shape[1]
            D = D / np.sqrt(T)
        return D


""" TODO: include optional kernels in each method
"""


class HilbertSchmidtIndependenceCriterion(Undirected, Unsigned):
    """Hilbert-Schmidt Independence Criterion (HSIC)"""

    name = "Hilbert-Schmidt Independence Criterion"
    identifier = "hsic"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = Hsic(bias=self._biased).statistic(x, y)
        return stat


class HellerHellerGorfine(Undirected, Unsigned):
    """Heller-Heller-Gorfine independence criterion"""

    name = "Heller-Heller-Gorfine Independence Criterion"
    identifier = "hhg"
    # Symmetric by construction: hhg(x, y) == hhg(y, x) exactly.
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = HHG().statistic(x, y)
        return stat


class DistanceCorrelation(Undirected, Unsigned):
    """Distance correlation"""

    name = "Distance correlation"
    identifier = "dcorr"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """ """
        x, y = data.to_numpy()[[i, j]]
        stat = Dcorr(bias=self._biased).statistic(x, y)
        return stat


class MultiscaleGraphCorrelation(Undirected, Unsigned):
    """Multiscale graph correlation"""

    name = "Multiscale graph correlation"
    identifier = "mgc"
    labels = ["distance", "unsigned", "unordered", "nonlinear", "undirected"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = MGC().statistic(x, y)
        return stat


class CrossDistanceCorrelation(Directed, Unsigned):
    """Cross-distance correlation"""

    name = "Cross-distance correlation"
    identifier = "dcorrx"
    labels = ["distance", "unsigned", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{max_lag}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        stat, _ = DcorrX(max_lag=self._max_lag).statistic(x, y)
        return stat


class CrossMultiscaleGraphCorrelation(Directed, Unsigned):
    """Cross-multiscale graph correlation"""

    name = "Cross-multiscale graph correlation"
    identifier = "mgcx"
    labels = ["unsigned", "distance", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{max_lag}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        stat, _, _ = MGCX(max_lag=self._max_lag).statistic(x, y)
        return stat


class TimeWarping(Undirected, Unsigned):

    labels = ["unsigned", "distance", "temporal", "undirected", "nonlinear"]

    def __init__(self, global_constraint=None):
        gcstr = global_constraint
        if gcstr is not None:
            gcstr = gcstr.replace("_", "-")
            self.identifier += f"_constraint-{gcstr}"
        self._global_constraint = global_constraint

    @property
    def simfn(self):
        try:
            return self._simfn
        except AttributeError:
            raise NotImplementedError(
                f"Add the similarity function for {self.identifier}"
            )

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        return self._simfn(z[i], z[j], global_constraint=self._global_constraint)


class DynamicTimeWarping(TimeWarping):
    """DTW with optional dtaidistance C backend and Sakoe-Chiba band constraint.

    Falls back to tslearn if dtaidistance is not available or for itakura constraint.
    """

    name = "Dynamic time warping"
    identifier = "dtw"

    def __init__(
        self,
        global_constraint=None,
        sakoe_chiba_radius=None,
        sakoe_chiba_ratio=None,
        normalise=False,
        **kwargs,
    ):
        if sakoe_chiba_radius is not None and sakoe_chiba_ratio is not None:
            raise ValueError("Set only one of sakoe_chiba_radius or sakoe_chiba_ratio.")
        # Validated, not coerced: `int(2.7)` silently becomes a band of 2 and
        # `float(True)` a ratio of 1.0, both of which look like the caller's
        # intent and are not.
        if sakoe_chiba_radius is not None:
            sakoe_chiba_radius = require_int("sakoe_chiba_radius",
                                             sakoe_chiba_radius, minimum=1)
        if sakoe_chiba_ratio is not None:
            sakoe_chiba_ratio = require_positive_float("sakoe_chiba_ratio",
                                                       sakoe_chiba_ratio)

        super().__init__(global_constraint=global_constraint, **kwargs)
        self._simfn = tslearn.metrics.dtw
        self._sakoe_chiba_radius = sakoe_chiba_radius
        self._sakoe_chiba_ratio = sakoe_chiba_ratio
        self._normalise = normalise
        self._warned_itakura_fallback = False
        self._warned_c_fallback = False

        if global_constraint == "sakoe_chiba":
            if sakoe_chiba_radius is not None:
                self.identifier += f"_radius-{sakoe_chiba_radius}"
            elif sakoe_chiba_ratio is not None:
                self.identifier += f"_ratio-{fmt_param(sakoe_chiba_ratio)}"
            else:
                self.identifier += "_radius-auto"
        if normalise:
            self.identifier += "_rmse"

    def _resolve_radius(self, n):
        if self._sakoe_chiba_radius is not None:
            return min(self._sakoe_chiba_radius, max(1, n - 1))
        if self._sakoe_chiba_ratio is not None:
            ratio_radius = int(np.ceil(self._sakoe_chiba_ratio * n))
            return min(max(1, ratio_radius), max(1, n - 1))
        return _auto_sakoe_radius(n)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        x = np.ascontiguousarray(z[i], dtype=np.double)
        y = np.ascontiguousarray(z[j], dtype=np.double)
        constraint = self._global_constraint
        n = min(len(x), len(y))

        if constraint == "itakura":
            if not self._warned_itakura_fallback:
                warnings.warn(
                    "DynamicTimeWarping(itakura) falls back to tslearn; "
                    "dtaidistance C backend does not support itakura.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_itakura_fallback = True
            d = tslearn.metrics.dtw(x, y, global_constraint="itakura")
            return d / np.sqrt(n) if self._normalise else d

        radius = None
        window = None
        if constraint == "sakoe_chiba":
            radius = self._resolve_radius(n)
            window = radius + 1

        try:
            from dtaidistance import dtw as _dtw_c
            from dtaidistance.exceptions import CythonException
            kwargs = {"use_c": True}
            if window is not None:
                kwargs["window"] = window
            d = _dtw_c.distance(x, y, **kwargs)
        except ImportError:
            # dtaidistance not installed — use tslearn
            if constraint == "sakoe_chiba":
                d = tslearn.metrics.dtw(
                    x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius,
                )
            else:
                d = tslearn.metrics.dtw(x, y, global_constraint=constraint)
        except (CythonException, ValueError):
            if not self._warned_c_fallback:
                warnings.warn(
                    "dtaidistance C backend unavailable for DynamicTimeWarping; "
                    "falling back to tslearn.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_c_fallback = True
            if constraint == "sakoe_chiba":
                d = tslearn.metrics.dtw(
                    x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius,
                )
            else:
                d = tslearn.metrics.dtw(x, y)
        return d / np.sqrt(n) if self._normalise else d

    @parse_multivariate
    def multivariate(self, data):
        """Batch DTW via dtaidistance.dtw.distance_matrix when available."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        series = [np.ascontiguousarray(Z[i], dtype=np.double) for i in range(M)]

        constraint = self._global_constraint

        if constraint == "itakura":
            # dtaidistance doesn't support itakura; fall back to bivariate loop
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = tslearn.metrics.dtw(
                        series[i], series[j], global_constraint="itakura"
                    )
                    A[i, j] = d
                    A[j, i] = d
            # The bivariate path and the dtaidistance path both divide by
            # sqrt(T) under `normalise`; this branch did not, so
            # `bivariate(data, i, j)` and `multivariate(data)[i, j]` differed by
            # a factor of sqrt(T) for the itakura constraint alone.
            if self._normalise:
                A = A / np.sqrt(Z.shape[1])
            return A

        try:
            from dtaidistance import dtw as _dtw_c

            kwargs = {"use_c": True, "compact": False}
            if constraint == "sakoe_chiba":
                n = min(len(s) for s in series)
                radius = self._resolve_radius(n)
                kwargs["window"] = radius + 1

            try:
                dm = _dtw_c.distance_matrix(series, **kwargs)
            except Exception:
                kwargs["use_c"] = False
                dm = _dtw_c.distance_matrix(series, **kwargs)

            dm = np.array(dm)
            mask_upper = np.triu(np.ones((M, M), dtype=bool), k=1)
            dm_sym = np.where(mask_upper, dm, dm.T)
            np.fill_diagonal(dm_sym, np.nan)
            if self._normalise:
                T = Z.shape[1]
                dm_sym = dm_sym / np.sqrt(T)
            return dm_sym

        except ImportError:
            # Fall back to bivariate loop with tslearn
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = self.bivariate(data, i=i, j=j)
                    A[i, j] = d
                    A[j, i] = d
            return A


class LongestCommonSubsequence(TimeWarping):

    name = "Longest common subsequence"
    identifier = "lcss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.lcss


class SoftDynamicTimeWarping(TimeWarping):

    name = "Dynamic time warping"
    identifier = "softdtw"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.soft_dtw(z[i], z[j])


class Barycenter(Directed, Signed):

    name = "Barycenter"
    identifier = "bary"
    labels = ["distance", "signed", "undirected", "temporal", "nonlinear"]
    _cache_namespace = "barycenter"

    @property
    def _cache_subkey(self):
        # Actual cache is keyed (mode, pair); statistic/squared are post-lookup
        # transforms. Sub-bucket amortization by mode so 4 separate caches don't
        # get lumped into one and mis-cost the cheap modes.
        return (self._mode,)

    def __init__(self, mode="euclidean", squared=False, statistic="mean"):
        if mode == "euclidean":
            self._fn = euclidean_barycenter
        elif mode == "dtw":
            self._fn = dtw_barycenter_averaging
        elif mode == "sgddtw":
            self._fn = dtw_barycenter_averaging_subgradient
        elif mode == "softdtw":
            self._fn = softdtw_barycenter
        else:
            raise NameError(f"Unknown Barycenter mode: {mode}")
        self._mode = mode

        self._squared = squared
        self._preproc = lambda x: x
        if squared:
            self._preproc = lambda x: x**2
            self.identifier += f"-sq"

        if statistic == "mean":
            self._statfn = lambda x: np.nanmean(self._preproc(x))
        elif statistic == "max":
            self._statfn = lambda x: np.nanmax(self._preproc(x))
        elif statistic == "max_time":
            self._statfn = lambda x: np.argmax(self._preproc(x))
        else:
            raise NameError(f"Unknown statistic: {statistic}")

        self.identifier += f"_{mode}_{statistic}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):

        try:
            bc = data.barycenter[self._mode][(i, j)]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True)
            bc = self._fn(z[[i, j]])
            try:
                data.barycenter[self._mode][(i, j)] = bc
            except AttributeError:
                data.barycenter = {self._mode: {(i, j): bc}}
            except KeyError:
                data.barycenter[self._mode] = {(i, j): bc}
            data.barycenter[self._mode][(j, i)] = data.barycenter[self._mode][(i, j)]

        return self._statfn(bc)


class GromovWasserstainTau(Undirected, Unsigned):
    """Gromov-Wasserstain distance (GWTau)"""

    name = "Gromov-Wasserstain Distance"
    identifier = "gwtau"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    @staticmethod
    def vec_geo_dist(x):
        diffs = np.diff(x, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.cumsum(distances)

    @staticmethod
    def wass_sorted(x1, x2):
        x1 = np.sort(x1)[::-1] # sort in descending order
        x2 = np.sort(x2)[::-1]

        if len(x1) == len(x2):
            res = np.sqrt(np.mean((x1 - x2) ** 2))
        else:
            N, M = len(x1), len(x2)
            i_ratios = np.arange(1, N + 1) / N
            j_ratios = np.arange(1, M + 1) / M


            min_values = np.minimum.outer(i_ratios, j_ratios)
            max_values = np.maximum.outer(i_ratios - 1/N, j_ratios - 1/M)

            lam = np.where(min_values > max_values, min_values - max_values, 0)

            diffs_squared = (x1[:, None] - x2) ** 2
            my_sum = np.sum(lam * diffs_squared)

            res = np.sqrt(my_sum)

        return res

    @staticmethod
    def gwtau(xi, xj):
        timei = np.arange(len(xi))
        timej = np.arange(len(xj))
        traji = np.column_stack([timei, xi])
        trajj = np.column_stack([timej, xj])

        vi = GromovWasserstainTau.vec_geo_dist(traji)
        vj = GromovWasserstainTau.vec_geo_dist(trajj)
        gw = GromovWasserstainTau.wass_sorted(vi, vj)

        return gw

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = self.gwtau(x, y)
        return stat


# ---------------------------------------------------------------------------
# CrossPairwiseDistance: lagged Euclidean distance
# ---------------------------------------------------------------------------

class CrossPairwiseDistance(Undirected, Unsigned):
    """Root-T-normalised cost of a lag-shifted alignment path, over lags 0..tau.

    At each lag ``t`` the two series are paired with an offset of ``t``, the
    ``t`` samples that fall off each end are stuttered against the opposite
    series' boundary sample, and the cost is
    ``sqrt(sum(differences**2) / T)``. The forward and backward offsets are
    minimised over, so the result is symmetric in the pair; ``min`` or ``mean``
    then reduces over ``t``.

    The ``_rmse`` in the identifier is the same house label
    `PairwiseDistance` uses: divided by ``sqrt(T)``. It is **not** the
    conventional DTW "path-length RMSE", which divides by the number of steps
    in the path -- here the divisor is the record length, and the boundary
    stutter means the path has ``T + t`` steps rather than ``T``. The point of
    the stutter is that the alignment is a *valid* DTW path (monotone,
    continuous, matched endpoints), and DTW minimises over all such paths, so
    ``dtw_rmse <= xpdist`` holds by construction rather than empirically.

    At ``tau=0`` the path is the identity alignment and the value is exactly
    ``pdist_euclidean`` with ``normalise=True``.
    """
    name = "Cross pairwise distance"
    labels = ["distance", "nonlinear", "undirected", "temporal"]

    def __init__(self, metric="euclidean", tau=1, statistic="min"):
        if metric != "euclidean":
            raise ValueError(f"Unsupported metric: {metric!r}. Only 'euclidean' supported.")
        # `int(tau) < 0` accepted `tau=1.7` (truncated to 1), `tau=True`
        # (silently 1) and raised an opaque conversion error on nan/inf. tau
        # counts samples, so it must be an integer.
        tau = require_int("tau", tau, minimum=0)
        stat = str(statistic).lower()
        if stat not in {"min", "mean"}:
            raise ValueError(f"statistic must be 'min' or 'mean', got: {statistic!r}")
        self._metric = metric
        self._tau = tau
        self._statistic = stat
        # `_rmse` is the house label for the sqrt(T) normalisation, shared with
        # PairwiseDistance and DynamicTimeWarping(normalise=True). See the class
        # docstring for what it does and does not claim.
        self.identifier = f"xpdist_{metric}_tau-{self._tau}_{stat}_rmse"

    @staticmethod
    def _shifted_dist_rmse(x, y, s):
        # Realises one specific DTW path: pair x[k+s] with y[k] over the overlap,
        # then stutter the dropped boundary samples against the corner of the
        # opposite series. Since the result is the cost of a valid DTW path and
        # DTW minimises over all paths, the resulting xpdist satisfies
        # dtw_rmse <= xpdist by construction. Normalise by sqrt(T) to match
        # DynamicTimeWarping(normalise=True).
        T = len(x)
        if s == 0:
            diff = x - y
        elif s > 0:
            left  = x[:s]      - y[0]
            mid   = x[s:]      - y[: T - s]
            right = x[T - 1]   - y[T - s :]
            diff = np.concatenate([left, mid, right])
        else:
            sa = -s
            left  = y[:sa]     - x[0]
            mid   = y[sa:]     - x[: T - sa]
            right = y[T - 1]   - x[T - sa :]
            diff = np.concatenate([left, mid, right])
        return float(np.sqrt(np.sum(diff ** 2) / T))

    def _cross_dist(self, x, y):
        tau = self._tau
        T = len(x)
        per_lag = np.empty(tau + 1)
        per_lag[0] = self._shifted_dist_rmse(x, y, 0)
        for t in range(1, tau + 1):
            if t >= T:
                per_lag[t] = np.inf
                continue
            fwd = self._shifted_dist_rmse(x, y, +t)
            bwd = self._shifted_dist_rmse(x, y, -t)
            per_lag[t] = min(fwd, bwd)
        if self._statistic == "min":
            return float(np.min(per_lag))
        finite = per_lag[np.isfinite(per_lag)]
        return float(np.mean(finite)) if finite.size > 0 else np.nan

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        Z = data.to_numpy(squeeze=True)
        return self._cross_dist(Z[i], Z[j])

    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        A = np.full((M, M), np.nan)
        for i in range(M):
            for j in range(i + 1, M):
                d = self._cross_dist(Z[i], Z[j])
                A[i, j] = d
                A[j, i] = d
        return A

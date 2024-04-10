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


class PairwiseDistance(Undirected, Unsigned):

    name = "Pairwise distance"
    identifier = "pdist"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, metric="euclidean", **kwargs):
        self._metric = metric
        self.identifier += f"_{metric}"

    @parse_multivariate
    def multivariate(self, data):
        return pairwise_distances(data.to_numpy(squeeze=True), metric=self._metric)


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


class HellerHellerGorfine(Directed, Unsigned):
    """Heller-Heller-Gorfine independence criterion"""

    name = "Heller-Heller-Gorfine Independence Criterion"
    identifier = "hhg"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "directed"]

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

    name = "Dynamic time warping"
    identifier = "dtw"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.dtw


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
        # insert compute SPI code here (computes on x and y)
        stat = self.gwtau(x, y)
        return stat

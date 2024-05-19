import numpy as np
import pandas as pd
from cdt.causality.pairwise import ANM, CDS, IGCI, RECI
import pyEDM
from sklearn.gaussian_process import GaussianProcessRegressor
from cdt.causality.pairwise.ANM import normalized_hsic

from pyspi.base import Directed, Unsigned, Signed, parse_bivariate, parse_multivariate


class AdditiveNoiseModel(Directed, Unsigned):

    name = "Additive noise model"
    identifier = "anm"
    labels = ["unsigned", "causal", "unordered", "linear", "directed"]
    
    # monkey-patch the anm_score function, see cdt PR #155
    def corrected_anm_score(self, x, y):
        gp = GaussianProcessRegressor(random_state=42).fit(x, y)
        y_predict = gp.predict(x).reshape(-1, 1) 
        indepscore = normalized_hsic(y_predict - y, x)
        return indepscore
    
    ANM.anm_score = corrected_anm_score

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return ANM().anm_score(z[i], z[j])


class ConditionalDistributionSimilarity(Directed, Unsigned):

    name = "Conditional distribution similarity statistic"
    identifier = "cds"
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return CDS().cds_score(z[i], z[j])


class RegressionErrorCausalInference(Directed, Unsigned):

    name = "Regression error-based causal inference"
    identifier = "reci"
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return RECI().b_fit_score(z[i], z[j])


class InformationGeometricConditionalIndependence(Directed, Unsigned):

    name = "Information-geometric conditional independence"
    identifier = "igci"
    labels = ["causal", "directed", "nonlinear", "unsigned", "unordered"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return IGCI().predict_proba((z[i], z[j]))


class ConvergentCrossMapping(Directed, Signed):

    name = "Convergent cross-mapping"
    identifier = "ccm"
    labels = ["causal", "directed", "nonlinear", "temporal", "signed"]

    def __init__(self, statistic="mean", embedding_dimension=None):
        self._statistic = statistic
        self._E = embedding_dimension

        self.identifier += f"_E-{embedding_dimension}_{statistic}"

    @property
    def key(self):
        return self._E

    def _from_cache(self, data):
        try:
            ccmf = data.ccm[self.key]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True)

            M = data.n_processes
            N = data.n_observations
            df = pd.DataFrame(
                np.concatenate([np.atleast_2d(np.arange(0, N)), z]).T,
                columns=["index"] + [f"proc{p}" for p in range(M)],
            )

            # Get the embedding
            if self._E is None:
                embedding = np.zeros((M, 1))

                # Infer optimal embedding from simplex projection
                for _i in range(M):
                    pred = str(10) + " " + str(N - 10)
                    embed_df = pyEDM.EmbedDimension(
                        dataFrame=df,
                        lib=pred,
                        pred=pred,
                        columns=df.columns.values[_i + 1],
                        showPlot=False,
                    )
                    embedding[_i] = embed_df.max()["E"]
            else:
                embedding = np.array([self._E] * M)

            # Compute CCM from the fixed or optimal embedding
            nlibs = 21
            ccmf = np.zeros((M, M, nlibs + 1))
            for _i in range(M):
                for _j in range(_i + 1, M):
                    try:
                        E = int(max(embedding[[_i, _j]]))
                    except NameError:
                        E = int(self._E)

                    # Get list of library sizes given nlibs and lower/upper bounds based on embedding dimension
                    upperE = int(np.floor((N - E - 1) / 10) * 10)
                    lowerE = int(np.ceil(2 * E / 10) * 10)
                    inc = int((upperE - lowerE) / nlibs)
                    lib_sizes = str(lowerE) + " " + str(upperE) + " " + str(inc)
                    srcname = df.columns.values[_i + 1]
                    targname = df.columns.values[_j + 1]
                    ccm_df = pyEDM.CCM(
                        dataFrame=df,
                        E=E,
                        columns=srcname,
                        target=targname,
                        libSizes=lib_sizes,
                        sample=100,
                        seed=42,
                    )
                    ccmf[_i, _j] = ccm_df.iloc[:, 1].values[: (nlibs + 1)]
                    ccmf[_j, _i] = ccm_df.iloc[:, 2].values[: (nlibs + 1)]

            try:
                data.ccm[self.key] = ccmf
            except AttributeError:
                data.ccm = {self.key: ccmf}
        return ccmf

    @parse_multivariate
    def multivariate(self, data):
        ccmf = self._from_cache(data)

        if self._statistic == "mean":
            return np.nanmean(ccmf, axis=2)
        elif self._statistic == "max":
            return np.nanmax(ccmf, axis=2)
        elif self._statistic == "diff":
            return np.nanmean(ccmf - np.transpose(ccmf, axes=[1, 0, 2]), axis=2)
        else:
            raise TypeError(f"Unknown statistic: {self._statistic}")

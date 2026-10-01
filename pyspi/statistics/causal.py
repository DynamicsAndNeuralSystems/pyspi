import warnings

import numpy as np
import pandas as pd
import pyEDM
from sklearn.gaussian_process import GaussianProcessRegressor

from pyspi.lib.pairwise_causal import (
    cds_score, igci_score, normalized_hsic, reci_score,
)
from pyspi.utils import require_int

from pyspi.base import Directed, Unsigned, Signed, parse_bivariate, parse_multivariate


class AdditiveNoiseModel(Directed, Unsigned):

    name = "Additive noise model"
    identifier = "anm"
    # Not `linear`: the fit is a Gaussian-process regression and the
    # independence test an RBF-kernel HSIC. Nothing about it is linear.
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """HSIC between x and the residual of a GP fit of y on x.

        This was already pyspi's own scoring function, monkey-patched over
        cdt's (which regressed the wrong way round -- cdt PR #155); only the
        HSIC came from cdt, and it is now `pyspi.lib.pairwise_causal`.
        """
        z = data.to_numpy()
        x, y = z[i], z[j]
        gp = GaussianProcessRegressor(random_state=42).fit(x, y)
        y_predict = gp.predict(x).reshape(-1, 1)
        return normalized_hsic(y_predict - y, x)


class ConditionalDistributionSimilarity(Directed, Unsigned):

    name = "Conditional distribution similarity statistic"
    identifier = "cds"
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return cds_score(z[i], z[j])


class RegressionErrorCausalInference(Directed, Unsigned):

    name = "Regression error-based causal inference"
    identifier = "reci"
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return reci_score(z[i], z[j])


class InformationGeometricCausalInference(Directed, Signed):
    """IGCI: Information-Geometric Causal Inference (Daniusis et al. 2010).

    Not "conditional independence" -- IGCI tests neither. It compares the two
    directions' entropies under a reference measure and leverages the asymmetry
    of a deterministic invertible map.

    The score is a *difference* of two entropies, hence exactly antisymmetric:
    `A[i, j] == -A[j, i]`, verified on the fixtures. Reporting it as unsigned
    was not cosmetic -- `Calculator._rmmin` shifts every column it believes
    unsigned by that column's minimum, which on an antisymmetric matrix moves
    `A[i,j]` and `A[j,i]` by the same amount and destroys the lead/lag the sign
    carries, and `set_group` folds the two directions together through `abs()`.

    It stays commented out of the bundled configs; this is a metadata and API
    correction, not a claim that the heuristic is reliable.
    """

    name = "Information-geometric causal inference"
    identifier = "igci"
    labels = ["causal", "antisymmetric", "signed", "nonlinear", "unordered"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return igci_score(z[i], z[j])


class InformationGeometricConditionalIndependence(InformationGeometricCausalInference):
    """Deprecated alias for :class:`InformationGeometricCausalInference`.

    The old name misdescribed the method. Kept so existing configs and scripts
    keep working; the identifier (`igci`) and the computation are unchanged.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "InformationGeometricConditionalIndependence is deprecated and "
            "will be removed in a future release: IGCI is Information-"
            "Geometric Causal *Inference*, and tests no conditional "
            "independence. Use InformationGeometricCausalInference; the "
            "identifier and the computed values are identical.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _optimal_embedding_dimension(df, column, lib_pred, max_e=10):
    """E in [1, max_e] maximising simplex-projection skill, computed serially.

    Replaces ``pyEDM.EmbedDimension``, for two independent reasons.

    **It was not selecting anything.** The call site read the winner as
    ``embed_df.max()["E"]``. ``DataFrame.max()`` reduces column-wise, so that is
    the largest *candidate* E, not the E at the largest rho -- ``max_e``, every
    time, for every process, whatever the data says. The three shipped
    ``ccm_E-None_*`` SPIs were therefore bit-identical to ``ccm_E-10_*`` on all
    three frozen fixtures (verified: max|difference| exactly 0), while their
    identifiers advertised an inferred embedding. ``rho.idxmax()`` is the
    selection that was intended.

    **It cannot be called safely.** ``EmbedDimension`` always builds a
    ``multiprocessing.Pool``; there is no serial path, and ``numProcess=1``
    still starts a child. pyEDM 2.5 starts pools with forkserver/spawn, never
    fork, so each child re-imports the caller's ``__main__`` -- see the note in
    ``ConvergentCrossMapping._from_cache``. Looping over ``pyEDM.Simplex``
    (public API, and exactly what ``PoolFunc.EmbedDimSimplexFunc`` calls in each
    child) gives the same rho per E with no pool at all.

    Ties go to the smallest E: a lower-dimensional embedding that predicts as
    well is the better model, and ``argmax`` returns the first maximum. Ties are
    not rare -- ``pyEDM.ComputeError`` rounds rho to 6 digits.
    """
    rho = np.empty(max_e)
    for k, E in enumerate(range(1, max_e + 1)):
        pred_df = pyEDM.Simplex(
            dataFrame=df, columns=column, target=column,
            lib=lib_pred, pred=lib_pred, E=E,
        )
        rho[k] = pyEDM.ComputeError(
            pred_df["Observations"], pred_df["Predictions"]
        )["rho"]
    return int(np.nanargmax(rho) + 1)


class ConvergentCrossMapping(Directed, Signed):

    name = "Convergent cross-mapping"
    identifier = "ccm"
    labels = ["causal", "directed", "nonlinear", "temporal", "signed"]
    # Variants (mean/max/diff x embedding dimension) share data.ccm[self.key],
    # keyed by embedding dimension — bucket them onto one parallel worker.
    _cache_namespace = "ccm"

    @property
    def _cache_subkey(self):
        # Cache is data.ccm[E]; statistic is post-lookup.
        return (self._E,)

    def __init__(self, statistic="mean", embedding_dimension=None):
        self._statistic = statistic
        self._E = (
            None if embedding_dimension is None
            else require_int("embedding_dimension", embedding_dimension, minimum=1)
        )

        # The "diff" statistic is ccm(i->j) - ccm(j->i), so A[i,j] == -A[j,i]
        # by construction. That is antisymmetric, not merely directed: the two
        # orientations are one quantity and its negation, not independent
        # values. "mean"/"max" remain plainly directed.
        if statistic == "diff":
            self.labels = [l for l in self.labels if l != "directed"] + ["antisymmetric"]

        self.identifier += f"_E-{self._E}_{statistic}"

    @property
    def key(self):
        return self._E

    def _from_cache(self, data):
        try:
            ccmf = data.ccm[self.key]
        except (AttributeError, KeyError):
            # pyEDM 2.5 self-parallelises (EmbedDimension over processes, CCM
            # over samples), and its pools are single-process here on purpose.
            #
            # pyEDM's own `_get_mp_context` documents "**fork is never used**":
            # it takes forkserver, else spawn. Both re-import the caller's
            # `__main__` in every child. pyspi is normally driven from a plain
            # script, and an unguarded script re-executed by a child raises
            # `RuntimeError: An attempt has been made to start a new process
            # before the current process has finished its bootstrapping phase`
            # -- which pyspi catches, so all nine `ccm_*` SPIs come back as an
            # all-NaN column, with the child having already re-run whatever ran
            # before `compute()`. It only looks fine from a REPL, a notebook,
            # or a `if __name__ == "__main__":`-guarded script (which is why
            # the baseline generator and `python -m pyspi` never saw it).
            # pyspi cannot know whether its caller is import-safe, so it does
            # not gamble on it.
            #
            # There is no speed argument on the other side either: at pyspi's
            # sizes the pool costs far more than it saves. Measured on an idle
            # machine, kuramoto_M7_T100, 21 pairs at E=1: 24.8s with
            # `parallel=True` against 6.8s with `parallel=False`, a 3.6x
            # *speedup* from turning it off. Parallelism belongs at the SPI
            # level, where `Calculator.compute(n_jobs=...)` already provides it.
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
                    col = df.columns.values[_i + 1]
                    embedding[_i] = _optimal_embedding_dimension(df, col, pred)
            else:
                embedding = np.array([self._E] * M)

            # Compute CCM from the fixed or optimal embedding
            nlibs = 21
            ccmf = np.zeros((M, M, nlibs + 1))
            for _i in range(M):
                for _j in range(_i + 1, M):
                    try:
                        E = int(np.max(embedding[[_i, _j]]))
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
                        parallel=False,
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

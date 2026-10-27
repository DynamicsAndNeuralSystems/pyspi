import warnings

import numpy as np
import pandas as pd
import pyEDM
from scipy.spatial import KDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

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
        # Explicit kernel: sklearn <=1.8's default (kernel=None) is this fixed
        # ConstantKernel * RBF, but 1.9 changed the default to an *optimised*
        # kernel, whose length scale collapses to its lower bound and changes
        # every value. No hyperparameters are tuned, so there is no optimiser
        # (and no randomness) in the fit.
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(x, y)
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


def _ccm_library_sizes(N, E, nlibs=21):
    """``nlibs + 1`` library sizes from ``ceil(2E/10)*10`` to ``floor((N-E-1)/10)*10``.

    Inclusive at both ends; fewer (the caller NaN-pads) when the span is under
    ``nlibs``. This used to be pyEDM's "start stop increment" with the increment
    rounded down, which yields more than ``nlibs + 1`` sizes whenever ``nlibs``
    does not divide the span (27 at N=100, E=1), and the excess -- the largest
    libraries, where cross-map skill converges -- was then cut off.
    """
    upper = int(np.floor((N - E - 1) / 10) * 10)
    lower = int(np.ceil(2 * E / 10) * 10)
    if upper < lower:
        raise ValueError(
            f"CCM at E={E} needs library sizes from {lower} to {upper}; "
            f"{N} observations are too few.")
    return [int(L) for L in np.unique(np.rint(np.linspace(lower, upper, nlibs + 1)))]


def _pearson(predictions, actuals):
    """pyEDM's `CCM._nan_safe_pearson`, operation for operation."""
    valid = ~(np.isnan(predictions) | np.isnan(actuals))
    if valid.sum() < 3:
        return np.nan
    p, a = predictions[valid], actuals[valid]
    pm, am = p - p.mean(), a - a.mean()
    denom = np.sqrt(np.dot(pm, pm) * np.dot(am, am))
    if denom == 0.0:
        return 0.0
    return np.dot(pm, am) / denom


def _cross_map_skill(library, targets, E, lib_sizes, sample=100, seed=42):
    """Cross-map skill of one variable's shadow manifold against several targets.

    Returns ``(len(targets), len(lib_sizes))``: entry ``[t, l]`` is the mean,
    over ``sample`` random libraries of ``lib_sizes[l]`` points drawn without
    replacement, of the Pearson correlation between ``targets[t]`` and its
    simplex cross-map estimate from the E-dimensional delay embedding of
    ``library`` (lag 1, Tp = 0, E + 1 neighbours, the query point itself left
    out) -- "library xmap target", in Sugihara et al.'s (2012) terms.

    This is pyEDM 2.5's CCM (``pyEDM.CCM._ccm_for_libsize``) reproduced
    operation for operation, so the values are bit-identical to
    ``pyEDM.CCM(columns=library, target=...)`` (pinned in
    tests/test_causal_audit.py). It exists for speed: everything up to the
    simplex weights -- the random libraries, the KD-tree, the neighbours and
    their weights -- depends on the library variable alone, never on the
    target, yet ``pyEDM.CCM`` recomputes it for every pair the variable is in
    (M - 1 times per variable over all pairs). Here it is computed once per
    library variable and applied to all of its targets.

    The random libraries match pyEDM's: it seeds each (direction, library
    size) task with ``SeedSequence(child.entropy)`` for a child spawned from
    ``SeedSequence(seed)``, and a spawned child carries its parent's entropy,
    so every task starts from ``SeedSequence(seed)``.

    Requires finite data (``Data`` guarantees it), so every row from ``E - 1``
    on is a valid embedding point.
    """
    library = np.asarray(library, dtype=np.float64)
    rows = np.arange(E - 1, library.shape[0])
    embed = np.ascontiguousarray(
        np.column_stack([library[rows - lag] for lag in range(E)]))
    actuals = [np.asarray(y, dtype=np.float64)[rows] for y in targets]
    n_points = embed.shape[0]
    r_M = np.arange(n_points)[:, np.newaxis]

    skill = np.empty((len(targets), len(lib_sizes)))
    rhos = np.empty((len(targets), sample))
    for l, L in enumerate(lib_sizes):
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        L = min(int(L), n_points)
        k = min(E + 1, L - 1)
        if k < 1:
            skill[:, l] = np.nan
            continue
        k_query = min(k + 1, L)
        for s in range(sample):
            lib_idx = rng.choice(n_points, size=L, replace=False)
            lib_idx.sort()
            dist, local = KDTree(embed[lib_idx]).query(embed, k=k_query)
            if dist.ndim == 1:
                dist, local = dist[:, np.newaxis], local[:, np.newaxis]
            neighbours = lib_idx[local]

            # First k neighbours other than the query point itself
            valid = ~(neighbours == r_M)
            cs = np.cumsum(valid, axis=1)
            first_k = valid & (cs <= k)
            insufficient = cs[:, -1] < k
            if np.all(insufficient):
                rhos[:, s] = np.nan
                continue
            if np.any(insufficient):
                first_k[insufficient, :] = False
            _, cols = np.where(first_k)
            nn_cols = np.zeros((n_points, k), dtype=np.intp)
            sufficient = ~insufficient
            n_sufficient = sufficient.sum()
            if n_sufficient > 0:
                nn_cols[sufficient] = cols.reshape(n_sufficient, k)
            nn_dist = dist[r_M, nn_cols]
            nn = neighbours[r_M, nn_cols]

            # Simplex weights exp(-d / d_min), normalised
            d_min = nn_dist[:, 0:1]
            weights = np.exp(-nn_dist / np.where(d_min > 0.0, d_min, 1.0))
            zero_mask = (d_min == 0.0)
            if np.any(zero_mask):
                weights = np.where(
                    zero_mask, np.where(nn_dist == 0.0, 1.0, 0.0), weights)
            w_sum = weights.sum(axis=1, keepdims=True)
            weights /= np.where(w_sum > 0.0, w_sum, 1.0)

            for t, actual in enumerate(actuals):
                predictions = np.sum(weights * actual[nn], axis=1)
                if np.any(insufficient):
                    predictions[insufficient] = np.nan
                rhos[t, s] = _pearson(predictions, actual)
        for t in range(len(targets)):
            skill[t, l] = np.nanmean(rhos[t])
    return skill


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
            # pyEDM's multiprocessing pools are never started: they use
            # forkserver/spawn, which re-import an unguarded caller's
            # `__main__` and left every `ccm_*` SPI all-NaN when pyspi ran from
            # a plain script (see CHANGELOG 3.0.0). Embedding selection loops
            # over `pyEDM.Simplex`, and cross-mapping is `_cross_map_skill`,
            # pyEDM 2.5's CCM computed serially and once per library variable.
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

            # Compute CCM from the fixed or optimal embedding. Each pair uses
            # the larger of its two dimensions in both directions.
            embedding = np.asarray(embedding, dtype=int).reshape(-1)
            pair_E = np.maximum.outer(embedding, embedding)
            nlibs = 21
            ccmf = np.zeros((M, M, nlibs + 1))
            for E in np.unique(pair_E[~np.eye(M, dtype=bool)]):
                lib_sizes = _ccm_library_sizes(N, int(E), nlibs)
                n = len(lib_sizes)
                for _i in range(M):
                    targets = [_j for _j in range(M)
                               if _j != _i and pair_E[_i, _j] == E]
                    if not targets:
                        continue
                    # The row of `skill` for target _j is "_i xmap _j": the
                    # shadow manifold of _i recovering _j, which Sugihara et
                    # al. (2012) read as evidence that _j drives _i. Row =
                    # source, column = target, so it is the _j -> _i cell.
                    skill = _cross_map_skill(
                        z[_i], [z[_j] for _j in targets], int(E), lib_sizes,
                        sample=100, seed=42)
                    for _j, row in zip(targets, skill):
                        ccmf[_j, _i, :n] = row
                        ccmf[_j, _i, n:] = np.nan

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

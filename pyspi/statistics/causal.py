import warnings

import numpy as np
import pandas as pd
import pyEDM
from scipy.linalg import cho_solve, cholesky
from scipy.spatial import KDTree
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from pyspi.lib.pairwise_causal import (
    cds_score, igci_score, normalized_hsic, reci_score,
)
from pyspi.utils import require_int

from pyspi.base import Directed, Unsigned, Signed, parse_bivariate, parse_multivariate


# A fixed (untuned) ConstantKernel * RBF, as in cdt's ANM; set explicitly
# because sklearn 1.9 changed the default kernel. 1e-10 is
# GaussianProcessRegressor's default `alpha`.
_ANM_KERNEL = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
    1.0, length_scale_bounds="fixed")
_ANM_ALPHA = 1e-10


def _anm_scores(x, ys):
    """``normalized_hsic(residual, x)`` for the GP regression of each y on x.

    The regression is ``GaussianProcessRegressor(kernel=_ANM_KERNEL,
    optimizer=None).fit(x, y).predict(x)``, the posterior mean at the training
    inputs (Rasmussen & Williams 2006, Alg. 2.1), computed with the same
    operations as scikit-learn. The kernel matrix and its Cholesky factor
    depend on the cause x alone, so they are built once and shared by every
    effect.
    """
    K = _ANM_KERNEL(x)
    K[np.diag_indices_from(K)] += _ANM_ALPHA
    L = cholesky(K, lower=True, check_finite=False)
    K_trans = _ANM_KERNEL(x, x)
    scores = []
    for y in ys:
        y_predict = K_trans @ cho_solve((L, True), y, check_finite=False)
        scores.append(normalized_hsic(y_predict.reshape(-1, 1) - y, x))
    return scores


class AdditiveNoiseModel(Directed, Unsigned):
    """Additive-noise-model score for row (cause) -> column (effect).

    Lower is stronger evidence for the row causing the column (a small HSIC
    means the residual of the column regressed on the row is independent of
    the row). ``cds`` and ``reci`` share this polarity; for ``ccm`` and
    ``igci`` higher means more.
    """

    name = "Additive noise model"
    identifier = "anm"
    # Not `linear`: the fit is a Gaussian-process regression and the
    # independence test an RBF-kernel HSIC. Nothing about it is linear.
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """HSIC between x and the residual of a GP fit of y on x.

        cdt's ANM regresses the other way round (cdt PR #155). The HSIC is
        in `pyspi.lib.pairwise_causal`.
        """
        z = data.to_numpy()
        return _anm_scores(z[i], [z[j]])[0]

    @parse_multivariate
    def multivariate(self, data):
        """`bivariate` for every pair, sharing each cause's GP factorisation."""
        z = data.to_numpy()
        M = data.n_processes
        A = np.full((M, M), np.nan)
        for i in range(M):
            effects = [j for j in range(M) if j != i]
            A[i, effects] = _anm_scores(z[i], [z[j] for j in effects])
        return A


class ConditionalDistributionSimilarity(Directed, Unsigned):
    """CDS score for row (cause) -> column (effect); lower is stronger evidence.

    See ``AdditiveNoiseModel`` for the polarity convention across the causal
    module.
    """

    name = "Conditional distribution similarity statistic"
    identifier = "cds"
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        # NaN when fewer than two x-bins can be compared.
        return cds_score(z[i], z[j], min_conditionals=2)


class RegressionErrorCausalInference(Directed, Unsigned):
    """RECI score for row (cause) -> column (effect); lower is stronger evidence.

    See ``AdditiveNoiseModel`` for the polarity convention across the causal
    module.
    """

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
    `A[i, j] == -A[j, i]`. It is therefore signed: `Calculator._rmmin` shifts
    every unsigned column by its minimum and `set_group` correlates unsigned
    SPIs through `abs()`, both of which would destroy the lead/lag the sign
    carries.

    It is commented out of the bundled configs; the heuristic is not claimed
    to be reliable.
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

    The serial equivalent of ``pyEDM.EmbedDimension``: it runs
    ``pyEDM.Simplex`` for each E (as ``PoolFunc.EmbedDimSimplexFunc`` does in
    each child) and selects ``rho.idxmax()``. ``EmbedDimension`` always builds a
    ``multiprocessing.Pool`` (even with ``numProcess=1``), which pyEDM starts
    with forkserver/spawn, so each child re-imports the caller's ``__main__``
    (see the note in ``ConvergentCrossMapping._from_cache``).

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
    ``nlibs``.
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

    This reproduces pyEDM 2.5's CCM (``pyEDM.CCM._ccm_for_libsize``),
    including its random libraries (every library size starts from
    ``SeedSequence(seed)``), so values equal
    ``pyEDM.CCM(columns=library, target=...)``. The difference is the
    organisation of the work: ``pyEDM.CCM`` handles one (library, target)
    pair per call, whereas here the random libraries, KD-tree neighbours and
    simplex weights, which depend on the library variable alone, are built
    once and applied to all of that variable's targets. pyEDM's process pools
    are not used (they re-import the caller's ``__main__``).

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
    """Convergent cross-mapping skill, row (source) -> column (target).

    ``table[i, j]`` is the skill with which the shadow manifold of *j*
    recovers *i*, which Sugihara et al. (2012) read as *i* driving *j*; higher
    means more (the opposite polarity to ``anm``/``cds``/``reci``). ``mean``
    and ``max`` are over library sizes; ``diff`` is ``ccm(i->j) - ccm(j->i)``
    and antisymmetric.

    With ``embedding_dimension=None`` each process gets its own
    simplex-optimal dimension, and each pair uses the larger of its two in
    both directions.
    """

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
            # Serial throughout: pyEDM's process pools (forkserver/spawn)
            # re-import the caller's `__main__`, which fails from an unguarded
            # script. Parallelism is at the SPI level (`Calculator.compute`).
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

            # Each pair uses the larger of its two embedding dimensions.
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
                    # "_i xmap _j" (the shadow manifold of _i recovering _j)
                    # is evidence that _j drives _i, so it fills cell [_j, _i].
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

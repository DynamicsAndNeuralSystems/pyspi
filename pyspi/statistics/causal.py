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

from pyspi.base import (
    Directed, Unsigned, Signed, fill_pairwise, parse_bivariate, parse_multivariate,
)


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

        This was already pyspi's own scoring function, monkey-patched over
        cdt's (which regressed the wrong way round -- cdt PR #155); only the
        HSIC came from cdt, and it is now `pyspi.lib.pairwise_causal`.
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


def _nli_embed(x, m, tau):
    """Delay vectors ``(x_n, x_{n-tau}, ..., x_{n-(m-1)tau})``, one per row.

    Rows run over ``n = (m-1)tau, ..., T-1``, so every process of a dataset
    gives the same ``N = T - (m-1)tau`` time indices and row ``n`` of two
    processes' embeddings is the same instant.
    """
    T = x.shape[0]
    span = (m - 1) * tau
    return np.ascontiguousarray(
        np.column_stack([x[span - l * tau:T - l * tau] for l in range(m)]))


def _nli_neighbours(emb, k, theiler):
    """Time indices of the ``k`` nearest neighbours of every delay vector.

    Neighbours with ``|n - r| <= theiler`` (the point itself included) are
    excluded. At most ``2 * theiler + 1`` points are excluded per row, so the
    ``k + 2 * theiler + 1`` nearest always contain ``k`` admissible ones and
    no second query is needed. Returns ``(N, k)``, nearest first.
    """
    N = emb.shape[0]
    _, idx = KDTree(emb).query(emb, k=k + 2 * theiler + 1)
    admissible = np.abs(idx - np.arange(N)[:, np.newaxis]) > theiler
    # A stable sort of the inadmissible flags keeps the admissible neighbours
    # first, in increasing distance; the first k of them are the k nearest.
    first = np.argsort(~admissible, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(idx, first, axis=1)


def _nli_mean_sq_dist(emb, idx):
    """``(1/k) sum_l |x_n - x_{idx[n, l]}|^2`` for every row ``n``."""
    diff = emb[idx] - emb[:, np.newaxis, :]
    return np.square(diff).sum(axis=2).mean(axis=1)


def _nli_mean_sq_dist_all(emb, theiler):
    """Mean squared distance from each ``x_n`` to every admissible ``x_j``.

    Admissible means ``|n - j| > theiler``. Returns ``(R, n_admissible)``.
    The sum over all ``j`` uses ``sum_j |c_n - c_j|^2 = N|c_n|^2 + sum_j
    |c_j|^2`` for the centred vectors ``c`` (``sum_j c_j = 0``), O(N m); the
    ``2 * theiler + 1`` excluded offsets are then subtracted one offset at a
    time, O(N m theiler).
    """
    N = emb.shape[0]
    c = emb - emb.mean(axis=0)
    sq = np.square(c).sum(axis=1)
    total = N * sq + sq.sum() - 2.0 * (c @ c.sum(axis=0))
    excluded = np.zeros(N)
    n_excluded = np.ones(N, dtype=np.int64)  # the point itself
    for d in range(1, min(theiler, N - 1) + 1):
        s = np.square(c[d:] - c[:-d]).sum(axis=1)
        excluded[d:] += s
        excluded[:-d] += s
        n_excluded[d:] += 1
        n_excluded[:-d] += 1
    n_admissible = N - n_excluded
    return (total - excluded) / n_admissible, n_admissible


def _nli_ranks(emb, theiler, queries, block=128):
    """Rank of ``|x_n - x_{queries[n, l]}|`` among ``x_n``'s admissible distances.

    For each row ``n`` the distances to every ``x_j`` with ``|n - j| >
    theiler`` are ranked in ascending order (1 = nearest); entry ``[n, l]`` is
    the rank of the distance to ``x_{queries[n, l]}``, which must itself be
    admissible. Tied distances take their average rank, as in
    ``scipy.stats.rankdata``, so the result does not depend on tie-breaking.

    One sort of each row serves every query column, so the ranks a process
    needs for all of its conditioning partners cost one O(N^2 log N) pass.
    """
    N = emb.shape[0]
    out = np.empty(queries.shape)
    times = np.arange(N)
    for a in range(0, N, block):
        b = min(N, a + block)
        D = np.zeros((b - a, N))
        for d in range(emb.shape[1]):
            D += np.square(emb[a:b, d, np.newaxis] - emb[np.newaxis, :, d])
        D[np.abs(times[a:b, np.newaxis] - times) <= theiler] = np.inf
        q = np.take_along_axis(D, queries[a:b], axis=1)
        D.sort(axis=1)
        for r in range(b - a):
            below = np.searchsorted(D[r], q[r], side="left")
            through = np.searchsorted(D[r], q[r], side="right")
            out[a + r] = (below + through + 1) / 2.0
    return out


class NonlinearInterdependence(Directed, Unsigned):
    """Nonlinear interdependence S, H, N or L from delay-embedded neighbourhoods.

    State-space measures of whether close states of one process occur at the
    same times as close states of another (Arnhold et al. 1999; Quian Quiroga
    et al. 2002; Chicharro & Andrzejak 2009). Each process is delay-embedded,
    ``x_n = (x_n, x_{n-tau}, ..., x_{n-(m-1)tau})`` for ``n = 1..N`` with
    ``N = T - (m-1)tau``. Let
    ``r_{n,l}`` and ``s_{n,l}`` (``l = 1..k``) be the time indices of the ``k``
    nearest neighbours of ``x_n`` and of ``y_n`` (Euclidean distance), with
    temporal neighbours ``|n - r| <= theiler`` excluded. Then

    * ``R_n^k(X)   = (1/k) sum_l |x_n - x_{r_{n,l}}|^2``  (own neighbours),
    * ``R_n^k(X|Y) = (1/k) sum_l |x_n - x_{s_{n,l}}|^2``  (Y's neighbours'
      time indices, measured in X),
    * ``R_n(X)``, the mean of ``|x_n - x_j|^2`` over every admissible ``j``,

    and, averaging over ``n``,

    * ``S(X|Y) = <R_n^k(X) / R_n^k(X|Y)>`` (Arnhold et al. 1999), in [0, 1];
    * ``H(X|Y) = <log(R_n(X) / R_n^k(X|Y))>`` (Arnhold et al. 1999), in nats,
      about 0 for independent processes;
    * ``N(X|Y) = <(R_n(X) - R_n^k(X|Y)) / R_n(X)>`` (Quian Quiroga et al.
      2002), at most 1, about 0 for independent processes;
    * ``L(X|Y) = <(G_n(X) - G_n^k(X|Y)) / (G_n(X) - G_n^k(X))>`` (Chicharro &
      Andrzejak 2009, eq. 4), at most 1, about 0 for independent processes.
      ``G_n^k(X|Y)`` is the mean rank of the ``k`` distances
      ``|x_n - x_{s_{n,l}}|`` among all of ``x_n``'s admissible distances
      (1 = nearest; ties take their average rank), ``G_n^k(X) = (k+1)/2`` is
      the smallest possible mean rank and ``G_n(X) = (N_n + 1)/2`` the mean
      rank, ``N_n`` being the number of admissible ``j`` for ``x_n``.

    With ``theiler = W`` every average over ``j`` (the neighbour searches,
    ``R_n(X)`` and the ranks) runs over ``|n - j| > W`` only, so ``N_n =
    N - 1 - #{j != n: |n - j| <= W}`` (``N - 2W - 1`` away from the ends), as
    in Chicharro & Andrzejak (2009), note 24.

    **Direction.** ``A[i, j] = A(x_i | x_j)`` (``A`` one of S, H, N, L): the
    row is the conditioned process X, whose distances are measured, and the
    column is the conditioning process Y, whose neighbourhoods are mapped onto
    X. The entry is large when close states of ``x_j`` are mapped to close
    states of ``x_i``, which Chicharro & Andrzejak (2009) read as a coupling
    ``x_i -> x_j`` (as in convergent cross-mapping). So with pyspi's
    row = source convention, ``A[i, j] > A[j, i]`` points to ``i -> j``.

    **Caveats.** Both entries of a coupled pair rise, so only the asymmetry
    indicates direction, and it is unreliable near generalised
    synchronisation. S and H depend on the effective dimension of each
    process, so their asymmetry can point the wrong way (Quian Quiroga et al.
    2000); L, which uses only ranks, is the most reliable for direction
    (Chicharro & Andrzejak 2009).

    **Undefined pairs.** S and H divide by ``R_n^k(X|Y)``, which is zero when
    every conditioned neighbour of some ``x_n`` coincides with it (repeated
    delay vectors, e.g. quantised data). That pair raises, and is NaN in the
    multivariate table.

    Args:
        statistic (str): ``"S"``, ``"H"``, ``"N"`` or ``"L"``.
        m (int): embedding dimension, default 5.
        tau (int): embedding delay in samples, default 1.
        k (int): number of nearest neighbours, default 5.
        theiler (int or None): Theiler window ``W`` in samples; neighbours
            with ``|n - j| <= W`` are excluded. ``None`` (default) uses
            ``m * tau``, which excludes every pair of delay vectors sharing a
            sample. Needs ``N = T - (m-1)tau > k + 2W + 1``.

    References:
        J. Arnhold, P. Grassberger, K. Lehnertz, C. E. Elger, Physica D 134,
        419 (1999).
        R. Quian Quiroga, J. Arnhold, P. Grassberger, Phys. Rev. E 61, 5142
        (2000).
        R. Quian Quiroga, A. Kraskov, T. Kreuz, P. Grassberger, Phys. Rev. E
        65, 041903 (2002).
        D. Chicharro, R. G. Andrzejak, Phys. Rev. E 80, 026217 (2009).
    """

    name = "Nonlinear interdependence"
    identifier = "nli"
    labels = ["causal", "directed", "nonlinear", "temporal", "unsigned"]
    _STATISTICS = ("S", "H", "N", "L")
    # All statistics share data.nli[(m, tau, k, theiler)]: embeddings,
    # neighbours, R_n and the conditioned ranks.
    _cache_namespace = "nli"

    @property
    def _cache_subkey(self):
        return self._key

    def __init__(self, statistic="L", m=5, tau=1, k=5, theiler=None):
        if statistic not in self._STATISTICS:
            raise ValueError(
                f"statistic must be one of {self._STATISTICS}, got {statistic!r}.")
        self._statistic = statistic
        self._m = require_int("m", m, minimum=1)
        self._tau = require_int("tau", tau, minimum=1)
        self._k = require_int("k", k, minimum=1)
        self._theiler = (self._m * self._tau if theiler is None
                         else require_int("theiler", theiler, minimum=0))
        self.identifier += f"_{statistic}_m-{self._m}_tau-{self._tau}_k-{self._k}"
        if theiler is not None:
            self.identifier += f"_theiler-{self._theiler}"

    @property
    def _key(self):
        return (self._m, self._tau, self._k, self._theiler)

    def _cache(self, data):
        try:
            return data.nli[self._key]
        except AttributeError:
            data.nli = {}
        except KeyError:
            pass
        data.nli[self._key] = {"process": {}, "ranks": {}}
        return data.nli[self._key]

    def _process(self, data, cache, p):
        """Embedding, neighbours, ``R_n^k``, ``R_n`` and ``N_n`` of process p."""
        if p not in cache["process"]:
            m, tau, k, W = self._key
            x = data.to_numpy()[p, :, 0]
            N = x.shape[0] - (m - 1) * tau
            if N <= k + 2 * W + 1:
                raise ValueError(
                    f"{self.identifier} needs more than k + 2*theiler + 1 = "
                    f"{k + 2 * W + 1} delay vectors so that every point keeps "
                    f"more than k admissible neighbours; T = {x.shape[0]} "
                    f"observations give N = T - (m-1)*tau = {N}.")
            emb = _nli_embed(np.asarray(x, dtype=np.float64), m, tau)
            nbrs = _nli_neighbours(emb, k, W)
            R, n_admissible = _nli_mean_sq_dist_all(emb, W)
            cache["process"][p] = (emb, nbrs, _nli_mean_sq_dist(emb, nbrs),
                                   R, n_admissible)
        return cache["process"][p]

    def _conditioned_ranks(self, data, cache, i, conditioners):
        """Fill ``cache["ranks"][(i, j)] = G_n^k(x_i | x_j)`` for each j given.

        All of process i's conditioning partners are ranked in one pass over
        its distance rows.
        """
        todo = [j for j in conditioners if (i, j) not in cache["ranks"]]
        if not todo:
            return
        emb = self._process(data, cache, i)[0]
        queries = np.hstack([self._process(data, cache, j)[1] for j in todo])
        ranks = _nli_ranks(emb, self._theiler, queries)
        k = self._k
        for c, j in enumerate(todo):
            cache["ranks"][(i, j)] = ranks[:, c * k:(c + 1) * k].mean(axis=1)

    def _pair(self, data, i, j):
        """``A(x_i | x_j)``: row i conditioned on column j's neighbourhoods."""
        cache = self._cache(data)
        emb, nbrs, Rk, R, n_admissible = self._process(data, cache, i)
        if self._statistic == "L":
            self._conditioned_ranks(data, cache, i, [j])
            G = (n_admissible + 1) / 2.0
            G_min = (self._k + 1) / 2.0
            return float(np.mean((G - cache["ranks"][(i, j)]) / (G - G_min)))

        Rk_cond = _nli_mean_sq_dist(emb, self._process(data, cache, j)[1])
        if self._statistic in ("S", "H") and not np.all(Rk_cond > 0):
            # Repeated delay vectors give R_n^k(X|Y) = 0: the ratio is undefined.
            raise ValueError(
                f"{self.identifier}: a Y-conditioned neighbourhood has zero "
                f"mean squared distance (repeated delay vectors), so "
                f"{self._statistic}(X|Y) is undefined.")
        if self._statistic == "S":
            return float(np.mean(Rk / Rk_cond))
        if self._statistic == "H":
            return float(np.mean(np.log(R / Rk_cond)))
        return float(np.mean((R - Rk_cond) / R))

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """``A(x_i | x_j)``, the ``[i, j]`` entry of ``multivariate``."""
        return self._pair(data, i, j)

    @parse_multivariate
    def multivariate(self, data):
        """Every ordered pair, from per-process embeddings and neighbours.

        Each process is embedded and searched once (O(N log N) with a
        KD-tree); a pair then costs O(N k m). L additionally ranks each
        process's distance rows once for all of its conditioning partners
        (O(N^2 log N) per process).
        """
        M = data.n_processes
        cache = self._cache(data)
        for p in range(M):
            self._process(data, cache, p)
        if self._statistic == "L":
            for i in range(M):
                self._conditioned_ranks(
                    data, cache, i, [j for j in range(M) if j != i])
        A = np.full((M, M), np.nan)
        pairs = [(i, j) for i in range(M) for j in range(M) if i != j]
        return fill_pairwise(self.identifier, A, pairs,
                             lambda i, j: self._pair(data, i, j))

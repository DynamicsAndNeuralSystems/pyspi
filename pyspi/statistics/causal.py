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


def _cross_map_skill(library, targets, E, lib_sizes, sample=100, seed=42, tp=0):
    """Cross-map skill of one variable's shadow manifold against several targets.

    Returns ``(len(targets), len(lib_sizes))``: entry ``[t, l]`` is the mean,
    over ``sample`` random libraries of ``lib_sizes[l]`` points drawn without
    replacement, of the Pearson correlation between ``targets[t]`` and its
    simplex cross-map estimate from the E-dimensional delay embedding of
    ``library`` (lag 1, E + 1 neighbours, the query point itself left out) --
    "library xmap target", in Sugihara et al.'s (2012) terms.

    ``tp`` is the prediction horizon, Ye et al.'s (2015) cross-map lag: the
    embedding point at time t estimates ``target[t + tp]``. Rows whose shifted
    target falls outside the series are dropped first, as pyEDM's ``Tp`` does,
    so the result equals ``pyEDM.CCM(..., Tp=tp)``.

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
    embed, actuals = _shadow_manifold(library, targets, E, tp)
    n_points = embed.shape[0]

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
            rhos[:, s] = _simplex_rhos(dist, lib_idx[local], k, actuals)
        for t in range(len(targets)):
            skill[t, l] = np.nanmean(rhos[t])
    return skill


def _shadow_manifold(library, targets, E, tp):
    """Delay embedding of ``library`` and the targets ``tp`` steps ahead of it.

    Row r of ``embed`` is ``(x(t), x(t-1), ..., x(t-E+1))`` for ``t =
    rows[r]``, and row r of each ``actuals`` entry is ``y(t + tp)``; ``rows``
    keeps every t with a full embedding (``t >= E - 1``) and an in-range
    target (``0 <= t + tp < N``).
    """
    if isinstance(tp, bool) or not isinstance(tp, (int, np.integer)):
        raise TypeError(f"tp must be an integer number of samples, got {tp!r}.")
    tp = int(tp)
    library = np.asarray(library, dtype=np.float64)
    rows = np.arange(max(E - 1, -tp), library.shape[0] - max(tp, 0))
    embed = np.ascontiguousarray(
        np.column_stack([library[rows - lag] for lag in range(E)]))
    actuals = [np.asarray(y, dtype=np.float64)[rows + tp] for y in targets]
    return embed, actuals


def _simplex_rhos(dist, neighbours, k, actuals, candidates=None):
    """Simplex cross-map skill of every target for one random library.

    ``dist`` and ``neighbours`` are ``(n_points, q)``, each row sorted by
    distance, ``neighbours`` holding embedding-row indices. The first ``k``
    entries of a row that are not the query point itself (and, if
    ``candidates`` is given, are marked in it) are its neighbours; a row with
    fewer than ``k`` is left out of the correlation. Returns one Pearson
    correlation per target.
    """
    n_points = dist.shape[0]
    r_M = np.arange(n_points)[:, np.newaxis]
    rhos = np.empty(len(actuals))

    # First k neighbours other than the query point itself
    valid = ~(neighbours == r_M)
    if candidates is not None:
        valid &= candidates
    cs = np.cumsum(valid, axis=1)
    first_k = valid & (cs <= k)
    insufficient = cs[:, -1] < k
    if np.all(insufficient):
        rhos[:] = np.nan
        return rhos
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
        rhos[t] = _pearson(predictions, actual)
    return rhos


def _cross_map_skill_dense(library, targets, E, L, sample=100, seed=42, tp=0):
    """``_cross_map_skill`` at the single library size ``L``, one KD-tree query in all.

    Returns ``(len(targets),)``, equal to
    ``_cross_map_skill(library, targets, E, [L], sample, seed, tp)[:, 0]``
    (same random libraries, neighbours and weights). A library of L of the n
    embedding points omits ``n - L`` of them, so the ``k_query`` nearest
    library points of a query point are the first ``k_query`` library members
    among its ``k_query + n - L`` nearest points overall: one query of the
    whole manifold serves every random library. It is fast when ``n - L`` is
    small, as at the largest CCM library size.

    With exactly tied distances (quantised data) this falls back to
    ``_cross_map_skill``, since which tied point a KD-tree returns depends on
    the tree.
    """
    embed, actuals = _shadow_manifold(library, targets, E, tp)
    n_points = embed.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    L = min(int(L), n_points)
    k = min(E + 1, L - 1)
    if k < 1:
        return np.full(len(targets), np.nan)
    k_query = min(k + 1, L)
    K = min(k_query + n_points - L, n_points)
    # One extra neighbour, to see a tie straddling the end of the list.
    dist, neighbours = KDTree(embed).query(embed, k=min(K + 1, n_points))
    if dist.ndim == 1:
        dist, neighbours = dist[:, np.newaxis], neighbours[:, np.newaxis]
    if np.any(dist[:, 1:] == dist[:, :-1]):
        # Tied distances: defer to the per-library KD-trees.
        return _cross_map_skill(library, targets, E, [L], sample=sample,
                                seed=seed, tp=tp)[:, 0]
    dist, neighbours = dist[:, :K], neighbours[:, :K]

    rhos = np.empty((len(targets), sample))
    member = np.zeros(n_points, dtype=bool)
    for s in range(sample):
        lib_idx = rng.choice(n_points, size=L, replace=False)
        member[:] = False
        member[lib_idx] = True
        in_lib = member[neighbours]
        # The k_query nearest library points, as the per-library tree finds.
        candidates = in_lib & (np.cumsum(in_lib, axis=1) <= k_query)
        rhos[:, s] = _simplex_rhos(dist, neighbours, k, actuals, candidates)
    return np.array([np.nanmean(rhos[t]) for t in range(len(targets))])


# Library-size steps of the cached CCM skill curve (nlibs + 1 sizes).
_CCM_NLIBS = 21


def _ccm_store(data, key, value):
    """Put ``value`` on ``data.ccm[key]``, creating the dict on first use."""
    try:
        data.ccm[key] = value
    except AttributeError:
        data.ccm = {key: value}


def _ccm_embedding(data, E, procs=None):
    """Embedding dimension of each process in ``procs`` (default: all), as ints.

    A fixed ``E`` is returned for every process. ``E=None`` gives each process
    its simplex-optimal dimension (``_optimal_embedding_dimension`` over E =
    1..10, library and prediction rows ``10 .. N - 10``). The whole dataset's
    selection is cached on ``data.ccm[("embedding",)]`` and shared by
    ``ccm_E-None_*`` and ``lccm_E-None_*``. A call for a subset of processes
    reads that cache when present and otherwise computes only those
    processes, without caching them.

    Selection loops over ``pyEDM.Simplex`` serially: pyEDM's multiprocessing
    pools (forkserver/spawn) re-import an unguarded caller's ``__main__``.
    """
    M = data.n_processes
    idx = np.arange(M) if procs is None else np.asarray(procs, dtype=int)
    if E is not None:
        return np.full(len(idx), int(E), dtype=int)
    cached = getattr(data, "ccm", {}).get(("embedding",))
    if cached is not None:
        return cached[idx]

    z = data.to_numpy(squeeze=True)
    N = data.n_observations
    df = pd.DataFrame(
        np.concatenate([np.atleast_2d(np.arange(0, N)), z]).T,
        columns=["index"] + [f"proc{p}" for p in range(M)],
    )
    pred = str(10) + " " + str(N - 10)
    embedding = np.array(
        [_optimal_embedding_dimension(df, f"proc{p}", pred) for p in idx],
        dtype=int)
    if procs is None:
        _ccm_store(data, ("embedding",), embedding)
    return embedding


def _ccm_curves(z, embedding, nlibs=_CCM_NLIBS):
    """CCM skill curves ``(P, P, nlibs + 1)`` of the processes ``z`` (P, N).

    ``[s, t, l]`` is "t xmap s" (the shadow manifold of t recovering s) at the
    l-th library size of ``_ccm_library_sizes(N, E_st)``, NaN-padded past the
    last size; ``E_st = max(embedding[s], embedding[t])`` in both directions.
    The diagonal is 0 (never computed).
    """
    P, N = z.shape
    pair_E = np.maximum.outer(embedding, embedding)
    ccmf = np.zeros((P, P, nlibs + 1))
    for E in np.unique(pair_E[~np.eye(P, dtype=bool)]):
        lib_sizes = _ccm_library_sizes(N, int(E), nlibs)
        n = len(lib_sizes)
        for _i in range(P):
            targets = [_j for _j in range(P)
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
    return ccmf


def _ccm_log_library_sizes(embedding, N, nlibs=_CCM_NLIBS):
    """``ln L`` aligned with ``_ccm_curves``: NaN past the last size and on the diagonal."""
    P = len(embedding)
    off = ~np.eye(P, dtype=bool)
    pair_E = np.maximum.outer(embedding, embedding)
    log_L = np.full((P, P, nlibs + 1), np.nan)
    for E in np.unique(pair_E[off]):
        sizes = np.log(np.asarray(_ccm_library_sizes(N, int(E), nlibs), float))
        log_L[(pair_E == E) & off, :len(sizes)] = sizes
    return log_L


def _ccm_convergence(curves):
    """Last finite skill minus first finite skill along the library axis.

    NaN where fewer than two library sizes have a finite skill.
    """
    finite = np.isfinite(curves)
    n_lib = curves.shape[2]
    first = np.argmax(finite, axis=2)[..., np.newaxis]
    last = (n_lib - 1 - np.argmax(finite[..., ::-1], axis=2))[..., np.newaxis]
    out = (np.take_along_axis(curves, last, axis=2)
           - np.take_along_axis(curves, first, axis=2))[..., 0]
    out[finite.sum(axis=2) < 2] = np.nan
    return out


def _ccm_slope(curves, log_L):
    """OLS slope of skill on ``ln L`` over the library sizes where both are finite.

    NaN where fewer than two such sizes.
    """
    ok = np.isfinite(curves) & np.isfinite(log_L)
    n = ok.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_bar = np.where(ok, log_L, 0.0).sum(axis=2) / n
        y_bar = np.where(ok, curves, 0.0).sum(axis=2) / n
        dx = np.where(ok, log_L - x_bar[..., np.newaxis], 0.0)
        dy = np.where(ok, curves - y_bar[..., np.newaxis], 0.0)
        out = (dx * dy).sum(axis=2) / (dx * dx).sum(axis=2)
    out[n < 2] = np.nan
    return out


class ConvergentCrossMapping(Directed, Signed):
    """Convergent cross-mapping skill, row (source) -> column (target).

    ``table[i, j]`` is read off the skill with which the shadow manifold of
    *j* recovers *i*, which Sugihara et al. (2012) read as *i* driving *j*;
    higher means more (the opposite polarity to ``anm``/``cds``/``reci``).
    That skill is computed once per Data and embedding dimension as a curve
    ``rho_ij(L)`` over 22 library sizes L from ``ceil(2E/10)*10`` to
    ``floor((N-E-1)/10)*10`` (``_ccm_library_sizes``), each the mean Pearson
    correlation over 100 random libraries (``_cross_map_skill``), and cached
    on ``data.ccm[E]``. The statistics are all read off that curve:

    * ``mean`` and ``max``: over library sizes.
    * ``diff``: ``ccm(i->j) - ccm(j->i)`` averaged over library sizes, and
      antisymmetric.
    * ``convergence``: ``rho_ij(L_max) - rho_ij(L_min)``, the last finite
      entry of the curve minus the first. Skill that grows with the library
      is Sugihara et al.'s (2012) criterion for causation. Signed and
      directed: negative when skill falls with L.
    * ``slope``: the least-squares slope of ``rho_ij`` against ``ln L`` over
      the finite library sizes, in skill per e-fold of library size.

    ``convergence`` and ``slope`` are NaN with fewer than two finite library
    sizes. A link strong enough to synchronise the pair saturates the skill
    from the smallest library on and so barely converges, and when both
    directions are causal the stronger one can converge less.

    With ``embedding_dimension=None`` each process gets its own
    simplex-optimal dimension, and each pair uses the larger of its two in
    both directions (Sugihara et al. use the library variable's own), so the
    two directions are computed on manifolds of the same dimension.

    ``bivariate`` reads the cached table when the Data has one and otherwise
    computes the pair alone. See ``LaggedConvergentCrossMapping`` for the
    time-delayed variant of Ye et al. (2015).

    Reference: G. Sugihara et al., "Detecting causality in complex
    ecosystems", Science 338, 496-500 (2012).
    """

    name = "Convergent cross-mapping"
    identifier = "ccm"
    labels = ["causal", "directed", "nonlinear", "temporal", "signed"]
    # Variants (statistic x embedding dimension) share data.ccm[self.key],
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
        # values. "mean"/"max"/"convergence"/"slope" remain plainly directed.
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
            # Compute CCM from the fixed or optimal embedding. Each pair uses
            # the larger of its two dimensions in both directions.
            ccmf = _ccm_curves(data.to_numpy(squeeze=True),
                               _ccm_embedding(data, self._E))
            _ccm_store(data, self.key, ccmf)
        return ccmf

    def _table(self, ccmf, embedding, N):
        """The statistic for every cell of a ``(P, P, nlibs + 1)`` curve block."""
        if self._statistic == "mean":
            return np.nanmean(ccmf, axis=2)
        elif self._statistic == "max":
            return np.nanmax(ccmf, axis=2)
        elif self._statistic == "diff":
            return np.nanmean(ccmf - np.transpose(ccmf, axes=[1, 0, 2]), axis=2)
        elif self._statistic == "convergence":
            out = _ccm_convergence(ccmf)
        elif self._statistic == "slope":
            out = _ccm_slope(ccmf, _ccm_log_library_sizes(embedding(), N))
        else:
            raise TypeError(f"Unknown statistic: {self._statistic}")
        np.fill_diagonal(out, np.nan)
        return out

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        pair = [i, j]
        try:
            ccmf = data.ccm[self.key][np.ix_(pair, pair)]
        except (AttributeError, KeyError):
            ccmf = _ccm_curves(data.to_numpy(squeeze=True)[pair],
                               _ccm_embedding(data, self._E, procs=pair))
        return self._table(
            ccmf, lambda: _ccm_embedding(data, self._E, procs=pair),
            data.n_observations)[0, 1]

    @parse_multivariate
    def multivariate(self, data):
        ccmf = self._from_cache(data)
        return self._table(ccmf, lambda: _ccm_embedding(data, self._E),
                           data.n_observations)


def _lagged_ccm_skill(z, embedding, max_lag, nlibs=_CCM_NLIBS, sample=100,
                      seed=42):
    """Cross-map skill by prediction horizon, ``(P, P, 2 * max_lag + 1)``.

    ``[s, t, max_lag + tp]`` is "t xmap s" at horizon ``tp``: the shadow
    manifold of t at time u estimating ``s[u + tp]``, at the largest library
    size of ``_ccm_library_sizes(N, E_st)`` only, with ``E_st`` the larger of
    the pair's two dimensions as in ``_ccm_curves``. Same seeds and sample
    count as ``_ccm_curves``, which reseeds each library size, so the
    ``tp = 0`` slice is exactly the last finite entry of the CCM curve. The
    diagonal is NaN.
    """
    P, N = z.shape
    pair_E = np.maximum.outer(embedding, embedding)
    tps = range(-max_lag, max_lag + 1)
    skill = np.full((P, P, len(tps)), np.nan)
    for E in np.unique(pair_E[~np.eye(P, dtype=bool)]):
        top = _ccm_library_sizes(N, int(E), nlibs)[-1]
        for _i in range(P):
            targets = [_j for _j in range(P)
                       if _j != _i and pair_E[_i, _j] == E]
            if not targets:
                continue
            ys = [z[_j] for _j in targets]
            for l, tp in enumerate(tps):
                # "_i xmap _j(t + tp)" is evidence for _j -> _i: cell [_j, _i].
                skill[targets, _i, l] = _cross_map_skill_dense(
                    z[_i], ys, int(E), top, sample=sample, seed=seed, tp=tp)
    return skill


class LaggedConvergentCrossMapping(Directed, Signed):
    """Lagged (time-delayed) convergent cross-mapping, row (source) -> column (target).

    Ye et al. (2015) extend CCM with a cross-map lag: the shadow manifold of
    the library variable at time t estimates the other variable at time
    ``t + Tp``. For each ordered pair and each ``Tp in -k..k`` this computes

        rho_ij(Tp) = corr( i(t + Tp), estimate of i(t + Tp) from M_j(t) ),

    "j xmap i" at horizon Tp, at the largest library size only (``L =
    floor((N-E-1)/10)*10``, 100 random libraries, seed 42, exactly as
    ``ConvergentCrossMapping``; the ``Tp = 0`` value equals the last point of
    the ``ccm`` curve). The ``(M, M, 2k + 1)`` array is cached on
    ``data.ccm[("lagged", E, k)]``.

    Orientation: row = source, so ``A[i, j]`` (``i -> j``) uses ``M_j``
    cross-mapping i, as in ``ccm``. If i reaches j after a delay ``d >= 1``,
    ``M_j(t)`` best recovers ``i(t - d)`` and ``rho_ij(Tp)`` peaks at
    ``Tp = -d < 0``; skill in the non-causal direction comes from generalized
    synchrony under strong forcing, where the peak sits at ``Tp >= 0``.

    Statistics:

    * ``tp``: the Tp of maximal skill, an integer in ``[-k, k]``. Negative
      ``A[i, j]`` is the signature of a causal i -> j. Its magnitude
      estimates ``d`` only roughly (the peak flattens for larger E). For an
      unlinked pair the argmax is noise, so read ``tp`` together with
      ``skill`` or ``ccm_*_convergence``. Ties go to the most negative Tp.
    * ``skill``: ``max over Tp of rho_ij(Tp)``, signed like ``ccm``.

    Unlike ``ccm``, which evaluates every library size at ``Tp = 0``, ``lccm``
    reads direction from when skill peaks, so it can separate a delayed
    causal link from synchrony; it does not test convergence.

    Parameters
    ----------
    statistic : {"tp", "skill"}
    embedding_dimension : int or None
        As in ``ConvergentCrossMapping``: ``None`` selects each process's
        simplex-optimal E and a pair uses the larger of its two, in both
        directions.
    max_lag : int
        k, the largest horizon examined, in samples (default 5).

    Rows whose shifted target falls outside the series are dropped (as
    pyEDM's ``Tp``), so a horizon of ``Tp`` costs ``|Tp|`` points at most.

    Reference: H. Ye, E. R. Deyle, L. J. Gilarranz & G. Sugihara,
    "Distinguishing time-delayed causal interactions using convergent cross
    mapping", Scientific Reports 5, 14750 (2015).
    """

    name = "Lagged convergent cross-mapping"
    identifier = "lccm"
    labels = ["causal", "directed", "nonlinear", "temporal", "signed"]
    # Both statistics read data.ccm[("lagged", E, k)].
    _cache_namespace = "ccm"

    _STATISTICS = ("tp", "skill")

    @property
    def _cache_subkey(self):
        return ("lagged", self._E, self._k)

    def __init__(self, statistic="tp", embedding_dimension=None, max_lag=5):
        if statistic not in self._STATISTICS:
            raise ValueError(
                f"statistic must be one of {self._STATISTICS}, got {statistic!r}.")
        self._statistic = statistic
        self._E = (
            None if embedding_dimension is None
            else require_int("embedding_dimension", embedding_dimension, minimum=1)
        )
        self._k = require_int("max_lag", max_lag, minimum=1)
        self.identifier += f"_E-{self._E}_k-{self._k}_{statistic}"

    @property
    def key(self):
        return ("lagged", self._E, self._k)

    def _from_cache(self, data):
        try:
            skill = data.ccm[self.key]
        except (AttributeError, KeyError):
            skill = _lagged_ccm_skill(data.to_numpy(squeeze=True),
                                      _ccm_embedding(data, self._E), self._k)
            _ccm_store(data, self.key, skill)
        return skill

    def _table(self, skill):
        finite = np.isfinite(skill)
        none = ~finite.any(axis=2)
        if self._statistic == "tp":
            peak = np.argmax(np.where(finite, skill, -np.inf), axis=2)
            out = (peak - self._k).astype(float)
        else:
            out = np.max(np.where(finite, skill, -np.inf), axis=2)
        out[none] = np.nan
        return out

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        pair = [i, j]
        try:
            skill = data.ccm[self.key][np.ix_(pair, pair)]
        except (AttributeError, KeyError):
            skill = _lagged_ccm_skill(
                data.to_numpy(squeeze=True)[pair],
                _ccm_embedding(data, self._E, procs=pair), self._k)
        return self._table(skill)[0, 1]

    @parse_multivariate
    def multivariate(self, data):
        return self._table(self._from_cache(data))

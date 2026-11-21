import warnings

import numpy as np
import pandas as pd
import pyEDM
from scipy.linalg import cho_solve, cholesky
from scipy.spatial import KDTree
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from pyspi.base import (
    Directed, Unsigned, Signed, fill_pairwise, parse_bivariate, parse_multivariate,
)
from pyspi.lib.pairwise_causal import (
    cds_score, igci_score, normalized_hsic, reci_score,
)
from pyspi.utils import require_int


# cdt's ANM kernel, a fixed (untuned) ConstantKernel * RBF, set explicitly
# because scikit-learn's default kernel differs between versions. 1e-10 is
# GaussianProcessRegressor's default `alpha`.
_ANM_KERNEL = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
    1.0, length_scale_bounds="fixed")
_ANM_ALPHA = 1e-10


def _anm_scores(x, ys):
    """
    ``normalized_hsic(residual, x)`` for the GP regression of each y on x.

    The regression is ``GaussianProcessRegressor(kernel=_ANM_KERNEL,
    optimizer=None).fit(x, y).predict(x)``, the posterior mean at the training
    inputs (Rasmussen & Williams 2006, Alg. 2.1), computed with the same
    operations as scikit-learn. The kernel matrix and its Cholesky factor
    depend on the cause x alone, so they are built once and shared by every
    effect. A pair with a constant cause or effect is NaN.
    """
    if np.ptp(x) == 0:
        return [np.nan] * len(ys)
    K = _ANM_KERNEL(x)
    K[np.diag_indices_from(K)] += _ANM_ALPHA
    L = cholesky(K, lower=True, check_finite=False)
    K_trans = _ANM_KERNEL(x, x)
    scores = []
    for y in ys:
        if np.ptp(y) == 0:
            scores.append(np.nan)
            continue
        y_predict = K_trans @ cho_solve((L, True), y, check_finite=False)
        scores.append(normalized_hsic(y_predict.reshape(-1, 1) - y, x))
    return scores


class AdditiveNoiseModel(Directed, Unsigned):
    """
    Additive-noise-model (ANM) score, row (cause) -> column (effect) [1]_.

    ``A[i, j]`` is the HSIC [2]_ between ``x_i`` and the residual of a
    Gaussian-process regression of ``x_j`` on ``x_i``. If
    ``x_j = f(x_i) + noise`` with the noise independent of ``x_i``, the
    residual is independent of the cause, so a lower value is stronger
    evidence for ``i -> j``.

    This is the ``anm_score`` of the Causal Discovery Toolbox (cdt) [3]_,
    reimplemented in `pyspi.lib.pairwise_causal`. The regression is the
    posterior mean of a Gaussian process with a fixed (untuned)
    ``ConstantKernel(1) * RBF(1)`` kernel and ``alpha = 1e-10``. HSIC is
    cdt's biased statistic ``tr(K H L H) / m`` with Gaussian kernels of
    median-heuristic width on the standardised variables, on at most 200
    evenly spaced samples. cdt's ``predict_proba`` reports the difference
    between the two directions' scores; here each direction is its own
    entry.

    Directed and unsigned. Lower is stronger evidence, as for ``cds`` and
    ``reci``; for ``ccm`` and ``igci`` higher means more. NaN for a pair
    involving a constant series.

    References
    ----------
    .. [1] P. O. Hoyer, D. Janzing, J. M. Mooij, J. Peters, B. Schölkopf,
           "Nonlinear causal discovery with additive noise models", Advances
           in Neural Information Processing Systems 21, 689-696 (2008).
    .. [2] A. Gretton, K. Fukumizu, C. H. Teo, L. Song, B. Schölkopf,
           A. J. Smola, "A kernel statistical test of independence", Advances
           in Neural Information Processing Systems 20, 585-592 (2007).
    .. [3] D. Kalainathan, O. Goudet, R. Dutta, "Causal Discovery Toolbox:
           uncovering causal relationships in Python", Journal of Machine
           Learning Research 21(37), 1-5 (2020).
    """

    name = "Additive noise model"
    identifier = "anm"
    # Not `linear`: a Gaussian-process fit and an RBF-kernel HSIC.
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """HSIC between ``x_i`` and the residual of a GP fit of ``x_j`` on ``x_i``."""
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
    """
    Conditional distribution similarity (CDS) score, row -> column [1]_.

    Row = cause, column = effect. Both series are standardised and binned at
    half a standard deviation, clipped at three (13 levels). For every bin of
    ``x_i`` holding more than 12 samples, the values of ``x_j`` in that bin
    are centred, scaled by the standard deviation of ``x_j`` and binned the
    same way; ``A[i, j]`` is the standard deviation of these conditional
    histograms (as probabilities) about their mean across bins. Conditionals
    of similar shape (a low score) are evidence for ``i -> j``. A discrete
    series (13 or fewer distinct values) is used as it is, and each
    conditional histogram of a discrete ``x_j`` is aligned to the marginal by
    the shift that maximises their cross-correlation.

    This is cdt's ``CDS`` [2]_, which is Fonollosa's implementation,
    reimplemented in `pyspi.lib.pairwise_causal`. Directed and unsigned;
    lower is stronger evidence, as for ``anm``. NaN for a pair involving a
    constant series and when fewer than two bins of ``x_i`` hold more than
    12 samples, where cdt returns 0.

    References
    ----------
    .. [1] J. A. R. Fonollosa, "Conditional distribution variability measures
           for causality detection", in Cause Effect Pairs in Machine
           Learning, Springer Series on Challenges in Machine Learning,
           339-347 (2019).
    .. [2] D. Kalainathan, O. Goudet, R. Dutta, "Causal Discovery Toolbox:
           uncovering causal relationships in Python", Journal of Machine
           Learning Research 21(37), 1-5 (2020).
    """

    name = "Conditional distribution similarity statistic"
    identifier = "cds"
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return cds_score(z[i], z[j], min_conditionals=2)


class RegressionErrorCausalInference(Directed, Unsigned):
    """
    Regression-error-based causal inference (RECI), row -> column [1]_.

    Row = cause, column = effect. ``A[i, j]`` is the mean squared error of
    the least-squares fit ``x_j ~ a x_i^3 + b``, with both series first
    min-max scaled to ``[0, 1]``. [1]_ infers ``i -> j`` when this error is
    smaller than that of the reverse fit, so a lower value is stronger
    evidence for ``i -> j``, read against ``A[j, i]``. The cubic monomial fit
    is cdt's [2]_, reimplemented in `pyspi.lib.pairwise_causal`. Directed and
    unsigned; lower is stronger evidence, as for ``anm``.

    References
    ----------
    .. [1] P. Blöbaum, D. Janzing, T. Washio, S. Shimizu, B. Schölkopf,
           "Cause-effect inference by comparing regression errors",
           Proceedings of the 21st International Conference on Artificial
           Intelligence and Statistics, PMLR 84, 900-909 (2018).
    .. [2] D. Kalainathan, O. Goudet, R. Dutta, "Causal Discovery Toolbox:
           uncovering causal relationships in Python", Journal of Machine
           Learning Research 21(37), 1-5 (2020).
    """

    name = "Regression error-based causal inference"
    identifier = "reci"
    labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return reci_score(z[i], z[j])


class InformationGeometricCausalInference(Directed, Signed):
    """
    Information-geometric causal inference (IGCI) [1]_ [2]_.

    ``A[i, j] = H(x_i) - H(x_j)``, the difference between the entropies of
    the two standardised series (IGCI with a Gaussian reference measure),
    each estimated from the gaps between consecutive order statistics (the
    1-D Kozachenko-Leonenko estimator; zero gaps from tied values are
    skipped). If ``x_j = f(x_i)`` for an invertible ``f`` chosen
    independently of the distribution of ``x_i``, the effect tends to have
    the lower entropy, so a positive value favours ``i -> j``. The method
    assumes a deterministic relationship and tests no conditional
    independence. It is cdt's ``IGCI`` [3]_ with the Gaussian reference
    measure and the entropy estimator, reimplemented in
    `pyspi.lib.pairwise_causal`.

    Signed and antisymmetric: ``A[i, j] = -A[j, i]``; positive means i
    causes j. It is in none of the bundled configs; the heuristic is not
    claimed to be reliable.

    References
    ----------
    .. [1] P. Daniušis, D. Janzing, J. M. Mooij, J. Zscheischler, B. Steudel,
           K. Zhang, B. Schölkopf, "Inferring deterministic causal
           relations", Proceedings of the 26th Conference on Uncertainty in
           Artificial Intelligence, 143-150 (2010).
    .. [2] D. Janzing, J. Mooij, K. Zhang, J. Lemeire, J. Zscheischler,
           P. Daniušis, B. Steudel, B. Schölkopf, "Information-geometric
           approach to inferring causal directions", Artificial Intelligence
           182-183, 1-31 (2012).
    .. [3] D. Kalainathan, O. Goudet, R. Dutta, "Causal Discovery Toolbox:
           uncovering causal relationships in Python", Journal of Machine
           Learning Research 21(37), 1-5 (2020).
    """

    name = "Information-geometric causal inference"
    identifier = "igci"
    labels = ["causal", "antisymmetric", "signed", "nonlinear", "unordered"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        return igci_score(z[i], z[j])


class InformationGeometricConditionalIndependence(InformationGeometricCausalInference):
    """
    Deprecated alias of `InformationGeometricCausalInference`.

    The name misdescribes IGCI, which tests no conditional independence. The
    identifier (``igci``) and the values are the same; it warns on use and
    will be removed in a future release.
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
    """
    E in [1, max_e] maximising simplex-projection skill, computed serially.

    The serial equivalent of ``pyEDM.EmbedDimension``: it runs
    ``pyEDM.Simplex`` for each E (as ``PoolFunc.EmbedDimSimplexFunc`` does in
    each child) and selects ``rho.idxmax()``. ``EmbedDimension`` always builds a
    ``multiprocessing.Pool`` (even with ``numProcess=1``), which pyEDM starts
    with forkserver/spawn, so each child re-imports the caller's ``__main__``.

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
    """
    ``nlibs + 1`` library sizes from ``ceil(2E/10)*10`` to ``floor((N-E-1)/10)*10``.

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
    """
    Cross-map skill of one variable's shadow manifold against several targets.

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
    """
    Delay embedding of ``library`` and the targets ``tp`` steps ahead of it.

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
    """
    Simplex cross-map skill of every target for one random library.

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
    """
    ``_cross_map_skill`` at the single library size ``L``, one KD-tree query in all.

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
    """
    Embedding dimension of each process in ``procs`` (default: all), as ints.

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
    """
    CCM skill curves ``(P, P, nlibs + 1)`` of the processes ``z`` (P, N).

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
    """
    Last finite skill minus first finite skill along the library axis.

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
    """
    OLS slope of skill on ``ln L`` over the library sizes where both are finite.

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
    """
    Convergent cross-mapping skill, row (source) -> column (target) [1]_.

    ``A[i, j]`` is read off the skill with which the shadow manifold (delay
    embedding) of ``x_j`` recovers ``x_i``, which [1]_ reads as ``i`` driving
    ``j``; higher means more (the opposite polarity to ``anm``, ``cds`` and
    ``reci``). That skill is computed once per Data and embedding dimension
    ``E`` as a curve ``rho_ij(L)`` over up to 22 library sizes ``L`` from
    ``ceil(2E/10)*10`` to ``floor((N-E-1)/10)*10``, each the mean Pearson
    correlation over 100 random libraries (seed 42) of simplex cross-map
    estimates (``E + 1`` neighbours, embedding delay of one sample), and
    cached on ``data.ccm[E]``. Each skill equals pyEDM 2.5's ``CCM`` at the
    same ``E``; pyEDM's process pools are not used. Every ``statistic`` is
    read off that curve.

    With ``embedding_dimension=None`` each process gets its own
    simplex-optimal dimension (``E = 1..10``), and each pair uses the larger
    of its two in both directions ([1]_ uses the library variable's own), so
    the two directions are computed on manifolds of the same dimension.

    ``convergence`` and ``slope`` are NaN with fewer than two finite library
    sizes. A link strong enough to synchronise the pair saturates the skill
    from the smallest library on and so barely converges, and when both
    directions are causal the stronger one can converge less. With a delay
    of one sample, a weakly autocorrelated source acting at a longer lag can
    leave the skill near zero.

    The "i drives j" reading holds for deterministic dynamics. Under
    stochastic lagged coupling it can reverse: if ``x_i`` drives ``x_j``
    with a delay, ``x_i``'s delay vector predicts ``x_j``, so the skill
    appears in the ``j -> i`` cell. On such data compare the ``tp`` of
    `LaggedConvergentCrossMapping`, the time-delayed variant, in the two
    cells instead.

    ``bivariate`` reads the cached table when the Data has one and otherwise
    computes the pair alone.

    References
    ----------
    .. [1] G. Sugihara, R. May, H. Ye, C.-H. Hsieh, E. Deyle, M. Fogarty,
           S. Munch, "Detecting causality in complex ecosystems", Science
           338(6106), 496-500 (2012).

    Parameters
    ----------
    statistic : str, optional
        What is read off the skill curve ``rho_ij(L)``: ``'mean'`` or
        ``'max'`` over library sizes; ``'diff'``, ``ccm(i->j) - ccm(j->i)``
        averaged over library sizes, so ``A[i, j] = -A[j, i]`` and positive
        means i drives j; ``'convergence'``, ``rho_ij(L_max) - rho_ij(L_min)``
        (the last finite entry minus the first; skill that grows with ``L``
        is the criterion for causation in [1]_, and a negative value means
        skill falls with ``L``); or ``'slope'``, the least-squares slope of
        ``rho_ij`` against ``ln L``, in skill per e-fold of library size.
        Default: ``'mean'``.
    embedding_dimension : int or None, optional
        Embedding dimension ``E`` (at least 1). None gives each process its
        own simplex-optimal dimension, as above. Default: None.
    """

    name = "Convergent cross-mapping"
    identifier = "ccm"
    labels = ["causal", "directed", "nonlinear", "temporal", "signed"]
    # Variants share data.ccm[E], so they run on one parallel worker.
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

        # diff is ccm(i->j) - ccm(j->i): antisymmetric rather than directed.
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
            # Each pair uses the larger of its two dimensions in both directions.
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
    """
    Cross-map skill by prediction horizon, ``(P, P, 2 * max_lag + 1)``.

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
    """
    Lagged convergent cross-mapping, row (source) -> column (target) [1]_.

    [1]_ extends CCM with a cross-map lag: the shadow manifold of the library
    variable at time t estimates the other variable at time ``t + Tp``. For
    each ordered pair and each ``Tp`` in ``-k..k`` this computes
    ``rho_ij(Tp)``, the skill with which the shadow manifold of ``x_j`` at
    time t recovers ``x_i(t + Tp)``, at the largest library size only
    (``L = floor((N-E-1)/10)*10``, 100 random libraries, seed 42, as in
    `ConvergentCrossMapping`; the ``Tp = 0`` value equals the last point of
    the ``ccm`` curve). The ``(M, M, 2k + 1)`` array is cached on
    ``data.ccm[("lagged", E, k)]``. Rows whose shifted target falls outside
    the series are dropped, as with pyEDM's ``Tp``, so a horizon ``Tp``
    costs at most ``|Tp|`` points.

    Orientation is as in ``ccm``: ``A[i, j]`` (``i -> j``) uses the manifold
    of ``x_j`` cross-mapping ``x_i``. If i reaches j after a delay
    ``d >= 1``, the manifold at time t best recovers ``x_i(t - d)`` and
    ``rho_ij(Tp)`` peaks at ``Tp = -d < 0``; skill in the non-causal
    direction comes from generalised synchrony under strong forcing, where
    the peak sits at ``Tp >= 0``. Unlike ``ccm``, which evaluates every
    library size at ``Tp = 0``, ``lccm`` reads direction from when skill
    peaks, so it can separate a delayed causal link from synchrony; it does
    not test convergence.

    References
    ----------
    .. [1] H. Ye, E. R. Deyle, L. J. Gilarranz, G. Sugihara, "Distinguishing
           time-delayed causal interactions using convergent cross mapping",
           Scientific Reports 5, 14750 (2015).

    Parameters
    ----------
    statistic : str, optional
        ``'tp'``: the ``Tp`` of maximal skill, an integer in ``[-k, k]``
        (ties go to the most negative). A negative ``A[i, j]`` is the
        signature of ``i -> j``; its magnitude estimates ``d`` only roughly
        (the peak flattens for larger E). For an unlinked pair the argmax is
        noise, so read ``tp`` together with ``skill`` or
        ``ccm_*_convergence``. Under noisy lagged coupling the reverse cell's
        ``tp`` is unreliable, and a hidden common driver makes both cells
        negative, so compare ``A[i, j]`` with ``A[j, i]`` (the more negative
        marks the driving direction) rather than reading one cell's sign.
        ``'skill'``: ``max over Tp of rho_ij(Tp)``, signed like ``ccm``.
        Default: ``'tp'``.
    embedding_dimension : int or None, optional
        As in `ConvergentCrossMapping`: None selects each process's
        simplex-optimal ``E``, and a pair uses the larger of its two in both
        directions. Default: None.
    max_lag : int, optional
        ``k``, the largest horizon examined, in samples (at least 1).
        Default: 5.
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


def _nli_embed(x, m, tau):
    """
    Delay vectors ``(x_n, x_{n-tau}, ..., x_{n-(m-1)tau})``, one per row.

    Rows run over ``n = (m-1)tau, ..., T-1``, so every process of a dataset
    gives the same ``N = T - (m-1)tau`` time indices and row ``n`` of two
    processes' embeddings is the same instant.
    """
    T = x.shape[0]
    span = (m - 1) * tau
    return np.ascontiguousarray(
        np.column_stack([x[span - l * tau:T - l * tau] for l in range(m)]))


def _nli_neighbours(emb, k, theiler):
    """
    Time indices of the ``k`` nearest neighbours of every delay vector.

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
    """
    Mean squared distance from each ``x_n`` to every admissible ``x_j``.

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
    """
    Rank of ``|x_n - x_{queries[n, l]}|`` among ``x_n``'s admissible distances.

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
    """
    Nonlinear interdependence S, H, N or L [1]_ [2]_ [3]_ [4]_.

    State-space measures of whether close states of one process occur at the
    same times as close states of another. Each process is delay-embedded,
    ``x_n = (x_n, x_{n-tau}, ..., x_{n-(m-1)tau})`` for ``n = 1..N`` with
    ``N = T - (m-1)tau``. Let ``r_{n,l}`` and ``s_{n,l}`` (``l = 1..k``) be
    the time indices of the ``k`` nearest neighbours of ``x_n`` and of
    ``y_n`` (Euclidean distance), with temporal neighbours
    ``|n - r| <= theiler`` excluded. Then

    * ``R_n^k(X)   = (1/k) sum_l |x_n - x_{r_{n,l}}|^2``  (own neighbours),
    * ``R_n^k(X|Y) = (1/k) sum_l |x_n - x_{s_{n,l}}|^2``  (Y's neighbours'
      time indices, measured in X),
    * ``R_n(X)``, the mean of ``|x_n - x_j|^2`` over every admissible ``j``,

    and, averaging over ``n``,

    * ``S(X|Y) = <R_n^k(X) / R_n^k(X|Y)>`` [1]_, in ``[0, 1]``;
    * ``H(X|Y) = <log(R_n(X) / R_n^k(X|Y))>`` [1]_, in nats, about 0 for
      independent processes;
    * ``N(X|Y) = <(R_n(X) - R_n^k(X|Y)) / R_n(X)>`` [3]_, at most 1, about 0
      for independent processes;
    * ``L(X|Y) = <(G_n(X) - G_n^k(X|Y)) / (G_n(X) - G_n^k(X))>`` ([4]_,
      eq. 4), at most 1, about 0 for independent processes.
      ``G_n^k(X|Y)`` is the mean rank of the ``k`` distances
      ``|x_n - x_{s_{n,l}}|`` among all of ``x_n``'s admissible distances
      (1 = nearest; ties take their average rank), ``G_n^k(X) = (k+1)/2`` is
      the smallest possible mean rank and ``G_n(X) = (N_n + 1)/2`` the mean
      rank, ``N_n`` being the number of admissible ``j`` for ``x_n``.

    With ``theiler = W`` every average over ``j`` (the neighbour searches,
    ``R_n(X)`` and the ranks) runs over ``|n - j| > W`` only, so ``N_n =
    N - 1 - #{j != n: |n - j| <= W}`` (``N - 2W - 1`` away from the ends), as
    in [4]_, note 24.

    Direction: ``A[i, j] = A(x_i | x_j)`` (``A`` one of S, H, N, L). The row
    is the conditioned process X, whose distances are measured, and the
    column is the conditioning process Y, whose neighbourhoods are mapped
    onto X. The entry is large when close states of ``x_j`` are mapped to
    close states of ``x_i``, which [4]_ reads as a coupling ``x_i -> x_j``
    (as in convergent cross-mapping). So with row = source,
    ``A[i, j] > A[j, i]`` points to ``i -> j``. Unsigned.

    Both entries of a coupled pair rise, so only the asymmetry indicates
    direction, and it is unreliable near generalised synchronisation. S and H
    depend on the effective dimension of each process, so their asymmetry
    can point the wrong way [2]_; L, which uses only ranks, is the most
    reliable for direction [4]_.

    S and H divide by ``R_n^k(X|Y)``, which is zero when every conditioned
    neighbour of some ``x_n`` coincides with it (repeated delay vectors, as
    in quantised data). That pair raises, and is NaN in the multivariate
    table.

    References
    ----------
    .. [1] J. Arnhold, P. Grassberger, K. Lehnertz, C. E. Elger, "A robust
           method for detecting interdependences: application to
           intracranially recorded EEG", Physica D 134(4), 419-430 (1999).
    .. [2] R. Quian Quiroga, J. Arnhold, P. Grassberger, "Learning
           driver-response relationships from synchronization patterns",
           Physical Review E 61(5), 5142-5148 (2000).
    .. [3] R. Quian Quiroga, A. Kraskov, T. Kreuz, P. Grassberger,
           "Performance of different synchronization measures in real data:
           a case study on electroencephalographic signals", Physical Review
           E 65(4), 041903 (2002).
    .. [4] D. Chicharro, R. G. Andrzejak, "Reliable detection of directional
           couplings using rank statistics", Physical Review E 80(2), 026217
           (2009).

    Parameters
    ----------
    statistic : str, optional
        ``'S'``, ``'H'``, ``'N'`` or ``'L'``. Default: ``'L'``.
    m : int, optional
        Embedding dimension (at least 1). Default: 5.
    tau : int, optional
        Embedding delay, in samples (at least 1). Default: 1.
    k : int, optional
        Number of nearest neighbours (at least 1). Default: 5.
    theiler : int or None, optional
        Theiler window ``W``, in samples: neighbours with ``|n - j| <= W``
        are excluded. None uses ``m * tau``, which excludes every pair of
        delay vectors sharing a sample; a value given explicitly is appended
        to the identifier. Needs ``N = T - (m-1)tau > k + 2W + 1``.
        Default: None.
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
        """
        Fill ``cache["ranks"][(i, j)] = G_n^k(x_i | x_j)`` for each j given.

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
        """
        Every ordered pair, from per-process embeddings and neighbours.

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


def _liang_moments(z, k):
    """
    The sample covariances every Liang-Kleeman estimator is built from.

    With the Euler forward difference at a step of one sample,
    ``dx_b[n] = (x_b[n + k] - x_b[n]) / k`` (Liang 2014; Liang 2021 eq. 7),
    defined for the first ``n = T - k`` samples, and every series cut to those
    same samples (as in Liang's reference code), returns read-only

    * ``C[a, b]  = cov(x_a, x_b)``   -- Liang's ``C_ab`` (2021 eq. 12),
    * ``Cd[a, b] = cov(x_a, dx_b)``  -- Liang's ``C_{a,db}`` (2021 eq. 13),
    * ``Vd[b]    = var(dx_b)``       -- Liang's ``C_{db,db}`` (2015 eq. 28).

    All three use the same ``1/(n - 1)`` normalisation, which cancels in every
    Liang quantity.
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    n = z.shape[1] - k
    if n < 3:
        raise ValueError(
            f"Liang-Kleeman information flow with k={k} needs at least k + 3 = "
            f"{k + 3} observations; got {z.shape[1]}.")
    x = z[:, :n]
    dx = (z[:, k:] - z[:, :n]) / k
    xc = x - x.mean(axis=1, keepdims=True)
    dxc = dx - dx.mean(axis=1, keepdims=True)
    C = xc @ xc.T / (n - 1)
    Cd = xc @ dxc.T / (n - 1)
    Vd = np.einsum("bt,bt->b", dxc, dxc) / (n - 1)
    for arr in (C, Cd, Vd):
        arr.setflags(write=False)
    return C, Cd, Vd


class LiangInformationFlow(Directed, Signed):
    r"""
    Liang-Kleeman information flow, row (source) -> column (target) [1]_.

    ``A[j, i]`` is ``T_{j->i}``, the rate at which the source ``x_j`` transfers
    information (entropy) to the target ``x_i``, in nats per sample, estimated
    under a linear stochastic model ``dx = (f + A x) dt + B dW`` with diagonal
    ``B`` [1]_ [3]_. Positive means ``x_j`` makes ``x_i`` more uncertain,
    negative means it reduces ``x_i``'s uncertainty (stabilises it), and zero
    means no causal influence (Liang's principle of nil causality, a theorem
    of the formalism: ``T_{j->i} = 0`` whenever ``x_i``'s dynamics do not
    depend on ``x_j``). Equation numbers below refer to [2]_ (2015) and
    [3]_ (2021).

    Estimators. With the Euler forward difference
    ``dx_i[n] = (x_i[n + k] - x_i[n]) / k`` (one sample is the time unit),
    ``C_ab = cov(x_a, x_b)``, ``C_{a,di} = cov(x_a, dx_i)`` and
    ``C_{di,di} = var(dx_i)``, all over the ``T - k`` samples that have a
    difference (``_liang_moments``):

    * ``conditional=False``: the bivariate maximum-likelihood estimator of
      [1]_ for each pair on its own (2015 eq. 3; 2021 eq. 5), with 1 the
      target and 2 the source,
      ``T_{2->1} = (C11 C12 C_{2,d1} - C12^2 C_{1,d1}) / D`` with
      ``D = C11^2 C22 - C11 C12^2``.

    * ``conditional=True``: the multivariate estimator with all ``M`` series
      as the system (2021 eq. 14),
      ``T_{j->i} = (1 / det C) sum_k Delta_jk C_{k,di} C_ij / C_ii``, which is
      ``a_ij C_ij / C_ii``, where ``Delta_jk`` are the cofactors of ``C`` and
      ``a_ij = sum_k (C^-1)_jk C_{k,di}`` (eq. 9;
      ``Delta_jk / det C = (C^-1)_kj``) is the coefficient of ``x_j`` in the
      least-squares regression of ``dx_i`` on all ``M`` series, computed with
      one ``np.linalg.solve``. For ``M = 2`` it is the bivariate estimator
      exactly. Conditioning on the other series removes indirect flows: in a
      chain ``x0 -> x1 -> x2`` the conditional ``T_{0->2}`` vanishes, the
      bivariate one does not.

    ``statistic='normalised'`` reports the relative flow of [2]_,
    ``tau_{j->i} = T_{j->i} / Z_i`` in ``[-1, 1]``, with the target's entropy
    balance as the normaliser (2015 eqs. 10-11 and 29-32; 2021 eqs. 17-20),
    ``Z_i = |dH_i*/dt| + sum_{j != i} |T_{j->i}| + |dH_i^noise/dt|``.
    Here ``dH_i*/dt = a_ii`` (2015 eq. 22/29; 2021 eq. 15),
    ``dH_i^noise/dt = g_ii / (2 C_ii)`` with ``g_ii = k Q_i / N`` (2015 eqs.
    24, 30), and ``Q_i / N = C_{di,di} - sum_j a_ij C_{j,di}`` (2015 eq. 28) is
    the residual variance of the regression of ``dx_i`` on the series in the
    fit (the pair, or all ``M``). The sum over sources runs over the pair's one
    source in the bivariate version [2]_ and over all ``M - 1`` others in the
    conditional one [3]_. ``|tau_{j->i}|`` is the share of the target's
    entropy budget due to ``x_j``, so ``tau`` is normalised within each target
    and ``|tau_{j->i}|`` and ``|tau_{i->j}|`` are not on one scale ([2]_,
    sec. VI). Liang's reference code mixes ``1/N`` (noise) and ``1/(N - 1)``
    (covariance) normalisations, so the normalised ``tau`` differs from it by
    a relative O(1/N).

    Units and sampling. The rate is per sample: for a continuous-time process
    sampled every ``h`` time units it is about ``h`` times the per-unit-time
    flow. ``k = 1`` is the usual choice, and Liang recommends ``k = 2`` for
    densely sampled chaotic series. ``g_ii`` uses ``k Delta t`` (the interval
    the difference spans) so that it estimates the diffusion coefficient
    whatever ``k``. The statistic is invariant under rescaling and shifting of
    each series. No significance test is applied (the confidence intervals of
    [1]_ are not implemented).

    References
    ----------
    .. [1] X. S. Liang, "Unraveling the cause-effect relation between time
           series", Physical Review E 90(5), 052150 (2014).
    .. [2] X. S. Liang, "Normalizing the causality between time series",
           Physical Review E 92(2), 022126 (2015).
    .. [3] X. S. Liang, "Normalized multivariate time series causality
           analysis and causal graph reconstruction", Entropy 23(6), 679
           (2021).

    Parameters
    ----------
    k : int, optional
        Stride of the forward difference, in samples (at least 1).
        Default: 1.
    conditional : bool, optional
        Use the multivariate estimator, conditioning on all other series
        (identifier stem ``liang-cond``). It needs a nonsingular ``M x M``
        covariance (``M < T - k``, no series an exact linear combination of
        the others). Default: False.
    statistic : str, optional
        ``'rate'`` (nats per sample) or ``'normalised'`` (``tau``, in
        ``[-1, 1]``). Default: ``'rate'``.
    """

    name = "Liang-Kleeman information flow"
    identifier = "liang"
    labels = ["causal", "directed", "linear", "temporal", "signed"]
    # All four variants read the same covariances, cached per stride k.
    _cache_namespace = "liang"

    _STATISTICS = {"rate": "rate", "normalised": "norm"}

    @property
    def _cache_subkey(self):
        return (self._k,)

    def __init__(self, k=1, conditional=False, statistic="rate"):
        self._k = require_int("k", k, minimum=1)
        if not isinstance(conditional, (bool, np.bool_)):
            raise TypeError(
                f"conditional must be a bool, got {conditional!r} "
                f"({type(conditional).__name__}).")
        if statistic not in self._STATISTICS:
            raise ValueError(
                f"Unknown statistic {statistic!r}; expected one of "
                f"{sorted(self._STATISTICS)}.")
        self._conditional = bool(conditional)
        self._statistic = statistic
        stem = "liang-cond" if self._conditional else "liang"
        self.identifier = f"{stem}_{self._STATISTICS[statistic]}_k-{self._k}"

    def _from_cache(self, data):
        try:
            moments = data.liang[self._k]
        except (AttributeError, KeyError):
            moments = _liang_moments(data.to_numpy(squeeze=True), self._k)
            try:
                data.liang[self._k] = moments
            except AttributeError:
                data.liang = {self._k: moments}
        return moments

    def _bivariate_flows(self, C, Cd, Vd):
        """
        Liang (2014, 2015) for every ordered pair at once; ``[j, i]`` is j -> i.

        For the pair (target i = "1", source j = "2"): ``a_self`` is
        ``a_11`` (2015 eq. 22), ``a_src`` is ``a_12`` (eq. 23), and
        ``a_src * C_ij / C_ii`` is eq. 3 rewritten.
        """
        var = np.diag(C)
        Cii = var[np.newaxis, :]          # target variance, by column
        Cjj = var[:, np.newaxis]          # source variance, by row
        Ci_di = np.diag(Cd)[np.newaxis, :]  # cov(x_i, dx_i), by column
        with np.errstate(divide="ignore", invalid="ignore"):
            det = Cii * Cjj - C ** 2
            # 1 - r^2 at rounding level: the pair is collinear, so undefined.
            det = np.where(det > 1e3 * np.finfo(float).eps * Cii * Cjj,
                           det, np.nan)
            a_self = (Cjj * Ci_di - C * Cd) / det
            a_src = (Cii * Cd - C * Ci_di) / det
            rate = a_src * C / Cii
            if self._statistic == "rate":
                return rate
            resvar = Vd[np.newaxis, :] - (a_self * Ci_di + a_src * Cd)
            noise = self._k * resvar / (2.0 * Cii)
            return rate / (np.abs(rate) + np.abs(a_self) + np.abs(noise))

    def _conditional_flows(self, C, Cd, Vd):
        """Liang (2021) with all M series as the system; ``[j, i]`` is j -> i."""
        M = C.shape[0]
        rank = np.linalg.matrix_rank(C)
        if rank < M:
            raise ValueError(
                f"Conditional Liang-Kleeman flow needs a nonsingular covariance "
                f"of all {M} series, but its rank is {rank}: too few "
                f"observations, or a series that is a linear combination of "
                f"the others. Use conditional=False.")
        # coef[j, i] = a_ij, the coefficient of x_j in dx_i's regression
        # (2021 eq. 9). The diagonal of `rate` is then a_ii = dH_i*/dt (eq. 15).
        coef = np.linalg.solve(C, Cd)
        var = np.diag(C)
        rate = coef * C / var[np.newaxis, :]
        if self._statistic == "rate":
            return rate
        resvar = Vd - np.einsum("ji,ji->i", coef, Cd)
        noise = self._k * resvar / (2.0 * var)
        # |a_ii| + sum_{j != i} |T_{j->i}| is the column sum of |rate|.
        Z = np.abs(rate).sum(axis=0) + np.abs(noise)
        return rate / Z[np.newaxis, :]

    def _matrix(self, data):
        """The full ``M x M`` table, diagonal and undefined pairs NaN."""
        C, Cd, Vd = self._from_cache(data)
        if self._conditional:
            A = self._conditional_flows(C, Cd, Vd)
        else:
            A = self._bivariate_flows(C, Cd, Vd)
        A = np.array(A, dtype=np.float64)
        A[~np.isfinite(A)] = np.nan
        np.fill_diagonal(A, np.nan)
        return A

    @parse_multivariate
    def multivariate(self, data):
        A = self._matrix(data)
        # Collinear or constant series give NaN pairs; report them as
        # `fill_pairwise` reports a failed pair.
        off = ~np.eye(A.shape[0], dtype=bool)
        bad = off & np.isnan(A)
        if bad.any():
            if bad.sum() == off.sum():
                raise ValueError(
                    f'SPI "{self.identifier}": every pair is undefined '
                    f"(singular pairwise covariances).")
            i, j = np.argwhere(bad)[0]
            warnings.warn(
                f'SPI "{self.identifier}": {int(bad.sum())} of '
                f"{int(off.sum())} pairs are undefined and set to NaN; first "
                f"({i}, {j}): singular pairwise covariance (collinear or "
                f"constant series).")
        return A

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """
        ``T_{i->j}`` (or ``tau_{i->j}``): i is the source, j the target.

        Read from the full table, which costs ``O(M^2)`` (``O(M^3)`` when
        conditional) once the cached covariances exist. Raises, rather than
        returning NaN, when the requested pair is undefined.
        """
        if i == j:
            return np.nan
        value = self._matrix(data)[i, j]
        if np.isnan(value):
            raise ValueError(
                f'SPI "{self.identifier}": pair ({i}, {j}) is undefined '
                f"(singular covariance: collinear or constant series).")
        return float(value)

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


def _liang_moments(z, k):
    """The sample covariances every Liang-Kleeman estimator is built from.

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
    r"""Liang-Kleeman rate of information flow, row (source) -> column (target).

    ``A[j, i]`` is ``T_{j->i}``, the rate at which the source ``x_j`` transfers
    information (entropy) to the target ``x_i``, in nats per sample, estimated
    under a linear stochastic model ``dx = (f + A x) dt + B dW`` with diagonal
    ``B`` (Liang 2014, 2021). Positive means ``x_j`` makes ``x_i`` *more*
    uncertain, negative means it *reduces* ``x_i``'s uncertainty (stabilises
    it), and zero means no causal influence (Liang's principle of nil
    causality, a theorem of the formalism: ``T_{j->i} = 0`` whenever ``x_i``'s
    dynamics do not depend on ``x_j``).

    **Estimators.** With the Euler forward difference
    ``dx_i[n] = (x_i[n + k] - x_i[n]) / k`` (one sample is the time unit),
    ``C_ab = cov(x_a, x_b)``, ``C_{a,di} = cov(x_a, dx_i)`` and
    ``C_{di,di} = var(dx_i)``, all over the ``T - k`` samples that have a
    difference (``_liang_moments``):

    * ``conditional=False``: Liang's (2014) bivariate maximum-likelihood
      estimator for each pair on its own (Liang 2015 eq. 3; Liang 2021 eq. 5),
      with 1 the target and 2 the source::

          T_{2->1} = (C11 C12 C_{2,d1} - C12^2 C_{1,d1})
                     / (C11^2 C22 - C11 C12^2)

    * ``conditional=True``: the multivariate estimator with all ``M`` series
      as the system (Liang 2021 eq. 14)::

          T_{j->i} = (1 / det C) sum_k Delta_jk C_{k,di} * C_ij / C_ii
                   = a_ij C_ij / C_ii,

      where ``Delta_jk`` are the cofactors of ``C`` and
      ``a_ij = sum_k (C^-1)_jk C_{k,di}`` (eq. 9; ``Delta_jk / det C =
      (C^-1)_kj``) is the coefficient of ``x_j`` in the least-squares
      regression of ``dx_i`` on all ``M`` series, computed with one
      ``np.linalg.solve``. For ``M = 2`` it is the bivariate estimator
      exactly. Conditioning on the other series removes indirect flows: in a
      chain ``x0 -> x1 -> x2`` the conditional ``T_{0->2}`` vanishes, the
      bivariate one does not.

    ``statistic="normalised"`` reports Liang's (2015) relative flow
    ``tau_{j->i} = T_{j->i} / Z_i`` in ``[-1, 1]``, with the target's entropy
    balance as the normaliser (Liang 2015 eqs. 10-11 and 29-32; Liang 2021
    eqs. 17-20)::

        Z_i = |dH_i*/dt| + sum_{j != i} |T_{j->i}| + |dH_i^noise/dt|,
        dH_i*/dt     = a_ii                          (2015 eq. 22/29; 2021 eq. 15)
        dH_i^noise/dt = g_ii / (2 C_ii),  g_ii = k * Q_i / N   (2015 eqs. 24, 30)
        Q_i / N      = C_{di,di} - sum_j a_ij C_{j,di}         (2015 eq. 28)

    ``Q_i / N`` is the residual variance of the regression of ``dx_i`` on the
    series in the fit (the pair, or all ``M``). The sum over sources runs over
    the pair's one source in the bivariate version (Liang 2015) and over all
    ``M - 1`` others in the conditional one (Liang 2021). ``|tau_{j->i}|`` is
    the share of the target's entropy budget due to ``x_j``, so ``tau`` is
    normalised within each target and ``|tau_{j->i}|`` and ``|tau_{i->j}|`` are
    not on one scale (Liang 2015, sec. VI). Liang's reference code mixes
    ``1/N`` (noise) and ``1/(N - 1)`` (covariance) normalisations, so the
    normalised ``tau`` differs from it by a relative O(1/N).

    **Units and sampling.** The rate is per sample: for a continuous-time
    process sampled every ``h`` time units it is about ``h`` times the
    per-unit-time flow. ``k`` sets the difference's stride; ``k = 1`` is the
    usual choice and Liang recommends ``k = 2`` for densely sampled chaotic
    series. ``g_ii`` uses ``k Delta t`` (the interval the difference spans) so
    that it estimates the diffusion coefficient whatever ``k``. The statistic
    is invariant under rescaling and shifting of each series.

    No significance test is applied (the confidence intervals of Liang 2014
    are not implemented).

    Parameters
    ----------
    k : int, default 1
        Stride of the Euler forward difference, in samples.
    conditional : bool, default False
        ``False``: Liang (2014)'s bivariate estimator for each pair.
        ``True``: Liang (2021)'s multivariate estimator, all series at once;
        requires a nonsingular ``M x M`` covariance (``M < T - k``, no series
        an exact linear combination of the others).
    statistic : {"rate", "normalised"}, default "rate"
        The flow ``T`` (nats per sample) or the normalised flow ``tau``.

    References
    ----------
    Liang, X. S. (2014). Unraveling the cause-effect relation between time
    series. *Phys. Rev. E* 90, 052150.
    Liang, X. S. (2015). Normalizing the causality between time series.
    *Phys. Rev. E* 92, 022126 (arXiv:1501.03548; equation numbers above).
    Liang, X. S. (2021). Normalized multivariate time series causality
    analysis and causal graph reconstruction. *Entropy* 23, 679
    (arXiv:2104.11360; equation numbers above).
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
        """Liang (2014, 2015) for every ordered pair at once; ``[j, i]`` is j -> i.

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
        """``T_{i->j}`` (or ``tau_{i->j}``): i is the source, j the target.

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

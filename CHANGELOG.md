# Changelog

## 3.0.0

A major overhaul. The Java/JIDT dependency is gone, the `Calculator` API is simplified, and several long-standing correctness bugs are fixed. **This release contains breaking changes** — see [Migrating from 2.x](#migrating-from-2x).

### Removed: Java and JIDT

Every information-theoretic estimator is now pure NumPy. `infodynamics.jar`, `jpype` and the JVM startup path are gone, so installing pyspi no longer requires a Java runtime.

The port was validated against JIDT 1.6.1 before the dependency was dropped. That work caught four bugs in the port (kernel-entropy normalisation, Theiler-windowed KSG neighbour counting, Gaussian auto-embed bias, and KSG auto-embed estimator consistency), all fixed. Mean absolute error against JIDT at T=1600:

| estimator | MI | TE | entropy |
|:----------|---:|---:|--------:|
| gaussian | 5.9e-17 | 4.7e-16 | 5.0e-09 |
| kernel | 6.8e-16 | 5.0e-05 | 3.1e-15 |
| symbolic | — | 2.4e-16 | — |
| kozachenko | — | — | 4.2e-05 |
| kraskov | 1.8e-03 | 2.5e-03 | — |

Gaussian/kernel MI, kernel entropy and symbolic TE agree to machine precision. The gaussian-entropy offset is a deterministic ridge term (the analogue of JIDT's stochastic `NOISE_LEVEL_TO_ADD`).

The k-NN rows are historical measurements under different input policies, not a parity claim: the harness used JIDT's default normalisation and random `1e-8` dither, whereas pyspi standardises continuous inputs and refuses tied coordinates. No convergence rate is inferred from this comparison; KSG bias depends on k, dimensionality and density smoothness (Gao, Oh & Viswanath 2018, *Demystifying fixed k-nearest neighbor information estimators*). The harness was removed with the JIDT dependency.

### Fixed: state, identity, and estimator contracts

A second pass, driven by a suite of red tests, closed a set of defects that produced valid-looking but wrong results. Full details in each commit; the scientifically material ones:

- **Stale caches survived data mutation.** Statistics cache results directly on the `Data` instance keyed by their own parameters, never by the data. Nothing invalidated them, so `set_data`/`add_process`/`remove_process` left every cached statistic serving the *previous* dataset's numbers, silently. All 15 cache attributes are now registered and dropped on mutation, and `Data` copies and freezes its input so nothing can mutate the series behind a cache.

- **Checkpoints were not bound to a run.** Resume validated only the SPI identifier and an `(M, M)` shape, so a different dataset, config, preprocessing setting, or process order silently inherited the earlier run's results. Checkpoints now carry a `run.json` manifest bound to `Calculator.run_digest` (config *contents*, dataset bytes in native dtype, and a computation-version token). Failed or non-finite checkpoints are retried by default, workers load the parent's config snapshot rather than rereading a path that may have changed, and a directory belonging to another run is refused rather than emptied.

- **Spectral caches ignored `fs`,** and were written under a `str` key but read under a `tuple` key — so the first write was unreachable and staleness only appeared from the *third* call, which is why a two-call probe found nothing.

- **Six measures advertised `kraskov` and ran Gaussian.** Joint/conditional/ crossmap/causal entropy, directed info and stochastic interaction are composed from marginal entropies and have no KSG estimator; the argument is now rejected. No bundled config used it, so no shipped result changed.

- **Cointegration `aeg` was forcibly symmetric.** It is not symmetric in its arguments (~0.8 mean absolute difference between orientations, up to ~1.6), yet the cache wrote each value to both `(i,j)` and `(j,i)`, so the reported value depended on process order. `aeg` is now `directed`; `johansen`, which is symmetric to ~3e-14, is unchanged.

- **KSG accepted inputs it cannot estimate from.** `k=30` on `N=20` returned 0.414 and `k=100` returned 1.63; negative Theiler windows were accepted. The effective-sample and window checks now guard the MI, TE and auto-embedding paths, and the auto-embedding search skips candidates it cannot support instead of ranking them and failing on the winner. Tied and quantised coordinates are explicitly rejected and directed to an external discrete estimator; pyspi has no general discrete MI/TLMI/DI plug-in estimator.

- **KSG lost JIDT's normalisation and tie policy.** Every KSG-family calculator in JIDT enables `normalise` and a small random dither by default. pyspi 2.x ran with both active, while the pure-NumPy port initially implemented neither. That is not a rounding difference:

  - **Ties.** Independent binary marginals (N=400, k=4) returned MI = **−3.35**; four-level, −1.96; one-decimal-rounded Gaussians, **+0.90** — a confident false positive on independent data. Every kth-nearest-neighbour radius is zero, so the digamma counts saturate on the tie structure and the estimate measures quantisation. Every process in the bundled `forex` dataset is tied: 24–212 distinct values in 250 samples.
  - **Scale.** KSG's L∞ radius is not invariant to per-coordinate rescaling. With `zscore=False`, scaling one of a correlated Gaussian pair (true MI 0.50) by 1e−3 or 1e3 collapsed the estimate from 0.49 to 0.05 and 0.06.

  Per-coordinate sample standardisation is restored for MI, TLMI, TE, CMI, AIS auto-embedding and directed information. Random dither is not: an unexposed noise draw changes a finite-sample estimate, while content-derived pseudo-random dither was found to depend on observation order. The continuous KSG estimator now requires every coordinate to be tie-free and directs quantised/discrete data to an external discrete plug-in estimator. This means every process in bundled `forex` is invalid for the `kraskov` variants. **Every `kraskov` SPI's values can change.**

  Kozachenko entropy likewise keeps its explicit tied-data error: `H(X + εξ) → −∞` as `ε → 0` for discrete `X`, so a dithered differential entropy on quantised data reports the dither level.

- **Symbolic TE packed symbols into an integer that overflowed** at `k_history=10` (reaching `(k!)^3`). Now counts distinct rows directly, validated against a tuple-keyed reference. `k_history=1` is rejected: a length-1 ordinal pattern has one symbol, so TE is identically zero. Those were the only two symbolic variants shipped, so **symbolic TE now has no bundled representation** — reintroducing it needs a defensible `k` with benchmark support. `k_history=10` remains constructible for long series, where the undersampling argument does not apply.

- **Results tables no longer use pickle.** Names are stored as `dtype='U'` and loaded with `allow_pickle=False`; files carry a schema version, run spec, digest and errors. Tables written by pyspi < 3.0.0 will not load — re-save them from a `Calculator`.

- **`load_table()` dropped everything `save()` wrote except the numbers.** The run spec, digest and error map were written and never read, so a loaded table could not be asked which SPIs failed — and a NaN column is otherwise indistinguishable from a legitimately undefined statistic. They now arrive in `DataFrame.attrs` (`schema`, `run_spec`, `run_digest`, `errors`). Validation also covered only `ndim` and the SPI axis; a file whose matrices were the wrong width reached `MultiIndex.from_product` and failed there with a reshape error rather than a statement about the file. The full `(n_spis, M, M)` shape is now checked.

- **`run_digest` did not bind to the algorithm.** It hashed the run spec, the config contents and the dataset bytes, so identical inputs computed by two different estimator implementations produced the same digest and a checkpoint written by one could be resumed by the other. It now covers the computation version too.

- **A successful recomputation left a stale error behind.** `compute(retry_failed=True)` on a resumed run kept the old `calc.errors` entry next to the good column it had just written, and `save()` froze that contradiction into the file.

- **Process names were not validated.** Duplicates surfaced as pandas' "Columns with duplicate values are not supported in stack" from four frames away, with nothing pointing at the names; non-string names were written to the NPZ as a `U` array and came back as their `str()`, so `procnames=[1, 2]` loaded as `["1", "2"]` and the file did not round-trip. Names are now coerced to `str` and required to be unique, on the internal constructor the parallel workers use as well as the public one.

- **`DirectedInfo` did not implement directed information.** It summed `H(Y^i)/i` minus causal entropy, so with a source statistically independent of the target it returned 0.007 at target autocorrelation 0 and 1.53 at 0.95 -- it measured target self-predictability. It now implements Massey's `sum_i [H(Y_i|Y^{i-1}) - H(Y_i|Y^{i-1},X^i)]`, validated against the closed form `0.5*ln(1+c^2)`.

  The kernel and kozachenko variants are dropped. Composing DI from four separately-estimated entropies leaves each with its own dimension-dependent bias, and those do not cancel: on independent data kernel sat at 3.8-4.4 for every `T` from 100 to 8000 (a fixed bandwidth in ~11 dimensions does not improve with sample size), and kozachenko returned negatives. In their place `di_kraskov` estimates each `I(X^i; Y_i | Y^{i-1})` term *directly* with the KSG/Frenzel-Pompe conditional-MI estimator, which fixes one neighbour radius in the joint space and reuses it across marginals so the biases cancel by construction. It matches the closed form as closely as the Gaussian variant. `n` now reaches the identifier for `DirectedInfo` and `CausalEntropy`.

- **Wavelet phase-slope index lost its direction.** `mne_connectivity` returns a lower-triangular matrix and pyspi filled the upper triangle *without* negating, so `psi[i,j] == psi[j,i]` — the sign is PSI's entire lead/lag content. The fill must also happen per frequency, *before* the band statistic: only a statistic commuting with negation may be applied first, and `max_f(-v) = -min_f(v)`, not `-max_f(v)`. `mean` is antisymmetric, `max` asymmetric. `fmin=0` also asked for an unbounded period, giving an ~11.1-million-sample Morlet wavelet at `T=100`; `fmin` is now resolved against the data-supported floor *and* the cycle count capped so the wavelet always fits the signal.

- **Kozachenko entropy returned `-inf` on tied data.** A duplicated observation puts a nearest neighbour at distance zero, and `log(0)` sends the estimate to `-inf`. Quantised series do this readily -- the bundled `forex` dataset has a process with 24 distinct values in 250 samples -- so several kozachenko SPIs silently produced infinities there. They now fail with the cause named.

- **Conditional mutual information did not reduce to MI on an empty conditioning set.** `_ksg_cmi` faked the conditioning count as a constant `N-(2w+1)`, which coincides with the MI estimator only at `w=0` and drifted with the Theiler window (0.005 at `w=1`, 0.051 at `w=10`). It now delegates to the MI estimator. Reachable only from `DirectedInfo`'s first term, whose history is empty; transfer entropy always has at least one history column, so it never took this branch. No bundled SPI is affected — `di_kraskov` ships without a Theiler window — but a hand-configured `dyn_corr_excl` would have hit it.

- **`ConditionalEntropy` is `directed`.** On `var1_M3_T100`, `max|A - Aᵀ|` is 0.115 (kozachenko) and 0.039 (kernel) under z-scoring, and the Gaussian form is asymmetric (0.73) with `zscore=False`. Structural labels describe the measure rather than one estimator under one preprocessing choice.

- **Importing pyspi reseeded NumPy's global RNG.** `pyspi.lib.ids` called `np.random.seed(1717)` at import, silently overriding the caller's seed -- stochastic SPIs looked reproducible but ignored it. Removed.

- **`filter_spis` matched raw YAML family labels**, so per-variant traits set in `__init__` were invisible (`filter_spis(["antisymmetric"])` returned nothing despite 18 matching SPIs) and a matching family selected all of its configs. It now resolves each config and matches on the labels the SPI actually carries.

- **Gaussian joint/conditional entropy used two different regularisations.** The vectorised multivariate path clipped `r^2` while the scalar path applied a ridge, so `bivariate()` and `multivariate()` disagreed by 8.4 nats on singular data. Both now share one primitive.

New: `Calculator.errors`, `Calculator.run_spec`, `Calculator.run_digest`, `Calculator.to_frame()` (long-form results, one row per `(spi, source, target)`) and `Calculator.summary()`; an `antisymmetric` structural label for measures satisfying `A[i,j] == -A[j,i]`; and a useful `repr` — a computed `Calculator` previously displayed as `<pyspi.calculator.Calculator at 0x...>`.


Three group-delay SPIs that shipped as silent all-NaN columns are now recorded failures (values unchanged).

### Fixed

- **Directed spectral SPIs were transposed.** `SpectralGrangerCausality` (both methods), `DirectedCoherence`, `PartialDirectedCoherence`, `GeneralizedPartialDirectedCoherence`, `DirectedTransferFunction` and `DirectDirectedTransferFunction` reported `A[i, j]` as the influence *of j on i*, the opposite of every other directed SPI in the library. The spectral backends follow the DTF/PDC literature convention; their output was passed through unchanged. pyspi's convention is **row = source, column = target**, as set by `base.Directed.multivariate`, and all directed SPIs now follow it. **Results computed with 2.x for these six SPIs need transposing.** Undirected spectral SPIs are unaffected and bit-identical.
- **`MutualInfo`, `TimeLaggedMutualInfo` and `TransferEntropy` silently returned NaN** when given `estimator="kozachenko"`; there is no Kozachenko-Leonenko path for these measures. They now raise `NotImplementedError` at construction. Use `estimator="kraskov"` instead.
- **`ccm_E-None_*` never inferred an embedding.** The auto-embedding path read the winning dimension as `pyEDM.EmbedDimension(...).max()["E"]`. `DataFrame.max()` reduces column-wise, so that is the largest *candidate* E — pyEDM's `maxE` default of 10 — for every process on every dataset, whatever the skill curve says. The three shipped `ccm_E-None_{mean,max,diff}` SPIs were therefore **bit-identical to `ccm_E-10_*` on all three frozen fixtures** (verified: max|difference| exactly 0) while their identifiers advertised an inferred embedding. Selection is now `argmax(rho)`, ties to the smaller E; on the fixtures it picks E ∈ {1, 2, 5, 10} depending on the process. **`ccm_E-None_*` values change** (3 SPIs); `ccm_E-1_*` and `ccm_E-10_*` are unaffected.
- **`ConvergentCrossMapping` broke whenever pyspi was driven from an unguarded script.** pyEDM 2.5's `_get_mp_context` documents that *"fork is never used"* — it takes forkserver, else spawn — and both re-import the caller's `__main__` in every child. Run from a plain `python analysis.py` with no `if __name__ == "__main__":` guard, which is how the README shows pyspi being used, each child re-executed the caller's script; the parent raised `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`, pyspi caught it, and all nine `ccm_*` SPIs came back as an **all-NaN column** — after the child had already re-run whatever preceded `compute()`. It looked fine from a REPL, a notebook, or a guarded script, which is why the baseline generator and `python -m pyspi` never saw it. pyEDM's nested pools are now off unconditionally: `parallel=False` for `CCM`, and `EmbedDimension` (which has no serial path and starts a child even at `numProcess=1`) replaced by a serial loop over the public `pyEDM.Simplex`, matching its per-E skill exactly. There is no speed cost — measured on an idle machine, `kuramoto_M7_T100`, 21 pairs at E=1, `parallel=True` took 24.8s against 6.8s serial, a **3.6× speedup** from switching it off. Parallelism belongs at the SPI level, where `compute(n_jobs=...)` already provides it.
- **A failed spectral factorisation was reported to nobody.** Wilson's algorithm is iterative; on hitting its iteration cap it logs `"Maximum iterations reached. N of M converged"` through `logging` and returns the unconverged factor anyway. Every Wilson-derived measure (`dcoh`, `dtf`, `ddtf`, `pdcoh`, `gpdcoh`, nonparametric `sgc`) is built from that factor. pyspi collects per-SPI diagnostics from the `warnings` channel only, so those numbers reached the results table with nothing recorded against them — including on the bundled `kuramoto_M7_T100` fixture, where 2 of 21 pairs fail to converge and the relative factorisation residual `max|S - GGᴴ| / max|S|` runs 0.14–6.0 across pairs. The backend's log warnings are now bridged into the `warnings` channel for the duration of each backend call, so they land in the per-SPI record. No value changes; the estimate is the user's to improve (longer series, a parametric fit), but it is no longer silent.
- **`python -m pyspi compute --quiet` reported success over a failed run.** Failed SPIs were named only in the computation summary, which `--quiet` suppresses, and the exit status was 0 unconditionally. Failures and empty (all-NaN) columns are now always reported on stderr, and the exit status is **1 whenever any SPI failed** — a results table with failed columns is one an automated pipeline must not ingest silently, and a NaN column is indistinguishable from a legitimately undefined statistic once the process has exited. `--allow-partial` opts back into exiting 0; the partial table is written either way, and a run in which *no* SPI produced a finite value exits 1 regardless.
- **All three `gd_*` (group delay) SPIs returned no finite value, on any input.** Not a power problem: `spectral_connectivity.statistics.coherence_fisher_z_transform` divides by `sqrt(coherence_bias(n_obs1) + coherence_bias(n_obs2))`, and the one-sample call passes `n_obs2 = 0`, for which `coherence_bias` returns `1/(2·0 − 2) = −0.5`. The radicand is negative for every `n_obs1`, so every p-value is NaN, nothing is ever significant, and the phase regression runs on a fully masked array. Reproduced at 5, 11, 19 and 39 tapers, T up to 4000, on a pair with median coherence 0.998. pyspi now computes the statistic itself with the standard one-sample form `(arctanh|C| − b)/sqrt(b)`, `b = 1/(2n − 2)` (Enochson & Goodman 1965; Bokil et al. 2007), then Benjamini–Hochberg over the band, the largest contiguous significant run thinned to independent points, and a least-squares fit of the unwrapped coherence phase against frequency. On `y(t) = x(t − L)` it recovers L to within 0.005 samples for L ∈ {1, 3, 5, 8}. Partial NaN remains correct — group delay is defined only where the coherence is significant. **`gd_*` values change** (3 SPIs, from nothing to something).

- **`MAX_CORR_AIS` searched the destination embedding only.** It selected `(k, k_tau)` from the destination and hard-coded the source to `(1, 1)`, while the method name denotes selection for both. It now selects `(l, l_tau)` from the source independently, by the same criterion and the same estimator, and passes all four to the estimator; `MAX_CORR_AIS_DEST_ONLY` is the previous behaviour under a name that describes it, and does accept a fixed source embedding. The three shipped auto-embedding SPIs are renamed to carry the method and change value. Auto-embedding and the search bounds are now **refused** by the kernel and symbolic estimators, which have no active-information-storage criterion and used to accept the argument and run a fixed embedding; a fixed embedding that the selected method would override is refused rather than ignored; and search bounds are refused when no auto method is set. Descriptions conflating this with Ragwitz local-prediction selection are corrected — that criterion is not implemented.

- **KSG's deterministic dither was observation-order dependent and introduced an arbitrary 12-decimal boundary.** On independently generated binary marginals at N=120/seed=0, jointly permuting aligned observations at Theiler window zero moved MI from **0.000276 to 0.048716**; complete reversal with `w=3` moved it from **0.032949 to 0.068347**, even though all values and `|i-j|` exclusions were unchanged. A four-level fixture moved by **0.047032** under permutation, and CMI moved as well. Separately, the affine-equivalent binary coordinate `x` versus `0.1*x + 0.3` straddled the 12-decimal rounding boundary and moved its paired MI from **0.030391 to −0.002863**. No deterministic procedure can assign distinct pseudo-random values to otherwise indistinguishable tied observations while remaining covariant under every observation permutation. The dither, digest keys and rounding are therefore removed: continuous KSG now rejects any coordinate with repeated values and directs the caller to an external discrete estimator. Valid tie-free inputs are standardised without rounding or random state and are sample-order, process-order, reflection and nonzero-affine covariant to floating-point tolerance. No finite-sample invariance under arbitrary nonlinear transforms is claimed.

- **KSG's strict marginal counts used a relative tolerance instead of `< ε`.** Every scalar MI, general MI and CMI branch queried at `ε*(1−1e−10)`. On tie-free N=8/k=1 data with `y=x+1e−10·noise`, this discarded genuine interior neighbours and returned **1.7178571429** instead of the exact all-pairs KSG1 value **1.5928571429**. Theiler-window paths had the same defect. Radius queries now use `nextafter(ε, −∞)`, the immediately smaller representable radius, and match independent quadratic-time references. `COMPUTATION_VERSION` is bumped to `3.0.0.r6`. All ten frozen KSG arrays were compared individually and remain bit-exact; their fixture geometries do not hit the corrected boundary.

- **`CorrelationFrame` reported invalid inferential p-values.** It correlated SPI edge vectors but used each source time series' length `T` as the F-test sample size. The observations are edges, not time samples, and substituting the edge count is still invalid because edges sharing a node are dependent. `get_pvalues()`, `compute_significant_values()` and `get_average_correlation(remove_insig=True)` now raise with that explanation and direct users to an independently validated network-preserving permutation/QAP analysis.

- **Coherence phase used linear statistics on wrapped angles.** The bundled spectral mean violated its antisymmetric label by **0.2463994238** on `var1_M3_T100`; its full-band maximum was exactly π for every off-diagonal entry, a branch-cut artefact rather than a useful pairwise summary. Spectral and nonbundled wavelet phase now use the circular mean `arg(mean(exp(i·phase)))` and construct the opposite orientation explicitly by negation. A resultant indistinguishable from zero under a count-scaled `8·eps·n` floating-point bound returns NaN, as does an indistinguishably antipodal mean: ±π is one valid circular location but has no unique signed ordinary-float orientation. Both classes refuse `statistic="max"`; the three `phase_multitaper_max_*` identifiers are removed from every bundled config, while the three mean identifiers remain and change value. `COMPUTATION_VERSION` is `3.0.0.r7`, so no r6 checkpoint can be reused for the final phase policy.

- **Group delay was reported in seconds, and its `rvalue` flipped under process permutation.** `C.frequencies` is in Hz, so `slope/(2π)` is a delay in seconds: a true 4-sample lag came back as 4.0, 2.0 and 1.0 at fs = 1, 2, 4 while the API and every other lagged SPI count samples. The delay is scaled by `fs` (a no-op at the shipped fs=1). `rvalue` stored the *signed* regression r symmetrically, but the fit is of the phase of `C_ij` and `phase(C_ji) = −phase(C_ij)`, so reversing the process order turned +0.99997 into −0.99997 at the mirrored position; it is now `|r|`, and that variant is labelled undirected and unsigned.

- **The KSG auto-embedding boundary overcounted by one.** The scorer used `N_eff = T − (dim−1)·delay`, but `_ksg_ais` aligns a one-step-ahead future against the embedding and so spends a sample on the shift as well: the aligned arrays have `T − 1 − (dim−1)·delay` rows. At T=5, kNN=4, dimension 1 the guard passed on a claimed N=5 while the real N is 4 — and `_ksg_mi_general` had no guard of its own, so the invalid candidate scored a finite **0.0**. The scorer now uses the aligned count, `_ksg_mi_general` validates the arrays it is given, and the search **raises** when no candidate is scorable rather than falling back to (1, 1) — an embedding it had just rejected. No shipped value moves: at T=100 the smallest aligned N is 63 against k=4.

- **IGCI was misnamed and mislabelled.** It is Information-Geometric Causal *Inference* and tests no conditional independence; `InformationGeometricCausalInference` is the accurate name, with the old one kept as a deprecated alias. Its score is a difference of two entropies and therefore exactly antisymmetric, so reporting it `Unsigned` let `Calculator._rmmin` shift both orientations equally and destroy the sign. It stays disabled in the bundled configs. `AdditiveNoiseModel` was labelled `linear` while fitting a Gaussian process — now `nonlinear`.

- **Structural traits now win over every competing config-declared trait.** The rule removed only `directed`/`undirected`, so an intrinsic `undirected` SPI plus YAML `asymmetric` retained both. The instance's one trait now displaces all of `directed`, `undirected`, `antisymmetric` and `asymmetric`; custom-YAML conflicts are tested in all four directions.

- **Integer parameters no longer advertise values they do not compute.** `ConvergentCrossMapping(embedding_dimension=2.7)` advertised `E-2.7` and later executed `E=2`; lagged-correlation config expansion similarly truncated fractional `max_tau` and accepted booleans/numeric strings. Both now require genuine integers and canonicalise NumPy integer scalars before identifiers and expansion.

- **`run_digest` hashed the config's absolute path.** Identical config contents over identical data therefore digested differently in a source checkout, an installed wheel and a temporary directory, so a reusable checkpoint was refused. Location-only fields are excluded from the digest and stay in `run_spec` as provenance; the config *contents* are hashed. A false negative rather than unsafe reuse, but it defeated the point of a content hash.

- **`Data.add_process()` could create a duplicate process name.** It appended `proc-<index>` unconditionally, so a caller who named their processes `["a", "proc-2"]` and appended twice produced a second `proc-2` — which `to_frame()` cannot stack, and which the constructor rejects.

- **Antisymmetric SPIs reported themselves as unsigned.** `issigned()` is not metadata: `Calculator._rmmin` subtracts the minimum from every SPI reporting unsigned — which on an antisymmetric matrix shifts `A[i,j]` and `A[j,i]` equally and destroys the lead/lag its sign carries — and `set_group` correlates unsigned SPIs through `abs()`. `phase`, `pli`, `wpli`, `psi` (multitaper and wavelet), `gd` and `ccm_*_diff` all declared `unsigned` over antisymmetric output. `coint_aeg_tstat` likewise returns a signed Engle–Granger t-statistic (more negative is stronger evidence; positive means none). All are now signed, and the merged label follows `issigned()` so a config cannot reintroduce the contradiction. `CrossPairwiseDistance` had no `issigned` at all, so `_rmmin()` raised `AttributeError` on any config containing it.

- **Cross-correlation was wrong four ways.** `correlate(x, y) / x.std() / y.std() / (T − 1)` is neither the biased (÷T) nor the unbiased (÷(T−|l|)) normalisation and mixed `std()`'s 1/T with a 1/(T−1) divisor, so a series against itself returned T/(T−1) — exactly **1.1111** at T = 10, for a quantity bounded by 1. The correlate call used the *raw* series while the divisor demeaned, giving **3.8384** for `arange(10)` against itself with `zscore=False`. The lag window was centred on index T rather than T−1, i.e. on lag +1, and the opposite orientation was cached unreversed although r_yx(l) = r_xy(−l) — both breaking the symmetry an *undirected* SPI must have. The significance band was `1.96/sqrt(len(r)//2)`, the half-width of the lag window, so it was twice too wide and scaled with the lag cut rather than the sample size; and the truncation walked a contiguous run outwards from lag 0, which on a pair where i leads j by one sample gave 0.9957 one way and −0.0202 the other. Now: biased normalisation (zero lag = Pearson's r), demeaned, a symmetric window, a `1.96/sqrt(T)` amplitude cut, and `sigonly` reducing over the lags above the cut — a set, hence invariant under l → −l — returning 0 when none clears it, rather than the largest of ~T/2 sample correlations under the null. **`xcorr_*` values change** (6 SPIs). `sigonly` is documented as what it is: a pointwise amplitude threshold `|r(l)| > 1.96/sqrt(T)`, kept under a historical name. It is *not* a significance test — the band is not inflated for the series' own autocorrelation and is not corrected for being applied at every lag, so under the null it keeps something almost surely (no seed in 400 gave an empty set at T=600).

- **Spectral Granger causality ignored `fs` and mis-oriented its NaN mask.** The parametric branch built `TimeSeries(..., sampling_interval=1)` unconditionally although `fs` is in the identifier *and* the cache key, so two SPIs advertising different sampling rates computed the same numbers. And the result was transposed into pyspi's (source, target) orientation while the NaN mask was left in the backend's, so with a directionally asymmetric NaN pattern the genuinely unestimable cell was already NaN and the mask blanked its mirror — a good estimate destroyed. Both fixed; no value change at the shipped `fs=1`.

- **Twelve `sgc_*` identifiers named a frequency band they did not use.** Spectral GC is undefined at zero frequency, so `fmin=0` is overridden to 1e-5 — but the identifier was built from the argument. **`sgc_*_fmin-0_*` is renamed to `sgc_*_fmin-1e-05_*`**; values unchanged.

- **Symbolic and kernel transfer entropy accepted embedding parameters they ignore.** `SymbolicTECalculator` reads only `k_HISTORY` and applies that one ordinal-pattern length to source and destination alike at unit delay — which is how Staniek & Lehnertz (2008) define it — yet `k_tau`, `l_history` and `l_tau` were accepted, stored, never read, and written into the identifier: `te_symbolic_k-3_kt-1_l-1_lt-1` advertised a destination history of 3 against a source history of 1 while computing 3 for both. The kernel calculator has the same contract and accepted them just as silently. They are now refused, symbolic identifiers become `te_symbolic_k-<k>`, and non-positive histories and delays are rejected at construction. Implementing separately aligned histories was the alternative and is the wrong call here: there is no oracle left to validate `k != l` against, and every symbolic variant is already commented out of every shipped config.

- **`dyn_corr_excl` named three different Theiler windows with one identifier.** The suffix was a bare `_DCE`, so 5, 10 and `"AUTO"` collided; a config setting two of them produced two identical identifiers. Now `_DCE-<value>`, which reaches `_getkey()` as well. **Shipped SPIs are renamed `..._DCE` → `..._DCE-AUTO`**; values unchanged.

- **Itakura-constrained DTW normalised inconsistently.** The `itakura` branch of `multivariate` skipped the `sqrt(T)` division that both the bivariate path and the dtaidistance path apply, so the two disagreed by `sqrt(T)` under `normalise=True`. Not shipped in any config.

- **Kozachenko SPIs were labelled `linear`.** They are k-nearest-neighbour estimators and are now labelled `nonlinear`, so `filter_spis(["linear"])` no longer returns them.
- **`Data(procnames=...)` was silently discarded.** The length was validated and then never assigned, so custom process names never reached the results table.
- **`CalculatorFrame.compute()` accepted no arguments**, raising `TypeError` for any keyword. It now forwards to `Calculator.compute`.
- The default config, and every bundled config, was **missing from built wheels** — an installed pyspi could not construct a `Calculator`. Package data now uses globs.
- `pyspi/lib/ids/LICENSE.txt` was not shipped, despite MIT requiring it.
- `LICENSE.txt` had been corrupted by a global find-and-replace, altering the verbatim GPLv3 text ("technological *statistics*"). Restored.

### Changed

- **`Calculator(subset=..., configfile=...)` collapsed into `config=`**, which accepts either a bundled name or a path to your own YAML.
- **`normalise=` renamed to `zscore=`** on `Calculator` and `Data`. Behaviour is unchanged (per-process z-score along time); the old name collided with `utils.normalise`, which was min-max, and with the per-SPI `normalise` arguments in `statistics/distance.py`.
- **Configs renamed and moved to `pyspi/configs/`.** The filename stem is now the lookup key, so the cost-pruned sets are reachable by name for the first time.
- `load_dataset()` exposes only the three demo datasets (`forex`, `cml`, `standard_normal`); `available_datasets()` lists them. The regression fixtures moved to `tests/` and no longer ship in the wheel.
- Per-SPI timings are printed after `compute()` (total and slowest five). `calc.timings` was always populated but never surfaced.
- **New `Calculator.save()` and `pyspi.load_table()`.** Results had no documented persistence path from the Python API at all -- only the CLI wrote files. `.npz` is now the canonical format: it stores the results in their natural `(n_spis, M, M)` shape plus names, round-trips exactly, and needs nothing beyond numpy. `.csv` remains as a one-way human-readable export.
- **Dropped pickle and parquet output.** Pickle is version-fragile and executes arbitrary code on load, which is wrong for an archival scientific artifact. Parquet is columnar and built for heterogeneous tabular data; for a dense float tensor it bought nothing over `.npz` while costing a ~40 MB pyarrow dependency. The `parquet` extra is gone.
- A config that keeps only part of a shared-cache group now warns, since the cache is built regardless and the remaining members are close to free. Only applies to user-written configs and to caches expensive enough to matter.
- **Every information-theoretic measure now reports nats.** The kernel and symbolic calculators reported bits while the Gaussian, KSG and Kozachenko ones reported nats — JIDT's split between base 2 for its box-kernel/discrete estimators and base e for the rest, carried into a single results table. `mi_kernel_W-0-5` and `mi_gaussian` therefore sat on axes differing by a factor of ln 2 with nothing in either identifier to say so, so any comparison of *magnitudes* across that boundary — a shared threshold, a "which SPI found the most information" ranking across estimators, a difference or ratio of two columns — was off by that factor. (Pearson and Spearman correlations *between* columns are invariant to a positive rescaling and were never affected; the earlier note overstated this.) Divide by ln 2 for the JIDT-comparable value. **All `kernel` and `symbolic` SPI values change by that factor.**
- **One singularity policy for every Gaussian quantity.** Gaussian MI clipped `r²` at `1 − 1e-15` while the entropy path ridged the covariance: on a pair of identical N=100 series the direct MI was **17.2698** nats and the same quantity assembled from entropies **8.8638**. Each variable is now treated as observed with independent noise of variance `1e-8 × its own variance`, expressed as one shared log-determinant primitive and its closed form, and used by MI, TLMI, entropy, joint/conditional entropy, TE and AIS alike. Both readings give 8.8638, the chain rule closes to 2e-16 on non-degenerate data, and the ridge is equivariant to per-variable rescaling rather than keyed to the loudest process. Gaussian MI acquires a bias of `ridge·r²/(1−r²)` — 5.6e-9 at ρ=0.6.
- **Parallel scheduling buckets on the cache SPIs actually share.** `build_tasks` grouped by `_cache_namespace` alone, so SPIs sharing no cache were serialised into one task: on `full` that produced a single 84-member `spectral_mv` task spanning 16 independent caches, and the longest task bounds the makespan. There is now one `cache_bucket()` definition, shared with the config advisory and the benchmark tooling; `full` goes from 149 tasks with a largest of 84 to 186 with a largest of 24, ordered by measured amortized cost rather than member count.
- `JIDTBase` renamed to `InfoTheoryBase`. Config files are unaffected.

### Dependencies

- **Requires Python 3.10+.**
- Dropped five unused runtime dependencies: `h5py`, `seaborn`, `plotly`, `matplotlib`, `nbformat`. Plotting and notebook packages moved to a `bench` extra.
- Dropped the `setuptools>=68,<80` pin. It existed because pyEDM imported `pkg_resources`; pyEDM 2.5 no longer does, so the floor is now `pyEDM>=2.5` and modern setuptools is usable.
- `pandas>=2.1` for `DataFrame.stack(future_stack=True)`, the pandas 3 semantics.
- **`spectral-connectivity` is now upper-bounded: `>=1.1,<3`.** `DirectedCoherence` reads two *private* `Connectivity` properties (`_transfer_function`, `_noise_covariance`) because the public `directed_coherence()` is wrong twice over (above), so an open-ended floor was a promise pyspi cannot keep. The range is verified end to end against the oldest published 1.1 (1.1.0) and the locked current release: both expose every symbol pyspi uses. 1.1.x additionally lacks `transforms.prepare_time_series` (there is a fallback) and returns a 400- rather than 401-point frequency grid, so band statistics differ marginally across the supported range. `tests/test_directionality.py` fails loudly if a private property disappears, rather than the measure silently changing meaning.
- **`cdt` and `torch` are gone.** pyspi used four functions from `cdt.causality.pairwise` — the ANM independence score, the conditional distribution similarity statistic, the RECI regression-error score and IGCI — now transcribed in `pyspi/lib/pairwise_causal.py` on NumPy/SciPy/scikit-learn, with cdt's MIT licence retained at `pyspi/lib/LICENSE-cdt.txt`. **No value changes**: verified bit-identical against cdt 0.6 over 60 mixed random pairs, with five frozen into `tests/data/fixtures/cdt_pairwise_reference.npz` so the evidence outlives the dependency. torch drove no pyspi computation — `InterDependenceScore` is NumPy — and entered only because cdt eagerly imports its Torch-backed models at package load. Measured: installed size **1.1 GB → 567 MB**, and the 1.5–1.7 s `import cdt.causality.pairwise` (paid once per worker under `compute(n_jobs=…)`) is gone. Per-call runtime is unchanged; the win is install size and start-up, not throughput.
- Build requires `setuptools>=77`, matching the PEP 639 licence metadata already in use; older setuptools would build a wheel with no licence metadata.
- New extra: `bench`. `testing` is unchanged.

### Testing

- The test suite previously **did not run at all** — collection aborted on an undeclared `dill` dependency. Fixed.
- New `tests/test_infotheory_analytic.py`: closed-form checks against `-0.5*ln(1-rho^2)`, `0.5*ln(2*pi*e*sigma^2)`, analytic Gaussian TE on a known AR(1), independence, and information-theoretic identities. Previously the suite contained three assertions comparing a computed value to an independently-known one, and five estimator classes were constructed but never computed.
- New `tests/test_directionality.py` pins the row=source convention for every directed SPI family.
- **Drift tolerances are per-SPI, not per-module.** The suite applied a 1e-2 relative band to every SPI in `causal` and `misc` on the assumption that cdt's optimisers, GP restarts and randomised independence tests made them irreproducible. Measured, that is false: computing the full config twice per fixture under the suite's own protocol reproduces **322 of 322 SPIs bit-exactly** on all three fixtures — comparing the non-finite masks as well as the finite values, so an SPI whose NaN pattern moved between the two runs could not be scored as identical — including every `anm`/`cds`/`reci`/`ccm`, every `coint_*`, `gpfit_*` (`GaussianProcessRegressor` defaults to `n_restarts_optimizer=0`, so there are no random restarts), `lmfit_*` (`random_state` pinned) and `ids`. A module-wide band over 62 SPIs, ~50 of them deterministic, is slack wide enough to hide the regressions this suite exists to catch. The map of loosened SPIs is now keyed by identifier and is empty; `tests/tools/measure_reproducibility.py` regenerates the evidence.
- New CLI exit-status tests, and a test that the spectral factorisation's non-convergence warning is not swallowed.
- **The drift suite reported violations and never failed.** Tolerance breaches went to a session-end banner, which made every tolerance in the file decorative: a deterministic SPI could move by any amount and the run still exited 0. It also returned early when a baseline had no finite entries, so an SPI frozen as an all-NaN column agreed with itself forever — which is how three `gd_*` SPIs stayed broken. Violations now fail, an empty baseline is rejected, and two new gates require that no SPI raises and that none produces an entirely non-finite column. The generator refuses to freeze a run with either condition. One documented exception, shared between generator and suite as `KNOWN_UNESTIMABLE`: on `kuramoto_M7_T100` nitime's automatic AR order search never turns over below `max_order=50`, because at 100 observations of a smooth oscillatory process BIC keeps improving with lag — the fixed-order variants are estimated normally on the same fixture and the automatic variant on the other two. The suite fails if a listed exception starts succeeding.
- `test_whether_calculator_computes` ran the default config and asserted nothing; it now requires an empty `calc.errors` and a finite value in every column, on a coupled VAR(1) rather than the particular white-noise fixture on which `gd_*` was observed to be empty.
- New `test_cache_sharing_never_changes_a_value`: computing the whole config against one Data, where every cache is shared, must equal computing each SPI against its own. This is the automatic coverage check for cache-key omissions — a per-SPI "different identifier implies different cache key" rule is the wrong invariant, since several classes cache a shared intermediate on purpose and apply the differing parameters after the lookup.
- New CI job builds the wheel, installs it into a clean environment, and from a directory containing no pyspi source resolves every bundled config and dataset, computes `fabfour`, and round-trips the NPZ through the CLI. A second job pins to the committed `uv.lock` alongside the floating matrix. The slow regression workflow also runs on pull requests, because a new workflow cannot be manually dispatched until it exists on the default branch.
- Frozen baselines regenerated from this fork as `.npz` (they were upstream 2.0.1 pickles, the wrong oracle for deliberately-changed estimators), and the drift suite now fails hard on a NaN-pattern change or a baseline/current SPI set mismatch, with tolerances split by estimator family.

### References

Definitions the corrected measures are checked against:

- Massey, J. (1990). Causality, feedback and directed information. *Proc. ISITA*. — the `sum_i I(X^i; Y_i | Y^{i-1})` form now implemented by `DirectedInfo`.
- Frenzel, S. & Pompe, B. (2007). Partial mutual information for coupling analysis of multivariate time series. *Phys. Rev. Lett.* 99, 204101. — the conditional-MI estimator behind `di_kraskov` and kraskov transfer entropy.
- Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). Estimating mutual information. *Phys. Rev. E* 69, 066138. — KSG estimator and its effective-sample conditions.
- Kozachenko, L. & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. *Probl. Inf. Transm.* 23, 95–101. — the k-NN entropy that is undefined on tied data.
- Baccalá, L., Sameshima, K., Ballester, G., Do Valle, A. & Timo-Iaria, C. (1998). Studying the interaction between brain structures via directed coherence and Granger causality. *Appl. Sig. Process.* 5, 40–48. — `DC_ij = sqrt(σ_jj)|H_ij| / sqrt(Σ_k σ_kk|H_ik|²)`, the bounded form now computed.
- Kamiński, M. & Blinowska, K. (1991). A new method of the description of the information flow in the brain structures. *Biol. Cybern.* 65, 203–210. — DTF, and the `[target, source]` convention that pyspi transposes to `row = source`.
- Enochson, L. & Goodman, N. (1965). *Gaussian approximations to the distribution of sample coherence.* — the one-sample coherence significance test the group-delay backend gets wrong.
- Bokil, H., Purpura, K., Schoffelen, J.-M., Thomson, D. & Mitra, P. (2007). Comparing spectra and coherences for groups of unequal size. *J. Neurosci. Methods* 159, 337–345. — the same z-transform, with its bias and variance both `1/(2n − 2)`.
- Gotman, J. (1983). Measurement of small time differences between EEG channels. *Electroencephalogr. Clin. Neurophysiol.* 56, 501–514. — group delay as the slope of coherence phase against frequency.
- Staniek, M. & Lehnertz, K. (2008). Symbolic transfer entropy. *Phys. Rev. Lett.* 100, 158101. — one ordinal-pattern length at unit delay, source and destination alike.
- Hoyer, P., Janzing, D., Mooij, J., Peters, J. & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *NIPS*. — `anm`.
- Fonollosa, J. A. R. (2016). Conditional distribution variability measures for causality detection. — `cds`.
- Blöbaum, P., Janzing, D., Washio, T., Shimizu, S. & Schölkopf, B. (2018). Cause-effect inference by comparing regression errors. *AISTATS*. — `reci`.
- Daniušis, P. et al. (2010). Inferring deterministic causal relations. *UAI*. — `igci`.
- Wibral, M., Vicente, R. & Lindner, M. (2014). Transfer entropy in neuroscience. — the maximum corrected AIS embedding criterion, as distinct from Ragwitz local-prediction selection (Ragwitz & Kantz 2002).
- Gao, W., Oh, S. & Viswanath, P. (2018). Demystifying fixed k-nearest neighbor information estimators. *IEEE Trans. Inf. Theory* 64, 5629–5661. — why the KSG error is not O(1/√N) in general.
- Lizier, J. (2014). JIDT: an information-theoretic toolkit. *Front. Robot. AI* 1, 11. — the reference implementation the NumPy port was validated against.

### Migrating from 2.x

| 2.x | 3.0 |
|:----|:----|
| `Calculator(subset="all")` | `Calculator(config="full")` |
| `Calculator(subset="fast")` | `Calculator(config="fast")` |
| `Calculator(configfile="my.yaml")` | `Calculator(config="my.yaml")` |
| `Calculator(normalise=False)` | `Calculator(zscore=False)` |
| `Data(..., normalise=False)` | `Data(..., zscore=False)` |
| `pyspi/config.yaml` | `pyspi/configs/full.yaml` |
| `pyspi/fast_config.yaml` | `pyspi/configs/fast.yaml` |
| `pyspi/sonnet_config.yaml` | `pyspi/configs/sonnet.yaml` |
| `pyspi/fabfour_config.yaml` | `pyspi/configs/fabfour.yaml` |
| `pyspi/benchmarked90_amortized_config.yaml` | `pyspi/configs/benchmarked_p90.yaml` |
| `load_dataset("cml7" \| "var1" \| "kuramoto")` | removed (test fixtures) |
| `utils.normalise` | removed (min-max; use `scipy.stats.zscore`) |
| `utils.standardise`, `utils.strshort` | removed (unused) |
| `utils.check_optional_deps` | removed (no optional runtime deps remain) |
| `JIDTBase` | `InfoTheoryBase` |
| `MutualInfo(estimator="kozachenko")` | raises; use `estimator="kraskov"` |
| `phase_multitaper_max_*` | removed; `CoherencePhase(statistic="max")` raises because ordinary maxima of wrapped angles are branch-cut dependent |

### SPI set changes

`full` goes from **328 SPIs to 322**. Every change below is deliberate; nothing else moved on the frozen test fixtures.

The unsupported coherence-phase maxima are removed from shipped configs. Other disabled variants remain commented with the evidence for switching them off and the condition that would justify switching them back on.

**Flagged, not removed.** `dspli_multitaper_max_*` and `dswpli_multitaper_max_*` saturate: the band maximum reaches exactly 1 as soon as the sign of the imaginary coherency is consistent across tapers at any *one* frequency. On the frozen fixtures the share of pairs at exactly 1 runs **29-100%** depending on the data, with 1 to 16 distinct values; the `mean` variants are graded normally. This is empirical, not a law - saturation is **not** monotone in `T`, since changing `T` recomputes the tapers and Fourier coefficients rather than adding to them (measured non-monotone in 7 of 36 seed/band combinations). They stay **enabled**; the statistic is behaving as defined.

**Directed coherence corrected (twice).** `spectral_connectivity.directed_coherence` is wrong in two independent ways.

1. It puts `|H|²` in the numerator while its denominator stays on the magnitude scale, making the ratio unbounded: baselines reached 3.27 (VAR), 1.84 (CML) and **1139.47** (Kuramoto).
2. Its `_get_noise_variance` reshapes `diag(Σ)` to `(…, 1, n, 1)`, which broadcasts the innovation variance along the **row** (target) axis of `H`. Baccalá's weight is indexed by the **source**. A row-indexed weight is constant across the summation index, so it factors out of numerator and denominator alike and cancels exactly — the innovation variances have no effect at all and the measure degenerates to `sqrt(directed_transfer_function())` for *every* noise covariance.

pyspi now recomputes `DC_ij = sqrt(σ_jj)|H_ij| / sqrt(Σ_k σ_kk|H_ik|²)` (Baccalá et al. 1998) from the same transfer function, with the variance on the source axis.

The previous release note claimed verification "under an identity noise covariance it reproduces `sqrt(DTF)` to 4e-16, the identity DC must satisfy when noise variances are equal". That check was **vacuous**: defect 2 makes the identity hold unconditionally. Measured with innovation standard deviations (1, 3, 0.2), the old form still reproduced `sqrt(DTF)` to 4e-16. Correctness is now pinned by an algebraic test against an explicit-loop transcription of the published formula at unequal variances, by `Σ_j DC_ij² == 1`, and by the `sqrt(DTF)` identity in *both* directions — it must hold at equal variances and must fail at unequal ones — driven from an exact analytic VAR(1) spectrum rather than a sampled estimate.

`dcoh_*` values (6 SPIs) differ from 2.x.

**Correlated innovations.** Baccalá's formula uses only `diag(Σ)`. Boundedness in [0,1] and `Σ_j DC_ij² == 1` hold regardless; what needs diagonal `Σ` is the reading of `DC_ij²` as the fraction of process *i*'s spectral power arriving from *j*. The Wilson-estimated innovation correlation on the bundled fixtures reaches 0.14 (VAR), 0.64 (CML) and 1.00 (Kuramoto), so the caveat is not academic. pyspi does **not** whiten: the minimum-phase factor `G = H·g₀` would give an exactly power-decomposing variant, but `g₀` is triangular and therefore order-dependent — recomputing the same pair as `[j, i]` yields a different `g₀`, so the result would depend on process order, the defect that made `coint_aeg` wrong. The published order-free form is computed, with the assumption documented on the class rather than hidden.

**Removed (7)**

| SPI | Why |
|:----|:----|
| `te_symbolic_k-1` | A length-1 ordinal pattern has one symbol, so TE is identically zero. |
| `te_symbolic_k-10` | Severely undersampled and unvalidated at `T=100`: `10!` symbols against ~91 usable samples. On var1 and cml every joint count is 1, so the value tracks sample size rather than dependence; that does not hold universally (kuramoto: 3 of 42 pairs). Still constructible, and defensible for long series. |
| `di_kernel_W-0.5` | ~3.8-4.4 on independent data at every `T` from 100 to 8000. |
| `di_kozachenko` | Negative values, for a nonnegative quantity. |
| `phase_multitaper_max_*` (3) | Ordinary maxima of wrapped phase depend on the branch cut and were exactly π off-diagonal on the VAR fixture. The constructors now refuse this statistic. |

**Added (1)**

| SPI | Why |
|:----|:----|
| `di_kraskov_NN-4_n-5` | Direct KSG/Frenzel-Pompe conditional-MI estimate of directed information; the validated nonlinear replacement for the two dropped variants. Not in the `benchmarked_p*` sets until it has been timed. |

**Values changed** — the comparison below is between final 3.0.0 and 2.0.1 on the frozen fixtures at rtol 1e-9. The three phase means change; the strict KSG fix does not alter a frozen array.

| SPIs | Cause |
|:-----|:------|
| `coint_aeg_*` (3) | No longer forced symmetric; each orientation is reported as computed. |
| `psi_wavelet_*` (6) | Sign restored, negation moved before the band statistic, wavelet length bounded. |
| `bary_sgddtw_*`, `bary-sq_sgddtw_*` (4) | Stochastic; they now honour the caller's seed instead of the import-time `seed(1717)`. |
| `dcoh_*` (6) | Directed coherence weights by the *source* innovation variance, which the backend's helper cancelled out. |
| `ccm_E-None_*` (3) | The auto-embedding search returns `argmax(rho)` instead of `max(E)`, which was pinning every process at E=10. |
| `gd_*` (3) | Group delay is computed rather than returned all-NaN by a backend whose one-sample significance test divides by the square root of a negative number. |
| `xcorr_*`, `xcorr-sq_*` (6) | Biased normalisation, demeaning, a symmetric lag window, a `1.96/sqrt(T)` amplitude cut, and a `sigonly` rule invariant under l → −l that returns 0 when nothing clears it. On `kuramoto_M7_T100` that last change alone moves 16 of 42 pairs, each from a value the old code had itself judged below the cut (0.109–0.195) to 0. |
| `sgc_*_fmin-0-25_*` (6) | The NaN mask is now transformed into the same orientation as the values it masks. |
| `mi_kraskov_*`, `tlmi_kraskov_*`, `te_kraskov_*`, `di_kraskov_*` (10) | Continuous coordinates are standardised; tied coordinates are refused instead of deterministically dithered. |
| `*_kernel_*` (9) | Reported in nats rather than bits. |
| `mi_gaussian`, `tlmi_gaussian`, `gc_gaussian_*`, `je_gaussian`, `ce_gaussian`, `cce_gaussian_*`, `xme_gaussian_*`, `si_gaussian`, `di_gaussian_n-5` (~14, ≲1e-8 relative) | One shared Gaussian ridge across every path, proportional to each variable's own variance. |
| `te_kraskov_NN-4_DCE-AUTO_MAX-CORR-AIS_*`, `te_kraskov_NN-4_MAX-CORR-AIS_*`, `gc_gaussian_MAX-CORR-AIS_*` (3) | `MAX_CORR_AIS` now selects the source embedding as well as the destination, instead of hard-coding the source to (1, 1). Auto-embedding also skips embeddings the estimator cannot support. |
| every `kraskov` SPI (10) | Content-derived dither and 12-decimal rounding are removed; valid continuous inputs can move only where that rounding changed neighbour geometry. |
| `phase_multitaper_mean_*` (3) | Circular rather than arithmetic phase mean, with the opposite orientation set by negation. Maximum absolute frozen-fixture changes by band are 1.0957–1.4568 (VAR), 0.5345–2.4907 (CML), and 3.0032–4.6345 (Kuramoto). |

**Renamed (23)** — no value change. In every case a parameter that changes the measure was absent from, or misreported by, the identifier.

| Was | Is | Why |
|:----|:---|:----|
| `sgc_*_fmin-0_*` (12) | `sgc_*_fmin-1e-05_*` | Spectral GC is undefined at zero frequency, so `fmin=0` is overridden — but the identifier was built from the argument. |
| `*_DCE` (4) | `*_DCE-AUTO` | The Theiler window's value, not just its presence: 5, 10 and `"AUTO"` shared one name. |
| `cce_gaussian`, `cce_kernel_W-0.5`, `cce_kozachenko`, `di_gaussian` (4) | `..._n-5` | `n` changes the measure. |
| `te_kraskov_*_k-max-*`, `gc_gaussian_k-max-*` (3) | `..._MAX-CORR-AIS_k-max-*` | The auto-embedding method changes what is computed, and `MAX_CORR_AIS` no longer means what it did. These three also change value — see below. |

Symbolic transfer entropy identifiers also lose the three embedding parameters the estimator never applied: `te_symbolic_k-<k>_kt-1_l-1_lt-1` → `te_symbolic_k-<k>`. No symbolic variant is shipped, so no bundled SPI is affected.

Values for the six directed spectral SPIs listed under **Fixed** are also transposed relative to 2.x.

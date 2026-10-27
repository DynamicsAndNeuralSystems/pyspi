# Changelog

## 3.0.0

A major overhaul. The Java/JIDT dependency is gone, the `Calculator` API is simplified, and several long-standing correctness bugs are fixed. **This release contains breaking changes** — see [Migrating from 2.x](#migrating-from-2x).

**Upgrading from 2.0.2 (the current PyPI release), at a glance:**

- Python **>= 3.11** (2.0.2: >= 3.8), numpy **>= 2** (2.0.2: `<2`), scikit-learn **>= 1.8**, pyEDM **>= 2.5** (2.0.2: `==1.15.2.0`), mne **>= 1.6** (2.0.2: `==0.23.0`). No Java, `jpype`, `cdt` or `torch`. See [Dependencies](#dependencies).
- The default `full` set goes from **285 SPIs (2.0.2) to 299**: 46 added, 32 removed, 23 renamed (identifier only), 230 unchanged by name. The exact lists are in [SPI set changes](#spi-set-changes) and [`docs/spi_changes_2.0.2_to_3.0.0.md`](docs/spi_changes_2.0.2_to_3.0.0.md). A change of name or value means results saved under 2.x cannot be joined to 3.0 by identifier without that table.
- The old `Calculator(subset=, configfile=, normalise=)` spellings still work, with a `DeprecationWarning`, until 4.0. See [Breaking changes and deprecations](#breaking-changes-and-deprecations).
- Results tables written by 2.x (pickle) do not load; six directed spectral SPIs are transposed; many estimator values change (see [Changed](#changed) and the "Values changed" table).
- The license is **GPL-3.0-or-later** (2.0.2: GPL-3.0), agreed by the maintainer, Ben Fulcher.

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

Gaussian/kernel MI, kernel entropy and symbolic TE agree to machine precision. **Kernel TE is the exception:** pyspi normalises each embedding column separately whereas JIDT normalises by the standard deviation of the whole series, so the two differ by up to 1.9e-4 bits (5.0e-05 mean absolute error above). "Matches JIDT" does not apply to kernel TE without that qualification. The gaussian-entropy offset is a deterministic ridge term (the analogue of JIDT's stochastic `NOISE_LEVEL_TO_ADD`).

The k-NN rows are historical measurements under different input policies, not a parity claim: the harness used JIDT's default normalisation and random `1e-8` dither, whereas pyspi standardises continuous inputs and refuses tied coordinates. No convergence rate is inferred from this comparison; KSG bias depends on k, dimensionality and density smoothness (Gao, Oh & Viswanath 2018, *Demystifying fixed k-nearest neighbor information estimators*). 
Tie-free series are not changed by pyspi's tie handling (see below), and for those the KSG MI and TE (including Theiler windows, dynamic correlation exclusion and auto-embedding) agree with JIDT 1.6.1 run without its default dither to about 1e-15. That evidence is now kept in the repository: `tests/test_jidt_reference.py` asserts pyspi against hard-coded JIDT 1.6.1 outputs for KSG MI at Theiler windows 0, 5 and 20, KSG TE (fixed embeddings, a Theiler window, and auto-embedding with it), Gaussian MI and TE, kernel MI, and symbolic TE (KSG to 1e-12, Gaussian to the 1e-8 ridge, kernel MI/symbolic exactly after the ln 2 conversion), and it records the kernel-TE discrepancy above as a bounded known difference. It needs no Java. The Java harness that generated the constants is in `tests/tools/jidt_reference/`; the jar is not committed.

**Provenance and licence.** The KSG, kernel, Gaussian and symbolic estimators follow, and were validated against, JIDT (Lizier, 2014; [github.com/jlizier/jidt](https://github.com/jlizier/jidt); GPL-3.0-or-later). `pyspi/statistics/infotheory.py` now carries that attribution and citation, and JIDT's copyright notice is retained in `pyspi/lib/LICENSE-jidt.txt` (listed in `license-files`, like the cdt notice).

### Breaking changes and deprecations

**Deprecated, not removed.** These 2.x spellings still work in 3.x with a `DeprecationWarning` and will be removed in 4.0:

- `subset="all"` becomes `config="full"`; other `subset` names are unchanged.
- `configfile=p` becomes `config=p`.
- `normalise=` becomes `zscore=`, on `Calculator`, `Data` and `CalculatorFrame`.
- Passing both an old and a new form raises `TypeError`.
- **Positional arguments after `subset`** (2.x `configfile`, `detrend`, `normalise`) no longer line up; pass them as keywords.

**Breaking, with no shim:**

- SPI constructors raise on parameters their estimator ignores. For example the symbolic and kernel TE calculators refuse `k_tau`, `l_history` and `l_tau`, so 2.x YAML files that set them now fail.
- `CorrelationFrame.get_pvalues()`, `compute_significant_values()` and `get_average_correlation(remove_insig=True)` raise `NotImplementedError` (the p-values were invalid; see Fixed).
- `Data` copies and freezes its input, and `Data.to_numpy()` is read-only.
- `utils.is_jpype_jvm_available` is removed, along with the other `utils` helpers in the migration table.
- `parse_bivariate` raises `ValueError` when `i` and `j` are ambiguous. In 2.x it constructed a `Warning` object and did nothing.
- Results tables written by pyspi < 3.0.0 (pickle) do not load; see Fixed.

**Behavior of the API itself.**

- **New since 2.0.2: parallel computation.** `Calculator.compute(n_jobs=...)` follows the joblib convention: `-1` is all cores, `-2` all but one, and `n_jobs=0` raises (`PYSPI_N_JOBS` sets the default). 2.0.2 had no `n_jobs`.
- `import pyspi` no longer replaces `sys.stdout`/`sys.stderr`. `Calculator(verbose=False)` no longer touches logging; `verbose=True` adds one handler, idempotently.
- The `OMP_NUM_THREADS` `setdefault` at import is removed. It ran after numpy had loaded, so it had no in-process effect, and it altered the environment of every importing program. Thread pinning for workers stays in `_parallel`.
- **New since 2.0.2: a command line.** `python -m pyspi compute` and a `pyspi` console command (2.0.2 had neither). The output suffix and `n_jobs` are validated before computing, and `PYSPI_N_JOBS` is honoured.

### Removed

- **Java and JIDT** (above): `infodynamics.jar`, `jpype1` and the JVM start-up path.
- **`cdt` and `torch`** (see Dependencies); `setup.py` and `requirements.txt` (packaging is `pyproject.toml` only); `pyspi/phiconf.yaml`; the 2.x config filenames (see the migration table); pickle and parquet output.
- **Thirty-two SPIs** (`full`: 285 to 299 net of 46 additions). See [SPI set changes](#spi-set-changes). Fourteen went with the port: six `dspli`/`dswpli` multitaper maxima, three `phase_multitaper_max_*`, two `te_symbolic_*`, `di_kernel_W-0.5`, `di_kozachenko`, and `igci`. Fifteen went after the per-SPI audit found them duplicate, constant or degenerate: the four `*_EllipticEnvelope` (bit-identical to `*_MinCovDet`), `lmfit_Lasso` and `lmfit_ElasticNet` (constant at `alpha = 1`), `bary_euclidean_mean` (identically 0), `cce_kernel_W-0.5_n-5` and `xme_kernel_W-0.5_k10` (box kernel in ~10 dimensions), the six `pli`/`wpli` multitaper maxima (exactly 1 at `T >= 1000`), and the three `psi_wavelet_max_*` (a bounded instantaneous slope whose maximum is positive under the null both ways round).

### Fixed: state, identity, and estimator contracts

A second pass, driven by a suite of red tests, closed a set of defects that produced valid-looking but wrong results. Full details in each commit; the scientifically material ones:

- **Stale caches survived data mutation.** Statistics cache results directly on the `Data` instance keyed by their own parameters, never by the data. Nothing invalidated them, so `set_data`/`add_process`/`remove_process` left every cached statistic serving the *previous* dataset's numbers, silently. All 15 cache attributes are now registered and dropped on mutation, and `Data` copies and freezes its input so nothing can mutate the series behind a cache.

- **Checkpoints were not bound to a run.** Resume validated only the SPI identifier and an `(M, M)` shape, so a different dataset, config, preprocessing setting, or process order silently inherited the earlier run's results. Checkpoints now carry a `run.json` manifest bound to `Calculator.run_digest` (config *contents*, dataset bytes in native dtype, and a computation-version token). Failed or non-finite checkpoints are retried by default, workers load the parent's config snapshot rather than rereading a path that may have changed, and a directory belonging to another run is refused rather than emptied.

- **Spectral caches ignored `fs`,** and were written under a `str` key but read under a `tuple` key — so the first write was unreachable and staleness only appeared from the *third* call, which is why a two-call probe found nothing.

- **Six measures advertised `kraskov` and ran Gaussian.** Joint/conditional/ crossmap/causal entropy, directed info and stochastic interaction are composed from marginal entropies and have no KSG estimator; the argument is now rejected. No bundled config used it, so no shipped result changed.

- **Cointegration `aeg` was forcibly symmetric.** It is not symmetric in its arguments (~0.8 mean absolute difference between orientations, up to ~1.6), yet the cache wrote each value to both `(i,j)` and `(j,i)`, so the reported value depended on process order. `aeg` is now `directed`; `johansen`, which is symmetric to ~3e-14, is unchanged.

- **KSG accepted inputs it cannot estimate from.** `k=30` on `N=20` returned 0.414 and `k=100` returned 1.63; negative Theiler windows were accepted. The effective-sample and window checks now guard the MI, TE and auto-embedding paths, and the auto-embedding search skips candidates it cannot support instead of ranking them and failing on the winner. Tied and quantised coordinates are handled by the seeded tie-breaking jitter described in the next entry, not rejected.

- **KSG lost JIDT's normalisation and tie policy.** Every KSG-family calculator in JIDT enables `normalise` and a small random dither by default. pyspi 2.x ran with both active, while the pure-NumPy port initially implemented neither. That is not a rounding difference:

  - **Ties.** Independent binary marginals (N=400, k=4) returned MI = **−3.35**; four-level, −1.96; one-decimal-rounded Gaussians, **+0.90** — a confident false positive on independent data. Every kth-nearest-neighbour radius is zero, so the digamma counts saturate on the tie structure and the estimate measures quantisation. Every process in the bundled `forex` dataset is tied: 24–212 distinct values in 250 samples.
  - **Scale.** KSG's L∞ radius is not invariant to per-coordinate rescaling. With `zscore=False`, scaling one of a correlated Gaussian pair (true MI 0.50) by 1e−3 or 1e3 collapsed the estimate from 0.49 to 0.05 and 0.06.

  Per-coordinate sample standardisation is restored for MI, TLMI, TE, CMI, AIS auto-embedding and directed information. Random dither is not: an unexposed noise draw changes a finite-sample estimate, while content-derived pseudo-random dither was found to depend on observation order. For ties, KSG now follows hctsa (`BF_TieBreakNoise`) and pyhctsa (`bf_tie_break_noise`): each raw series with fewer than 90% unique values (and a positive standard deviation) gets Gaussian jitter of SD `1e-10 × std` from a fixed-seed local generator before embedding (seeds: source 0, target 1, embedding selection 2). Tie-free series are untouched, so existing tie-free results are bit-identical and still match JIDT 1.6.1 to about 1e-15. The same independent binary, four-level and rounded-Gaussian data (N=400, k=4) that previously gave MI −3.35, −1.96 and +0.90 now give about 0 ± 0.03, and every `kraskov` SPI now computes on the bundled `forex` dataset. The jitter is deterministic but depends on observation order. NaN, infinite and constant series still raise. **Every `kraskov` SPI's values can change**: 2.x ran JIDT with its default dither (seeded, `NOISE_SEED=42`), which 3.0 does not reproduce.

  Kozachenko entropy keeps an explicit tied-data error (it is not jittered): `H(X + εξ) → −∞` as `ε → 0` for discrete `X`, so a dithered differential entropy on quantised data reports the dither level.

- **Symbolic TE packed symbols into an integer that overflowed** at `k_history=10` (reaching `(k!)^3`). Now counts distinct rows directly, validated against a tuple-keyed reference. `k_history=1` is rejected: a length-1 ordinal pattern has one symbol, so TE is identically zero. Those were the only two symbolic variants shipped, so **symbolic TE now has no bundled representation** — reintroducing it needs a defensible `k` with benchmark support. `k_history=10` remains constructible for long series, where the undersampling argument does not apply.

- **Results tables no longer use pickle.** Names are stored as `dtype='U'` and loaded with `allow_pickle=False`; files carry a schema version, run spec, digest and errors. Tables written by pyspi < 3.0.0 will not load — re-save them from a `Calculator`.

- **`load_table()` dropped everything `save()` wrote except the numbers.** The run spec, digest and error map were written and never read, so a loaded table could not be asked which SPIs failed — and a NaN column is otherwise indistinguishable from a legitimately undefined statistic. They now arrive in `DataFrame.attrs` (`schema`, `run_spec`, `run_digest`, `errors`). Validation also covered only `ndim` and the SPI axis; a file whose matrices were the wrong width reached `MultiIndex.from_product` and failed there with a reshape error rather than a statement about the file. The full `(n_spis, M, M)` shape is now checked.

- **`run_digest` did not bind to the algorithm.** It hashed the run spec, the config contents and the dataset bytes, so identical inputs computed by two different estimator implementations produced the same digest and a checkpoint written by one could be resumed by the other. It now covers the computation version too.
- **`pyspi.COMPUTATION_VERSION` is public.** It is the value checkpoints and `run_digest` are keyed on, bumped whenever a change can alter a valid SPI output. Record it with saved results and do not pool results computed under different values.

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

- **`MAX_CORR_AIS` searched the destination embedding only (in the intermediate NumPy port).** JIDT 1.6.1, and so pyspi 2.0.2, searched the source embedding as well, so relative to 2.0.2 **only the identifiers change, not the values**. The port had selected `(k, k_tau)` from the destination and hard-coded the source to `(1, 1)`, while the method name denotes selection for both. It now selects `(l, l_tau)` from the source independently, by the same criterion and the same estimator, and passes all four to the estimator; `MAX_CORR_AIS_DEST_ONLY` is the previous behaviour under a name that describes it, and does accept a fixed source embedding. The three shipped auto-embedding SPIs are renamed to carry the method; their values match 2.0.2's search. Auto-embedding and the search bounds are now **refused** by the kernel and symbolic estimators, which have no active-information-storage criterion and used to accept the argument and run a fixed embedding; a fixed embedding that the selected method would override is refused rather than ignored; and search bounds are refused when no auto method is set. Descriptions conflating this with Ragwitz local-prediction selection are corrected — that criterion is not implemented.

- **KSG's deterministic dither was observation-order dependent and introduced an arbitrary 12-decimal boundary.** On independently generated binary marginals at N=120/seed=0, jointly permuting aligned observations at Theiler window zero moved MI from **0.000276 to 0.048716**; complete reversal with `w=3` moved it from **0.032949 to 0.068347**, even though all values and `|i-j|` exclusions were unchanged. A four-level fixture moved by **0.047032** under permutation, and CMI moved as well. Separately, the affine-equivalent binary coordinate `x` versus `0.1*x + 0.3` straddled the 12-decimal rounding boundary and moved its paired MI from **0.030391 to −0.002863**. No deterministic procedure can assign distinct pseudo-random values to otherwise indistinguishable tied observations while remaining covariant under every observation permutation. That content-derived dither, its digest keys and the rounding were therefore removed. They were replaced by hctsa-style jitter from a fixed-seed generator applied only to series with fewer than 90% unique values (previous entry), which is deterministic for a given observation order but is not claimed to be order-covariant. Tie-free inputs are standardised without rounding or random state and are sample-order, process-order, reflection and nonzero-affine covariant to floating-point tolerance. No finite-sample invariance under arbitrary nonlinear transforms is claimed.

- **KSG's strict marginal counts used a relative tolerance instead of `< ε`.** Every scalar MI, general MI and CMI branch queried at `ε*(1−1e−10)`. On tie-free N=8/k=1 data with `y=x+1e−10·noise`, this discarded genuine interior neighbours and returned **1.7178571429** instead of the exact all-pairs KSG1 value **1.5928571429**. Theiler-window paths had the same defect. Radius queries now use `nextafter(ε, −∞)`, the immediately smaller representable radius, and match independent quadratic-time references. `COMPUTATION_VERSION` is bumped to `3.0.0.r6`. All ten frozen KSG arrays were compared individually and remain bit-exact; their fixture geometries do not hit the corrected boundary.

- **`CorrelationFrame` reported invalid inferential p-values.** It correlated SPI edge vectors but used each source time series' length `T` as the F-test sample size. The observations are edges, not time samples, and substituting the edge count is still invalid because edges sharing a node are dependent. `get_pvalues()`, `compute_significant_values()` and `get_average_correlation(remove_insig=True)` now raise with that explanation and direct users to an independently validated network-preserving permutation/QAP analysis.

- **Coherence phase used linear statistics on wrapped angles.** The bundled spectral mean violated its antisymmetric label by **0.2463994238** on `var1_M3_T100`; its full-band maximum was exactly π for every off-diagonal entry, a branch-cut artefact rather than a useful pairwise summary. Spectral and nonbundled wavelet phase now use the circular mean `arg(mean(exp(i·phase)))` and construct the opposite orientation explicitly by negation. A resultant indistinguishable from zero under a count-scaled `8·eps·n` floating-point bound returns NaN, as does an indistinguishably antipodal mean: ±π is one valid circular location but has no unique signed ordinary-float orientation. Both classes refuse `statistic="max"`; the three `phase_multitaper_max_*` identifiers are removed from every bundled config, while the three mean identifiers remain and change value. `COMPUTATION_VERSION` is `3.0.0.r7`, so no r6 checkpoint can be reused for the final phase policy.

- **Group delay was reported in seconds, and its `rvalue` flipped under process permutation.** `C.frequencies` is in Hz, so `slope/(2π)` is a delay in seconds: a true 4-sample lag came back as 4.0, 2.0 and 1.0 at fs = 1, 2, 4 while the API and every other lagged SPI count samples. The delay is scaled by `fs` (a no-op at the shipped fs=1). `rvalue` stored the *signed* regression r symmetrically, but the fit is of the phase of `C_ij` and `phase(C_ji) = −phase(C_ij)`, so reversing the process order turned +0.99997 into −0.99997 at the mirrored position; it is now `|r|`, and that variant is labelled undirected and unsigned.

- **The KSG auto-embedding boundary overcounted by one.** The scorer used `N_eff = T − (dim−1)·delay`, but `_ksg_ais` aligns a one-step-ahead future against the embedding and so spends a sample on the shift as well: the aligned arrays have `T − 1 − (dim−1)·delay` rows. At T=5, kNN=4, dimension 1 the guard passed on a claimed N=5 while the real N is 4 — and `_ksg_mi_general` had no guard of its own, so the invalid candidate scored a finite **0.0**. The scorer now uses the aligned count, `_ksg_mi_general` validates the arrays it is given, and the search **raises** when no candidate is scorable rather than falling back to (1, 1) — an embedding it had just rejected. No shipped value moves: at T=100 the smallest aligned N is 63 against k=4.

- **IGCI was misnamed and mislabelled.** It is Information-Geometric Causal *Inference* and tests no conditional independence; `InformationGeometricCausalInference` is the accurate name, with the old one kept as a deprecated alias. Its score is a difference of two entropies and therefore exactly antisymmetric, so reporting it `Unsigned` let `Calculator._rmmin` shift both orientations equally and destroy the sign. It is **not in any bundled config in 3.0**; it was active in 2.0.2's `config.yaml` (as `InformationGeometricConditionalIndependence`, identifier `igci`), so `full` loses that SPI. `AdditiveNoiseModel` was labelled `linear` while fitting a Gaussian process — now `nonlinear`.

- **Structural traits now win over every competing config-declared trait.** The rule removed only `directed`/`undirected`, so an intrinsic `undirected` SPI plus YAML `asymmetric` retained both. The instance's one trait now displaces all of `directed`, `undirected`, `antisymmetric` and `asymmetric`; custom-YAML conflicts are tested in all four directions.

- **Integer parameters no longer advertise values they do not compute.** `ConvergentCrossMapping(embedding_dimension=2.7)` advertised `E-2.7` and later executed `E=2`; lagged-correlation config expansion similarly truncated fractional `max_tau` and accepted booleans/numeric strings. Both now require genuine integers and canonicalise NumPy integer scalars before identifiers and expansion.

- **`run_digest` hashed the config's absolute path.** Identical config contents over identical data therefore digested differently in a source checkout, an installed wheel and a temporary directory, so a reusable checkpoint was refused. Location-only fields are excluded from the digest and stay in `run_spec` as provenance; the config *contents* are hashed. A false negative rather than unsafe reuse, but it defeated the point of a content hash.

- **`Data.add_process()` could create a duplicate process name.** It appended `proc-<index>` unconditionally, so a caller who named their processes `["a", "proc-2"]` and appended twice produced a second `proc-2` — which `to_frame()` cannot stack, and which the constructor rejects.

- **Antisymmetric SPIs reported themselves as unsigned.** `issigned()` is not metadata: `Calculator._rmmin` subtracts the minimum from every SPI reporting unsigned — which on an antisymmetric matrix shifts `A[i,j]` and `A[j,i]` equally and destroys the lead/lag its sign carries — and `set_group` correlates unsigned SPIs through `abs()`. `phase`, `pli`, `wpli`, `psi` (multitaper and wavelet), `gd` and `ccm_*_diff` all declared `unsigned` over antisymmetric output. `coint_aeg_tstat` likewise returns a signed Engle–Granger t-statistic (more negative is stronger evidence; positive means none). All are now signed, and the merged label follows `issigned()` so a config cannot reintroduce the contradiction. `CrossPairwiseDistance` had no `issigned` at all, so `_rmmin()` raised `AttributeError` on any config containing it.

- **Cross-correlation was wrong four ways.** `correlate(x, y) / x.std() / y.std() / (T − 1)` is neither the biased (÷T) nor the unbiased (÷(T−|l|)) normalisation and mixed `std()`'s 1/T with a 1/(T−1) divisor, so a series against itself returned T/(T−1) — exactly **1.1111** at T = 10, for a quantity bounded by 1. The correlate call used the *raw* series while the divisor demeaned, giving **3.8384** for `arange(10)` against itself with `zscore=False`. The lag window was centred on index T rather than T−1, i.e. on lag +1, and the opposite orientation was cached unreversed although r_yx(l) = r_xy(−l) — both breaking the symmetry an *undirected* SPI must have. The significance band was `1.96/sqrt(len(r)//2)`, the half-width of the lag window, so it was twice too wide and scaled with the lag cut rather than the sample size; and the truncation walked a contiguous run outwards from lag 0, which on a pair where i leads j by one sample gave 0.9957 one way and −0.0202 the other. Now: biased normalisation (zero lag = Pearson's r), demeaned, a symmetric window, a `1.96/sqrt(T)` amplitude cut, and `sigonly` reducing over the lags above the cut — a set, hence invariant under l → −l — returning 0 when none clears it, rather than the largest of ~T/2 sample correlations under the null. **`xcorr_*` values change** (6 SPIs). `sigonly` is documented as what it is: a pointwise amplitude threshold `|r(l)| > 1.96/sqrt(T)`, kept under a historical name. It is *not* a significance test — the band is not inflated for the series' own autocorrelation and is not corrected for being applied at every lag, so under the null it keeps something almost surely (no seed in 400 gave an empty set at T=600).

- **Spectral Granger causality ignored `fs` and mis-oriented its NaN mask.** The parametric branch built `TimeSeries(..., sampling_interval=1)` unconditionally although `fs` is in the identifier *and* the cache key, so two SPIs advertising different sampling rates computed the same numbers. And the result was transposed into pyspi's (source, target) orientation while the NaN mask was left in the backend's, so with a directionally asymmetric NaN pattern the genuinely unestimable cell was already NaN and the mask blanked its mirror — a good estimate destroyed. Both fixed; no value change at the shipped `fs=1`.

- **Eleven `sgc_*` identifiers (in `full`) named a frequency band they did not use.** Spectral GC is undefined at zero frequency, so `fmin=0` is overridden to 1e-5 — but the identifier was built from the argument. **`sgc_*_fmin-0_*` is renamed to `sgc_*_fmin-1e-05_*`**; values unchanged.

- **Symbolic and kernel transfer entropy accepted embedding parameters they ignore.** `SymbolicTECalculator` reads only `k_HISTORY` and applies that one ordinal-pattern length to source and destination alike at unit delay — which is how Staniek & Lehnertz (2008) define it — yet `k_tau`, `l_history` and `l_tau` were accepted, stored, never read, and written into the identifier: `te_symbolic_k-3_kt-1_l-1_lt-1` advertised a destination history of 3 against a source history of 1 while computing 3 for both. The kernel calculator has the same contract and accepted them just as silently. They are now refused, symbolic identifiers become `te_symbolic_k-<k>`, and non-positive histories and delays are rejected at construction. Implementing separately aligned histories was the alternative and is the wrong call here: there is no oracle left to validate `k != l` against, and every symbolic variant is already commented out of every shipped config.

- **`dyn_corr_excl` named three different Theiler windows with one identifier.** The suffix was a bare `_DCE`, so 5, 10 and `"AUTO"` collided; a config setting two of them produced two identical identifiers. Now `_DCE-<value>`, which reaches `_getkey()` as well. **Shipped SPIs are renamed `..._DCE` → `..._DCE-AUTO`**; values unchanged.

- **Itakura-constrained DTW normalised inconsistently.** The `itakura` branch of `multivariate` skipped the `sqrt(T)` division that both the bivariate path and the dtaidistance path apply, so the two disagreed by `sqrt(T)` under `normalise=True`. Not shipped in any config.

- **`dtw_constraint-sakoe-chiba` is renamed `dtw_constraint-sakoe-chiba_radius-auto` and its values change.** 2.x used tslearn's default Sakoe-Chiba radius of 1; the radius is now chosen from the series length (10% of `T` or `1.5·√T`, whichever is smaller, with a floor of 10 and a cap of `T-1`), tunable with `PYSPI_DTW_SAKOE_LINEAR_FRAC`, `PYSPI_DTW_SAKOE_SQRT_COEFF` and `PYSPI_DTW_SAKOE_MIN_RADIUS`. Values differ from 2.0.1 by up to 8.5.
- **Kozachenko SPIs were labelled `linear`.** They are k-nearest-neighbour estimators and are now labelled `nonlinear`, so `filter_spis(["linear"])` no longer returns them.
- **`Data(procnames=...)` was silently discarded.** The length was validated and then never assigned, so custom process names never reached the results table.
- **`CalculatorFrame.compute()` accepted no arguments**, raising `TypeError` for any keyword. It now forwards to `Calculator.compute`.
- The default config, and every bundled config, was **missing from built wheels** — an installed pyspi could not construct a `Calculator`. Package data now uses globs.
- `pyspi/lib/ids/LICENSE.txt` was not shipped, despite MIT requiring it.
- `LICENSE.txt` had been corrupted by a global find-and-replace, altering the verbatim GPLv3 text ("technological *statistics*"). Restored.
- **`lcss_constraint-sakoe-chiba` and `lcss_constraint-itakura` were plain `lcss`.**
  tslearn before 0.10 ignored `global_constraint` in `lcss` (tslearn #526), so in every earlier pyspi release, 2.x included, both constrained SPIs were identical to `lcss` (0.0 difference on every drift fixture).
  tslearn 0.10 applies the band, but by holding every cell outside it at zero.
  That is correct for the Sakoe-Chiba band; the Itakura parallelogram, however, narrows to a single cell at each end, so any pair whose last samples differ by more than `eps` scored exactly 0 (43% of random pairs at T=100), and 40 of 94 of the remaining pairs read up to 0.03 low.
  pyspi now computes constrained LCSS itself (`_banded_lcss` in `pyspi/statistics/distance.py`), with tslearn's own band masks restricting which pairs may match but not where the recursion may pass (Vlachos et al. 2002).
  It equals tslearn 0.10 exactly for Sakoe-Chiba, and plain `lcss` still calls tslearn and is unchanged.
  **Both constrained values change**: by up to 0.4 for Sakoe-Chiba and 0.09 for Itakura on the drift fixtures.
  `lcss_constraint-sakoe-chiba` uses tslearn's default radius of 1, the narrow band that `dtw_constraint-sakoe-chiba` used in 2.x.

The remaining entries in this section come from a per-SPI audit of the whole `full` set, run after the LCSS finding above.
Each fix was first pinned by an independent reference test that the previous code failed; the tests live in `tests/test_*_audit.py` and the module test files.
`COMPUTATION_VERSION` is `3.0.0.r12` (r11 was the audit's fixes; r12 the deferred findings the maintainer then decided on: the `coint_aeg` orientation, the GroupDelay thinning step and the phase measures' real-valued bins), and the identifiers whose values change are listed under **Values changed in r11 and r12** below.

- **All nine `ccm_*` SPIs were transposed, so every `ccm_*_diff` had the wrong sign.**
  pyEDM labels its output columns `library:target`: `a:b` is `a`'s shadow manifold predicting `b`, which Sugihara et al. (2012) read as evidence that `b` drives `a`.
  pyspi stored the `i:j` column at `A[i, j]`, so `A[i, j]` was the evidence for `j -> i`, the reverse of the row = source convention; this dates from the first CCM commit and holds in 2.x as well.
  On Sugihara's coupled logistic maps with X driving Y the shipped `ccm_*_mean` gave `A[X, Y] = -0.09` and `A[Y, X] = 0.76`.
  The cause is now on the row; the new matrices are exactly the transpose of the old ones.
  Separately, the library sizes were built as `lower upper inc` with `inc = int((upper - lower) / 21)`, which yields more than 22 sizes, and only the first 22 were kept, so at T=100 the largest library actually used was 73 of 90 at E=1 and 62 of 80 at E=10 and the `max` statistic was not the maximum over the intended range.
  There are now 22 rounded, inclusive library sizes.
  **All nine values change**: by up to 0.7 (var1), 1.2 (cml) and 2.4 (kuramoto, the `diff` variants) relative to the stored baselines, almost all of it the transpose.
  **Results computed with 2.x for `ccm_*` need transposing** (and `ccm_*_diff` negating).
- **`cds` returned 0, its strongest evidence for `i -> j`, when it was undefined.**
  With fewer than two bins of the cause holding more than 12 points, the conditional-distribution spread cannot be compared and cdt returns a hard-coded 0; over 200 random nonlinear pairs that was every pair at T <= 40, 76% at T=64 and 4% at T=80, and on independent noise at T=64 two of three processes reported 0 against every target.
  The SPI now requires at least two usable conditionals and returns NaN otherwise.
  The frozen cdt fixture is unchanged (the default keeps cdt's behaviour) and no drift value moves, since the fixtures have no fallback zeros.
  The IGCI slope estimator (not used by any SPI) also had the opposite sign to the documented and entropy-branch convention and is corrected, and the RECI docstring now describes the `a x^3 + b` regressor the code fits.
- **`softdtw_constraint-sakoe-chiba` and `softdtw_constraint-itakura` were plain `softdtw`.**
  The same defect as LCSS, on pyspi's side this time: `SoftDynamicTimeWarping` never passed `global_constraint` to `tslearn.metrics.soft_dtw`, so both constrained SPIs were identical to `softdtw` on every fixture.
  The constraint is now passed through; tslearn >= 0.10 holds cells outside the band at `+inf`, which is the right semantics for soft-DTW, and both variants now equal a cell-by-cell Cuturi & Blondel recursion with the band mask.
  **Both values change**, by up to 240 (absolute) on the drift fixtures.
- **`dcorrx_*` and `mgcx_*` were transposed, and their lagged terms were wrong.**
  hyppo's `DcorrX(x, y)` pairs the present of `x` with the past of `y`, so it measures `y -> x`; pyspi passed `(z[i], z[j])` and so stored the `j -> i` value at `A[i, j]`, the reverse of every other directed SPI.
  Separately, hyppo 0.5.2 (and its current main) hands sliced distance matrices to a `Dcorr`/`MGC` built with `compute_distance="euclidean"`, so every lag `l >= 1` term measured dependence between rows of distance matrices rather than between the lagged series; only the `l = 0` term was right.
  pyspi now sums `(n - l)/n * S(target[l:], source[:n-l])` itself, the definition of Mehta et al. (2020) and of hyppo's own docstring, passing the aligned distance blocks with `compute_distance=None`, which is exactly equal to the statistic on the raw lagged series.
  This also drops the MGC optimal-scale search that `MGCX` ran on every pair and pyspi discarded, so `mgcx_*` is about 2x faster.
  **All four values change** (`dcorrx_maxlag-{1,10}`, `mgcx_maxlag-{1,10}`), by up to 1.7 on the drift fixtures; the `maxlag-1` change is dominated by the transpose.
- **`hsic` and `hsic_biased` depended on the process order.**
  hyppo's `compute_kern(x, y)` takes the median-heuristic Gaussian width from `x` and applies it to `y` as well, so `hsic(x, y) != hsic(y, x)` (by up to 0.016 on the fixtures) and `hsic(x, 3x)` was 0.72 instead of 1.
  Each variable now gets its own median width, as in Gretton et al. (2005); the result is bit-identical to hyppo whenever the two widths agree.
  **Both values change**, by up to 0.016.
- **`Barycenter` was declared `Directed` and its direct `bivariate` disagreed with `multivariate`.**
  The matrix was symmetric by construction, since each pair's barycenter was cached under both orientations, and the config labels already said undirected; the class is now `Undirected`.
  The subgradient barycenter (`sgddtw`) depends on the order of its two inputs, and the cache kept whichever orientation was asked for first, so `bivariate(i=0, j=1)` on fresh data disagreed with `multivariate()[0, 1]`.
  The pair is now always fitted in one fixed orientation (the one `multivariate` already used), so no stored value moves.
- **KSG mutual information with a Theiler window was biased upward.**
  The window removes `|i - j| <= w` from each point's neighbour pool, but `mi_kraskov_*_DCE-*` and `tlmi_kraskov_*_DCE-*` kept the `psi(N)` term for the full sample, adding about `ln(N / (N - 2w))` to every estimate; the amount varied pair to pair with the automatic window, which reaches `w = 33` on `kuramoto_M7_T100`.
  On i.i.d. independent data at N=500 the shipped estimator reported +0.10 nats at `w = 20` and +0.27 at `w = 60`, where the truth is 0; on the JIDT reference fixture MI rose from 0.045 to 0.127 as `w` went from 0 to 20.
  JIDT 1.6.1 has the same term (its source questions it in a comment), so this is a deliberate departure from JIDT: `psi(N)` is replaced by the mean of `psi(m_i + 1)` over each point's reduced pool, after which the fixture value is flat in `w` (0.0452, 0.0459, 0.0457) and the `w = 0` path is bit-identical.
  Transfer entropy is unaffected: the `N` term cancels in the Frenzel-Pompe conditional form, and all eight `te_*` SPIs are unchanged to the bit.
  **Two values change**: `mi_kraskov_NN-4_DCE-AUTO` and `tlmi_kraskov_NN-4_DCE-AUTO`, by up to 0.84 nats on `kuramoto_M7_T100`.
- **`phi_Geo_*` stopped its optimiser before the model was disconnected.**
  The augmented-Lagrangian solver for geometric integrated information (`phi_g_gauss_al`, ported from the PhiToolbox, which has the same stopping rule) stopped when the objective stabilised, without checking that the cross-unit entries of the "disconnected" model had reached zero.
  On weakly coupled, strongly autocorrelated pairs it stopped at iteration 3 with about 80% of the coupling still present, so phi_G was far too small: the worst fixture pair, `kuramoto_M7_T100` (1, 5), read 1.4e-6 against the true constrained minimum 6.1e-5.
  Convergence now also requires the constraint residual to vanish, the value is evaluated at the block-diagonal projection of the final iterate, and exhausting `maxiter` warns.
  Against a direct BFGS minimisation of the Oizumi, Tsuchiya & Amari (2016) objective the error is now at most 3e-14 on every fixture pair.
  **Both `phi_Geo_t-1_norm-{0,1}` values change**, by up to 6e-5 on `kuramoto_M7_T100` and below 1e-9 elsewhere.
- **Debiased squared PLI was wrong wherever the cross-spectrum is real.**
  The backend computes `(n * PLI^2 - 1) / (n - 1)`, which equals Vinck et al.'s (2011) mean over taper pairs of `sign(Im S_k) * sign(Im S_l)` only when every sign is nonzero.
  At 0 Hz and Nyquist the tapered coefficients of a real signal are real, so every sign is 0, the correct value is 0, and the backend returned `-1/(n - 1) = -0.25` for any input; every band containing those bins was biased by that constant divided by the number of in-band bins.
  pyspi now computes the pairwise-sign-product estimator from the backend's Fourier coefficients (also about 1.4x faster).
  **The three `dspli_multitaper_mean_*` values change**, by 0.0098 (full band) and 0.0096 (half bands), the same shift for every pair.
- **Spectral SPIs accepted a `statistic` they could not apply.**
  `PhaseSlopeIndex(statistic="max")` returned the default variant's matrix under a `psi_multitaper_max_*` identifier; `statistic="delay"`, `"slope"` or `"rvalue"` on any other multitaper class returned the spectrum's 0 Hz bin, through an `except TypeError` branch meant for `GroupDelay`; and `GroupDelay(statistic="mean")` raised an `IndexError` from inside `multivariate`.
  Each class now declares the statistics it supports and the constructor raises `ValueError` otherwise.
  Every bundled config still loads; no shipped value moves.
- **The bivariate directed spectral band was open, not closed.**
  `dtf`, `dcoh`, `pdcoh`, `gpdcoh` and `ddtf` selected frequencies with `fmin < f < fmax`, dropping the 0 Hz, 0.25 and Nyquist bins that their identifiers name and that every other band in the module includes; on low-pass pairs the maximum sits at 0 Hz.
  The band is now closed.
  **All 30 values change**, by up to 0.55 for the `max` variants and 0.034 for the `mean` variants on the drift fixtures.
- **`ddtf_*` multiplied by `sqrt(PDC)` instead of the partial coherence.**
  Korzeniewska et al. (2003) define the direct DTF as the full-frequency DTF times the partial coherence; the backend multiplies by `sqrt(PDC)` although its own docstring says partial coherence, and for a pair the bivariate PDC equals the DTF.
  pyspi now takes the partial coherence from the model spectrum `H Sigma H^H`, using the same two private backend properties `dcoh` already relies on.
  **All six `ddtf_*` values change**, by up to 0.31.
- **Parametric spectral Granger causality was fitted on an indefinite autocovariance.**
  nitime's `autocov_vector` averages lag `k` over `T - k` products and does not remove the mean, so the block-Toeplitz matrix handed to the Levinson-Wiggins-Robinson recursion need not be positive semidefinite.
  On `kuramoto_M7_T100` this gave an indefinite innovation covariance on 7 of 21 pairs at order 1 and on all 21 at orders 2 and 20, unstable models on 18 of 21 pairs at order 20, and negative or NaN "causality" values that `nanmean` then averaged.
  It is also why `sgc_parametric_*_order-None` was recorded as unestimable on that fixture: the NaN in BIC's log-determinant never compares greater, so the order search never stopped.
  The earlier explanation in these notes ("BIC improves all the way to lag 49") was wrong.
  pyspi now fits the pair itself with the same LWR recursion, criterion and first-turn-over rule, on the Yule-Walker autocovariance (about the mean, divided by `T`), which is positive semidefinite by construction; `max_order` is now the largest order tried, where nitime stopped at `max_order - 2`.
  Against a dense Yule-Walker solve plus Geweke's formula the fixture values agree to rounding; in a Monte Carlo on a bivariate VAR(1) at T=100 the order-20 RMSE falls 1.5-3.7x and invalid fits on a persistent process go from 37 of 300 runs to 0.
  Automatic order selection at T=100 now picks 1 on `var1_M3_T100` (the true order), 1-2 on `cml_M5_T100` and 1-5 on `kuramoto_M7_T100`.
  **All 18 `sgc_parametric_*` values change**; the six `order-None` SPIs on `kuramoto_M7_T100` go from unestimable to finite, and `KNOWN_UNESTIMABLE` was left empty (the GroupDelay entry below later adds the six `gd_*` cases on `var1` and `cml`).
- **Parametric spectral Granger causality labelled its frequencies wrongly.**
  nitime's `GrangerAnalyzer.frequencies` is `linspace(0, fs/2, 513)`, but the values are evaluated by `freqz` at `k/1026` cycles per sample, so the bin labelled 0.5 is really 0.499.
  pyspi now uses the frequencies the spectrum was evaluated at.
  Only the `sgc_parametric_*_fmin-0-25_fmax-0-5_*` band statistics move, by at most 0.04.
- **Wavelet phase slope index applied its frequency floor in the wrong units.**
  The floor on `fmin` was `5 / T`, in cycles per sample, where the band is in Hz; it is now `5 * fs / T`.
  At the shipped `fs = 1` the two are the same float, so no value moves; at `fs = 250` the old floor let `fmin = 0` reach 1/250 of the supported value and PSI was not invariant to the unit of frequency.
- **`pec`, `pec_orth`, `pec_log` and `pec_orth_log` reported themselves unsigned.**
  Power envelope correlations lie in `[-1, 1]` and reach -0.72 on the fixtures; as unsigned, `_rmmin` shifted each column and `set_group` correlated it through `abs()`, the defect fixed for `coint_aeg_tstat` above.
  They are now signed unless `orth` and `absolute` are both set; `pec_orth_abs` and `pec_orth_log_abs` stay unsigned.
  `Cointegration` was labelled `nonlinear`; Johansen and Engle-Granger are linear, and every config said so, but the class label won the merge.
  Labels and `issigned()` only; no value moves.
- **Smaller contract fixes, no value changes.**
  `Covariance.multivariate` and `Precision.multivariate` returned the cached fit's own array, so an edit to a result through the direct API was written into `data.covariance` and read by every later SPI; they now return a copy.
  `scipy.stats.spearmanr` on a matrix checks only its first two variables for constancy and returns a scalar NaN when one is constant, so a constant series in process 0 or 1 made `SpearmanR` and `LaggedCorrelation(spearman)` fail for every pair, while the same series in another position gave NaN only for itself; the rank correlation is now assembled from `rankdata` and `corrcoef`, which is what `spearmanr` does internally and is bit-identical.
  `LaggedCorrelation.bivariate` returned one of the two one-sided correlations when the other was undefined, where `multivariate` returned NaN; both now return NaN.
  `CrossmapEntropy(history_length=-2)` returned a finite value and `StochasticInteraction(delay=-1)` returned NaN; both now refuse non-positive integers.
  `CrossmapEntropy`'s docstring said `k` gives `k - 1` source lags; the shipped `k = 1` uses one lag and so computes the same quantity as `k = 2`, which is now stated and pinned by a test.
  `CrossCorrelation` had no docstring (its description sat after the class attributes); it now documents that the normalisation is biased and that `max` is the largest signed value.
- **`coint_aeg_*` put the regressor on the column; `lmfit_*` puts it on the row.** The Engle-Granger call was `coint(z[i], z[j])`, which regresses the row *i* on the column *j*, while `LinearModel` fits `z[j]` on `z[i]`, so the two directed regression families in `full` disagreed about which index is the regressor. The asymmetry is real (the two orientations differ by ~0.8 on average on random walks, up to ~1.6), so this is not cosmetic. `aeg` now calls `coint(z[j], z[i])`: `table[i, j]` is the t-statistic of the column regressed on the row, regressor on the row, as for `lmfit_*` and as the source-by-target convention reads. **The three `coint_aeg_tstat_*` matrices are transposed** relative to r11 (and relative to 2.x, where the class was symmetrised and the orientation you got depended on visit order). `test_aeg_is_the_engle_granger_tstat_of_the_column_regressed_on_the_row` and `test_aeg_and_lmfit_agree_on_which_index_is_the_regressor` pin it against `statsmodels` and `sklearn` directly.
- **Group delay fitted its phase slope to correlated frequencies.**
  `gd_*` regresses the coherence phase on at least three independent frequencies from the largest significant coherence run, and it thinned that run to one point per `(fmax - fmin) / 50` whatever the series length.
  At T=100 that is one bin, so three adjacent bins, which share the tapers' main lobes, passed as three independent estimates, and 8% of independent white-noise pairs got a value; at T=20000 it is 200 bins, so a 0.016-wide coupled band gave three points and missed a 4-sample delay by 0.5 to 2.0 samples, and a 0.005-wide band gave nothing.
  The step is now one multitaper bandwidth, six bins at the backend's defaults, as `psi` already used: the same delay comes back within 0.13 samples, and none of 600 independent pairs gets a value at T=100 or T=1000.
  The per-bin Fisher-z coherence screen is kept; its docstring now says it is an approximation that passes about 16% of bins at nominal 5% on independent noise at five tapers.
  **All three `gd_*` values change.** On `var1_M3_T100` and `cml_M5_T100` no pair has the 13-bin significant run that three independent points need at T=100 (the longest are 12 and 10 bins), so all six are now unestimable and listed in `KNOWN_UNESTIMABLE`; on `kuramoto_M7_T100` they move by up to 0.34 samples and 30 of 68 finite entries become NaN.
- **Phase measures included the two frequencies where phase does not exist.**
  At 0 Hz and at the Nyquist frequency every tapered Fourier coefficient of a real signal is real, so the cross-spectrum's phase is exactly 0 or π and no phase statistic estimates anything there: PLV and PPC are 1 whenever the five tapers agree in sign, PLI, dsPLI and imaginary coherence are 0, wPLI is 0/0 (returned as 0), and the coherence phase is 0 or π.
  Each shipped band (`[0, 0.5]`, `[0, 0.25]`, `[0.25, 0.5]` at fs=1) contains one of the two, so every band mean carried a term that measured nothing, `plv_*_max` and `ppc_*_max` were exactly 1.0 on 16 of 42 `kuramoto_M7_T100` entries (2 of 20 on `cml_M5_T100`) for that reason alone, and `pli_*_max` and `wpli_*_max` could never fall below 0.
  The eight phase measures (`phase`, `icoh`, `plv`, `pli`, `wpli`, `dspli`, `dswpli`, `ppc`) now leave those bins out of the band before the mean or maximum, and a band holding nothing else is refused with a `ValueError`.
  Amplitude measures (`cohmag` and the directed measures) keep the closed band and `psi` its open one; the rule is stated once, in the `NonparametricSpectral` docstring.
  **33 values change**: the 18 `icoh`/`plv`/`ppc`/`pli`/`wpli`/`dspli` means by up to 0.039 and the three `phase` means by up to 0.44 rad, `plv_*_max` by up to 0.15, `ppc_*_max` by up to 0.35, and `pli_*_max`/`wpli_*_max` by up to 1.0 where every value inside the band is negative (these six maxima are no longer bundled, see **SPI set changes**). `dswpli_*` (already NaN at those bins) and `icoh_*_max` do not move.
  `cohmag`'s `name` is now "Magnitude-squared coherence", which is what it computes; PSI's docstring says its `mean` is a sum over the band; the stale "multiple sampling frequencies not yet handled" warning is gone from `spectral.py` and `wavelet.py` (both pass `fs` through).
- **The phase slope index used the backend's open band.**
  `psi_multitaper_*` summed over `fmin < f < fmax`, so the interior edge an identifier names (`0.25` in `fmin-0_fmax-0-25` and `fmin-0-25_fmax-0-5`) was left out while every other spectral SPI included it.
  PSI now follows the phase rule above (closed band less the real-valued bins), then keeps one bin per multitaper bandwidth as before.
  For `[0, 0.5]` the two bands coincide and nothing moves; **the six half-band `psi_multitaper_mean_*` values change** (by up to 0.72 on the fixtures, which is one term of a sum that scales with the band).
  The wavelet PSI is unaffected.
- **Group delay fitted its slope through the real-valued bins.**
  At 0 Hz and `fs / 2` the phase is forced to 0 or π whatever the delay, so when GroupDelay's significant coherence run reached either edge that bin entered the regression as an observation.
  For a pure delay it lies on the line anyway; for anything else it pulls the fit.
  Those two bins are now left out of the run search (the closed band less the real-valued bins, as for the phase measures).
  `gd_*` moves by at most 0.13 samples on `kuramoto_M7_T100` in the two bands that reach 0 Hz; the `[0.25, 0.5]` band is unchanged there.
- **The closed band dropped the Nyquist bin at `fs != 1`.**
  The band was `freq <= fmax` exactly, and the backend's grid is `k * fs / nfft`, whose last bin at `fs = 3, T = 100` is `1.5000000000000002`, so `cohmag`, the bivariate directed measures, `sgc` and GroupDelay silently lost the bin their `fmax` names.
  One helper, `_closed_band`, now compares both edges with a relative tolerance of 1e-9.
  At `fs = 1` every shipped edge is exact on the grid and no value moves.
- **`gwtau` left the trajectory's start out of its distance vector.**
  Kravtsova, McGee & Dawes (2023) define GW_tau on the vector of intrinsic distances `d(r, x_k)` of *every* point from the start, `k = 1..n`, whose first entry is 0.
  pyspi, like the authors' reference MATLAB (`vec_geo_dist.m`), returned the `n - 1` cumulative sums without it, so the closed-form 2-Wasserstein distance divided the same sum of squares by `n - 1` instead of `n`.
  The vector now has one entry per point and **`gwtau` changes by the constant factor `sqrt((T - 1) / T)`** (0.5% at T=100); no comparison between pairs is affected, and `gwtau(x, x)` is still 0.
  The class keeps its historical misspelling `GromovWasserstainTau`, which configs and saved results refer to; its `name` is now "Gromov-Wasserstein distance (GW_tau)".

### Changed

- **The wavelet phase-slope index's `max` variants carried no information.**
  With one epoch, mne-connectivity's wavelet coherency is a unit phasor at every time point, so `psi_wavelet_*` is the time average (`mean`) or maximum (`max`) of an instantaneous phase slope bounded by 7, not Nolte's coherence-weighted index (that is `psi_multitaper_*`).
  The maximum tends to its bound whatever the data: on independent white noise it exceeded the value for a pair at lag 2 in 85-100% of 200 runs in every band, and it is positive both ways round.
  The three `psi_wavelet_max_*` identifiers are removed from every bundled config; the class still accepts `statistic="max"`.
  The three `mean` identifiers are unchanged, and their docstring now gives the lag range their sign holds over: right in every run for lags up to 5 samples, wrong in every run at lag 12 on the full band, where the fixed 10-point grid's phase step wraps (`1 / (2 df)`).
  A time-averaged Nolte wavelet PSI on a bandwidth-spaced grid was prototyped and rejected: its null sd at T=2000 was 0.29 against the current 0.11 and it reversed at the same lag on `[0.25, 0.5]`.
- **`si_kernel_W-0.5_k-1` checked and kept.**
  It was a candidate for removal alongside the fixed-width `di`/`cce`/`xme` kernel estimators.
  Against the closed-form stochastic interaction of a Gaussian VAR(1) it converges: its bias on independent data falls from +1.0 nats at T=100 to +0.05 at T=10000 (about `T**-0.65`, a factor of 20 to 30), its bias on a coupled pair from 6 times the true value at T=100 to 0.18 times at T=10000, and it rises monotonically with coupling strength (Spearman 0.92 at T=1000).
  No other fixed width does better: W=0.25 is worse on every count and W=1.0 trades the vanishing positive bias for a permanent negative one (27-48% of the true value at T=1e5).
  At T of about 100 its absolute values are dominated by finite-sample bias, which the class docstring now says.
- **Conventions the audit found undocumented are now stated in the class docstrings** (no value changes). `gc_gaussian_*` is the Gaussian transfer entropy `0.5 ln(var_restricted / var_full)` in nats, **half** of Geweke's Granger causality (`TransferEntropy`). `xcorr_*_max` is the largest *signed* lagged correlation, so an anticorrelated pair scores a small positive value; `squared=True` gives strength without sign (`CrossCorrelation`). Score polarity differs across the causal module and is not harmonised: lower is stronger evidence for `anm`, `cds` and `reci`, higher for `ccm` and `igci` (`AdditiveNoiseModel` and siblings). `ccm_E-None_*` gives each pair the larger of its two simplex-optimal embedding dimensions in both directions, not the library variable's own (`ConvergentCrossMapping`). `softdtw*` is the raw soft-minimum cost at `gamma = 1`, which can be negative and is not the Blondel et al. (2021) divergence; its `name` was "Dynamic time warping" and is now "Soft dynamic time warping" (`SoftDynamicTimeWarping`). `lmfit_*` is the in-sample MSE of the column regressed on the row, lower meaning a better linear prediction (`LinearModel`). `prec_GraphicalLassoCV`/`cov_GraphicalLassoCV` are solver-path dependent when cross-validation picks `alpha` near 0, and the `ConvergenceWarning` is suppressed (`Estimators`). Two spectral facts are recorded here rather than in code: the multitaper estimates use `NW = 3` tapers whose bias does not shrink with `T`, and the Wilson factorisation behind the nonparametric `sgc_*` leaves residuals of 0.07-0.29 on the fixtures when run on the unpadded grid.

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
- **scikit-learn >= 1.8 changes some covariance values.** scikit-learn 1.8 added the Croux-Haesbroeck consistency correction to `MinCovDet`/`EllipticEnvelope` (scikit-learn #32117), and it cannot be switched off. `cov_`, `cov-sq_`, `prec_` and `prec-sq_` for these two estimators therefore differ from 2.0.x: maximum absolute change 0.038 (var1), 0.53 (cml) and 35.9 (kuramoto, `prec-sq`). `GraphicalLasso` and `GraphicalLassoCV` now converge at `tol=enet_tol=1e-8`; before, at `tol=1e-4`, the value tracked the solver path, which scikit-learn 1.8 also changed (#31987, #31906). Values move by at most 1e-4, and now agree across scikit-learn versions to about 1e-9. The baselines for exactly these keys were regenerated; every other array is bit-identical.
- **`anm` pins its Gaussian-process kernel.** `AdditiveNoiseModel` now passes a fixed `ConstantKernel(1.0)*RBF(1.0)` with no optimiser. It had relied on scikit-learn's default, which in 1.9 became an optimised kernel and shifted values by up to 7.6 (absolute). Values are unchanged under scikit-learn <= 1.8 and identical under 1.9. The kernel matrix is ill-conditioned (condition number ~6e11), so `anm` varies by about 1e-4 across platforms; its drift tolerance is rtol 1e-3.
- **Cross-platform drift tolerances.** The families solved through dense linear algebra (`gc_gaussian`, `di_gaussian`, `xme_gaussian`, `phi_star`, `sgc_parametric`) differ at the rounding level between BLAS/LAPACK builds even with identical package versions. Measured on Linux CI against baselines frozen on another platform, they differed by at most 2.4e-7 relative with the dependencies locked, and 7e-7 with the latest ones. They are checked at rtol 1e-6; every other SPI stays at 1e-9.
- **Config labels are consistent across all eight bundled configs.** Covariance in `fabfour`/`sonnet` is now signed and multivariate, matching `full`; DTW and Barycenter in `sonnet` are `nonlinear` again (#97 had set `linear`); the PSI time-frequency labels, stray commas and an `unisgned` typo are fixed. Label merging lets a class's `linear`/`nonlinear` win and raises on any remaining contradictory pair (linear/nonlinear, directed/undirected, signed/unsigned); `tests/test_config_labels.py` guards this.
- **No SPI reads the global NumPy RNG any more.** Two families did, so their values changed with the caller's seed, with which SPIs had run earlier, and between serial and parallel execution. (1) tslearn's subgradient barycenter took its update order from the global stream, so each of the four `bary*_sgddtw_*` values depended on how many pairs had been fitted before it and could not be reproduced for a subset of pairs at all. (2) `MinCovDet` and `EllipticEnvelope` search from random subsets of the observations, which made the eight `cov`/`prec` SPIs built on them differ from run to run. Each fit now uses its own fixed-seed generator, so a value is a function of the data alone. **These twelve SPI values change**, by about the estimators' own run-to-run spread.
- **Multitaper phase slope index ignored its frequency band.** `psi_multitaper_mean_*` passes `fmin`/`fmax` to the backend only from a fallback that ran on `TypeError`; spectral-connectivity 2.0 accepts the no-argument call, so every band variant returned the same full-band, unthinned matrix (identical to 1e-12 across the three p90 bands, and hundreds of times larger than the banded value at T=2000). The band is now always applied. **The three `psi_multitaper_mean_*` values change.** Wavelet PSI was not affected. With both changes `COMPUTATION_VERSION` is `3.0.0.r8`, so no r7 checkpoint is reused.
- **Multitaper phase slope index now follows Nolte et al. (2008).** `psi_multitaper_mean_*` is `Im sum_f conj(C(f)) C(f + df)` over neighbouring in-band frequencies one multitaper bandwidth (six FFT bins) apart, computed in pyspi from the backend's coherency. It had been the backend's sum over *all* pairs of about fifty thinned frequencies, which is not the published index: that value fell as the lag between two processes grew and changed sign near 50 samples, and no other implementation (Nolte's `data2psi`, FieldTrip, MNE-connectivity, HERMES) computes it; spectral-connectivity has since replaced it upstream, so pyspi's values would have changed on a dependency upgrade. The index now grows in proportion to the lag and its sign is reliable up to `T/12` samples. The cost is sensitivity to very short lags in a single recording: at `T=1000` a one-sample lead has d' of about 1.4, where the all-pairs sum gave about 90; from five samples up the new index is comparable or better. It stays unnormalised (the published jackknife needs epochs). **The three `psi_multitaper_mean_*` values change**; `COMPUTATION_VERSION` is `3.0.0.r9`.
- **tslearn >= 0.10 changes the four `bary*_sgddtw_*` values.**
  tslearn 0.10 changed how `dtw_barycenter_averaging_subgradient` updates and when it stops (it now stops on the cost over a whole epoch; tslearn #764), so the subgradient barycenter converges elsewhere.
  Against tslearn 0.8.1/0.9.0 the four values move by up to 0.75 (absolute) on the drift fixtures.
  The floor is raised from `>=0.6` to `>=0.10` so that a fresh install reproduces the stored baselines, and `COMPUTATION_VERSION` is `3.0.0.r10`.
- **A parallel run no longer leaves its thread pins in the parent's environment.** `compute(n_jobs>1)` set `NUMBA_NUM_THREADS=1` (and the BLAS equivalents) for its workers and never restored them. If numba had already launched its threads in that process, every later numba compilation there raised, so a serial run after a parallel one failed any SPI whose kernel was not yet compiled. The variables are restored when the pool closes.
- **Distance correlation and the Kraskov estimators are faster, with identical values.** `dcorr` and `dcorr_biased` evaluated both orders of every pair and discarded one; they now evaluate the retained order only (2.0x at M=8, T=5000). Kraskov MI, time-lagged MI and transfer entropy counted marginal neighbours by materialising a Python list per observation, and with a Theiler window by one tree query per observation; counts now come from one batched query, less the temporally excluded neighbours found inside the ball, one time offset at a time in O(N) memory (3-10x on the seven p90 Kraskov SPIs at T=5000, more with `dyn_corr_excl`). Radii, strict boundaries, the exclusion window and errors are unchanged and every output is bit-identical, so `COMPUTATION_VERSION` does not move.
- **The five directed spectral measures share one spectral factorisation per pair.** `dtf`, `dcoh`, `pdcoh`, `gpdcoh` and `ddtf` (30 SPIs in `benchmarked_p90`) are all read off the Wilson minimum-phase factor of a pair's cross-spectrum, which the backend recomputed on every access: two to four times per measure, fourteen per ordered pair across the five. It is now computed once per ordered pair and the five classes form one scheduling task, so workers no longer repeat it per class. 7.6x faster on those SPIs (184 s to 24 s at M=40, T=1000; they were 44% of a full `benchmarked_p90` run at M=64), with about 40% less memory held per pair, since the Fourier coefficients are released once factorised. Every output is bit-identical.
- **Faster, with bit-identical values, from the r11 audit.**
  `hhg` built each 2x2 contingency table with an O(n^3) scan in hyppo; the counts are row-wise max-ranks plus a 2-D dominance count and are now computed in O(n^2 log n) with a Fenwick tree, combined with hyppo's own integer arithmetic (71 s to 8 s at M=4, T=1000, and 11.3 s to 0.68 s per pair).
  `hsic`, `hhg`, `mgc`, `lcss`, `softdtw` and `gwtau` evaluate only the retained orientation of each pair, as `dcorr` already did (another 2x).
  `LaggedCorrelation(kendall)` ran `kendalltau` on all pairs of the 2M stacked lead and lag rows and discarded the lead-lead and lag-lag half; it now computes the M x M block only (2.2x on the fourteen `corr_kendall_*`).
  `StochasticInteraction` and kernel `MutualInfo` compute `i < j` only, and `StochasticInteraction` and `CrossmapEntropy` cache each process's own conditional entropy on `data.entropy` (2-3.7x on `si_*`, `mi_kernel_*` and `xme_*`).
  The bivariate directed spectral measures recomputed the same DPSS tapers for every pair; they are now cached per `(T, fs)` (1.6x on the 30 `dtf`/`dcoh`/`pdcoh`/`gpdcoh`/`ddtf` SPIs at M=10, T=1000).
  `ccm_*` builds each shadow manifold's random libraries, KD-tree, neighbours and weights once per library variable and applies them to all of its targets, reproducing pyEDM 2.5's `CCM` operation for operation, including its seeding (6-7x at M=10, T=1000; pinned against `pyEDM.CCM` in both directions).
  `anm` factorises each cause's GP kernel once rather than once per pair and skips the log marginal likelihood it never used (6.9x at M=10, T=1000).
  Every output was checked bit-identical on the drift fixtures and at M=10, T=1000.

### Dependencies

- **Floors, relative to 2.0.2** (`pyproject.toml` is the source of truth):

  | | 2.0.2 | 3.0.0 |
  |:--|:--|:--|
  | Python | `>=3.8` | **`>=3.11`** (scikit-learn >= 1.8 needs it; 3.10 reaches end of life in October 2026). CI matrix 3.11-3.13 |
  | numpy | `<2.0.0` | **`>=2.0`** |
  | scikit-learn | unpinned | **`>=1.8`** |
  | pyEDM | `==1.15.2.0` | **`>=2.5`** |
  | mne | `==0.23.0` | `>=1.6`, plus new `mne-connectivity>=0.5` |
  | spectral-connectivity | unpinned | `>=1.1,<3` |
  | scipy / pandas / statsmodels / nitime / hyppo / tslearn | unpinned | `>=1.11` / `>=2.1` / `>=0.14` / `>=0.10` / `>=0.4` / `>=0.10`; new `dtaidistance>=2.3` |
  | removed | `jpype1`, `cdt==0.5.23`, `torch`, `h5py`, `seaborn`, `sktime`, `dill`, `future`, `setuptools<81` | (`aeon` is an optional `tsfile` extra for `.ts` files) |

  The pinned versions were checked against the 2.0.2 wheel metadata and the 3.0.0 `pyproject.toml`.
- Dropped unused runtime dependencies (`h5py`, `seaborn`, and the intermediate `plotly`, `matplotlib`, `nbformat`). Plotting and notebook packages moved to a `bench` extra.
- Dropped the `setuptools>=68,<80` pin. It existed because pyEDM imported `pkg_resources`; pyEDM 2.5 no longer does, so the floor is now `pyEDM>=2.5` and modern setuptools is usable.
- `pandas>=2.1` for `DataFrame.stack(future_stack=True)`, the pandas 3 semantics.
- **`spectral-connectivity` is now upper-bounded: `>=1.1,<3`.** `DirectedCoherence` reads two *private* `Connectivity` properties (`_transfer_function`, `_noise_covariance`) because the public `directed_coherence()` is wrong twice over (above), so an open-ended floor was a promise pyspi cannot keep. The range is verified end to end against the oldest published 1.1 (1.1.0) and the locked current release: both expose every symbol pyspi uses. 1.1.x additionally lacks `transforms.prepare_time_series` (there is a fallback) and returns a 400- rather than 401-point frequency grid, so band statistics differ marginally across the supported range. `tests/test_directionality.py` fails loudly if a private property disappears, rather than the measure silently changing meaning.
- **`cdt` and `torch` are gone.** pyspi used four functions from `cdt.causality.pairwise` — the ANM independence score, the conditional distribution similarity statistic, the RECI regression-error score and IGCI — now transcribed in `pyspi/lib/pairwise_causal.py` on NumPy/SciPy/scikit-learn, with cdt's MIT licence retained at `pyspi/lib/LICENSE-cdt.txt`. **No value changes**: verified bit-identical against cdt 0.6 over 60 mixed random pairs, with five frozen into `tests/data/fixtures/cdt_pairwise_reference.npz` so the evidence outlives the dependency. torch drove no pyspi computation — `InterDependenceScore` is NumPy — and entered only because cdt eagerly imports its Torch-backed models at package load. Measured: installed size **1.1 GB → 567 MB**, and the 1.5–1.7 s `import cdt.causality.pairwise` (paid once per worker under `compute(n_jobs=…)`) is gone. Per-call runtime is unchanged; the win is install size and start-up, not throughput.
- Build requires `setuptools>=77`, matching the PEP 639 licence metadata already in use; older setuptools would build a wheel with no licence metadata.
- New extras: `bench` and `tsfile`; `testing` now requires `pytest>=7` (2.0.2 pinned `pytest==5.4.2`).
- Licence: **GPL-3.0-or-later** (2.0.2: GPL-3.0), as agreed by the maintainer. `pyspi/lib/LICENSE-jidt.txt` joins the cdt and ids notices in `license-files`.

### Testing

- The test suite previously **did not run at all** — collection aborted on an undeclared `dill` dependency. Fixed.
- New `tests/test_infotheory_analytic.py`: closed-form checks against `-0.5*ln(1-rho^2)`, `0.5*ln(2*pi*e*sigma^2)`, analytic Gaussian TE on a known AR(1), independence, and information-theoretic identities. Previously the suite contained three assertions comparing a computed value to an independently-known one, and five estimator classes were constructed but never computed.
- New `tests/test_directionality.py` pins the row=source convention for every directed SPI family.
- **Drift tolerances are per-SPI, not per-module.** The suite applied a 1e-2 relative band to every SPI in `causal` and `misc` on the assumption that cdt's optimisers, GP restarts and randomised independence tests made them irreproducible. Measured, that is false: computing the full config twice per fixture under the suite's own protocol reproduces **322 of 322 SPIs bit-exactly** (the 322-SPI `full` of that date; `full` is now 316 after the dspli/dswpli removals) on all three fixtures — comparing the non-finite masks as well as the finite values, so an SPI whose NaN pattern moved between the two runs could not be scored as identical — including every `anm`/`cds`/`reci`/`ccm`, every `coint_*`, `gpfit_*` (`GaussianProcessRegressor` defaults to `n_restarts_optimizer=0`, so there are no random restarts), `lmfit_*` (`random_state` pinned) and `ids`. A module-wide band over 62 SPIs, ~50 of them deterministic, is slack wide enough to hide the regressions this suite exists to catch. The map of loosened SPIs is now keyed by identifier and is empty; `tests/tools/measure_reproducibility.py` regenerates the evidence.
- Merged from `main`: `CITATION.cff` and the PyPI trusted-publishing workflow (`publish.yaml`, which reads the static version in `pyproject.toml`). `setup.py` and `requirements.txt` are deleted in favour of `pyproject.toml`.
- New `tests/test_jidt_reference.py`: JIDT 1.6.1 outputs hard-coded as constants and asserted against the NumPy estimators, with the Java generator in `tests/tools/jidt_reference/`. No Java is needed to run it (see Removed: Java and JIDT).
- New `tests/test_config_labels.py` and `tests/test_import_hygiene.py`; the deprecation shims (`tests/test_api_compat.py`) and `n_jobs` resolution are tested.
- New CLI exit-status tests, and a test that the spectral factorisation's non-convergence warning is not swallowed.
- **The drift suite reported violations and never failed.** Tolerance breaches went to a session-end banner, which made every tolerance in the file decorative: a deterministic SPI could move by any amount and the run still exited 0. It also returned early when a baseline had no finite entries, so an SPI frozen as an all-NaN column agreed with itself forever — which is how three `gd_*` SPIs stayed broken. Violations now fail, an empty baseline is rejected, and two new gates require that no SPI raises and that none produces an entirely non-finite column. The generator refuses to freeze a run with either condition. Documented exceptions are shared between generator and suite as `KNOWN_UNESTIMABLE`, and the suite fails if a listed exception starts succeeding.
  The map is currently empty: the six `sgc_parametric_*_order-None` SPIs on `kuramoto_M7_T100` were listed until r11, when the parametric VAR fit moved to the Yule-Walker autocovariance and the order search started turning over (see **Fixed**).
- `test_whether_calculator_computes` ran the default config and asserted nothing; it now requires an empty `calc.errors` and a finite value in every column, on a coupled VAR(1) rather than the particular white-noise fixture on which `gd_*` was observed to be empty.
- New `test_cache_sharing_never_changes_a_value`: computing the whole config against one Data, where every cache is shared, must equal computing each SPI against its own. This is the automatic coverage check for cache-key omissions — a per-SPI "different identifier implies different cache key" rule is the wrong invariant, since several classes cache a shared intermediate on purpose and apply the differing parameters after the lookup.
- New CI job builds the wheel, installs it into a clean environment, and from a directory containing no pyspi source resolves every bundled config and dataset, computes `fabfour`, and round-trips the NPZ through the CLI. The blocking slow regression job installs from `uv.lock`, the resolution the baselines were frozen against. A non-blocking job, "regression (latest deps, advisory)", reports upstream drift with `PYSPI_DRIFT_TIER=latest`. The unit and packaging workflows run on pull requests and on pushes to `main` only (no duplicate runs), and the slow suite also runs on pushes to `main`. The CI matrix is Python 3.11-3.13. The baseline generator records provenance in each `.npz` (dependency versions, platform, `COMPUTATION_VERSION`), and the workflow that regenerates baselines installs from `uv.lock`.
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
- Vlachos, M., Kollios, G. & Gunopulos, D. (2002). Discovering similar multidimensional trajectories. *Proc. ICDE*, 673–684. — LCSS with a band that limits which pairs may match, as `lcss_constraint-*` now computes.
- Mehta, R., Chung, J., Shen, C., Xu, T. & Vogelstein, J. (2020). Independence testing for multivariate time series. *arXiv:1908.06486*. — the lagged sum behind `dcorrx` and `mgcx`, `sum_l (n - l)/n * S(target[l:], source[:n-l])`.
- Gretton, A., Bousquet, O., Smola, A. & Schölkopf, B. (2005). Measuring statistical dependence with Hilbert-Schmidt norms. *ALT*, 63–77. — HSIC, with a median-heuristic kernel width per variable.
- Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F. & Pennartz, C. (2011). An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. *NeuroImage* 55, 1548–1565. — the debiased squared PLI and wPLI as means over taper pairs.
- Oizumi, M., Tsuchiya, N. & Amari, S. (2016). Unified framework for information integration based on information geometry. *PNAS* 113, 14817–14822. — geometric integrated information as the KL divergence to the nearest disconnected model, the constrained minimum `phi_Geo_*` must reach.
- Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. & Kasicki, S. (2003). Determination of information flow direction among brain structures by a modified directed transfer function (dDTF) method. *J. Neurosci. Methods* 125, 195–207. — the direct DTF as full-frequency DTF times partial coherence.
- Geweke, J. (1982). Measurement of linear dependence and feedback between multiple time series. *J. Am. Stat. Assoc.* 77, 304–313. — spectral Granger causality, checked on an exact VAR spectrum and through a Yule-Walker fit.
- Cuturi, M. & Blondel, M. (2017). Soft-DTW: a differentiable loss function for time-series. *ICML*. — the soft-DTW recursion that the constrained `softdtw_constraint-*` variants are checked against.
- Lizier, J. (2014). JIDT: an information-theoretic toolkit. *Front. Robot. AI* 1, 11. — the reference implementation the NumPy port was validated against.

### Migrating from 2.x

| 2.x | 3.0 |
|:----|:----|
| `Calculator(subset="all")` | `Calculator(config="full")` (old form still works, with a `DeprecationWarning`, until 4.0) |
| `Calculator(subset="fast")` (also `fabfour`, `sonnet`) | `Calculator(config="fast")` (other names unchanged) |
| `Calculator(configfile="my.yaml")` | `Calculator(config="my.yaml")` (deprecated shim as above) |
| `Calculator(normalise=False)` | `Calculator(zscore=False)` (deprecated shim; also on `Data` and `CalculatorFrame`) |
| `Calculator(dataset, name, labels, "all", "my.yaml", False, False)` (positional) | pass `config=`, `detrend=`, `zscore=` as keywords; positions after `subset` no longer line up |
| `Data(..., normalise=False)` | `Data(..., zscore=False)` |
| `pyspi/config.yaml` | `pyspi/configs/full.yaml` |
| `pyspi/fast_config.yaml` | `pyspi/configs/fast.yaml` |
| `pyspi/sonnet_config.yaml` | `pyspi/configs/sonnet.yaml` |
| `pyspi/fabfour_config.yaml` | `pyspi/configs/fabfour.yaml` |
| `pyspi/phiconf.yaml` | removed |
| *(no 2.x equivalent)* | `pyspi/configs/benchmarked_p{99,95,90,80}.yaml`, new cost-pruned sets |
| `load_dataset("cml7" \| "var1" \| "kuramoto")` | removed (test fixtures) |
| `utils.normalise` | removed (min-max; use `scipy.stats.zscore`) |
| `utils.standardise`, `utils.strshort` | removed (unused) |
| `utils.check_optional_deps` | removed (no optional runtime deps remain) |
| `JIDTBase` | `InfoTheoryBase` |
| `utils.is_jpype_jvm_available` | removed (no JVM) |
| `setup.py`, `requirements.txt` | removed; install metadata is in `pyproject.toml` |
| `MutualInfo`/`TimeLaggedMutualInfo`/`TransferEntropy(estimator="kozachenko")` | raises `NotImplementedError`; use `estimator="kraskov"` (2.x returned NaN) |
| SPI classes given parameters their estimator ignores (e.g. symbolic/kernel TE with `k_tau`, `l_history`, `l_tau`) | raises; remove them from 2.x YAML files |
| `CorrelationFrame.get_pvalues()`, `.compute_significant_values()`, `.get_average_correlation(remove_insig=True)` | raise `NotImplementedError` (the p-values were invalid) |
| `Data.to_numpy()` returning a writable view; `Data` aliasing its input | read-only; `Data` copies and freezes its input |
| `parse_bivariate` with ambiguous `i`/`j` (built a `Warning` and carried on) | raises `ValueError` |
| pickled results tables | cannot be loaded; re-save with `Calculator.save()` (`.npz`) and read with `pyspi.load_table()` |
| `A[i, j]` of `sgc`, `dcoh`, `pdcoh`, `gpdcoh`, `dtf`, `ddtf` (influence of j on i) | transposed: row = source, column = target; transpose saved 2.x results |
| SPI identifiers in saved 2.x results | 23 renamed, 14 removed, 46 added; see [SPI set changes](#spi-set-changes) |
| `import pyspi` replacing `sys.stdout`/`sys.stderr`, seeding NumPy's global RNG, setting `OMP_NUM_THREADS` | no longer |
| `phase_multitaper_max_*` | removed; `CoherencePhase(statistic="max")` raises because ordinary maxima of wrapped angles are branch-cut dependent |

### SPI set changes

**Relative to pyspi 2.0.2 (the current PyPI release), `full` goes from 285 SPIs to 299.** The delta below was computed by instantiating `Calculator(subset="all")` under 2.0.2 and `Calculator(config="full")` under 3.0.0 (no computation) and comparing identifiers, with renames matched by class and parameters.

| Change | SPIs | Families |
|:-------|-----:|:---------|
| 2.0.2 `config.yaml` | 285 | |
| Unchanged identifier | 230 | (values may still change; see "Values changed") |
| **Renamed** (identifier only, except DTW) | 23 | `sgc` (11), Kraskov DCE (4), `cce`/`di_gaussian` `n` (4), auto-embedding method (3), DTW Sakoe-Chiba (1) |
| **Removed** | 32 | with the port (14): multitaper `dspli`/`dswpli` maxima (6), `phase_multitaper_max` (3), symbolic TE (2), `di_kernel`, `di_kozachenko`, `igci`; after the audit (15): `*_EllipticEnvelope` (4), `pli`/`wpli` multitaper maxima (6), `lmfit_Lasso`, `lmfit_ElasticNet`, `bary_euclidean_mean`, `cce_kernel_W-0.5_n-5`, `xme_kernel_W-0.5_k10`; after the round-3 simulations (3): `psi_wavelet_max_*` |
| **Added** | 46 | `LaggedCorrelation` (42), `CrossPairwiseDistance` (2), `di_kraskov_NN-4_n-5` (1), `PhaseAlignedCorrelation` (1) |
| 3.0.0 `full.yaml` | 299 | 285 − 32 + 46 |

Other bundled sets: `fast` goes from 217 SPIs (2.0.2 `fast_config.yaml`) to 194, `fabfour` (4) keeps its size, and `sonnet` keeps its 14 with one substitution (its `pli_multitaper_max_fs-1_fmin-0_fmax-0-5`, degenerate as below, becomes the band `mean`). The cost-pruned `benchmarked_p99`, `p95`, `p90` and `p80` sets are new in 3.0 (294, 281, 266 and 239 SPIs). The complete added, removed and renamed identifier lists are in [`docs/spi_changes_2.0.2_to_3.0.0.md`](docs/spi_changes_2.0.2_to_3.0.0.md).

Every removal and rename below is deliberate. (An earlier draft of these notes said "328 to 322, nothing else moved"; that compared against an intermediate state of this branch, not against 2.0.2, and omitted the 44 additions and the IGCI removal.)

The unsupported coherence-phase maxima are removed from shipped configs. Other disabled variants (including `InformationGeometricCausalInference`, active as `igci` in 2.0.2) remain commented with the evidence for switching them off and the condition that would justify switching them back on.

**Removed: the multitaper `dspli`/`dswpli` maxima (6).** The `dspli_multitaper_max_*` and `dswpli_multitaper_max_*` variants (bands 0-0.25, 0.25-0.5 and 0-0.5) return exactly 1.0 for essentially any input, including white noise at T=100, 1,000 and 10,000, and coupled AR processes. With a single recording the debiased statistics are estimated across only a few tapers, so at any frequency where the tapers agree in sign the value is exactly 1 (28 of 501 frequency bins in a T=1000 noise test), and the maximum over frequencies is 1. The `mean` variants are kept. (Counts: `full` 322 to 316, `fast` 213 to 207, `benchmarked_p99` 318 to 312, `p95` 305 to 299, `p90` 289 to 283, `p80` 261 to 255.)

**Removed after the audit (15).** The per-SPI audit (see **Fixed**) measured every bundled SPI on the three drift fixtures and at `M = 10, T = 1000`, and found fifteen that carry no information of their own. Bit-identical duplicates: the four `*_EllipticEnvelope` SPIs equal `*_MinCovDet` on every input, because `sklearn.covariance.EllipticEnvelope.fit` is `MinCovDet.fit` plus an outlier threshold. Constants: `lmfit_Lasso` is `(T - 1) / T` for every pair (`Lasso(alpha=1)` zeroes the slope of any z-scored regressor), `lmfit_ElasticNet` likewise unless `|r| > 0.5` (constant on two of three fixtures), and `bary_euclidean_mean` is identically 0 (the Euclidean barycenter of two zero-mean series). Degenerate: the six `pli_multitaper_max_*` and `wpli_multitaper_max_*` are exactly 1.0 for every pair once `T >= 1000`, by the same mechanism as the `dspli`/`dswpli` maxima above (a mean of a few taper signs takes a handful of values, and one bin in the band always reaches 1); and `cce_kernel_W-0.5_n-5` and `xme_kernel_W-0.5_k10` are fixed-width box-kernel entropy estimates in ten to eleven dimensions whose bias does not shrink with `T` (the `di_kernel` defect). Every class remains constructible with these arguments; only the bundled configs change. (Counts: `full` 317 to 302, `fast` 207 to 197, `benchmarked_p99` 312 to 297, `p95` 299 to 284, `p90` 283 to 269, `p80` 255 to 242; the three `psi_wavelet_max_*` removed after the round-3 simulations below take them to 299, 194, 294, 281, 266 and 239.) Deterministic derived quantities were *kept*: `je_gaussian`/`ce_gaussian` are a constant minus `mi_gaussian` under z-scoring, `ccm_*_diff` is `ccm_*_mean - ccm_*_mean.T`, bivariate `pdcoh_*`/`gpdcoh_*` reduce to `dtf_*`/`dcoh_*^2` by 2×2 algebra, and `phi_*_norm-1` is about `phi_*_norm-0 / 1.42`; they are redundant only given their partner and are listed here so a user selecting features knows.

**Removed (32)**

| SPI | Why |
|:----|:----|
| `te_symbolic_k-1` | A length-1 ordinal pattern has one symbol, so TE is identically zero. |
| `te_symbolic_k-10` | Severely undersampled and unvalidated at `T=100`: `10!` symbols against ~91 usable samples. On var1 and cml every joint count is 1, so the value tracks sample size rather than dependence; that does not hold universally (kuramoto: 3 of 42 pairs). Still constructible, and defensible for long series. |
| `di_kernel_W-0.5` | ~3.8-4.4 on independent data at every `T` from 100 to 8000. |
| `di_kozachenko` | Negative values, for a nonnegative quantity. |
| `phase_multitaper_max_*` (3) | Ordinary maxima of wrapped phase depend on the branch cut and were exactly π off-diagonal on the VAR fixture. The constructors now refuse this statistic. |
| `dspli_multitaper_max_*`, `dswpli_multitaper_max_*` (6) | Constant 1.0 on essentially any input (above). |
| `igci` | Active in 2.0.2's `config.yaml`; commented out of `full.yaml` as a disabled heuristic. The class remains as `InformationGeometricCausalInference`. |
| `cov_EllipticEnvelope`, `cov-sq_EllipticEnvelope`, `prec_EllipticEnvelope`, `prec-sq_EllipticEnvelope` (4) | Bit-identical to the `*_MinCovDet` SPIs on every input (audit). |
| `lmfit_Lasso`, `lmfit_ElasticNet` (2) | Constant `(T - 1) / T` (Lasso, always; ElasticNet unless `|r| > 0.5`) at sklearn's `alpha = 1` on z-scored data (audit). |
| `bary_euclidean_mean` | Identically 0: the Euclidean barycenter of two zero-mean series has mean 0 (audit). |
| `cce_kernel_W-0.5_n-5`, `xme_kernel_W-0.5_k10` (2) | Fixed-width box kernel in ten to eleven dimensions; the bias does not shrink with `T` (the `di_kernel` defect). `xme_kernel_W-0.5_k1` stays (audit). |
| `pli_multitaper_max_*`, `wpli_multitaper_max_*` (6) | Exactly 1.0 for every pair at `T >= 1000` (the `dspli` maxima defect); `mean` variants kept (audit). |
| `psi_wavelet_max_*` (3) | The maximum over time of a bounded instantaneous phase slope: positive under the null in both orientations and above a lag-2 coupled pair in 85-100% of independent runs; `mean` variants kept (round-3 simulation). |

**Added (46)**

| SPI | Why |
|:----|:----|
| `corr_{pearson,spearman,kendall}_tau-{1,2,3,4,5,10,20}` and `..._tau-N-sq` (42) | `LaggedCorrelation`: the mean of the two one-sided lag-τ correlations, `0.5 * (corr(x[τ:], y[:-τ]) + corr(y[τ:], x[:-τ]))`, plain and squared, for three correlation types at seven lags. It is symmetric (hence undirected), and a pure delay `y(t) = x(t - τ)` scores about 0.5, not 1. |
| `xpdist_euclidean_tau-1_{mean,min}_rmse` (2) | `CrossPairwiseDistance`: shifted-path Euclidean upper bound on DTW. |
| `di_kraskov_NN-4_n-5` (1) | Direct KSG/Frenzel-Pompe conditional-MI estimate of directed information; the validated nonlinear replacement for the two dropped variants. Not in the `benchmarked_p*` sets until it has been timed. |
| `phasealigned_welch_nperseg-auto` (1) | `PhaseAlignedCorrelation`: zero-lag correlation maximised over all-pass (phase-only) filtering of one process, `sum_f w_f |S_jk(f)| / sqrt(sum_f w_f S_jj(f) sum_f w_f S_kk(f))` from Hann-window Welch cross-spectra. Completes the ladder `r <= max xcorr <= P`; a pure delay gives `max xcorr == P`. Undirected, unsigned, in [0, 1], with a null floor of about `sqrt(pi / (4K))` for `K` segments; `nperseg` defaults to the largest power of two not above `T / 8` within [8, 256], refusing `T < 20`. Not in the `benchmarked_p*` sets until it has been timed. |

**Renamed (23)** — no value change, except the DTW row. In every case a parameter that changes the measure was absent from, or misreported by, the identifier.

| Was | Is | Why |
|:----|:---|:----|
| `sgc_*_fmin-0_*` (11) | `sgc_*_fmin-1e-05_*` | Spectral GC is undefined at zero frequency, so `fmin=0` is overridden — but the identifier was built from the argument. |
| `mi_kraskov_NN-4_DCE`, `tlmi_kraskov_NN-4_DCE`, `te_kraskov_NN-4_DCE_k-1_...`, `te_kraskov_NN-4_DCE_k-2_...` (4) | `..._DCE-AUTO...` | The Theiler window's value, not just its presence: 5, 10 and `"AUTO"` shared one name. |
| `cce_gaussian`, `cce_kernel_W-0.5`, `cce_kozachenko`, `di_gaussian` (4) | `..._n-5` | `n` changes the measure. |
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4`, `te_kraskov_NN-4_k-max-10_tau-max-4`, `gc_gaussian_k-max-10_tau-max-2` (3) | `..._MAX-CORR-AIS_k-max-*` (the first also `DCE` to `DCE-AUTO`) | The auto-embedding method changes what is computed. Values are those of 2.0.2's search (it already selected the source embedding too). |
| `dtw_constraint-sakoe-chiba` (1) | `dtw_constraint-sakoe-chiba_radius-auto` | The radius is now chosen from the series length instead of tslearn's default of 1. **Values change** (up to 8.5 versus 2.0.1). |

Symbolic transfer entropy identifiers also lose the three embedding parameters the estimator never applied: `te_symbolic_k-<k>_kt-1_l-1_lt-1` → `te_symbolic_k-<k>`. No symbolic variant is shipped, so no bundled SPI is affected.

Values for the six directed spectral SPIs listed under **Fixed** are also transposed relative to 2.x.

**Values changed in r11 and r12** (the per-SPI audit and its deferred findings; see **Fixed**).
In r11, 72 of the then 317 `full` identifiers changed value relative to r10; r12 changes the further identifiers listed in the second table below, relative to r11.
Maximum absolute change on the drift fixtures `var1_M3_T100` / `cml_M5_T100` / `kuramoto_M7_T100`:

| SPIs | Count | Max change | Cause |
|:-----|------:|:-----------|:------|
| `ccm_E-{None,1,10}_{mean,max,diff}` | 9 | 0.68 / 1.23 / 2.44 | Transposed to source-by-target; library sizes reach the largest library. |
| `softdtw_constraint-{sakoe-chiba,itakura}` | 2 | 204 / 46 / 239 | The constraint is now applied. |
| `dcorrx_maxlag-{1,10}`, `mgcx_maxlag-{1,10}` | 4 | 1.42 / 1.75 / 1.16 | Transposed to source-by-target; lagged terms follow the published definition. |
| `hsic`, `hsic_biased` | 2 | 0.004 / 0.005 / 0.016 | Each variable's own kernel width. |
| `mi_kraskov_NN-4_DCE-AUTO`, `tlmi_kraskov_NN-4_DCE-AUTO` | 2 | 0.10 / 0.17 / 0.84 | Theiler-window digamma term. |
| `phi_Geo_t-1_norm-{0,1}` | 2 | 3e-10 / 9e-10 / 6e-5 | Optimiser runs to a disconnected model. |
| `dspli_multitaper_mean_*` | 3 | 0.0098 on every fixture | Debiased squared PLI at real-valued bins. |
| `dtf_*`, `pdcoh_*`, `gpdcoh_*`, `dcoh_*` | 24 | 0.038 / 0.55 / 0.36 | Closed frequency band. |
| `ddtf_*` | 6 | 0.037 / 0.060 / 0.31 | Closed band, and partial coherence in place of `sqrt(PDC)`. |
| `sgc_parametric_*` | 18 | 2.8 / 6.7 / 7.2 | Yule-Walker VAR fit and corrected frequency labels; the six `order-None` SPIs on `kuramoto_M7_T100` go from unestimable to finite, and the `[0.25, 0.5]` `order-1` and all `order-20` ones there from partly NaN to finite. |

Everything else in `full` was unchanged to the bit on all three fixtures in r11 (`anm` to its documented platform tolerance).

r12 (relative to r11; same fixtures, same columns):

| SPIs | Count | Max change | Cause |
|:-----|------:|:-----------|:------|
| `coint_aeg_tstat_*` | 3 | 1.64 / 25.1 / 2.32 | Transposed: the column is regressed on the row, as for `lmfit_*`. |
| `gd_multitaper_delay_*` | 3 | all NaN / all NaN / 0.34 (30 of 68 finite entries become NaN) | Independent frequencies one multitaper bandwidth apart; `var1` and `cml` have no 13-bin significant run at T=100. |
| `phase_multitaper_mean_*` | 3 | 0.19 / 0.44 / 0.41 rad | Real-valued 0 Hz and Nyquist bins left out of the band. |
| `icoh`, `plv`, `ppc`, `pli`, `wpli`, `dspli` `_multitaper_mean_*` | 18 | 0.035 / 0.035 / 0.039 | As above. |
| `plv_multitaper_max_*`, `ppc_multitaper_max_*` | 6 | 0 / 0.26 / 0.35 | As above; the exact-1.0 entries (16 of 42 on kuramoto, 2 of 20 on cml) are gone. |
| `pli_multitaper_max_*`, `wpli_multitaper_max_*` | 6 | 0.60 / 0 / 1.0 | As above (can now be negative); no longer bundled. |
| `psi_multitaper_mean_fs-1_fmin-0_fmax-0-25`, `..._fmin-0-25_fmax-0-5` | 2 | 0.57 / 0.72 / 0.48 | Closed band less the real-valued bins in place of the backend's open band; the full-band PSI is unchanged. |
| `gd_multitaper_delay_*` (again) | 2 | NaN / NaN / 0.13 | Real-valued bins left out of the fit; `[0.25, 0.5]` unchanged. |
| `gwtau` | 1 | 0.041 / 0.053 / 0.017 | Constant factor `sqrt((T - 1) / T)`: the start of the trajectory is in the distance vector. |

Everything else in `full` is unchanged to the bit between r11 and r12 on all three fixtures (`dswpli_*`, `icoh_*_max`, `cohmag_*`, the full-band `psi` and the wavelet `psi` included).

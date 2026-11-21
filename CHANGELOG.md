# Changelog

## 3.0.0

A major overhaul.
The Java/JIDT dependency is gone, the `Calculator` API is simplified, and several long-standing correctness bugs are fixed.
**This release contains breaking changes**: see [Migrating from 2.x](#migrating-from-2x).

**Upgrading from 2.0.2 (the current PyPI release), at a glance:**

- Python **>= 3.11** (2.0.2: >= 3.8), numpy **>= 2** (2.0.2: `<2`), scikit-learn **>= 1.8**, pyEDM **>= 2.5** (2.0.2: `==1.15.2.0`), mne **>= 1.6** (2.0.2: `==0.23.0`), spectral-connectivity **>= 2.0** (2.0.2: unpinned).
  No Java, `jpype`, `cdt` or `torch`.
  See [Dependencies](#dependencies).
- The default `full` set goes from **285 SPIs (2.0.2) to 231**: 74 added, 128 removed, 22 renamed (identifier only), 135 unchanged by name.
  The exact lists are in [SPI set changes](#spi-set-changes) and [`docs/spi_changes_2.0.2_to_3.0.0.md`](docs/spi_changes_2.0.2_to_3.0.0.md).
  A change of name or value means results saved under 2.x cannot be joined to 3.0 by identifier without that table.
- The old `Calculator(subset=, configfile=, normalise=)` spellings still work, with a `DeprecationWarning`, until 4.0.
  See [Breaking changes and deprecations](#breaking-changes-and-deprecations).
- Results tables written by 2.x (pickle) do not load; the directed spectral SPIs are transposed; many estimator values change (see [Changed](#changed) and "Further value changes").
- The license is **GPL-3.0-or-later** (2.0.2: GPL-3.0), agreed by the maintainer, Ben Fulcher.

### Removed: Java and JIDT

Every information-theoretic estimator is now pure NumPy.
`infodynamics.jar`, `jpype` and the JVM startup path are gone, so installing pyspi no longer requires a Java runtime.

The KSG, kernel, Gaussian and symbolic estimators follow JIDT 1.6.1 and are checked against it: hard-coded JIDT 1.6.1 outputs are asserted in the test suite, which needs no Java, for KSG MI and TE (with Theiler windows and auto-embedding), Gaussian MI and TE, kernel MI and TE, and symbolic TE.
They agree to 1e-12 (KSG), to the 1e-8 ridge (Gaussian), and exactly (kernel and symbolic, after the ln 2 conversion).
Values still differ from 2.0.2 where pyspi departs from JIDT's defaults on purpose: KSG breaks ties with a seeded jitter instead of JIDT's random dither, the kernel MI, TLMI and TE leave each point out of its own neighbour counts, and every estimator reports nats (see **Fixed** and **Changed**).

**Provenance and licence.**
The KSG, kernel, Gaussian and symbolic estimators follow, and were validated against, JIDT (Lizier, 2014; [github.com/jlizier/jidt](https://github.com/jlizier/jidt); GPL-3.0-or-later).
`pyspi/statistics/infotheory.py` carries that attribution and citation, and JIDT's copyright notice is retained in `pyspi/lib/LICENSE-jidt.txt` (listed in `license-files`, like the cdt notice).

### Breaking changes and deprecations

**Deprecated, not removed.** These 2.x spellings still work in 3.x with a `DeprecationWarning` and will be removed in 4.0:

- `subset="all"` becomes `config="full"`; other `subset` names are unchanged.
- `configfile=p` becomes `config=p`.
- `normalise=` becomes `zscore=`, on `Calculator`, `Data` and `CalculatorFrame`.
- The SPI class `GromovWasserstainTau` (misspelled) becomes `GromovWassersteinTau`, in Python and in YAML configs; the identifier `gwtau` and its values are unchanged.
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
- **New since 2.0.2: a command line.**
  `python -m pyspi compute` and a `pyspi` console command (2.0.2 had neither).
  The output directory, suffix and `n_jobs` are validated before computing, and `PYSPI_N_JOBS` is honoured.
  Failed SPIs and empty columns are reported on stderr, and the exit status is 1 whenever any SPI failed; `--allow-partial` exits 0 with the partial table written.

### Removed

- **Java and JIDT** (above): `infodynamics.jar`, `jpype1` and the JVM start-up path.
- **`cdt` and `torch`** (see Dependencies); `setup.py` and `requirements.txt` (packaging is `pyproject.toml` only); `pyspi/phiconf.yaml`; the 2.x config filenames (see the migration table); pickle and parquet output.
- **128 SPIs** from `full` (285 to 231, with 74 added): 30 duplicate, constant or degenerate, 63 carried by an SPI that stays bundled, 27 that track something other than coupling, and 8 other estimators.
  All but four can still be computed from a custom config.
  See [SPI set changes](#spi-set-changes).
- **Nine wavelet classes that were never in a bundled config**: `statistics.wavelet`'s `CoherenceMagnitude`, `CoherencePhase`, `ImaginaryCoherence`, `PhaseLockingValue`, `PairwisePhaseConsistency`, `PhaseLagIndex`, `weighted_PhaseLagIndex`, `DebiasedSquaredWeightedPhaseLagIndex` and `debiased_weighted_squared_PhaseLagIndex`.
  Their shared estimate used a wavelet about one sample wide and averaged the cross-spectrum over epochs only, of which pyspi passes one, so the coherency was exactly `1 + 0j` at every frequency and time whatever the data.
  The multitaper classes of the same names in `statistics.spectral` are unaffected, and the wavelet `PhaseSlopeIndex` has its own estimate.
  `WaveletCoherence` (see **Added**) is a separate, smoothed estimate.

### Fixed

**Results and their identity.**

- **Caches survived data mutation.**
  Statistics cache results on the `Data` instance, keyed by their own parameters; `set_data`, `add_process` and `remove_process` did not invalidate them, so cached statistics served the previous dataset's numbers.
  Every cache is now dropped on mutation, and `Data` copies and freezes its input.
- **Spectral caches ignored `fs`**, so two SPIs differing only in sampling rate could share a cached spectrum.
- **Importing pyspi reseeded NumPy's global RNG** (`pyspi.lib.ids` called `np.random.seed(1717)` at import), overriding the caller's seed.
- **Process names** are coerced to `str` and must be unique; `Data(procnames=...)` was validated and then discarded, so custom names never reached the results table.
  `Data.add_process()` no longer creates a duplicate name.
- **`CalculatorFrame.compute()` accepted no arguments**; it now forwards to `Calculator.compute`.
- **`CorrelationFrame` reported invalid p-values.**
  It used each time series' length as the F-test sample size, but the observations are edges, and edges sharing a node are dependent.
  `get_pvalues()`, `compute_significant_values()` and `get_average_correlation(remove_insig=True)` now raise, directing users to a network-preserving permutation (QAP) analysis.
- **`CalculatorFrame` and `CorrelationFrame` bookkeeping.**
  `init_from_yaml` appended a dataset's name and labels before loading it, so after one failed entry every later dataset carried its predecessor's name and labels.
  `get_correlation_df(rmmin=True)` shifted the calculator's table in place; it now shifts a copy.
  `CorrelationFrame.ddf` averaged datasets that share a name, including several unnamed ones; unnamed datasets are now named by position (`dataset_0`, `dataset_1`, ...), and a name given to two datasets raises `ValueError`.
  The SPI-by-SPI correlation is computed as two whole-matrix Spearman correlations (identical output, about 60x faster).
- **Loading and config errors that were silent or late.**
  A `.csv` dataset never loaded (`np.genfromtxt` was given `","` as its dtype); it now does.
  A config entry with an unknown key (for example `config:` for `configs:`) built the class with its default parameters, and a module or class named twice silently lost the first block; both now raise, and a bare `Class:` entry means the defaults.
- `pyspi/lib/ids/LICENSE.txt` was not shipped, despite MIT requiring it, and `LICENSE.txt` had been altered by a global find-and-replace ("technological *statistics*"); both are fixed.

**Orientation and sign.**

- **Directed spectral SPIs were transposed.**
  `SpectralGrangerCausality` (both methods), `DirectedCoherence`, `PartialDirectedCoherence`, `GeneralizedPartialDirectedCoherence`, `DirectedTransferFunction` and `DirectDirectedTransferFunction` reported `A[i, j]` as the influence *of j on i*, the opposite of every other directed SPI.
  pyspi's convention is **row = source, column = target**, and all directed SPIs now follow it.
  **Results computed with 2.x for these SPIs need transposing.**
  Undirected spectral SPIs are unaffected.
- **`ccm_*` was transposed.**
  pyEDM's `a:b` column is `a`'s shadow manifold predicting `b`, which is evidence that `b` drives `a`; it is now stored at `A[b, a]`.
- **`dcorrx_*`, `mgcx_*` and `coint_aeg_*` are transposed relative to 2.x**: the lagged distance measures follow the published definition with the source on the row, and `coint_aeg_*` regresses the column on the row (as `lmfit_*` does).
- **Cointegration `aeg` was forced to be symmetric.**
  The cache wrote each value to both `(i, j)` and `(j, i)`, so the reported value depended on process order; `aeg` is now `directed`.
- **Wavelet phase-slope index lost its direction.**
  pyspi filled the upper triangle of the backend's lower-triangular result without negating it, so `psi[i, j] == psi[j, i]`.
  The fill now happens per frequency, before the band statistic, so `mean` is antisymmetric.
  `fmin=0` also asked for an unbounded period; `fmin` is now resolved against the data-supported floor, and the cycle count is capped so the wavelet fits the signal.
- **Antisymmetric SPIs reported themselves as unsigned.**
  `Calculator._rmmin` subtracts the minimum from every unsigned SPI, which on an antisymmetric matrix destroys the lead/lag its sign carries, and `set_group` correlates unsigned SPIs through `abs()`.
  `phase`, `pli`, `wpli`, `psi` (multitaper and wavelet), `gd`, `ccm_*_diff` and `coint_aeg_tstat` (a signed Engle-Granger t-statistic: more negative is stronger evidence) are now signed, and an SPI's merged label follows its `issigned()`.
  `CrossPairwiseDistance` had no `issigned`, so `_rmmin()` raised on any config containing it.
- **`IGCI` was misnamed and mislabelled.**
  It is Information-Geometric Causal *Inference*, so `InformationGeometricCausalInference` is the class name, with the old one kept as a deprecated alias; its score is exactly antisymmetric and is now signed.
  It is not in any bundled config.

**Estimators.**

- **`DirectedInfo` did not implement directed information.**
  It summed `H(Y^i)/i` minus the causal entropy, which measures the target's self-predictability: with an independent source it returned 0.007 at target autocorrelation 0 and 1.53 at 0.95.
  It now implements Massey's `sum_i [H(Y_i|Y^{i-1}) - H(Y_i|Y^{i-1},X^i)]`, which matches the closed form `0.5*ln(1+c^2)` on a Gaussian pair.
  `di_kraskov` estimates each term directly with the KSG/Frenzel-Pompe conditional-MI estimator, whose biases cancel by construction; the kernel and Kozachenko variants, composed from separately estimated entropies, are not bundled.
  `n` reaches the identifier of `DirectedInfo` and `CausalEntropy`.
- **KSG ties and scale.**
  KSG standardises each coordinate, as JIDT does, and breaks ties deterministically: in each series with a positive standard deviation, every repeated value gets Gaussian jitter of SD `1e-10 × std` from a fixed-seed generator (seeds: source 0, target 1, embedding selection 2), repeated until all values are distinct, as in pyhctsa.
  Tie-free series are untouched and match JIDT 1.6.1 run without its dither to about 1e-15.
  Without this, tied data give confident nonzero MI on independent series, since every kth-neighbour radius is zero; every process of the bundled `forex` dataset is tied.
  The jitter is deterministic but depends on observation order; NaN, infinite and constant series raise.
  **Every `kraskov` SPI's values can change**: 2.x ran JIDT with its seeded random dither, which 3.0 does not reproduce.
- **KSG refuses inputs it cannot estimate from**: `k` too large for the effective sample, and negative Theiler windows; the auto-embedding search skips candidates it cannot support.
- **Kozachenko entropy returned `-inf` on tied data** (a nearest neighbour at distance zero); it now raises with the cause named.
- **Six measures advertised `kraskov` and ran Gaussian.**
  Joint, conditional, cross-map and causal entropy, directed information and stochastic interaction are composed from marginal entropies and have no KSG estimator; `estimator="kraskov"` is now refused for them.
- **`MutualInfo`, `TimeLaggedMutualInfo` and `TransferEntropy` returned NaN** for `estimator="kozachenko"`; they now raise `NotImplementedError` at construction (use `"kraskov"`).
- **KSG mutual information with a Theiler window was biased upward.**
  `mi_kraskov_*_DCE-*` and `tlmi_kraskov_*_DCE-*` take the `psi(N)` term over each point's window-reduced neighbour pool (JIDT keeps `psi(N)`); transfer entropy is unaffected.
- **Auto-embedding names its method.**
  `MAX_CORR_AIS` selects the destination and source embeddings independently, as JIDT 1.6.1 did, so the three bundled auto-embedding SPIs keep 2.0.2's values and are renamed to carry the method.
  `MAX_CORR_AIS_DEST_ONLY` selects the destination embedding and takes a fixed source embedding.
  The kernel and symbolic estimators, which have no active-information-storage criterion, refuse auto-embedding and its search bounds.
- **Symbolic and kernel transfer entropy accepted embedding parameters they ignore.**
  Symbolic TE applies one ordinal-pattern length to source and destination at unit delay (Staniek & Lehnertz 2008), yet `k_tau`, `l_history` and `l_tau` were accepted and written into the identifier.
  They are now refused, symbolic identifiers become `te_symbolic_k-<k>`, and `k_history=1`, whose single symbol gives an identically zero TE, is refused.
- **`dyn_corr_excl` named three Theiler windows with one identifier.**
  The suffix was a bare `_DCE`, so 5, 10 and `"AUTO"` collided; it is now `_DCE-<value>`, and the shipped SPIs are renamed `..._DCE` → `..._DCE-AUTO` (values unchanged).
- **`ccm_E-None_*` never inferred an embedding.**
  The dimension was read as `EmbedDimension(...).max()["E"]`, the largest candidate (10), so `ccm_E-None_*` equalled `ccm_E-10_*`.
  Selection is now `argmax(rho)`, ties to the smaller E, and **`ccm_E-None_*` values change**.
  The library sizes now run up to the largest library.
- **CCM runs serially inside pyEDM.**
  pyEDM 2.5 starts its pools with forkserver or spawn, which re-import the caller's `__main__`, so from a script without an `if __name__ == "__main__":` guard every `ccm_*` SPI came back as NaN.
  pyspi now calls pyEDM without its process pools (3.6x faster on a 7-process example); parallelism comes from `compute(n_jobs=...)`.
- **Spectral factorisation failures are reported.**
  Wilson's algorithm logs non-convergence through `logging` and returns the unconverged factor, so every Wilson-derived measure (`dcoh`, `dtf`, `ddtf`, `pdcoh`, `gpdcoh`, nonparametric `sgc`) reached the table silently; the warning now lands in the SPI's record.
- **Group delay (`GroupDelay`) returned no finite value on any input**: the backend's one-sample coherence test takes the square root of a negative number.
  pyspi computes the standard one-sample test `(arctanh|C| − b)/sqrt(b)`, `b = 1/(2n − 2)` (Enochson & Goodman 1965; Bokil et al. 2007), applies Benjamini-Hochberg over the band, thins the longest significant run to one point per multitaper bandwidth, and fits the unwrapped phase against frequency.
  The delay is in samples (it was in seconds), and `rvalue` is `|r|`, undirected and unsigned.
  It is not bundled (see [SPI set changes](#spi-set-changes)).
- **Coherence phase used linear statistics on wrapped angles.**
  `CoherencePhase` uses the circular mean `arg(mean(exp(i·phase)))` and builds the opposite orientation by negation; a resultant indistinguishable from zero, or an antipodal mean, gives NaN.
  `statistic="max"` is refused, since a maximum of wrapped angles depends on the branch cut.
  **The three `phase_multitaper_mean_*` values change.**
- **Spectral Granger causality ignored `fs`** in its parametric branch and mis-oriented its NaN mask, so an estimable cell could be blanked; no value changes at the shipped `fs=1`.
  Eleven `sgc_*` identifiers named `fmin=0`, which is overridden to 1e-5 because spectral GC is undefined at zero frequency: **`sgc_*_fmin-0_*` is renamed `sgc_*_fmin-1e-05_*`** (values unchanged).
- **Parametric spectral Granger causality (`sgc_parametric_*`) was fitted on an indefinite autocovariance.**
  The VAR is now fitted on the Yule-Walker autocovariance, which is positive semidefinite, the band uses the frequencies the model is evaluated at, and the automatic order search terminates.
- **Directed spectral measures used an open frequency band.**
  `dtf`, `dcoh`, `pdcoh`, `gpdcoh` and `ddtf` use the closed band their identifiers name, and `ddtf_*` uses the partial coherence in place of `sqrt(PDC)` (Korzeniewska et al. 2003).
- **Phase measures included 0 Hz and Nyquist**, where a real signal's cross-spectrum has no phase.
  `phase`, `icoh`, `plv`, `pli`, `wpli`, `dspli`, `dswpli`, `ppc` and `psi_multitaper_*` leave those bins out of the band; amplitude measures keep the closed band.
- **Debiased squared PLI (`dspli_*`) was wrong where the cross-spectrum is real.**
  It is now the mean of taper-pair sign products (Vinck et al. 2011).
- **Cross-correlation (`CrossCorrelation`) was wrong four ways.**
  It mixed normalisations (a series against itself gave 1.11 at T = 10), correlated the raw series while dividing by demeaned ones, centred its lag window on lag +1, and set a significance band twice too wide that scaled with the lag window.
  It now uses the biased normalisation (zero lag is Pearson's r), demeaned series, a symmetric window and the amplitude cut `|r(l)| > 1.96/sqrt(T)`, which `sigonly` applies; this is not a significance test.
  `LagResolvedCorrelation` reads the same profile.
- **`softdtw_constraint-*` ignored its constraint** and equalled plain `softdtw`; the band is now passed to tslearn.
- **`lcss_constraint-sakoe-chiba` and `lcss_constraint-itakura` were plain `lcss`**, because tslearn before 0.10 ignored `global_constraint` in `lcss` (tslearn #526).
  pyspi now computes constrained LCSS itself, with tslearn's band masks restricting which pairs may match but not where the recursion may pass (Vlachos et al. 2002); plain `lcss` still calls tslearn.
  **Both constrained values change**, by up to 0.4 for Sakoe-Chiba and 0.09 for Itakura.
- **`dtw_constraint-sakoe-chiba` is renamed `dtw_constraint-sakoe-chiba_radius-auto` and its values change.**
  2.x used tslearn's default radius of 1; the radius is now 10% of `T` or `1.5·√T`, whichever is smaller, with a floor of 10 and a cap of `T-1`, tunable with `PYSPI_DTW_SAKOE_LINEAR_FRAC`, `PYSPI_DTW_SAKOE_SQRT_COEFF` and `PYSPI_DTW_SAKOE_MIN_RADIUS`.
  Itakura-constrained DTW also applies the `sqrt(T)` normalisation on every path.
- **`hsic` and `hsic_biased` depended on the order of the two processes.**
  Each variable now gets its own median-heuristic kernel width, the median distance between its samples (Gretton et al. 2007).
- **`phi_Geo_*` stopped before its model was disconnected.**
  The solver now also requires the constraint to be met, and phi_G is evaluated at the block-diagonal projection (Oizumi et al. 2016).
- **`gwtau` left the start of the trajectory out of its distance vector** (Kravtsova et al. 2023); values change by the factor `sqrt((T - 1) / T)`.
- **Undefined values.**
  `cds` returned 0, its strongest evidence for `i -> j`, when undefined, and `anm`, `cds` and `xcorr_*` returned 0 or an arbitrary number for a pair with a constant series; all are now NaN.
  `prec_*` and `pcorr_*` reported a pseudo-inverse when the covariance estimate is singular; for `EmpiricalCovariance` and `MinCovDet` they are now NaN, with a warning giving the rank.
- **Integer parameters** (`ConvergentCrossMapping(embedding_dimension=2.7)`, a fractional `max_tau`) were truncated after the identifier was built; they must now be integers.
- **SPI metadata (no value changes).**
  `InterDependenceScore` left `terms`, `pnorm` and `bandwidth` out of its identifier, and `SpectralGrangerCausality` left out `max_order` (when the order is selected), `ignore_nan` and `nan_threshold`, so two parameterisations could share a table column; non-default values are now appended (e.g. `ids_bw-0-25`), and every bundled identifier is unchanged.
  `IntegratedInformation` rejects an unknown `phitype`, a `normalization` other than 0 or 1 and a non-integer `delay` at construction.
  Smaller fixes: `Covariance`/`Precision` return a copy of the cached array; `SpearmanR` and `LaggedCorrelation(spearman)` handle a constant series in any position; `LaggedCorrelation.bivariate` returns NaN when either direction is undefined; `CrossmapEntropy` and `StochasticInteraction` reject non-positive `history_length`/`delay`; the wavelet PSI frequency floor is in Hz; spectral classes reject a `statistic` they cannot apply.

### Added: per-cell quality codes

Every cell of `calc.table` has a quality code saying what it is, in `calc.quality`: a `uint8` DataFrame with the same index and columns.
The design is shared with hctsa and pyhctsa, in a module (`pyspi/quality.py`) that the two packages keep identical: the value array holds a real, finite number or NaN, so it can be used directly, and the code says why a cell is NaN.
Codes 0-7 are hctsa's `TS_Quality` codes, so results can be compared code for code.
`pyspi.Quality` names them: `GOOD` (0), `ERROR` (1), `NAN` (2, undefined or not appropriate, and every diagonal cell), `INF` (3), `NEG_INF` (4), `COMPLEX` (5), and `NOT_COMPUTED` (255) for an SPI that has not run yet.
`pyspi.quality.quality_labels` and `quality_summary` turn codes into text and counts.

- **An Inf does not fail the whole SPI.**
  The finite cells are kept, the infinite cells are NaN with code `INF` or `NEG_INF`, and a warning gives the count.
  The SPI is listed in `calc.errors` only if no off-diagonal value is finite, as for an all-NaN result.
- **Complex output is labelled, not truncated.**
  The raw output is classified before it is cast to float, so a cell with a non-zero imaginary part is NaN with code `COMPLEX`.
- **A failed pair is `ERROR`, not an anonymous NaN.**
  A pair that raises inside an SPI that otherwise succeeds (see "One undefined pair no longer voids an SPI" under Changed) is NaN with one warning, labelled `ERROR`, with its message in `calc.pair_errors` (`{spi: [(source, target, message), ...]}`).
  `calc.errors` holds SPI-level failures only.
  An undirected SPI evaluates each pair once, as `(i, j)` with `i < j`, and mirrors it, so a failure is mirrored onto its lower cell and counted once in the warning; such an SPI raises when every pair fails, giving the estimator's message in `calc.errors`.
- An SPI that raises or returns the wrong shape is `ERROR` in every off-diagonal cell, with its message in `calc.errors`.
  An SPI that returns only NaN keeps code `NAN` and its `calc.errors` entry.
- **Files and checkpoints keep the codes.**
  `.npz` files hold a `quality` array (`(n_spis, M, M)`, `uint8`) and `pair_errors`.
  `pyspi.load_results(path)` returns `Results(table, quality, errors, run_spec, pair_errors)`; `load_table` keeps only its small metadata in `attrs`.
  Checkpoints store the codes in a `<spi>.quality` sidecar (and pair failures in `<spi>.pairs.json`).
- The summary printed after `compute()` counts the off-diagonal cells by code and lists each incomplete SPI with its counts, e.g. `one_infinite (1 Inf, 1 -Inf)`.

### Changed

- **`Calculator(subset=..., configfile=...)` collapsed into `config=`**, which accepts either a bundled name or a path to your own YAML.
- **`normalise=` renamed to `zscore=`** on `Calculator` and `Data`.
  Behaviour is unchanged (per-process z-score along time); the old name collided with `utils.normalise`, which was min-max, and with the per-SPI `normalise` arguments in `statistics/distance.py`.
- **Configs renamed and moved to `pyspi/configs/`.**
  The filename stem is the lookup key, so the cost-pruned sets are reachable by name.
- `load_dataset()` exposes only the three demo datasets (`forex`, `cml`, `standard_normal`); `available_datasets()` lists them.
- Per-SPI timings are printed after `compute()` (total and slowest five).
- **`Calculator.save()` and `pyspi.load_table()`** store results as `.npz`: the results in their natural `(n_spis, M, M)` shape plus names, the run spec, digest and errors, round-tripping exactly with nothing beyond numpy.
  Names are stored as `dtype='U'` and loaded with `allow_pickle=False`.
  `.csv` remains as a one-way human-readable export.
- **Dropped pickle and parquet output.**
  Pickle is version-fragile and executes arbitrary code on load, which is wrong for an archival scientific artifact, and parquet bought nothing over `.npz` for a dense float tensor while costing a ~40 MB pyarrow dependency.
- A config that keeps only part of a shared-cache group warns, since the cache is built regardless and the remaining members are close to free.
- `JIDTBase` is renamed `InfoTheoryBase`; config files are unaffected.
- **Every information-theoretic measure reports nats.**
  In 2.x the kernel and symbolic calculators reported bits while the Gaussian, KSG and Kozachenko ones reported nats (JIDT's split), so magnitudes on either side of that boundary differed by a factor of ln 2.
  Correlations between columns are unaffected; divide by ln 2 for the JIDT-comparable value.
  **All `kernel` and `symbolic` SPI values change by that factor.**
- **Kernel MI, time-lagged MI and TE exclude the point itself from its neighbour counts**, as the KSG estimators do: a leave-one-out density estimate (JIDT's `DYN_CORR_EXCL = 1`; JIDT's default counts the point itself).
  Counting it adds 1 to every count, which inflates the estimate most where counts are small.
  Kernel TE also scales the destination's next value like its past, as JIDT does.
  **`mi_kernel_*`, `tlmi_kernel_*` and `te_kernel_*` change value.**
  The kernel entropies, and the measures composed from them (`je`, `ce`, `si`, `xme`, `cce`), still count the point itself: a leave-one-out box-kernel entropy is infinite wherever a point's box is otherwise empty.
- **One singularity policy for every Gaussian quantity.**
  Each variable is treated as observed with independent noise of variance `1e-8 ×` its own variance (a deterministic analogue of JIDT's `NOISE_LEVEL_TO_ADD`), through one shared log-determinant, in MI, TLMI, entropy, joint and conditional entropy, TE and AIS.
  Gaussian MI acquires a bias of `ridge·r²/(1−r²)`: 5.6e-9 at ρ=0.6.
- **The AUTO Theiler window (`dyn_corr_excl="AUTO"`) is capped at `N // 10`** (hctsa's rule), so it can no longer approach the series length.
- **One undefined pair no longer voids an SPI.**
  A pair whose estimate raises (e.g., tied values for a Kozachenko-Leonenko entropy) is set to NaN with one warning, and the SPI fails only if every pair does.
- **`gpfit_*` is the leave-one-out mean squared error** of the fitted Gaussian process (Rasmussen & Williams 2006, eq. 5.12), not the in-sample error, which the RBF fit can drive to zero on unrelated pairs.
- **`Barycenter` gains `statistic="corr"`**, `2 mean(b^2) / mean_energy - 1`: Pearson's r for the Euclidean barycenter and an alignment-adjusted correlation for the DTW geometries.
  `bary_dtw_corr` and `bary_softdtw_corr` replace the `mean` SPIs, since the barycenter of two zero-mean series has mean close to zero.
- **Multitaper phase slope index follows Nolte et al. (2008) and applies its band.**
  `psi_multitaper_mean_*` is `Im sum_f conj(C(f)) C(f + df)` over neighbouring in-band frequencies one multitaper bandwidth (six FFT bins) apart.
  2.x took spectral-connectivity's sum over all pairs of thinned frequencies, which is not the published index, and its band variants all returned the full-band value.
  The index grows in proportion to the lag, and its sign is reliable up to `T/12` samples; in a single recording it is less sensitive to one-sample leads.
  It stays unnormalised (the published jackknife needs epochs).
  **The three `psi_multitaper_mean_*` values change.**
- **No SPI reads the global NumPy RNG.**
  tslearn's subgradient barycenter took its update order from the global stream, and `MinCovDet` and `EllipticEnvelope` search from random subsets of the observations, so their values changed with the caller's seed and with which SPIs ran earlier.
  Each fit now uses its own fixed-seed generator, so a value is a function of the data alone.
  **`cov_MinCovDet` changes value**, by about the estimator's own run-to-run spread, as do the `sgddtw` barycenter SPIs, which are not bundled.
- **scikit-learn >= 1.8 changes some covariance values.**
  scikit-learn 1.8 added the Croux-Haesbroeck consistency correction to `MinCovDet`/`EllipticEnvelope` (scikit-learn #32117), which cannot be switched off, so their covariance, precision and partial-correlation SPIs differ from 2.0.x.
  `GraphicalLasso` converges at `tol=enet_tol=1e-8`, so the value no longer tracks the solver path, which scikit-learn 1.8 also changed (#31987, #31906); values move by at most 1e-4 and agree across scikit-learn versions to about 1e-9.
- **`anm` pins its Gaussian-process kernel** to a fixed `ConstantKernel(1.0)*RBF(1.0)` with no optimiser, where it had relied on scikit-learn's default, which in 1.9 became an optimised kernel and shifted values by up to 7.6.
  Values are unchanged under scikit-learn <= 1.8.
  The kernel matrix is ill-conditioned, so `anm` varies by about 1e-4 relative across platforms.
- **Labels describe each SPI as computed.**
  The labels a bundled config writes for an SPI agree with the SPI: its sign with `issigned()`, its directedness with the class trait, and its linearity with the class where the class sets one; where these vary by variant (`lagcorr_*`, `coint_*`, `es_*`, `pec_*`), the labels are set per config entry.
  When a config's labels are merged with a class's, the class's directedness trait, its `linear`/`nonlinear` and its `issigned()` win, and any remaining contradictory pair raises.
  2.0.2 applied only the class labels, not those in its config files.
  Labels that differ from 2.0.2's: the Kozachenko SPIs, `anm` and the rank correlations `SpearmanR` and `KendallTau` (which capture monotone, not only linear, dependence) are `nonlinear`; `gpfit_DotProduct` and `Cointegration` are `linear`; `coint_aeg_*` is `directed` and signed; `hhg` is `undirected`; `phase`, `pli`, `wpli` and `psi` are `antisymmetric` and signed; `pec`, `pec_orth`, `pec_log` and `pec_orth_log` are signed; and the spectral SPIs that had no linearity label are `linear`.
  Every SPI carries exactly one module label (`M01`-`M14`, or `MXX` for one not assigned to a module).
  `filter_spis` matches on the labels each SPI carries after merging, not on the raw YAML family labels, so per-variant traits are found and a matching family no longer selects all of its configs.
- **Conventions stated in the class docstrings:** `gc_gaussian_*` is half of Geweke's Granger causality; `xcorr_*_max` is the largest signed value; lower is stronger evidence for `anm`, `cds` and `reci`, higher for `ccm` and `igci`; `ccm_E-None_*` uses the larger of the pair's two embedding dimensions; `softdtw*` is the raw soft-DTW cost (it can be negative), named "Soft dynamic time warping"; `cohmag` is the magnitude-squared coherence.
- **Every SPI class has a docstring in one layout, shared with pyhctsa**: a summary line, what is computed with its direction and sign, a `References` section and a `Parameters` section with defaults.
  `Calculator`, `CalculatorFrame`, `CorrelationFrame`, `Data`, `load_table`, `load_results` and the functions in `pyspi.utils` have `Parameters` and `Returns` sections.
  Antisymmetric SPIs state their sign convention (`A[i, j] = -A[j, i]`; positive means the row leads).

**Faster, with identical values.**

- **Distance correlation and the Kraskov estimators.**
  `dcorr` and `dcorr_biased` evaluate only the retained order of each pair (2.0x at M=8, T=5000).
  Kraskov MI, time-lagged MI and transfer entropy count marginal neighbours with one batched tree query, less the temporally excluded neighbours, in O(N) memory (3-10x at T=5000, more with `dyn_corr_excl`).
- **The five directed spectral measures share one spectral factorisation per pair.**
  `dtf`, `dcoh`, `pdcoh`, `gpdcoh` and `ddtf` are all read off the Wilson minimum-phase factor of a pair's cross-spectrum, which is now computed once per ordered pair, and the five classes form one scheduling task.
  7.6x faster on those measures (184 s to 24 s at M=40, T=1000), with about 40% less memory held per pair.
- **Itakura-constrained DTW.**
  pyspi builds tslearn's Itakura mask with NumPy and runs tslearn's recursion over each row's admitted columns only, in a numba kernel cached on disk; values are bit-identical to tslearn's.
  At M=16, T=800 the SPI takes 0.05 s instead of 0.23 s, and 0.14 s instead of 1.8 s in a new process.
  `lcss_constraint-itakura` uses the same mask.
- **The distance-correlation family centres each series once.**
  `dcorr`, `hsic` and `dcorrx` share each series' centred distance matrix (one per lag for `dcorrx`) across its pairs, holding at most 256 MB of matrices at a time.
  At T=800, `dcorr` is 7x faster, `hsic` 11x and `dcorrx_maxlag-1`/`-10` 5x and 4x.
  pyspi applies hyppo's final step itself, with the same values as hyppo 0.5.2.
- `hhg` in O(n^2 log n) (about 9x); `corr_kendall_*` (2.2x); `si_*`, `mi_kernel_*` and `xme_*` cache per-process entropies (2-3.7x); the directed spectral measures cache their tapers (1.6x); `ccm_*` (6-7x) and `anm` (6.9x) share each cause's work across its targets.
- **Undirected SPIs evaluate each unordered pair once** and mirror the value and any failure (`phi_*` 1.9x faster).
  For the distance SPIs among them (`hsic*`, `hhg`, `dcorr*`, `mgc`, `lcss*`, `softdtw*`, `gwtau`), a pair that raises is NaN with a warning, as for every other SPI, instead of failing the SPI.
  A direct `bivariate(data, i=i, j=j)` call with `i < j` on a fresh `Data` can differ in the last bit for `xcorr_*`, `je_*` and `ce_*`.

### Added

- **74 new SPIs in `full`**, by family (the sets other than `full` that bundle them in brackets):
  - **`LagResolvedCorrelation`** (13): directed, lag-resolved cross-correlation `r_{i->j}(tau) = corr(x_i[t], x_j[t + tau])` over `L = min(max_lag, T // 4)` lags each way (Box et al. 2015; Mitra et al. 2014).
    `lagcorr_{lag,peak,asym}_L-{None,10}` give the signed lag of the largest `|r|` (antisymmetric), the signed `r` at that lag (undirected) and the lead/lag asymmetry `(S_{i->j} - S_{j->i}) / (S_{i->j} + S_{j->i})` (antisymmetric); `lagcorr_onesided_tau-{1,2,3,5,10,1e}_L-None` is the directed `r_{i->j}(tau)` at one lag shared by both directions of a pair (`1e` from the series' 1/e autocorrelation lengths); `lagcorr_onesidedpeak_L-None` is the largest lead `|r|` within the sum of those lengths.
    They replace the bundled `xcorr_*` (all 13 in the `benchmarked_p*` sets; `lag`, `peak`, `asym` and `onesided_tau-1` at `L-None` in `fast`).
  - **Conditional VAR measures** (15), from one OLS VAR of all processes, so an indirect link or a common driver gives about 0: conditional Granger causality `cgc_p-{1,2,bic}` (Geweke 1984; Barnett & Seth 2014; on Geweke's scale, twice that of `gc_gaussian_*`), and `cpdc`, `cgpdc`, `cddtf` and `pcohmag` (partial directed coherence, generalised PDC, direct DTF and squared partial coherence) at `order=bic` over [0, 0.5], [0, 0.25] and [0.25, 0.5] cycles per sample (`cgc_p-bic`, `cpdc_p-bic_mean_fmin-0_fmax-0-5` and `pcohmag_p-bic_mean_fmin-0_fmax-0-5` in `fast`).
  - **`PartialCorrelation`** (6): `pcorr_<estimator>`, `-P_ij / sqrt(P_ii P_jj)` from the precision matrix of each bundled covariance estimator (`EmpiricalCovariance`, `GraphicalLasso`, `LedoitWolf`, `MinCovDet`, `OAS`, `ShrunkCovariance`), the correlation of `x_i` and `x_j` conditional on all other processes (Marrelec et al. 2006; Smith et al. 2011) (all in the `benchmarked_p*` sets; `EmpiricalCovariance` and `LedoitWolf` in `fast`).
  - **`NonlinearInterdependence`** (4): `nli_{L,N,H}_m-5_tau-1_k-5` and `nli_L_m-3_tau-1_k-5`, how well close delay vectors of the column process map onto close delay vectors of the row process (Arnhold et al. 1999; Quian Quiroga et al. 2002; Chicharro & Andrzejak 2009); only the asymmetry `A[i, j] - A[j, i]` indicates direction (`nli_L_m-5_tau-1_k-5` in `fast`).
  - **`LiangInformationFlow`** (4): `liang_{rate,norm}_k-1` and `liang-cond_{rate,norm}_k-1`, the Liang-Kleeman rate of information flow under a linear stochastic model, in nats per sample and signed, pairwise (Liang 2014) or conditioned on every other series (Liang 2021); `norm` divides by the target's entropy budget (Liang 2015) (`liang_norm_k-1` in `fast`).
  - **`EventSynchronisation`** (4): `es_Q_q-{0-95,0-9}`, the synchronisation strength of extremes in [0, 1] (undirected), and `es_q_q-{0-95,0-9}`, their delay asymmetry in [-1, 1] (antisymmetric; positive when the extremes of the row precede those of the column) (Quian Quiroga, Kreuz & Grassberger 2002; Odenweller & Donner 2020) (`es_q_q-0-95` in `fast`).
  - **`TailDependenceAsymmetry`** (2): `tailasym_q-{0-95,0-9}`, the empirical upper minus lower tail-dependence coefficient on ranks, in [-1, 1], zero under any elliptical copula (Joe 1997; Schmidt & Stadtmueller 2006) (`tailasym_q-0-95` in `fast`).
  - **`EdgeTimeSeries`** (3): `edge_excessvar`, `edge_topshare_f-0-05` and `edge_acf1`, whether a correlation is steady or bursty, from the edge time series `e_t = z_x(t) z_y(t)` (Faskowitz et al. 2020; Esfahlani et al. 2020), each about 0 for a Gaussian pair (`edge_excessvar` in `fast`).
  - **Convergent cross-mapping** (5): `ccm_E-{None,1,10}_convergence`, the cross-map skill at the largest library size minus that at the smallest (Sugihara et al. 2012), and lagged CCM `lccm_E-None_k-5_{tp,skill}` (Ye et al. 2015), the prediction horizon in `[-5, 5]` of peak skill (negative for a causal row -> column link) and that peak skill.
  - **Kernel MI, TLMI and TE with dynamic correlation exclusion** (3): `mi_kernel_W-0.25_DCE-AUTO`, `tlmi_kernel_W-0.25_DCE-AUTO` and `te_kernel_W-0.25_DCE-AUTO_k-1` (all three in `benchmarked_p90`, `p95` and `p99`; MI and TLMI also in `p80` and `fast`).
  - **`WaveletCoherence`** (3): `wtc_wavelet_mean_*` over the bands [0, 0.5], [0, 0.25] and [0.25, 0.5], Morlet wavelet coherence smoothed in time and scale (Torrence and Webster 1999; Grinsted et al. 2004), averaged over time outside the cone of influence.
    It detects coupling that is transient or changes phase lag during the record; the local estimate is biased upwards, so independent series give values well above zero.
  - **`ChatterjeeXi`** (2): `xi_tau-{0,1}`, Chatterjee's rank correlation xi (Chatterjee 2021), directed, tending to 1 iff the column process is a measurable function of the row process and to 0 iff they are independent (both in the `benchmarked_p*` sets; `xi_tau-0` in `fast`).
  - **`CrossPairwiseDistance`** (2): `xpdist_euclidean_tau-1_{mean,min}_rmse`, a shifted-path Euclidean upper bound on DTW (in the `benchmarked_p*` sets).
  - **`Barycenter`** (2): `bary_dtw_corr` and `bary_softdtw_corr` (see **Changed**) (`bary_dtw_corr` in the `benchmarked_p*` sets and `sonnet`; `bary_softdtw_corr` in `p95` and `p99`).
  - **`PhaseAmplitudeCoupling`** (2): `pac_mi_excess_p-0-01-0-05_a-0-15-0-4` and `pac_mi_excess_p-0-05-0-1_a-0-2-0-45`, the modulation index (Tort et al. 2010) of the column's amplitude envelope by the row's phase, less its mean over 100 circular time-shift surrogates.
    The `mi`, `mvl` (Özkurt & Schnitzler 2011), `_excess` and `_z` statistics are available at any band pair; an amplitude band narrower than twice the top phase frequency is refused (Aru et al. 2015).
    Time-shift surrogates do not remove the spurious coupling of sharp non-sinusoidal waveforms (Kramer et al. 2008).
  - **`di_kraskov_NN-4_n-5`** (1): directed information from the KSG/Frenzel-Pompe conditional-MI estimator (see **Fixed**).
  - **`te_symbolic_k-3`** (1): symbolic transfer entropy with 3-sample ordinal patterns (Staniek & Lehnertz 2008), with a bias under independence of about 0.13, 0.03 and 0.01 nats at T = 300, 1000 and 3000 (in the `benchmarked_p*` sets).
  - **`phasealigned_welch_nperseg-auto`** (1): the zero-lag correlation maximised over all-pass (phase-only) filtering of one process, `sum_f w_f |S_jk(f)| / sqrt(sum_f w_f S_jj(f) sum_f w_f S_kk(f))` from Welch cross-spectra, undirected and in [0, 1], with a null floor of about `sqrt(pi / (4K))` for `K` segments.
  - **`ppc_hilbert_adaptive`** (1): the pairwise phase consistency (Vinck et al. 2010) of band-passed Hilbert phases, averaged over log-spaced bands weighted by the rhythmic power the two processes share above their `1/f` backgrounds, so no band has to be chosen.
- **Statistics available by name but not bundled:** `LaggedCorrelation` (`corr_{pearson,spearman,kendall}_tau-<k>[-sq]`), the undirected average of the two one-sided lag-k correlations; `ConvergentCrossMapping(statistic="slope")`, the slope of skill against `ln L`; `ConditionalGrangerCausality(statistic="coef")`, the signed net lagged coefficient; and the `PhaseAmplitudeCoupling` statistics above.
- **Dynamic correlation exclusion for the kernel estimators** of MI, time-lagged MI and TE: `dyn_corr_excl` (an integer or `"AUTO"`) applies to `estimator="kernel"` as well as `"kraskov"`, with the same `_DCE-<value>` suffix.
  As in JIDT (`DYN_CORR_EXCL = w + 1`), each probability is normalised by the number of samples compared and an empty joint box contributes 0; if every joint box is empty the result is NaN (JIDT gives 0).
  The window removes the dependence that autocorrelation adds and can take MI below zero, so read values against surrogates.
- **The multitaper bandwidth is a parameter.**
  Every multitaper SPI in `pyspi/statistics/spectral.py` takes `time_halfbandwidth_product` (`NW`, default 3, the value every bundled SPI uses).
  A larger `NW` averages more tapers (`floor(2 NW) - 1`), lowering the variance and upward bias of coherence-type measures at the cost of frequency resolution.
  A value other than 3 is appended to the identifier as `_nw-<value>`, with dots written as dashes.
- **Feature vectors.**
  `Calculator.feature_vector()` and `pyspi.feature_vector(table)` flatten a results table to one value per independent pairwise feature: only `i < j` for `undirected` and `antisymmetric` SPIs, both orientations for `directed` and `asymmetric` ones.
  Self-pairs are dropped unless `include_self=True`, and NaNs are kept, so the length and order depend only on the config and `M`.
  `pyspi.feature_vector(table)` rebuilds each SPI's trait from the config recorded in `table.attrs["run_spec"]`, or takes an explicit `directedness={identifier: trait}`.
- **`Directed.directedness()`** returns the one symmetry trait an SPI carries: `undirected`, `antisymmetric` (`A[i, j] == -A[j, i]`, a new label), `directed` or `asymmetric`.
- **Parallel computation and checkpoints.**
  `Calculator.compute(n_jobs=...)` follows the joblib convention (see Breaking changes), and `compute(checkpoint_dir=...)` writes each SPI as it finishes and resumes from it.
  A checkpoint directory carries a `run.json` manifest bound to `Calculator.run_digest`, which hashes the config contents, the dataset bytes and the computation version, so a different dataset, config, preprocessing or process order is refused rather than reused; failed checkpoints are retried by default.
- **Run records.**
  `Calculator.errors`, `Calculator.run_spec`, `Calculator.run_digest`, `Calculator.to_frame()` (long-form results, one row per `(spi, source, target)`), `Calculator.summary()` and a `repr` for `Calculator`.
  `run_spec` records `computation_version`, `pyspi_version` and `dependency_versions`, saved with every `.npz` and returned by `load_table` in `attrs`.
  `pyspi.COMPUTATION_VERSION` is bumped whenever a change can alter a valid SPI output; record it with saved results, and do not pool results computed under different values.
- `ConvergentCrossMapping` has a `bivariate` method.

### Dependencies

- **Floors, relative to 2.0.2** (`pyproject.toml` is the source of truth):

  | | 2.0.2 | 3.0.0 |
  |:--|:--|:--|
  | Python | `>=3.8` | **`>=3.11`** (scikit-learn >= 1.8 needs it; 3.10 reaches end of life in October 2026). CI matrix 3.11-3.13 |
  | numpy | `<2.0.0` | **`>=2.0`** |
  | scikit-learn | unpinned | **`>=1.8`** |
  | pyEDM | `==1.15.2.0` | **`>=2.5`** |
  | mne | `==0.23.0` | `>=1.6`, plus new `mne-connectivity>=0.5` |
  | spectral-connectivity | unpinned | **`>=2.0,<3`** |
  | scipy / pandas / statsmodels / nitime / hyppo / tslearn | unpinned | `>=1.13` / `>=2.2.2` / `>=0.14.2` / `>=0.11` / `>=0.5.2` / `>=0.10`; new `dtaidistance>=2.3.12` |
  | removed | `jpype1`, `cdt==0.5.23`, `torch`, `h5py`, `seaborn`, `sktime`, `dill`, `future`, `setuptools<81` | (`aeon` is an optional `tsfile` extra for `.ts` files) |

- Dropped unused runtime dependencies (`h5py`, `seaborn`, and the intermediate `plotly`, `matplotlib`, `nbformat`). Plotting and notebook packages moved to a `bench` extra.
- Every floor is a release that installs and runs with NumPy 2 on Python 3.11.
  Older pandas and statsmodels wheels were built against NumPy 1 and fail to import under NumPy 2 ("numpy.dtype size changed"), and hyppo releases before 0.5.2 do not declare the `future` package they import.
- **`spectral-connectivity` is bounded: `>=2.0,<3`.**
  `DirectedCoherence` reads two private `Connectivity` properties (`_transfer_function`, `_noise_covariance`), because the public `directed_coherence()` does not compute the bounded form of Baccalá et al. (1998) that pyspi reports.
  1.x returns a frequency grid without the Nyquist bin, which changes the statistics of bands that end at the Nyquist frequency, so 2.0 is the floor.
- **`cdt` and `torch` are gone.**
  pyspi used four functions from `cdt.causality.pairwise` (the ANM independence score, the conditional distribution similarity statistic, the RECI regression-error score and IGCI), now transcribed in `pyspi/lib/pairwise_causal.py` on NumPy/SciPy/scikit-learn, with cdt's MIT licence retained at `pyspi/lib/LICENSE-cdt.txt`.
  **No value changes**: the transcription is bit-identical to cdt 0.6.
  torch entered only because cdt imports its Torch-backed models at load; the installed size goes from 1.1 GB to 567 MB, and each worker of `compute(n_jobs=…)` starts 1.5–1.7 s faster.
- `numba>=0.61` is a direct dependency (`DynamicTimeWarping`'s Itakura kernel); tslearn already requires the same floor.
- Build requires `setuptools>=77`, matching the PEP 639 licence metadata.
- New extras: `bench` and `tsfile`; `testing` now requires `pytest>=7` (2.0.2 pinned `pytest==5.4.2`).
- Licence: **GPL-3.0-or-later** (2.0.2: GPL-3.0), as agreed by the maintainer.
  `pyspi/lib/LICENSE-jidt.txt` joins the cdt and ids notices in `license-files`.

### Testing

- The test suite runs; in 2.0.2 collection aborted on an undeclared `dill` dependency.
- The information-theoretic estimators are checked against closed forms (Gaussian MI, entropy and TE on a known AR(1), independence, and information-theoretic identities) and against hard-coded JIDT 1.6.1 outputs, with no Java needed.
- The corrected measures are each checked against a closed form, a textbook implementation or a third-party reference (see [References](#references)); every directed SPI family is checked for the row = source convention; and each bundled SPI's config labels are checked against the SPI.
- A drift suite recomputes every SPI in `full` on three small frozen datasets and fails on any change in value beyond rounding, in the NaN pattern, or in the SPI set, and on any SPI that raises or returns no finite value.
- The source distribution includes the whole test suite with its baselines and fixtures, so `pytest` runs from an unpacked sdist.
- CI builds the wheel, installs it into a clean environment, and from outside the source tree resolves every bundled config and dataset, computes `fabfour`, and round-trips the NPZ through the CLI. The regression job installs from `uv.lock`, the resolution the baselines were frozen against, and an advisory job reports drift under the latest dependencies.

### References

Definitions the corrected measures are checked against:

- Massey, J. (1990). Causality, feedback and directed information. *Proc. ISITA*, 303–305. — the `sum_i I(X^i; Y_i | Y^{i-1})` form now implemented by `DirectedInfo`.
- Frenzel, S. & Pompe, B. (2007). Partial mutual information for coupling analysis of multivariate time series. *Phys. Rev. Lett.* 99, 204101. — the conditional-MI estimator behind `di_kraskov` and kraskov transfer entropy.
- Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). Estimating mutual information. *Phys. Rev. E* 69, 066138. — KSG estimator and its effective-sample conditions.
- Kozachenko, L. & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. *Probl. Inf. Transm.* 23, 95–101. — the k-NN entropy that is undefined on tied data.
- Baccalá, L., Sameshima, K., Ballester, G., Do Valle, A. & Timo-Iaria, C. (1998). Studying the interaction between brain structures via directed coherence and Granger causality. *Appl. Sig. Process.* 5, 40–48. — `DC_ij = sqrt(σ_jj)|H_ij| / sqrt(Σ_k σ_kk|H_ik|²)`, the bounded form now computed.
- Kamiński, M. & Blinowska, K. (1991). A new method of the description of the information flow in the brain structures. *Biol. Cybern.* 65, 203–210. — DTF, and the `[target, source]` convention that pyspi transposes to `row = source`.
- Enochson, L. & Goodman, N. (1965). *Gaussian approximations to the distribution of sample coherence.* — the one-sample coherence significance test the group-delay backend gets wrong.
- Bokil, H., Purpura, K., Schoffelen, J.-M., Thomson, D. & Mitra, P. (2007). Comparing spectra and coherences for groups of unequal size. *J. Neurosci. Methods* 159, 337–345. — the same z-transform, with its bias and variance both `1/(2n − 2)`.
- Gotman, J. (1983). Measurement of small time differences between EEG channels. *Electroencephalogr. Clin. Neurophysiol.* 56, 501–514. — group delay as the slope of coherence phase against frequency.
- Staniek, M. & Lehnertz, K. (2008). Symbolic transfer entropy. *Phys. Rev. Lett.* 100, 158101. — one ordinal-pattern length at unit delay, source and destination alike.
- Hoyer, P., Janzing, D., Mooij, J., Peters, J. & Schölkopf, B. (2008). Nonlinear causal discovery with additive noise models. *NIPS* 21, 689–696. — `anm`.
- Fonollosa, J. A. R. (2019). Conditional distribution variability measures for causality detection. In *Cause Effect Pairs in Machine Learning*, Springer, 339–347. — `cds`.
- Blöbaum, P., Janzing, D., Washio, T., Shimizu, S. & Schölkopf, B. (2018). Cause-effect inference by comparing regression errors. *AISTATS*. — `reci`.
- Daniušis, P. et al. (2010). Inferring deterministic causal relations. *UAI*. — `igci`.
- Wibral, M., Vicente, R. & Lindner, M. (2014). Transfer entropy in neuroscience. — the maximum corrected AIS embedding criterion, as distinct from Ragwitz local-prediction selection (Ragwitz & Kantz 2002).
- Gao, W., Oh, S. & Viswanath, P. (2018). Demystifying fixed k-nearest neighbor information estimators. *IEEE Trans. Inf. Theory* 64, 5629–5661. — why the KSG error is not O(1/√N) in general.
- Vlachos, M., Kollios, G. & Gunopulos, D. (2002). Discovering similar multidimensional trajectories. *Proc. ICDE*, 673–684. — LCSS with a band that limits which pairs may match, as `lcss_constraint-*` now computes.
- Shen, C., Chung, J., Mehta, R., Xu, T. & Vogelstein, J. T. (2024). Independence testing for temporal data. *Trans. Mach. Learn. Res.* — the lagged sum behind `dcorrx` and `mgcx`, `sum_l (n - l)/n * S(target[l:], source[:n-l])`.
- Gretton, A., Bousquet, O., Smola, A. & Schölkopf, B. (2005). Measuring statistical dependence with Hilbert-Schmidt norms. *ALT*, LNCS 3734, 63–77. — HSIC.
- Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B. & Smola, A. J. (2007). A kernel statistical test of independence. *NIPS* 20, 585–592. — the median-heuristic kernel width, which `hsic` sets per variable.
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
| SPI identifiers in saved 2.x results | 22 renamed, 128 removed, 74 added; see [SPI set changes](#spi-set-changes) |
| `import pyspi` replacing `sys.stdout`/`sys.stderr`, seeding NumPy's global RNG, setting `OMP_NUM_THREADS` | no longer |
| `phase_multitaper_max_*` | removed; `CoherencePhase(statistic="max")` raises because ordinary maxima of wrapped angles are branch-cut dependent |

### SPI set changes

**Relative to pyspi 2.0.2, `full` goes from 285 SPIs to 231.**
The delta below was computed by instantiating `Calculator(subset="all")` under 2.0.2 and `Calculator(config="full")` under 3.0.0 (no computation) and comparing identifiers, with renames matched by class and parameters.

| Change | SPIs | Families |
|:-------|-----:|:---------|
| 2.0.2 `config.yaml` | 285 | |
| Unchanged identifier | 135 | (values may still change; see "Further value changes") |
| **Renamed** (identifier only, except DTW) | 22 | `sgc` (11), Kraskov DCE (4), `cce`/`di_gaussian` `n` (3), auto-embedding method (3), DTW Sakoe-Chiba (1) |
| **Removed** | 128 | duplicate, constant or degenerate (30); carried by an SPI that stays bundled (63); tracking something other than coupling (27); other estimators not bundled (8) |
| **Added** | 74 | `LagResolvedCorrelation` (13), conditional spectral measures (12), `PartialCorrelation` (6), `NonlinearInterdependence` (4), `LiangInformationFlow` (4), `EventSynchronisation` (4), `ConvergentCrossMapping` convergence (3), `ConditionalGrangerCausality` (3), `EdgeTimeSeries` (3), kernel MI/TLMI/TE with dynamic correlation exclusion (3), `WaveletCoherence` (3), `ChatterjeeXi` (2), `CrossPairwiseDistance` (2), `Barycenter` correlation (2), `LaggedConvergentCrossMapping` (2), `PhaseAmplitudeCoupling` (2), `TailDependenceAsymmetry` (2), and one each of `di_kraskov`, `te_symbolic_k-3`, `PhaseAlignedCorrelation` and `AdaptivePhaseConsistency` |
| 3.0.0 `full.yaml` | 231 | 285 − 128 + 74 |

Other bundled sets: `fast` goes from 217 SPIs (2.0.2 `fast_config.yaml`) to 138.
`fabfour` keeps its 4, with `spearmanr` in place of `spearmanr-sq` (no squared rank correlation is bundled).
`sonnet` keeps its 14, with two substitutions (`pli_multitaper_max_fs-1_fmin-0_fmax-0-5` becomes the band `mean`, and `bary_dtw_mean` becomes `bary_dtw_corr`) and three renamed identifiers.
The cost-pruned `benchmarked_p99`, `p95`, `p90` and `p80` sets are new in 3.0 (183, 173, 161 and 145 SPIs).
The complete added, removed and renamed identifier lists are in [`docs/spi_changes_2.0.2_to_3.0.0.md`](docs/spi_changes_2.0.2_to_3.0.0.md).

Every removal except `phase_multitaper_max_*` and `te_symbolic_k-1_kt-1_l-1_lt-1` is from the bundled configs only: the classes accept those parameters, so each SPI can still be computed from a custom config.
An SPI counts as carried by another when it is an exact function of an SPI that stays bundled, or agrees with one at Spearman |ρ| ≥ 0.99 on simulated multivariate systems of coupled processes.

**Removed (128)**

| SPI | Why |
|:----|:----|
| `dspli_multitaper_max_*`, `dswpli_multitaper_max_*` (6) | Constant 1.0 on essentially any input: with one recording the debiased statistics are estimated across a few tapers, so the band maximum is 1. `mean` variants kept. |
| `phase_multitaper_max_*` (3) | The maximum of a wrapped phase depends on the branch cut. `CoherencePhase` refuses `statistic="max"`. |
| `te_symbolic_k-1_kt-1_l-1_lt-1` | A length-1 ordinal pattern has one symbol, so TE is identically zero; `k_history=1` is refused. |
| `te_symbolic_k-10_kt-1_l-1_lt-1` | `10!` symbols against about 90 usable samples at T=100. Still available as `te_symbolic_k-10` for long series. |
| `di_kernel_W-0.5`, `di_kozachenko` (2) | Composed from separately estimated entropies whose biases do not cancel: 3.8-4.4 on independent data at every T (kernel), and negative values (Kozachenko). `di_kraskov_NN-4_n-5` replaces them. |
| `igci` | Not bundled; the class remains as `InformationGeometricCausalInference`. |
| `cov_EllipticEnvelope`, `cov-sq_EllipticEnvelope`, `prec_EllipticEnvelope`, `prec-sq_EllipticEnvelope` (4) | Equal to the `*_MinCovDet` SPIs (`EllipticEnvelope.fit` is `MinCovDet.fit` plus an outlier threshold). |
| `lmfit_Lasso`, `lmfit_ElasticNet` (2) | Constant `(T - 1) / T` (Lasso always; ElasticNet unless `|r| > 0.5`) at sklearn's `alpha = 1` on z-scored data. |
| `bary_euclidean_mean` | Identically 0: the Euclidean barycenter of two zero-mean series has mean 0. |
| `cce_kernel_W-0.5`, `xme_kernel_W-0.5_k10` (2) | Fixed-width box kernel in ten to eleven dimensions; the bias does not shrink with T. `xme_kernel_W-0.5_k1` stays. |
| `pli_multitaper_max_*`, `wpli_multitaper_max_*` (6) | Saturate at 1 for every pair at T >= 1000, as the `dspli` maxima do; `mean` variants kept. |
| `psi_wavelet_max_*` (3) | The maximum over time of a bounded instantaneous phase slope tends to its bound whatever the data; `mean` variants kept. |
| `bary_{dtw,sgddtw,softdtw}_mean` (3) | The barycenter of two zero-mean series has mean close to zero. Replaced by `bary_*_corr`. |
| `bary-sq_{dtw,sgddtw,softdtw}_mean` (3) | A monotone function of `bary_*_corr`. |
| `cov_ShrunkCovariance`, `cov_LedoitWolf`, `cov_OAS` (3) | On z-scored data, the empirical correlation times one shrinkage factor per dataset, so `cov_EmpiricalCovariance` carries them. |
| `cov_GraphicalLasso` | At `alpha = 0.01` the penalty barely moves the covariance, so `cov_EmpiricalCovariance` carries it. `pcorr_GraphicalLasso` stays. |
| `cov-sq_{EmpiricalCovariance,ShrunkCovariance,GraphicalLasso,LedoitWolf,MinCovDet,OAS}` (6) | The exact square of a covariance SPI. |
| `prec{,-sq}_{EmpiricalCovariance,GraphicalLasso,LedoitWolf,MinCovDet,OAS,ShrunkCovariance}` (12) | `pcorr_<estimator>` is the same precision entry rescaled to [-1, 1], and replaces them. |
| `{cov,cov-sq,prec,prec-sq}_GraphicalLassoCV` (4) | The cross-validated penalty can set every off-diagonal entry to exactly 0, and the empirical and fixed-penalty estimators detect coupling as well at a small fraction of the cost. |
| `spearmanr-sq` | The exact square of `spearmanr`. |
| `kendalltau`, `kendalltau-sq` (2) | Ranks pairs as `spearmanr` does. |
| `je_gaussian`, `ce_gaussian` (2) | On z-scored data, a constant minus `mi_gaussian`. |
| `lmfit_Ridge`, `lmfit_BayesianRidge`, `lmfit_SGDRegressor` (3) | In-sample error of a linear fit of one z-scored series on another, a function of r² that `mi_gaussian` carries (`SGDRegressor` is the same fit by a noisier optimiser). |
| `pdist_cosine`, `pdist_euclidean`, `bary-sq_euclidean_mean` (3) | On z-scored series, `1 − r`, `sqrt(2T(1 − r))` and `(1 + r)/2` of the Pearson correlation r. |
| `phi_Geo_t-1_norm-1`, `phi_star_t-1_norm-1` (2) | The `norm-0` value divided by a constant under z-scoring. |
| `pdcoh_multitaper_*`, `gpdcoh_multitaper_*` (12) | Fitted on one pair, PDC equals DTF and gPDC equals DC² at every frequency. |
| `ppc_multitaper_*` (6) | PPC = (N·PLV² − 1)/(N − 1) at every frequency, so `plv_*` carries it. |
| `xcorr_max_sig-True`, `xcorr_mean_sig-{True,False}` and their `xcorr-sq_*` forms (6) | The directed `lagcorr_*` SPIs read the same cross-correlation profile and replace them. |
| `dtf_multitaper_*`, `ddtf_multitaper_*` (12) | Fitted on one pair, DTF's direction follows which process is less predictable rather than which drives the other. `dcoh_*` is the bundled directed spectral family. |
| `gd_multitaper_delay_*` (3) | NaN unless a run of significant coherence spans three independent frequencies, which most pairs lack; `lagcorr_lag_*` carries the lead/lag. |
| `bary{,-sq}_{euclidean,dtw,sgddtw,softdtw}_max` (8) | On z-scored input the barycenter's maximum is `sqrt((1 + corr) / 2)` times its crest factor, which follows the two marginals more than their coupling. |
| `ccm_E-{None,1,10}_diff` (3) | Has the wrong sign under stochastic coupling. `lccm_E-None_k-5_tp` is the directed CCM statistic. |
| `gwtau` | On z-scored series it mostly tracks the difference between the two lag-1 autocorrelations rather than the coupling. |

**Added (74)**: see the new SPIs under **Added**.

**Renamed (22)**: no value change, except the DTW row.
In every case a parameter that changes the measure was absent from, or misreported by, the identifier.

| Was | Is | Why |
|:----|:---|:----|
| `sgc_*_fmin-0_*` (11) | `sgc_*_fmin-1e-05_*` | Spectral GC is undefined at zero frequency, so `fmin=0` is overridden, but the identifier was built from the argument. |
| `mi_kraskov_NN-4_DCE`, `tlmi_kraskov_NN-4_DCE`, `te_kraskov_NN-4_DCE_k-1_...`, `te_kraskov_NN-4_DCE_k-2_...` (4) | `..._DCE-AUTO...` | The Theiler window's value, not just its presence: 5, 10 and `"AUTO"` shared one name. |
| `cce_gaussian`, `cce_kozachenko`, `di_gaussian` (3) | `..._n-5` | `n` changes the measure. |
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4`, `te_kraskov_NN-4_k-max-10_tau-max-4`, `gc_gaussian_k-max-10_tau-max-2` (3) | `..._MAX-CORR-AIS_k-max-*` (the first also `DCE` to `DCE-AUTO`) | The auto-embedding method changes what is computed. Values are those of 2.0.2's search, which selects the source embedding too. |
| `dtw_constraint-sakoe-chiba` (1) | `dtw_constraint-sakoe-chiba_radius-auto` | The radius is chosen from the series length instead of tslearn's default of 1. **Values change** (by up to 8.5). |

Symbolic transfer entropy identifiers also lose the three embedding parameters the estimator never applied: `te_symbolic_k-<k>_kt-1_l-1_lt-1` → `te_symbolic_k-<k>`.

Values for the directed spectral classes listed under **Fixed** are also transposed relative to 2.x.

**Further value changes** (beyond those above, relative to 2.0.2; see **Fixed** and **Changed**):

| SPIs | Count | Cause |
|:-----|------:|:------|
| `ccm_E-{None,1,10}_{mean,max}` | 6 | Transposed to source-by-target; library sizes reach the largest library. |
| `softdtw_constraint-{sakoe-chiba,itakura}` | 2 | The constraint is applied. |
| `dcorrx_maxlag-{1,10}`, `mgcx_maxlag-{1,10}` | 4 | Transposed; lagged terms follow the published definition. |
| `coint_aeg_tstat_*` | 3 | Transposed: the column is regressed on the row. |
| `hsic`, `hsic_biased` | 2 | Each variable's own kernel width. |
| `mi`, `tlmi` `_kraskov_NN-4_DCE-AUTO` | 2 | Theiler-window digamma term. |
| `mi`, `tlmi`, `te` `_kraskov_NN-4_DCE-AUTO*` | 5 | AUTO Theiler window capped at `N // 10`. |
| `phi_Geo_t-1_norm-0` | 1 | Optimiser runs to a disconnected model. |
| `dcoh_*` | 6 | Closed frequency band. |
| `sgc_parametric_*` | 18 | Yule-Walker VAR fit and corrected frequency labels. |
| `phase`, `icoh`, `plv`, `pli`, `wpli`, `dspli` `_multitaper_mean_*`; `plv_multitaper_max_*` | 21 | Real-valued 0 Hz and Nyquist bins left out of the band (`dspli` also: taper-pair sign products). |
| `psi_multitaper_mean_*` (half bands) | 2 | Closed band less the real-valued bins. |
| `gpfit_DotProduct`, `gpfit_RBF` | 2 | Leave-one-out MSE. |

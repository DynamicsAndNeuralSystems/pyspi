# SPI set changes: pyspi 2.0.2 to 3.0.0

Computed by instantiating `Calculator(subset="all")` under pyspi 2.0.2 (285 SPIs; Java/JIDT available) and `Calculator(config="full")` under 3.0.0 (231 SPIs), without computing, and comparing the identifiers. Renames are matched by class and parameters.

| | count |
|:--|--:|
| 2.0.2 `config.yaml` | 285 |
| identifier unchanged | 135 |
| renamed (same class and parameters; identifier changed) | 22 |
| removed | 128 |
| added | 74 |
| 3.0.0 `full.yaml` | 231 |

285 - 128 removed + 74 added = 231; 135 + 22 renamed + 74 added = 231.
The removals include 14 made while porting away from JIDT, 24 duplicate, constant or degenerate SPIs, 37 unbundled because an SPI that stays carries them, and the 6 `xcorr` SPIs, replaced by the directed `lagcorr` SPIs.
The last 43 are removed from the bundled configs only: their classes are unchanged, and each can still be computed from a custom config.
The default kernel SPIs `mi_kernel_W-0.25`, `tlmi_kernel_W-0.25` and `te_kernel_W-0.25_k-1` keep their identifiers, but their values differ from 2.0.2: each point is now left out of its own neighbour counts.

## Added (74)

| Family | Count | Identifiers |
|:--|--:|:--|
| `LagResolvedCorrelation` (directed, lag-resolved cross-correlation: peak lag, peak value, lead/lag asymmetry, and the one-sided correlation at lags 1, 2, 3, 5, 10 and one set by the autocorrelation lengths) | 13 | `lagcorr_{lag,peak,asym}_L-{None,10}`, `lagcorr_onesided_tau-{1,2,3,5,10,1e}_L-None`, `lagcorr_onesidedpeak_L-None` |
| `PartialCorrelation` (`-P_ij / sqrt(P_ii P_jj)` from each bundled covariance estimator's precision matrix) | 7 | `pcorr_{EmpiricalCovariance,GraphicalLasso,GraphicalLassoCV,LedoitWolf,MinCovDet,OAS,ShrunkCovariance}` |
| `ChatterjeeXi` (Chatterjee's rank correlation xi at lags 0 and 1) | 2 | `xi_tau-0`, `xi_tau-1` |
| `CrossPairwiseDistance` | 2 | `xpdist_euclidean_tau-1_mean_rmse`, `xpdist_euclidean_tau-1_min_rmse` |
| `Barycenter` (alignment-adjusted correlation `2 mean(b^2) / mean_energy - 1`; replaces the `mean` statistic) | 3 | `bary_dtw_corr`, `bary_sgddtw_corr`, `bary_softdtw_corr` |
| `DirectedInfo` (KSG/Frenzel-Pompe) | 1 | `di_kraskov_NN-4_n-5` |
| `PhaseAlignedCorrelation` (Welch, automatic segment length) | 1 | `phasealigned_welch_nperseg-auto` |
| `TransferEntropy` (symbolic, 3-sample ordinal patterns) | 1 | `te_symbolic_k-3` |
| `MutualInfo`, `TimeLaggedMutualInfo`, `TransferEntropy` (box kernel with AUTO dynamic correlation exclusion) | 3 | `mi_kernel_W-0.25_DCE-AUTO`, `tlmi_kernel_W-0.25_DCE-AUTO`, `te_kernel_W-0.25_DCE-AUTO_k-1` |
| `WaveletCoherence` (Morlet coherence smoothed in time and scale, three bands) | 3 | `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-5`, `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-25`, `wtc_wavelet_mean_fs-1_fmin-0-25_fmax-0-5` |

Full list:

- `lagcorr_lag_L-None`
- `lagcorr_peak_L-None`
- `lagcorr_asym_L-None`
- `lagcorr_onesided_tau-1_L-None`
- `lagcorr_onesided_tau-2_L-None`
- `lagcorr_onesided_tau-3_L-None`
- `lagcorr_onesided_tau-5_L-None`
- `lagcorr_onesided_tau-10_L-None`
- `lagcorr_onesided_tau-1e_L-None`
- `lagcorr_onesidedpeak_L-None`
- `lagcorr_lag_L-10`
- `lagcorr_peak_L-10`
- `lagcorr_asym_L-10`
- `di_kraskov_NN-4_n-5`
- `phasealigned_welch_nperseg-auto`
- `bary_dtw_corr`
- `bary_sgddtw_corr`
- `bary_softdtw_corr`
- `xpdist_euclidean_tau-1_mean_rmse`
- `xpdist_euclidean_tau-1_min_rmse`
- `pcorr_EmpiricalCovariance`
- `pcorr_GraphicalLasso`
- `pcorr_GraphicalLassoCV`
- `pcorr_LedoitWolf`
- `pcorr_MinCovDet`
- `pcorr_OAS`
- `pcorr_ShrunkCovariance`
- `xi_tau-0`
- `xi_tau-1`
- `te_symbolic_k-3`
- `mi_kernel_W-0.25_DCE-AUTO`
- `tlmi_kernel_W-0.25_DCE-AUTO`
- `te_kernel_W-0.25_DCE-AUTO_k-1`
- `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-5`
- `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-25`
- `wtc_wavelet_mean_fs-1_fmin-0-25_fmax-0-5`

## Removed (128)

| Identifier (2.0.2) | Class | Why |
|:--|:--|:--|
| `cov_EllipticEnvelope`, `cov-sq_EllipticEnvelope` | `Covariance` | sklearn's `EllipticEnvelope.fit` is `MinCovDet.fit` plus a decision threshold, so the matrices are MinCovDet's. |
| `prec_EllipticEnvelope`, `prec-sq_EllipticEnvelope` | `Precision` | As above. |
| `lmfit_Lasso` | `LinearModel` | `Lasso(alpha=1)` zeroes the slope for any z-scored regressor, so every entry is `var(y) = (T-1)/T`: a constant. |
| `lmfit_ElasticNet` | `LinearModel` | `ElasticNet(alpha=1)` zeroes the slope unless `|r| > 0.5`, so it is mostly constant. |
| `bary_euclidean_mean` | `Barycenter` | The Euclidean barycenter of two z-scored series has mean identically zero. |
| `cce_kernel_W-0.5_n-5` | `CausalEntropy` | Fixed-width box kernel over the `n = 5` histories of both series (about ten dimensions) does not converge with `T`. |
| `xme_kernel_W-0.5_k10` | `CrossmapEntropy` | Fixed-width box kernel in 11 dimensions; as above. `xme_kernel_W-0.5_k1` is kept. |
| `pli_multitaper_max_*` (3) | `PhaseLagIndex` | With one recording PLI at each frequency is a mean of signs over a few tapers, so the band maximum saturates at 1 once `T >= 1000`. `mean` variants kept. |
| `wpli_multitaper_max_*` (3) | `WeightedPhaseLagIndex` | As above. |
| `psi_wavelet_max_*` (3) | `PhaseSlopeIndex` (wavelet) | Maximum over time of a bounded instantaneous phase slope; tends to its bound whatever the data. `mean` variants kept. |
| `bary_dtw_mean`, `bary_sgddtw_mean`, `bary_softdtw_mean` (3) | `Barycenter` | The mean of a barycenter of two zero-mean series is close to zero. Replaced by `bary_*_corr`. |
| `bary-sq_dtw_mean`, `bary-sq_sgddtw_mean`, `bary-sq_softdtw_mean` (3) | `Barycenter` | Monotone function of `bary_*_corr`. |
| `di_kernel_W-0.5` | `DirectedInfo` | DirectedInfo kernel variant dropped (3.8-4.4 on independent data at every T). |
| `di_kozachenko` | `DirectedInfo` | DirectedInfo kozachenko variant dropped (negative values). |
| `dspli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | `DebiasedSquaredPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dspli_multitaper_max_fs-1_fmin-0_fmax-0-25` | `DebiasedSquaredPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dspli_multitaper_max_fs-1_fmin-0_fmax-0-5` | `DebiasedSquaredPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dswpli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | `DebiasedSquaredWeightedPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dswpli_multitaper_max_fs-1_fmin-0_fmax-0-25` | `DebiasedSquaredWeightedPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dswpli_multitaper_max_fs-1_fmin-0_fmax-0-5` | `DebiasedSquaredWeightedPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `igci` | `InformationGeometricConditionalIndependence` | IGCI is no longer in `full.yaml` (it was active in 2.0.2). |
| `phase_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | `CoherencePhase` | Maximum of wrapped phase depends on the branch cut; the constructor now refuses `statistic="max"`. |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-25` | `CoherencePhase` | Maximum of wrapped phase depends on the branch cut; the constructor now refuses `statistic="max"`. |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-5` | `CoherencePhase` | Maximum of wrapped phase depends on the branch cut; the constructor now refuses `statistic="max"`. |
| `te_symbolic_k-10_kt-1_l-1_lt-1` | `TransferEntropy` | `10!` symbols against ~91 usable samples at T=100; unvalidated. |
| `te_symbolic_k-1_kt-1_l-1_lt-1` | `TransferEntropy` | A length-1 ordinal pattern has one symbol, so TE is identically 0. |
| `cov_ShrunkCovariance`, `cov_LedoitWolf`, `cov_OAS` | `Covariance` | Unbundled: on z-scored data, the empirical correlation times one shrinkage factor per dataset, so `cov_EmpiricalCovariance` carries them. |
| `cov-sq_EmpiricalCovariance`, `cov-sq_ShrunkCovariance`, `cov-sq_GraphicalLasso`, `cov-sq_LedoitWolf`, `cov-sq_MinCovDet`, `cov-sq_OAS` | `Covariance` | Unbundled: the exact square of a covariance SPI. |
| `lmfit_Ridge`, `lmfit_BayesianRidge`, `lmfit_SGDRegressor` | `LinearModel` | Unbundled: a function of r², which `mi_gaussian` carries (|ρ| 1.000; `SGDRegressor` is the same fit by a noisier optimiser). |
| `pdcoh_multitaper_{mean,max}_*` (6) | `PartialDirectedCoherence` | Unbundled: fitted on one pair, PDC equals DTF at every frequency. |
| `gpdcoh_multitaper_{mean,max}_*` (6) | `GeneralizedPartialDirectedCoherence` | Unbundled: fitted on one pair, gPDC equals DC² at every frequency. |
| `ppc_multitaper_{mean,max}_*` (6) | `PairwisePhaseConsistency` | Unbundled: PPC = (N·PLV² − 1)/(N − 1) at every frequency, so `plv_*` carries it. |
| `kendalltau`, `kendalltau-sq` | `KendallTau` | Unbundled: ranks pairs as `spearmanr` does (|ρ| 0.998 within datasets). |
| `pdist_cosine`, `pdist_euclidean` | `PairwiseDistance` | Unbundled: on z-scored series, `1 − r` and `sqrt(2T(1 − r))` of the Pearson correlation r. |
| `bary-sq_euclidean_mean` | `Barycenter` | Unbundled: `(1 + r)/2` on z-scored series. |
| `phi_Geo_t-1_norm-1`, `phi_star_t-1_norm-1` | `IntegratedInformation` | Unbundled: the `norm-0` value divided by a constant under z-scoring. |
| `xcorr_max_sig-True`, `xcorr_mean_sig-True`, `xcorr_mean_sig-False`, `xcorr-sq_max_sig-True`, `xcorr-sq_mean_sig-True`, `xcorr-sq_mean_sig-False` | `CrossCorrelation` | Unbundled: replaced by the directed `lagcorr` SPIs, which read the same cross-correlation profile. |

## Renamed (23)

| 2.0.2 | 3.0.0 | Change |
|:--|:--|:--|
| `cce_gaussian` | `cce_gaussian_n-5` | identifier now carries `n` |
| `cce_kernel_W-0.5` | `cce_kernel_W-0.5_n-5` | identifier now carries `n` |
| `cce_kozachenko` | `cce_kozachenko_n-5` | identifier now carries `n` |
| `di_gaussian` | `di_gaussian_n-5` | identifier now carries `n` |
| `dtw_constraint-sakoe-chiba` | `dtw_constraint-sakoe-chiba_radius-auto` | Sakoe-Chiba radius is now auto (tslearn default radius 1 in 2.x); **values change** |
| `gc_gaussian_k-max-10_tau-max-2` | `gc_gaussian_MAX-CORR-AIS_k-max-10_tau-max-2` | auto-embedding method added to the identifier |
| `mi_kraskov_NN-4_DCE` | `mi_kraskov_NN-4_DCE-AUTO` | `_DCE` -> `_DCE-AUTO` (the Theiler window's value) |
| `sgc_nonparametric_max_fs-1_fmin-0_fmax-0-25` | `sgc_nonparametric_max_fs-1_fmin-1e-05_fmax-0-25` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_nonparametric_max_fs-1_fmin-0_fmax-0-5` | `sgc_nonparametric_max_fs-1_fmin-1e-05_fmax-0-5` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_nonparametric_mean_fs-1_fmin-0_fmax-0-25` | `sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-25` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_nonparametric_mean_fs-1_fmin-0_fmax-0-5` | `sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-5` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_parametric_max_fs-1_fmin-0_fmax-0-25_order-1` | `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-25_order-1` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_parametric_max_fs-1_fmin-0_fmax-0-25_order-20` | `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-25_order-20` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_parametric_max_fs-1_fmin-0_fmax-0-25_order-None` | `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-25_order-None` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-25_order-1` | `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-1` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-25_order-20` | `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-20` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-25_order-None` | `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-None` | identifier now shows the `fmin` actually used (1e-05) |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-5_order-None` | `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-None` | identifier now shows the `fmin` actually used (1e-05) |
| `te_kraskov_NN-4_DCE_k-1_kt-1_l-1_lt-1` | `te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1` | `_DCE` -> `_DCE-AUTO` (the Theiler window's value) |
| `te_kraskov_NN-4_DCE_k-2_kt-1_l-1_lt-1` | `te_kraskov_NN-4_DCE-AUTO_k-2_kt-1_l-1_lt-1` | `_DCE` -> `_DCE-AUTO` (the Theiler window's value) |
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4` | `te_kraskov_NN-4_DCE-AUTO_MAX-CORR-AIS_k-max-10_tau-max-4` | `_DCE` -> `_DCE-AUTO` and auto-embedding method added to the identifier |
| `te_kraskov_NN-4_k-max-10_tau-max-4` | `te_kraskov_NN-4_MAX-CORR-AIS_k-max-10_tau-max-4` | auto-embedding method added to the identifier |
| `tlmi_kraskov_NN-4_DCE` | `tlmi_kraskov_NN-4_DCE-AUTO` | `_DCE` -> `_DCE-AUTO` (the Theiler window's value) |

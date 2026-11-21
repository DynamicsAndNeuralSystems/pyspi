# SPI set changes: pyspi 2.0.2 to 3.0.0

Computed by instantiating `Calculator(subset="all")` under pyspi 2.0.2 (285 SPIs; Java/JIDT available) and `Calculator(config="full")` under 3.0.0 (231 SPIs), without computing, and comparing the identifiers.
Renames are matched by class and parameters.

| | count |
|:--|--:|
| 2.0.2 `config.yaml` | 285 |
| identifier unchanged | 135 |
| renamed (same class and parameters; identifier changed) | 22 |
| removed | 128 |
| added | 74 |
| 3.0.0 `full.yaml` | 231 |

285 - 128 removed + 74 added = 231; 135 + 22 renamed + 74 added = 231.
Of the 128 removals, 30 are duplicate, constant or degenerate, 63 are carried by an SPI that stays bundled (the six `xcorr_*`, for example, by the `lagcorr_*` SPIs), 27 track something other than the coupling they are meant to measure, and 8 are estimators that 3.0.0 does not bundle for other reasons.
All but four can still be computed from a custom config: `CoherencePhase` refuses `statistic="max"`, and symbolic transfer entropy refuses `k_history=1`.
The default kernel SPIs `mi_kernel_W-0.25`, `tlmi_kernel_W-0.25` and `te_kernel_W-0.25_k-1` keep their identifiers, but their values differ from 2.0.2: each point is now left out of its own neighbour counts.

## Added (74)

| Family | Count | Identifiers |
|:--|--:|:--|
| `LagResolvedCorrelation`: directed, lag-resolved cross-correlation (peak lag, peak value, lead/lag asymmetry, and the one-sided correlation at set lags) | 13 | `lagcorr_{lag,peak,asym}_L-{None,10}`, `lagcorr_onesided_tau-{1,2,3,5,10,1e}_L-None`, `lagcorr_onesidedpeak_L-None` |
| `PartialCorrelation`: `-P_ij / sqrt(P_ii P_jj)` from each bundled covariance estimator's precision matrix | 6 | `pcorr_{EmpiricalCovariance,GraphicalLasso,LedoitWolf,MinCovDet,OAS,ShrunkCovariance}` |
| `ChatterjeeXi`: Chatterjee's rank correlation xi at lags 0 and 1 | 2 | `xi_tau-{0,1}` |
| `CrossPairwiseDistance`: shifted-path Euclidean distance | 2 | `xpdist_euclidean_tau-1_{mean,min}_rmse` |
| `Barycenter`: alignment-adjusted correlation `2 mean(b^2) / mean_energy - 1` | 2 | `bary_{dtw,softdtw}_corr` |
| `ConvergentCrossMapping`: CCM convergence, `rho(L_max) - rho(L_min)` | 3 | `ccm_E-{None,1,10}_convergence` |
| `LaggedConvergentCrossMapping`: lagged CCM, horizon and value of peak skill | 2 | `lccm_E-None_k-5_{tp,skill}` |
| `NonlinearInterdependence`: S, H, N and L from delay-embedded neighbourhoods | 4 | `nli_{L,N,H}_m-5_tau-1_k-5`, `nli_L_m-3_tau-1_k-5` |
| `LiangInformationFlow`: Liang-Kleeman information flow, pairwise and conditional | 4 | `liang_{rate,norm}_k-1`, `liang-cond_{rate,norm}_k-1` |
| `DirectedInfo`: KSG/Frenzel-Pompe estimate | 1 | `di_kraskov_NN-4_n-5` |
| `MutualInfo`, `TimeLaggedMutualInfo`, `TransferEntropy`: box kernel with AUTO dynamic correlation exclusion | 3 | `mi_kernel_W-0.25_DCE-AUTO`, `tlmi_kernel_W-0.25_DCE-AUTO`, `te_kernel_W-0.25_DCE-AUTO_k-1` |
| `TransferEntropy`: symbolic, 3-sample ordinal patterns | 1 | `te_symbolic_k-3` |
| `PhaseAlignedCorrelation`: zero-lag correlation maximised over phase-only filtering (Welch, automatic segment length) | 1 | `phasealigned_welch_nperseg-auto` |
| `AdaptivePhaseConsistency`: spectrally weighted Hilbert phase consistency | 1 | `ppc_hilbert_adaptive` |
| `WaveletCoherence`: Morlet coherence smoothed in time and scale, three bands | 3 | `wtc_wavelet_mean_fs-1_{fmin-0_fmax-0-5,fmin-0_fmax-0-25,fmin-0-25_fmax-0-5}` |
| `PhaseAmplitudeCoupling`: surrogate-corrected modulation index at two band pairs | 2 | `pac_mi_excess_p-0-01-0-05_a-0-15-0-4`, `pac_mi_excess_p-0-05-0-1_a-0-2-0-45` |
| `EdgeTimeSeries`: burstiness of a correlation | 3 | `edge_excessvar`, `edge_topshare_f-0-05`, `edge_acf1` |
| `ConditionalGrangerCausality`: Granger causality from one VAR of all processes | 3 | `cgc_p-{1,2,bic}` |
| `ConditionalPartialDirectedCoherence`, `ConditionalGeneralizedPartialDirectedCoherence`, `ConditionalDirectDirectedTransferFunction`, `PartialCoherence`: conditional spectral measures from the same VAR, three bands each | 12 | `{cpdc,cgpdc,cddtf,pcohmag}_p-bic_mean_{fmin-0_fmax-0-5,fmin-0_fmax-0-25,fmin-0-25_fmax-0-5}` |
| `EventSynchronisation`: synchronisation strength and delay asymmetry of extremes | 4 | `es_{strength,asym}_q-{0-95,0-9}` |
| `TailDependenceAsymmetry`: upper minus lower tail dependence | 2 | `tailasym_q-{0-95,0-9}` |

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
- `pcorr_EmpiricalCovariance`
- `pcorr_GraphicalLasso`
- `pcorr_LedoitWolf`
- `pcorr_MinCovDet`
- `pcorr_OAS`
- `pcorr_ShrunkCovariance`
- `xi_tau-0`
- `xi_tau-1`
- `xpdist_euclidean_tau-1_min_rmse`
- `xpdist_euclidean_tau-1_mean_rmse`
- `bary_dtw_corr`
- `bary_softdtw_corr`
- `ccm_E-None_convergence`
- `ccm_E-1_convergence`
- `ccm_E-10_convergence`
- `lccm_E-None_k-5_tp`
- `lccm_E-None_k-5_skill`
- `nli_L_m-5_tau-1_k-5`
- `nli_N_m-5_tau-1_k-5`
- `nli_H_m-5_tau-1_k-5`
- `nli_L_m-3_tau-1_k-5`
- `liang_rate_k-1`
- `liang_norm_k-1`
- `liang-cond_rate_k-1`
- `liang-cond_norm_k-1`
- `di_kraskov_NN-4_n-5`
- `mi_kernel_W-0.25_DCE-AUTO`
- `tlmi_kernel_W-0.25_DCE-AUTO`
- `te_kernel_W-0.25_DCE-AUTO_k-1`
- `te_symbolic_k-3`
- `phasealigned_welch_nperseg-auto`
- `ppc_hilbert_adaptive`
- `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-5`
- `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-25`
- `wtc_wavelet_mean_fs-1_fmin-0-25_fmax-0-5`
- `pac_mi_excess_p-0-01-0-05_a-0-15-0-4`
- `pac_mi_excess_p-0-05-0-1_a-0-2-0-45`
- `edge_excessvar`
- `edge_topshare_f-0-05`
- `edge_acf1`
- `cgc_p-1`
- `cgc_p-2`
- `cgc_p-bic`
- `cpdc_p-bic_mean_fmin-0_fmax-0-5`
- `cpdc_p-bic_mean_fmin-0_fmax-0-25`
- `cpdc_p-bic_mean_fmin-0-25_fmax-0-5`
- `cgpdc_p-bic_mean_fmin-0_fmax-0-5`
- `cgpdc_p-bic_mean_fmin-0_fmax-0-25`
- `cgpdc_p-bic_mean_fmin-0-25_fmax-0-5`
- `cddtf_p-bic_mean_fmin-0_fmax-0-5`
- `cddtf_p-bic_mean_fmin-0_fmax-0-25`
- `cddtf_p-bic_mean_fmin-0-25_fmax-0-5`
- `pcohmag_p-bic_mean_fmin-0_fmax-0-5`
- `pcohmag_p-bic_mean_fmin-0_fmax-0-25`
- `pcohmag_p-bic_mean_fmin-0-25_fmax-0-5`
- `es_strength_q-0-95`
- `es_asym_q-0-95`
- `es_strength_q-0-9`
- `es_asym_q-0-9`
- `tailasym_q-0-95`
- `tailasym_q-0-9`

## Removed (128)

| Identifier (2.0.2) | Class | Why |
|:--|:--|:--|
| `dspli_multitaper_max_*`, `dswpli_multitaper_max_*` (6) | `DebiasedSquaredPhaseLagIndex`, `DebiasedSquaredWeightedPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `phase_multitaper_max_*` (3) | `CoherencePhase` | The maximum of a wrapped phase depends on the branch cut; the constructor refuses `statistic="max"`. |
| `te_symbolic_k-1_kt-1_l-1_lt-1` | `TransferEntropy` | A length-1 ordinal pattern has one symbol, so TE is identically 0; `k_history=1` is refused. |
| `te_symbolic_k-10_kt-1_l-1_lt-1` | `TransferEntropy` | `10!` symbols against about 90 usable samples at T=100; still available as `te_symbolic_k-10` for long series. |
| `di_kernel_W-0.5`, `di_kozachenko` (2) | `DirectedInfo` | Composed from separately estimated entropies whose biases do not cancel: the kernel variant is 3.8-4.4 on independent data at every T, and the Kozachenko variant can be negative. `di_kraskov_NN-4_n-5` replaces them. |
| `igci` | `InformationGeometricConditionalIndependence` | Not bundled; the class remains as `InformationGeometricCausalInference`. |
| `cov_EllipticEnvelope`, `cov-sq_EllipticEnvelope`, `prec_EllipticEnvelope`, `prec-sq_EllipticEnvelope` (4) | `Covariance`, `Precision` | sklearn's `EllipticEnvelope.fit` is `MinCovDet.fit` plus a decision threshold, so the matrices are MinCovDet's. |
| `lmfit_Lasso`, `lmfit_ElasticNet` (2) | `LinearModel` | At `alpha=1` on z-scored data the slope is zeroed (Lasso always, ElasticNet unless `|r| > 0.5`), so the value is the constant `(T-1)/T`. |
| `bary_euclidean_mean` | `Barycenter` | The Euclidean barycenter of two z-scored series has mean identically zero. |
| `cce_kernel_W-0.5` | `CausalEntropy` | Fixed-width box kernel over the `n = 5` histories of both series (about ten dimensions); the bias does not shrink with T. |
| `xme_kernel_W-0.5_k10` | `CrossmapEntropy` | Fixed-width box kernel in 11 dimensions; as above. `xme_kernel_W-0.5_k1` is kept. |
| `pli_multitaper_max_*`, `wpli_multitaper_max_*` (6) | `PhaseLagIndex`, `WeightedPhaseLagIndex` | With one recording, PLI at each frequency is a mean of signs over a few tapers, so the band maximum saturates at 1 once T >= 1000; `mean` variants kept. |
| `psi_wavelet_max_*` (3) | `PhaseSlopeIndex` (wavelet) | The maximum over time of a bounded instantaneous phase slope tends to its bound whatever the data; `mean` variants kept. |
| `bary_{dtw,sgddtw,softdtw}_mean` (3) | `Barycenter` | The barycenter of two zero-mean series has mean close to zero; `bary_*_corr` replaces them. |
| `bary-sq_{dtw,sgddtw,softdtw}_mean` (3) | `Barycenter` | A monotone function of `bary_*_corr`. |
| `cov_{ShrunkCovariance,LedoitWolf,OAS}` (3) | `Covariance` | On z-scored data, the empirical correlation times one shrinkage factor per dataset, so `cov_EmpiricalCovariance` carries them. |
| `cov_GraphicalLasso` | `Covariance` | At `alpha = 0.01` the penalty barely moves the covariance: Spearman |ρ| >= 0.99 with `cov_EmpiricalCovariance`. `pcorr_GraphicalLasso` stays. |
| `cov-sq_{EmpiricalCovariance,ShrunkCovariance,GraphicalLasso,LedoitWolf,MinCovDet,OAS}` (6) | `Covariance` | The exact square of a covariance SPI. |
| `prec{,-sq}_{EmpiricalCovariance,GraphicalLasso,LedoitWolf,MinCovDet,OAS,ShrunkCovariance}` (12) | `Precision` | `pcorr_<estimator>` is the same precision entry rescaled to [-1, 1], and replaces them. |
| `cov_GraphicalLassoCV`, `cov-sq_GraphicalLassoCV`, `prec_GraphicalLassoCV`, `prec-sq_GraphicalLassoCV` (4) | `Covariance`, `Precision` | The cross-validated penalty can set every off-diagonal entry to exactly 0, and the fixed-penalty and empirical estimators detect coupling as well at a small fraction of the cost. |
| `spearmanr-sq` | `SpearmanR` | The exact square of `spearmanr`. |
| `kendalltau`, `kendalltau-sq` (2) | `KendallTau` | Ranks pairs as `spearmanr` does (|ρ| 0.998 within datasets). |
| `je_gaussian`, `ce_gaussian` (2) | `JointEntropy`, `ConditionalEntropy` | On z-scored data, a constant minus `mi_gaussian`. |
| `lmfit_{Ridge,BayesianRidge,SGDRegressor}` (3) | `LinearModel` | A function of r², which `mi_gaussian` carries (`SGDRegressor` is the same fit by a noisier optimiser). |
| `pdist_cosine`, `pdist_euclidean` (2) | `PairwiseDistance` | On z-scored series, `1 - r` and `sqrt(2T(1 - r))` of the Pearson correlation r. |
| `bary-sq_euclidean_mean` | `Barycenter` | `(1 + r)/2` on z-scored series. |
| `phi_Geo_t-1_norm-1`, `phi_star_t-1_norm-1` (2) | `IntegratedInformation` | The `norm-0` value divided by a constant under z-scoring. |
| `pdcoh_multitaper_{mean,max}_*` (6) | `PartialDirectedCoherence` | Fitted on one pair, PDC equals DTF at every frequency. |
| `gpdcoh_multitaper_{mean,max}_*` (6) | `GeneralizedPartialDirectedCoherence` | Fitted on one pair, gPDC equals DC² at every frequency, so `dcoh_*` carries it. |
| `ppc_multitaper_{mean,max}_*` (6) | `PairwisePhaseConsistency` | PPC = (N·PLV² - 1)/(N - 1) at every frequency, so `plv_*` carries it. |
| `xcorr_max_sig-True`, `xcorr_mean_sig-{True,False}` and their `xcorr-sq_*` forms (6) | `CrossCorrelation` | The directed `lagcorr_*` SPIs read the same cross-correlation profile and replace them. |
| `dtf_multitaper_{mean,max}_*`, `ddtf_multitaper_{mean,max}_*` (12) | `DirectedTransferFunction`, `DirectDirectedTransferFunction` | Fitted on one pair, DTF's direction follows which process is less predictable rather than which drives the other; `dcoh_*` is the bundled directed spectral family. |
| `gd_multitaper_delay_*` (3) | `GroupDelay` | NaN unless a run of significant coherence spans three independent frequencies, which most pairs lack; `lagcorr_lag_*` carries the lead/lag. |
| `bary{,-sq}_{euclidean,dtw,sgddtw,softdtw}_max` (8) | `Barycenter` | On z-scored input the barycenter's maximum is `sqrt((1 + corr) / 2)` times its crest factor, which follows the two marginals more than their coupling. |
| `ccm_E-{None,1,10}_diff` (3) | `ConvergentCrossMapping` | Has the wrong sign under stochastic coupling; `lccm_E-None_k-5_tp` is the directed CCM statistic. |
| `gwtau` | `GromovWasserstainTau` | On z-scored series it mostly tracks the difference between the two lag-1 autocorrelations rather than the coupling. |

## Renamed (22)

| 2.0.2 | 3.0.0 | Change |
|:--|:--|:--|
| `cce_gaussian` | `cce_gaussian_n-5` | identifier now carries `n` |
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

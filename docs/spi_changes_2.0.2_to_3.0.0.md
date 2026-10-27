# SPI set changes: pyspi 2.0.2 to 3.0.0

Computed by instantiating `Calculator(subset="all")` under pyspi 2.0.2 (285 SPIs; Java/JIDT available) and `Calculator(config="full")` under 3.0.0 (299 SPIs), without computing, and comparing the identifiers. Renames are matched by class and parameters.

| | count |
|:--|--:|
| 2.0.2 `config.yaml` | 285 |
| identifier unchanged | 230 |
| renamed (same class and parameters; identifier changed) | 23 |
| removed | 32 |
| added | 46 |
| 3.0.0 `full.yaml` | 299 |

285 - 32 removed + 46 added = 299; 230 + 23 renamed + 46 added = 299.
The 32 removals are the 14 made while porting away from JIDT, the 15 made after the per-SPI audit (duplicate, constant or degenerate SPIs), and the 3 `psi_wavelet_max_*` made after the follow-up simulations; all are in the table below.

## Added (46)

| Family | Count | Identifiers |
|:--|--:|:--|
| `LaggedCorrelation` (Pearson, Spearman, Kendall at lags 1, 2, 3, 4, 5, 10, 20; each plain and squared) | 42 | `corr_{pearson,spearman,kendall}_tau-{1,2,3,4,5,10,20}[-sq]` |
| `CrossPairwiseDistance` | 2 | `xpdist_euclidean_tau-1_mean_rmse`, `xpdist_euclidean_tau-1_min_rmse` |
| `DirectedInfo` (KSG/Frenzel-Pompe) | 1 | `di_kraskov_NN-4_n-5` |
| `PhaseAlignedCorrelation` (Welch, automatic segment length) | 1 | `phasealigned_welch_nperseg-auto` |

Full list:

- `corr_kendall_tau-1`
- `corr_kendall_tau-1-sq`
- `corr_kendall_tau-10`
- `corr_kendall_tau-10-sq`
- `corr_kendall_tau-2`
- `corr_kendall_tau-2-sq`
- `corr_kendall_tau-20`
- `corr_kendall_tau-20-sq`
- `corr_kendall_tau-3`
- `corr_kendall_tau-3-sq`
- `corr_kendall_tau-4`
- `corr_kendall_tau-4-sq`
- `corr_kendall_tau-5`
- `corr_kendall_tau-5-sq`
- `corr_pearson_tau-1`
- `corr_pearson_tau-1-sq`
- `corr_pearson_tau-10`
- `corr_pearson_tau-10-sq`
- `corr_pearson_tau-2`
- `corr_pearson_tau-2-sq`
- `corr_pearson_tau-20`
- `corr_pearson_tau-20-sq`
- `corr_pearson_tau-3`
- `corr_pearson_tau-3-sq`
- `corr_pearson_tau-4`
- `corr_pearson_tau-4-sq`
- `corr_pearson_tau-5`
- `corr_pearson_tau-5-sq`
- `corr_spearman_tau-1`
- `corr_spearman_tau-1-sq`
- `corr_spearman_tau-10`
- `corr_spearman_tau-10-sq`
- `corr_spearman_tau-2`
- `corr_spearman_tau-2-sq`
- `corr_spearman_tau-20`
- `corr_spearman_tau-20-sq`
- `corr_spearman_tau-3`
- `corr_spearman_tau-3-sq`
- `corr_spearman_tau-4`
- `corr_spearman_tau-4-sq`
- `corr_spearman_tau-5`
- `corr_spearman_tau-5-sq`
- `di_kraskov_NN-4_n-5`
- `phasealigned_welch_nperseg-auto`
- `xpdist_euclidean_tau-1_mean_rmse`
- `xpdist_euclidean_tau-1_min_rmse`

## Removed (32)

| Identifier (2.0.2) | Class | Why |
|:--|:--|:--|
| `cov_EllipticEnvelope`, `cov-sq_EllipticEnvelope` | `Covariance` | sklearn's `EllipticEnvelope.fit` is `MinCovDet.fit` plus a decision threshold, so the matrices are MinCovDet's bit for bit. Duplicate (audit). |
| `prec_EllipticEnvelope`, `prec-sq_EllipticEnvelope` | `Precision` | As above. Duplicate (audit). |
| `lmfit_Lasso` | `LinearModel` | `Lasso(alpha=1)` zeroes the slope for any z-scored regressor, so every entry is `var(y) = (T-1)/T`: a constant (audit). |
| `lmfit_ElasticNet` | `LinearModel` | `ElasticNet(alpha=1)` zeroes the slope unless `|r| > 0.5`; constant on two of three fixtures (audit). |
| `bary_euclidean_mean` | `Barycenter` | The Euclidean barycenter of two z-scored series has mean identically zero (`|value| < 1e-15`) (audit). |
| `cce_kernel_W-0.5_n-5` | `CausalEntropy` | Fixed-width box kernel over the `n = 5` histories of both series (about ten dimensions) does not improve with `T`; the `di_kernel` defect (audit). |
| `xme_kernel_W-0.5_k10` | `CrossmapEntropy` | Fixed-width box kernel in 11 dimensions; as above. `xme_kernel_W-0.5_k1` is kept (audit). |
| `pli_multitaper_max_*` (3) | `PhaseLagIndex` | With one recording PLI at each frequency is a mean of signs over a few tapers, so the band maximum is exactly 1 for every pair once `T >= 1000`; the `dspli` maxima defect. `mean` variants kept (audit). |
| `wpli_multitaper_max_*` (3) | `WeightedPhaseLagIndex` | As above (audit). |
| `psi_wavelet_max_*` (3) | `PhaseSlopeIndex` (wavelet) | Maximum over time of a bounded instantaneous phase slope; positive under the null both ways round (simulation). `mean` variants kept. |
| `di_kernel_W-0.5` | `DirectedInfo` | DirectedInfo kernel variant dropped (3.8-4.4 on independent data at every T). |
| `di_kozachenko` | `DirectedInfo` | DirectedInfo kozachenko variant dropped (negative values). |
| `dspli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | `DebiasedSquaredPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dspli_multitaper_max_fs-1_fmin-0_fmax-0-25` | `DebiasedSquaredPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dspli_multitaper_max_fs-1_fmin-0_fmax-0-5` | `DebiasedSquaredPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dswpli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | `DebiasedSquaredWeightedPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dswpli_multitaper_max_fs-1_fmin-0_fmax-0-25` | `DebiasedSquaredWeightedPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `dswpli_multitaper_max_fs-1_fmin-0_fmax-0-5` | `DebiasedSquaredWeightedPhaseLagIndex` | Constant 1.0 for essentially any input (few-taper debiased estimate); `mean` variants kept. |
| `igci` | `InformationGeometricConditionalIndependence` | IGCI is commented out of `full.yaml` (it was active in 2.0.2). |
| `phase_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | `CoherencePhase` | Maximum of wrapped phase depends on the branch cut; the constructor now refuses `statistic="max"`. |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-25` | `CoherencePhase` | Maximum of wrapped phase depends on the branch cut; the constructor now refuses `statistic="max"`. |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-5` | `CoherencePhase` | Maximum of wrapped phase depends on the branch cut; the constructor now refuses `statistic="max"`. |
| `te_symbolic_k-10_kt-1_l-1_lt-1` | `TransferEntropy` | `10!` symbols against ~91 usable samples at T=100; unvalidated. |
| `te_symbolic_k-1_kt-1_l-1_lt-1` | `TransferEntropy` | A length-1 ordinal pattern has one symbol, so TE is identically 0. |

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

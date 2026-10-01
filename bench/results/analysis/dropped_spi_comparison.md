# Dropped-SPI comparison across percentiles and anchors

Cost = amortized walltime (group cache-amortized, per `cut_config.py --mode amortized`).

Layout: for each percentile, three lists —
(a) dropped at **both** anchors (= robust drop),
(b) dropped **only at M=16,T=800** (cheap at M=32,T=1600),
(c) dropped **only at M=32,T=1600** (cheap at M=16,T=800).

## p80  (keep top 80% fastest — drop 66 SPIs)

### dropped at BOTH anchors (65)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `hhg` | 505.80 | 16607.16 |
| `gpfit_RBF` | 411.13 | 9505.94 |
| `gpfit_DotProduct` | 257.96 | 6083.66 |
| `mgcx_maxlag-10` | 342.69 | 5998.78 |
| `dcorrx_maxlag-10` | 77.37 | 1881.63 |
| `mgcx_maxlag-1` | 105.55 | 1810.04 |
| `ccm_E-None_mean` | 292.74 | 1445.05 |
| `ccm_E-1_mean` | 292.74 | 1445.05 |
| `ccm_E-10_mean` | 292.74 | 1445.05 |
| `ccm_E-None_max` | 292.74 | 1445.05 |
| `ccm_E-1_max` | 292.74 | 1445.05 |
| `ccm_E-None_diff` | 292.74 | 1445.05 |
| `ccm_E-10_max` | 292.74 | 1445.05 |
| `ccm_E-10_diff` | 292.74 | 1445.05 |
| `ccm_E-1_diff` | 292.74 | 1445.05 |
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4` | 31.35 | 428.69 |
| `te_kraskov_NN-4_k-max-10_tau-max-4` | 28.30 | 403.06 |
| `mgc` | 23.17 | 365.21 |
| `dcorrx_maxlag-1` | 12.68 | 290.20 |
| `cce_kernel_W-0.5` | 16.11 | 190.27 |
| `bary-sq_dtw_mean` | 11.14 | 181.23 |
| `bary_softdtw_max` | 11.14 | 181.23 |
| `bary-sq_sgddtw_mean` | 11.14 | 181.23 |
| `bary-sq_softdtw_max` | 11.14 | 181.23 |
| `bary-sq_euclidean_mean` | 11.14 | 181.23 |
| `bary_softdtw_mean` | 11.14 | 181.23 |
| `bary_euclidean_max` | 11.14 | 181.23 |
| `bary_dtw_mean` | 11.14 | 181.23 |
| `bary_dtw_max` | 11.14 | 181.23 |
| `bary-sq_euclidean_max` | 11.14 | 181.23 |
| `bary-sq_sgddtw_max` | 11.14 | 181.23 |
| `bary_sgddtw_mean` | 11.14 | 181.23 |
| `bary-sq_softdtw_mean` | 11.14 | 181.23 |
| `bary-sq_dtw_max` | 11.14 | 181.23 |
| `bary_sgddtw_max` | 11.14 | 181.23 |
| `bary_euclidean_mean` | 11.14 | 181.23 |
| `hsic` | 6.79 | 129.78 |
| `hsic_biased` | 6.84 | 126.02 |
| `anm` | 5.52 | 122.34 |
| `softdtw` | 7.92 | 118.82 |
| `softdtw_constraint-itakura` | 7.75 | 118.80 |
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-25_mean` | 62.26 | 118.63 |
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-5_mean` | 62.24 | 118.61 |
| `softdtw_constraint-sakoe-chiba` | 7.93 | 118.13 |
| `dcorr` | 3.56 | 87.11 |
| `dcorr_biased` | 3.64 | 82.55 |
| `te_kraskov_NN-4_DCE_k-1_kt-1_l-1_lt-1` | 7.30 | 73.19 |
| `te_kraskov_NN-4_DCE_k-2_kt-1_l-1_lt-1` | 7.67 | 72.97 |
| `xme_kernel_W-0.5_k10` | 5.07 | 60.11 |
| `di_kernel_W-0.5` | 5.04 | 59.87 |
| `si_kernel_W-0.5_k-1` | 4.34 | 50.31 |
| `cce_kozachenko` | 4.04 | 48.52 |
| `te_kraskov_NN-4_k-1_kt-1_l-1_lt-1` | 3.60 | 41.53 |
| `lcss_constraint-itakura` | 1.91 | 25.66 |
| `lcss_constraint-sakoe-chiba` | 1.92 | 23.56 |
| `lcss` | 1.79 | 23.41 |
| `xme_kozachenko_k10` | 1.75 | 23.34 |
| `te_kernel_W-0.25_k-1` | 2.00 | 21.40 |
| `cds` | 3.49 | 20.55 |
| `tlmi_kraskov_NN-4` | 1.88 | 19.03 |
| `tlmi_kraskov_NN-4_DCE` | 1.71 | 18.89 |
| `ddtf_multitaper_max_fs-1_fmin-0_fmax-0-5` | 1.59 | 12.17 |
| `ddtf_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | 1.59 | 12.17 |
| `ddtf_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | 1.59 | 12.17 |
| `ddtf_multitaper_max_fs-1_fmin-0_fmax-0-25` | 1.59 | 12.17 |

### dropped ONLY at M=16,T=800 (1)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `ddtf_multitaper_mean_fs-1_fmin-0_fmax-0-25` | 1.59 | 12.17 |

### dropped ONLY at M=32,T=1600 (1)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `xme_kernel_W-0.5_k1` | 1.11 | 12.38 |

## p90  (keep top 90% fastest — drop 33 SPIs)

### dropped at BOTH anchors (31)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `hhg` | 505.80 | 16607.16 |
| `gpfit_RBF` | 411.13 | 9505.94 |
| `gpfit_DotProduct` | 257.96 | 6083.66 |
| `mgcx_maxlag-10` | 342.69 | 5998.78 |
| `dcorrx_maxlag-10` | 77.37 | 1881.63 |
| `mgcx_maxlag-1` | 105.55 | 1810.04 |
| `ccm_E-None_mean` | 292.74 | 1445.05 |
| `ccm_E-1_mean` | 292.74 | 1445.05 |
| `ccm_E-10_mean` | 292.74 | 1445.05 |
| `ccm_E-None_max` | 292.74 | 1445.05 |
| `ccm_E-1_max` | 292.74 | 1445.05 |
| `ccm_E-None_diff` | 292.74 | 1445.05 |
| `ccm_E-10_max` | 292.74 | 1445.05 |
| `ccm_E-10_diff` | 292.74 | 1445.05 |
| `ccm_E-1_diff` | 292.74 | 1445.05 |
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4` | 31.35 | 428.69 |
| `te_kraskov_NN-4_k-max-10_tau-max-4` | 28.30 | 403.06 |
| `mgc` | 23.17 | 365.21 |
| `dcorrx_maxlag-1` | 12.68 | 290.20 |
| `cce_kernel_W-0.5` | 16.11 | 190.27 |
| `bary_softdtw_max` | 11.14 | 181.23 |
| `bary-sq_dtw_mean` | 11.14 | 181.23 |
| `bary-sq_sgddtw_mean` | 11.14 | 181.23 |
| `bary-sq_softdtw_max` | 11.14 | 181.23 |
| `bary-sq_euclidean_mean` | 11.14 | 181.23 |
| `bary_softdtw_mean` | 11.14 | 181.23 |
| `bary-sq_euclidean_max` | 11.14 | 181.23 |
| `bary-sq_sgddtw_max` | 11.14 | 181.23 |
| `bary-sq_softdtw_mean` | 11.14 | 181.23 |
| `bary-sq_dtw_max` | 11.14 | 181.23 |
| `bary_sgddtw_max` | 11.14 | 181.23 |

### dropped ONLY at M=16,T=800 (2)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-25_mean` | 62.26 | 118.63 |
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-5_mean` | 62.24 | 118.61 |

### dropped ONLY at M=32,T=1600 (2)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `bary_dtw_max` | 11.14 | 181.23 |
| `bary_sgddtw_mean` | 11.14 | 181.23 |

## p95  (keep top 95% fastest — drop 16 SPIs)

### dropped at BOTH anchors (15)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `hhg` | 505.80 | 16607.16 |
| `gpfit_RBF` | 411.13 | 9505.94 |
| `gpfit_DotProduct` | 257.96 | 6083.66 |
| `mgcx_maxlag-10` | 342.69 | 5998.78 |
| `dcorrx_maxlag-10` | 77.37 | 1881.63 |
| `mgcx_maxlag-1` | 105.55 | 1810.04 |
| `ccm_E-None_mean` | 292.74 | 1445.05 |
| `ccm_E-None_diff` | 292.74 | 1445.05 |
| `ccm_E-10_max` | 292.74 | 1445.05 |
| `ccm_E-1_diff` | 292.74 | 1445.05 |
| `ccm_E-10_diff` | 292.74 | 1445.05 |
| `ccm_E-1_mean` | 292.74 | 1445.05 |
| `ccm_E-None_max` | 292.74 | 1445.05 |
| `ccm_E-1_max` | 292.74 | 1445.05 |
| `ccm_E-10_mean` | 292.74 | 1445.05 |

### dropped ONLY at M=16,T=800 (1)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-25_mean` | 62.26 | 118.63 |

### dropped ONLY at M=32,T=1600 (1)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4` | 31.35 | 428.69 |

## p99  (keep top 99% fastest — drop 3 SPIs)

### dropped at BOTH anchors (2)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `hhg` | 505.80 | 16607.16 |
| `gpfit_RBF` | 411.13 | 9505.94 |

### dropped ONLY at M=16,T=800 (1)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `mgcx_maxlag-10` | 342.69 | 5998.78 |

### dropped ONLY at M=32,T=1600 (1)

| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |
|---|---:|---:|
| `gpfit_DotProduct` | 257.96 | 6083.66 |

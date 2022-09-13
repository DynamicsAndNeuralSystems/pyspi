# pynats

Python-based network analysis for time series.

# Installation

Requires Octave if using the integrated information statistics.

## Getting started

When it's ready you'll be able to download it via `pip`
> pip install pynats

## Issues

- You need `llvmlite`, which requires `LLVM 10.0.x` or `LLVM 9.0.x` and should be fine for most systems however in `arch` this needs to be installed via the AUR (`python-llvmlite`) or by installing `llvm10` using `pacman` (which will overwrite the latest version)
- Same for `PyTorch` (install with `python-pytorch`). If you cannot, then torch might be able to be installed via `pip`, however I would use the `--no-cache-dir` flag otherwise there's a `MemoryError` raised.

# List of functions

### Correlation coefficients

Association coefficients that assume the observations are paired but not necessarily values of a time series. As coefficients these statistics are signed in their raw form.

| Function | Description |
| ----------- | ----------- |
| `pearsonr` | Pearson's product-moment correlation coefficient |
| `spearmanr` | Spearman's rank-correlation coefficient |
| `kendalltau` | Kendall's rank-correlation coefficient |
| `pcorr` | Partial correlation (conditioned on all other processes) |
| `prec` | Precision (inverse of partial correlation) |
| `xcorr` | Cross correlation (with output statistic dependent on parameters, see below) |

### Independence criterion

statistics that assume a certain model of paired observations (not necessarily time series) to be important in distinguishing independence or integration.

| Function | Description |
| ----------- | ----------- |
| `hsic` | Hilbert-Schmidt Independence Criterion |
| `hhg` | Heller-Heller-Gorfine independence criterion |
| `dcorr` | Distance correlation |
| `mgc` | Multi-scale graph correlation |
| `anm` | Additive noise model |
| `gpfit` | Gaussian process bivariate fit |
| `cds` | Conditional distribution similarity fit |
| `igci` | Information-geometric conditional independence |
| `reci` | Neural correlation coefficient |

### Discrete-time statistics

statistics that assume the temporal precendence in discrete-time time series are important in distinguishing independence or integration.

| Function | Description |
| -------- | ----------- |
| `coint_aeg` | Cointegration computed with the Engle-Granger two-step method |
| `coint_johansen` | Cointegration computed with the Johansen test |
| `ccm` | Convergent-cross mapping |
| `dcorrx` | Distance correlation for time series |
| `mgc` | Multi-scale graph correlation for time series |
| `dtw` | (Fast) dynamic time warping |

### Spectral statistics

statistics that involve a Fourier or wavelet transformation prior to computing statistics.
Each statistic is averaged over some frequency (and time, for wavelet transformations) range specified by the parameters (see below).

| Function | Description |
| -------- | ----------- |
| `coherency` | Coherency |
| `phase` | Coherence phase |
| `cohmag` | Coherence magnitude |
| `icoh` | Imaginary part of coherence |
| `plv` | Phase-locking value |
| `pli` | Phase-lag index |
| `wpli` | Weighted phase-lag index |
| `dspli` | Weighted phase-lag index |
| `dswpli` | Debiased squared weighted phase-lag index |
| `pcoh` | Partial coherence |
| `pdcoh` | Partial directed coherence |
| `gpdcoh` | Generalized partial directed coherence |
| `dtf` | Directed transfer function |
| `ddtf` | Direct directed transfer function |
| `psi` | Phase-slope index |
| `gd` | Group delay |
| `sgc` | Spectral Granger causality |
| `ppc` | Pairwise-phase consistency |
| `pec` | Power envelope correlation |

### Information-theoretic statistics

General bivariate information-theoretic statistics that are computed with either a kernel or a Gaussian estimator.

| Function | Description |
| -------- | ----------- |
| `mi` | Mutual information |
| `tl_mi` | Time-lagged mutual information |
| `te` | Transfer entropy |
| `ce` | Conditional entropy |
| `cce` | Causally-conditioned entropy |
| `di` | Directed information |
| `si` | Stochastic interaction |
| `xme` | Cross-map entropy (similarity index) |

# List of parameters

The output of a number of the functions use parameters to define the statistics.
The shorthand for each parameter (LHS of the table) is appended to the function name with underscores between each parameter.

| Parameter | Description |
| -------- | ----------- |
| `sq` | Square the output |
| `mean` | Take the mean of the output sequence (for functions such as `xcorr` and `ccm`) |
| `max` | Take the max of the output sequence |
| `diff` | Take the mean of the difference between two output sequences |
| `empirical` | Maximum likelihood estimator for covariance matrix (non rank-based correlation coefficients only) |
| `shrunk` | Shrunk estimate for the covariance matrix |
| `ledoit_wolf` | Ledoit-Wolf estimator for the covariance matrix |
| `oas` | Oracle Approximating Shrinkage estimator for the covariance matrix |
| `tstat` | Outputs a t-statistic (for cointegration) |
| `pvalue` | Outputs a p-value (for cointegration) |
| `max_eig_stat` | Outputs the maximum eigenvalue (for cointegration) |
| `trace_stat` | Outputs the trace of the matrix (for cointegration) |
| `kernel_W-X` | Kernel estimator for information-theoretic statistics with width of `X` (default: `0.5`) |
| `kraskov_NN-X` | Kraskov-Strogaz-Grassberger estimator for mutual information-based statistics with nearest-neighbours `X` (default: `4`) |
| `gaussian` | Gaussian estimator for information-theoretic statistics |
| `kozachenko` | Kozachenko estimator for entropy-based statistics |
| `k-X` | History length of target process for transfer entropy/Granger causality (default: `1`) |
| `kt-X` | Time delay of target process for transfer entropy/Granger causality (default: `1`) |
| `l-X` | History length of source process for transfer entropy/Granger causality (default: `1`) |
| `lt-X` | Time delay of source process for transfer entropy/Granger causality (default: `1`) |
| `DCE` | Dynamic correlation exclusion (a.k.a Theiler Window) for information-theoretic statistics (not yet suitable for Gaussian estimator) |
| `fs-X` | Sampling frequency of `X` (default: `1`) |
| `fmin-X` | Minimum frequency for averaging spectral/wavelet statistics (default: `0`) |
| `fmax-X` | Maximum frequency for averaging  spectral/wavelet statistics (default: `nyquist = fs/2`) |
| `order-X` | AR Order for parametric spectral Granger causality, choose `None` for optimisation by BIC (default: `None`) |
| `cwt` | Continuous wavelet transformation (for spectral statistics) |

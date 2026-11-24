<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyspi/main/img/pyspi_logo_darkmode.png" media="(prefers-color-scheme: dark)">
    <img src="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyspi/main/img/pyspi_logo.png" alt="pyspi logo" height="200"/>
  </picture>
</p>

<h1 align="center"><em>pyspi</em>: Python Toolkit of Statistics for Pairwise Interactions</h1>



<p align="center">
 	<a href="https://zenodo.org/badge/latestdoi/601919618"><img src="https://zenodo.org/badge/601919618.svg" height="20"/></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg" alt="License: GPL-3.0-or-later" height="20"/></a>
    <a href="https://github.com/DynamicsAndNeuralSystems/pyspi/actions/workflows/run_unit_tests.yaml"><img src="https://github.com/DynamicsAndNeuralSystems/pyspi/actions/workflows/run_unit_tests.yaml/badge.svg" height="20"/></a><br>
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.11 | 3.12 | 3.13"></a><br>
    <a href="https://pepy.tech/projects/pyspi"><img src="https://static.pepy.tech/personalized-badge/pyspi?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads" alt="pyspi downloads" height="20"/></a>
</p>

_pyspi_ is a comprehensive python library for computing statistics of pairwise interactions (SPIs) from multivariate time-series (MTS) data.
The toolbox provides easy access to hundreds of methods for evaluating the relationship between pairs of time series, from simple statistics (like correlation) to advanced multi-step algorithms (like Granger causality).
The code is licensed under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0), version 3 or any later version (GPL-3.0-or-later).

**Feel free to reach out for help with real-world applications.**
Feedback is much appreciated through [issues](https://github.com/DynamicsAndNeuralSystems/pyspi/issues), or [pull requests](https://github.com/DynamicsAndNeuralSystems/pyspi/pulls).

| Section       | Description           |
|:--------------|:----------------------|
| [Installation](#installation-)       | Installing _pyspi_ and its dependencies                      |
| [Getting Started](#getting-started-) | A quick introduction on how to get started with _pyspi_      |
| [Choosing an SPI set](#choosing-an-spi-set) | The bundled configs, and how to pick one |
| [Running at scale](#running-at-scale) | Parallelism, checkpointing, and HPC clusters |
| [SPI Descriptions](#spi-descriptions-) | A link to the full table of SPIs and detailed descriptions   |
| [Documentation](#documentation)     | A link to our API reference and full documentation on GitBooks |
| [Contributing to _pyspi_](#contributing-to-pyspi-) | A guide for community members willing to contribute to _pyspi_ |
| [Acknowledgement](#acknowledgement-) | A citation for _pyspi_ for scholarly articles                |
| [Our Contributors](#our-contributors-) | A summary of our primary contributors                        |
<hr style="border-top: 3px solid #bbb;">

## Installation 📥

_pyspi_ requires **Python 3.11 or newer**.

> **Upgrading from 2.x?** Version 3.0 removes the Java/JIDT dependency and changes the `Calculator` API.
> See the [CHANGELOG](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/CHANGELOG.md#migrating-from-2x) for the migration table.

```bash
pip install pyspi
```

_pyspi_ depends on a large scientific stack, so installing into a dedicated
environment is strongly recommended:

```bash
python -m venv .venv && source .venv/bin/activate
pip install pyspi
```

Developing on _pyspi_ itself, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DynamicsAndNeuralSystems/pyspi.git
cd pyspi
uv sync                       # runtime + dev dependencies from uv.lock
uv run pytest -q              # fast test suite
```

Optional extras: `.[tsfile]` to load sktime/aeon `.ts` files, `.[bench]` for the compute-cost benchmark suite in [`bench/`](https://github.com/DynamicsAndNeuralSystems/pyspi/tree/main/bench).

## Getting Started 🚀

```python
import numpy as np
import pyspi
from pyspi.calculator import Calculator

dataset = np.random.randn(5, 500)      # 5 processes, 500 observations
calc = Calculator(dataset=dataset)     # z-scores each process by default
calc.compute()                         # compute every SPI

calc.table                             # rows = processes, columns = (SPI, process)
calc.table["cov_EmpiricalCovariance"]  # one SPI's 5x5 matrix
calc.feature_vector()                  # one value per independent pair, for ML

calc.quality                           # one code per cell: 0 good, else why it is NaN
calc.save("results.npz")               # canonical on-disk format
pyspi.load_table("results.npz")        # round-trips exactly
pyspi.load_results("results.npz")      # table, quality codes, run spec, errors, reasons
```

The dataset is `(M, T)`, one row per process; for a `(T, M)` array or a DataFrame with one column per process, pass `dim_order="sp"` (`--dim-order sp` on the command line), and a DataFrame's labels become the process names.
With more processes than observations and no `dim_order`, pyspi warns; pass `dim_order="ps"` to state that the layout is intended.

Every value in `calc.table` is a real, finite number or NaN.
`calc.quality` says what each cell is, with the codes of `pyspi.Quality` (shared with hctsa and pyhctsa): good, error, NaN (undefined or not appropriate, including every self-pair), +/-Inf, or complex.
`calc.quality` is the record of what happened to each cell, and the other records explain its codes.
An error means that something went wrong, and its message is in `calc.errors` (or `calc.pair_errors` for a single pair).
Not appropriate means that the SPI ran as intended and the data cannot support its estimate (too few observations, tied values, a constant series): the reason is in `calc.not_appropriate` (or `calc.pair_not_appropriate`), and the SPI is not counted as failed.
An SPI that returns no finite value without raising anything is recorded in `calc.not_appropriate` too.

Results are stored as an `(n_spis, M, M)` array, the data's natural shape, plus the SPI and process names and each SPI's symmetry trait.
`.csv` is also accepted for eyeballing small results, but it is one-way and impractical for the full SPI set.

`feature_vector()` turns a table into features for classification or regression, one value per independent pair.
Undirected SPIs store each pair twice (the lower triangle mirrors the upper), and antisymmetric SPIs such as `phase`, `pli`, `wpli`, `psi` and `es_asym` store it twice with the sign flipped.
For these the vector keeps only the pairs `(i, j)` with `i < j`, and a positive antisymmetric value means process `i` leads process `j`.
Directed SPIs keep both orientations.
Self-pairs are dropped unless `include_self=True`.
`pyspi.feature_vector(pyspi.load_table("results.npz"))` does the same for a saved table, using the symmetry traits saved with it.

Or from the command line, writing a results table next to your data:

```bash
python -m pyspi compute --data ts.npy --config fast --n-jobs 4 --output results.npz
```

Try it on a bundled example dataset:

```python
from pyspi.data import load_dataset, available_datasets

available_datasets()                   # forex, cml, standard_normal
calc = Calculator(dataset=load_dataset("forex"), config="fabfour")
calc.compute()
```

`forex` is quantised: every one of its seven processes contains repeated values (24–212 distinct values across 250 observations).
The Kozachenko-Leonenko entropy is undefined on tied values, so on `forex` the SPIs built on it (`je_kozachenko`, `ce_kozachenko`, `cce_kozachenko_n-5`, `xme_kozachenko_k1` and `si_kozachenko_k-1`) are not appropriate for it: their values are NaN, with the reason in `calc.not_appropriate` and the code `Quality.NAN` in `calc.quality`.
The KSG (`kraskov`) SPIs run, because they break ties with a tiny fixed-seed jitter.

The [tutorial notebook](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/demos/pyspi_tutorial.ipynb) walks through the 3.0 API.
Walkthrough tutorials in the online documentation, written for 2.x:
[simple demonstration](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/getting-started-a-simple-demonstration) ·
[finance](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/finance-stock-price-time-series) ·
[neuroimaging](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/neuroimaging-fmri-time-series)

## Choosing an SPI set

Computing all 231 SPIs is expensive, and cost grows steeply in both the number
of processes *M* and the series length *T*. `config=` takes either a bundled
name or a path to your own YAML:

| `config=` | SPIs | Use when |
|:----------|-----:|:---------|
| `"full"` (default) | 231 | You want everything and can afford it. |
| `"fast"` | 138 | General use; drops the slowest SPIs. |
| `"benchmarked_p99"` | 229 | Everything except `bary_softdtw_corr` and `gpfit_RBF`, which cost more than the other 229 together. |
| `"benchmarked_p95"` | 213 | p90 plus costlier settings of its methods (`ccm_E-10`, `dcorrx_maxlag-10`, `cce_kozachenko`, Itakura soft-DTW) and `hsic`. |
| `"benchmarked_p90"` | 206 | Large batches. |
| `"benchmarked_p80"` | 185 | Cost-constrained sweeps. |
| `"sonnet"` | 14 | One representative SPI per module (M01-M14); M07 has no bundled SPI, so M14 has two. |
| `"fabfour"` | 4 | Smoke tests and quick sanity checks. |

Each `benchmarked_p<N>` set holds about the cheapest *N*% of the 231 SPIs in `full`, ranked by amortized cost (SPIs that share a cache split its cost) on one core at *M* = 16, *T* = 1000, with cache groups kept whole.
A few SPIs are placed by hand, and each config's header lists them: an SPI whose cost grows about as *T*² or faster moves to a larger set, one that a cheaper SPI tracks closely goes to `benchmarked_p99`, and `cds` and `te_kraskov_NN-4_DCE-AUTO_k-2_kt-1_l-1_lt-1` are in `benchmarked_p90` for coverage although they cost more than its cutoff.
See [`bench/README.md`](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/bench/README.md) for the benchmark's methodology, and [`bench/results/analysis/report.md`](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/bench/results/analysis/report.md) for how cost scales across a grid of *M* ∈ {4..64} and *T* ∈ {200..3200}.

To build your own subset by keyword:

```python
from pyspi.utils import filter_spis
filter_spis(["nonlinear", "directed"], output_name="my_spis")
calc = Calculator(config="my_spis.yaml")
```

## Frequency bands

pyspi is sampling-rate agnostic: a dataset carries no sampling rate.
The bundled spectral SPIs use normalised frequency, in cycles per sample, so a band `[0, 0.25]` covers 0 to `fs / 4` whatever the true sampling rate `fs`, and a frequency in Hz is the normalised one times `fs`.
To define bands in Hz, set `fs`, `fmin` and `fmax` on the multitaper and wavelet entries of a custom config; for example, coherence over 8–12 Hz at 250 Hz, with the identifier `cohmag_multitaper_mean_fs-250_fmin-8_fmax-12`:

```yaml
.statistics.spectral:
  CoherenceMagnitude:
    configs:
      - {fs: 250, fmin: 8, fmax: 12}
```

The conditional VAR (`cpdc`, `cgpdc`, `cddtf`, `pcohmag`) and phase-amplitude coupling (`pac`) entries take cycles per sample only, so divide their band edges by `fs`.
A band outside `[0, fs / 2]` is refused when the config is loaded.
A band that holds no frequency of the series' frequency grid, which depends on its length, gives NaN labelled not appropriate, with its reason in `calc.not_appropriate`.

## Running at scale

**Prefer one dataset per process.** `Calculator.compute(n_jobs=...)` parallelises
*within* a single dataset, but SPIs sharing a cache are grouped into one task
that runs serially in one worker, so the makespan is floored by the longest
single task. Measured on the benchmark grid at *M*=64, *T*=1600, that ceiling is
**2.3-4.5x regardless of `n_jobs`**, on every config. If you have more datasets
than cores — the usual case on a cluster — run each dataset in its own
single-core process instead. That scales close to linearly, isolates failures to
one dataset, and schedules far faster as an array job.

Use `n_jobs > 1` when you have fewer datasets than cores, when one dataset must
finish under a walltime limit, or when memory forces you to share a single copy
of a large dataset across workers.

```bash
# PBS array job: one dataset per task, one core each
#PBS -J 0-499
#PBS -l ncpus=1,mem=8GB
python -m pyspi compute --data "datasets/${PBS_ARRAY_INDEX}.npy" \
    --config benchmarked_p90 --checkpoint-dir "results/${PBS_ARRAY_INDEX}/"
```

`--checkpoint-dir` writes each SPI to `<dir>/<identifier>.npy` as it finishes and
resumes from those on a re-run, so a job killed at the walltime limit picks up
where it stopped. `PYSPI_N_JOBS` sets `n_jobs` from the environment.

When `n_jobs > 1`, workers pin their nested BLAS/OpenMP thread pools to one
thread each to avoid oversubscription. macOS is
an exception: its Accelerate BLAS cannot be pinned this way, so prefer `n_jobs=1`
there for a single dataset.

On macOS and Windows, workers start by re-importing your script, so a script that calls `compute(n_jobs=...)` with more than one worker must do so under an `if __name__ == "__main__":` guard.
Without it, or for a script piped to `python -`, pyspi raises a `RuntimeError` saying so.

```python
if __name__ == "__main__":
    calc = Calculator(dataset=X, config="fast")
    calc.compute(n_jobs=4)
    calc.save("results.npz")
```

## SPI Descriptions 📋
To access a table with a high-level overview of the _pyspi_ library of SPIs, including their associated identifiers, see the [table of SPIs](https://time-series-features.gitbook.io/pyspi/spis/table-of-spis) in the full documentation.
For detailed descriptions of each SPI, as well as its associated estimators, we provide a full breakdown in the [SPI descriptions](https://time-series-features.gitbook.io/pyspi/spis/spi-descriptions) page of our documentation. 
Both pages describe the 2.x set; for 3.0, [`full.yaml`](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/pyspi/configs/full.yaml) lists every bundled SPI, each SPI class's docstring describes it, and [`docs/spi_changes_2.0.2_to_3.0.0.md`](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/docs/spi_changes_2.0.2_to_3.0.0.md) lists what was added, removed and renamed.

## Documentation
The online documentation on [GitBook](https://time-series-features.gitbook.io/pyspi/) describes pyspi 2.x; for 3.0, this README, the [CHANGELOG](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/CHANGELOG.md) (with its [migration table](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/CHANGELOG.md#migrating-from-2x)) and the docstrings are current.
Use the following links to quickly access some of the key sections:

- [Usage guide](https://time-series-features.gitbook.io/pyspi/usage)
- [Distributing _pyspi_ computations](https://time-series-features.gitbook.io/pyspi/usage/advanced-usage/distributing-calculations-on-a-cluster)
- [Table of SPIs and descriptions](https://time-series-features.gitbook.io/pyspi/spis)
- [FAQ](https://time-series-features.gitbook.io/pyspi/usage/faq)
- [API Reference](https://time-series-features.gitbook.io/pyspi/api-reference)
- [Development guide](https://time-series-features.gitbook.io/pyspi/development)

## Contributing to _pyspi_ 👨‍👨‍👦‍👦
Contributions play a vital role in the continual development and enhancement of _pyspi_, a project built and enriched through community collaboration.
If you would like to contribute to _pyspi_, or explore the many ways in which you can participate in the project, please have a look at our 
detailed [contribution guidelines](https://time-series-features.gitbook.io/pyspi/development/contributing-to-pyspi) about how to proceed.
[CONTRIBUTING.md](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/CONTRIBUTING.md) sets out the code and docstring conventions.
In contributing to _pyspi_, all participants are expected to adhere to our [code of conduct](https://time-series-features.gitbook.io/pyspi/development/code-of-conduct).

### SPI Wishlist
We strive to provide the most comprehensive toolkit of SPIs. If you have ideas for new SPIs or suggestions for improvements to existing ones, we are eager to hear from and collaborate with you! 
Any pairwise dependence measure, provided it is accompanied by a published research paper, typically falls within the scope for consideration in the 
_pyspi_ library.
To request an SPI, please [open an issue](https://github.com/DynamicsAndNeuralSystems/pyspi/issues) with a link to the paper that describes it.

## Acknowledgement 👍

If you use this software, please read and cite this article:

- &#x1F4D7; O.M. Cliff, A.G. Bryant, J.T. Lizier, N. Tsuchiya, B.D. Fulcher. [Unifying pairwise interactions in complex dynamics](https://doi.org/10.1038/s43588-023-00519-x), _Nature Computational Science_ (2023).

Note that [preprint](https://arxiv.org/abs/2201.11941) and [free-to-read](https://rdcu.be/dn3JB) versions of this article are available.

<details closed>
    <summary>Click here for a BibTex reference:</summary>

```
@article{Cliff2023:UnifyingPairwiseInteractions,
	title = {Unifying pairwise interactions in complex dynamics},
	volume = {3},
	issn = {2662-8457},
	url = {https://www.nature.com/articles/s43588-023-00519-x},
	doi = {10.1038/s43588-023-00519-x},
	number = {10},
	journal = {Nature Computational Science},
	author = {Cliff, Oliver M. and Bryant, Annie G. and Lizier, Joseph T. and Tsuchiya, Naotsugu and Fulcher, Ben D.},
	month = sep,
	year = {2023},
	pages = {883--893},
}
```

</details>

## Other highly comparative toolboxes 🧰
If you are interested in trying other highly comparative toolboxes like _pyspi_, see the below list:

- [_hctsa_](https://github.com/benfulcher/hctsa), the _highly comparative time-series analysis_ toolkit, computes 6392 time-series features from univariate time series.
- [_hcga_](https://github.com/barahona-research-group/hcga), a _highly comparative graph analysis_ toolkit, computes several thousands of graph features directly from any given network.
- [_pyhctsa_](https://github.com/DynamicsAndNeuralSystems/pyhctsa), a python implementation of the _highly comparative time-series analysis_ toolkit.


## Our Contributors 🌟
We are thankful for the contributions of each and everyone who has helped make this project better. 
Whether you've added a line of code, improved our documentation, or reported an issue, your contributions are greatly appreciated! 
Below are some of the leading contributors to _pyspi_:

<a href="https://github.com/DynamicsAndNeuralSystems/pyspi/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DynamicsAndNeuralSystems/pyspi" />
</a>

## License 🧾
_pyspi_ is released under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0), version 3 or any later version (GPL-3.0-or-later); see [LICENSE.txt](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/LICENSE.txt).

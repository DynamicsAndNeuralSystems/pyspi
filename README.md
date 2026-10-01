<p align="center">
  <picture>
    <source srcset="img/pyspi_logo_darkmode.png" media="(prefers-color-scheme: dark)">
    <img src="img/pyspi_logo.png" alt="pyspi logo" height="200"/>
  </picture>
</p>

<h1 align="center"><em>pyspi</em>: Python Toolkit of Statistics for Pairwise Interactions</h1>



<p align="center">
 	<a href="https://zenodo.org/badge/latestdoi/601919618"><img src="https://zenodo.org/badge/601919618.svg" height="20"/></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" height="20"/></a>
    <a href="https://github.com/DynamicsAndNeuralSystems/pyspi/actions/workflows/run_unit_tests.yaml"><img src="https://github.com/DynamicsAndNeuralSystems/pyspi/actions/workflows/run_unit_tests.yaml/badge.svg" height="20"/></a>
    <a href="https://twitter.com/compTimeSeries"><img src="https://img.shields.io/twitter/url/https/twitter.com/compTimeSeries.svg?style=social&label=Follow%20%40compTimeSeries" height="20"/></a><br>
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.10 | 3.11 | 3.12"></a><br>
    <a href="https://pepy.tech/projects/pyspi"><img src="https://static.pepy.tech/personalized-badge/pyspi?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads" alt="pyspi downloads" height="20"/></a>
</p>

_pyspi_ is a comprehensive python library for computing statistics of pairwise interactions (SPIs) from multivariate time-series (MTS) data.
The toolbox provides easy access to hundreds of methods for evaluating the relationship between pairs of time series, from simple statistics (like correlation) to advanced multi-step algorithms (like Granger causality).
The code is licensed under the [GNU GPL v3 license](http://www.gnu.org/licenses/gpl-3.0.html) (or later).

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

_pyspi_ requires **Python 3.10 or newer**.

> **Upgrading from 2.x?** Version 3.0 removes the Java/JIDT dependency and
> changes the `Calculator` API. See [CHANGELOG.md](CHANGELOG.md) for the
> migration table.

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

Optional extras: `.[tsfile]` to load sktime/aeon `.ts` files, `.[bench]` for the
compute-cost benchmark suite in [`bench/`](bench/).

For a more detailed guide, see the [full documentation](https://time-series-features.gitbook.io/pyspi/installation/installing-pyspi),
the [troubleshooting guide](https://time-series-features.gitbook.io/pyspi/installation/troubleshooting),
and [alternative installation options](https://time-series-features.gitbook.io/pyspi/installation/alternative-installation-options).

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

calc.save("results.npz")               # canonical on-disk format
pyspi.load_table("results.npz")        # round-trips exactly
```

Results are stored as an `(n_spis, M, M)` array plus the SPI and process names,
which is the data's natural shape. `.csv` is also accepted for eyeballing small
results, but it is one-way and impractical for the full SPI set.

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

`forex` is quantised: every one of its seven processes contains repeated
values (24–212 distinct values across 250 observations). Consequently, all
KSG/`kraskov` variants refuse it under pyspi's continuous, tie-free input
contract. pyspi does not provide a general discrete MI, TLMI or DI plug-in
estimator; its symbolic support is TE-specific. Use an external discrete
estimator when the missing quantities are needed.

Walkthrough tutorials in the full documentation:
[simple demonstration](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/getting-started-a-simple-demonstration) ·
[finance](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/finance-stock-price-time-series) ·
[neuroimaging](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/neuroimaging-fmri-time-series)

## Choosing an SPI set

Computing all 322 SPIs is expensive, and cost grows steeply in both the number
of processes *M* and the series length *T*. `config=` takes either a bundled
name or a path to your own YAML:

| `config=` | SPIs | Use when |
|:----------|-----:|:---------|
| `"full"` (default) | 322 | You want everything and can afford it. |
| `"fast"` | 213 | General use; drops the slowest SPIs. |
| `"benchmarked_p99"` | 318 | Near-complete, with only the worst cost outliers removed. |
| `"benchmarked_p95"` | 305 | Good coverage/cost trade-off. |
| `"benchmarked_p90"` | 290 | Recommended default for large batches. |
| `"benchmarked_p80"` | 261 | Cost-constrained sweeps. |
| `"sonnet"` | 14 | One representative SPI per module (M01-M14). |
| `"fabfour"` | 4 | Smoke tests and quick sanity checks. |

The `benchmarked_p<N>` sets keep the fastest *N*% of SPIs by **measured**
amortized compute cost, so the cut point is the *N*th percentile of the cost
distribution. They were derived from a benchmark grid spanning *M* ∈ {4..64} and
*T* ∈ {200..3200}; see [`bench/README.md`](bench/README.md) for the methodology
and [`bench/results/analysis/report.md`](bench/results/analysis/report.md) for
the measurements.

To build your own subset by keyword:

```python
from pyspi.utils import filter_spis
filter_spis(["nonlinear", "directed"], output_name="my_spis")
calc = Calculator(config="my_spis.yaml")
```

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

## SPI Descriptions 📋
To access a table with a high-level overview of the _pyspi_ library of SPIs, including their associated identifiers, see the [table of SPIs](https://time-series-features.gitbook.io/pyspi/spis/table-of-spis) in the full documentation.
For detailed descriptions of each SPI, as well as its associated estimators, we provide a full breakdown in the [SPI descriptions](https://time-series-features.gitbook.io/pyspi/spis/spi-descriptions) page of our documentation. 

## Documentation
The full documentation is hosted on [GitBooks](https://time-series-features.gitbook.io/pyspi/). 
Use the following links to quickly access some of the key sections:

- [Full installation guide](https://time-series-features.gitbook.io/pyspi/installation)
- [Troubleshooting](https://time-series-features.gitbook.io/pyspi/installation/troubleshooting)
- [Alternative installation options](https://time-series-features.gitbook.io/pyspi/installation/alternative-installation-options)
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
In contributing to _pyspi_, all participants are expected to adhere to our [code of conduct](https://time-series-features.gitbook.io/pyspi/development/code-of-conduct).

### SPI Wishlist
We strive to provide the most comprehensive toolkit of SPIs. If you have ideas for new SPIs or suggestions for improvements to existing ones, we are eager to hear from and collaborate with you! 
Any pairwise dependence measure, provided it is accompanied by a published research paper, typically falls within the scope for consideration in the 
_pyspi_ library.
You can access our SPI wishlist via the [projects tab](https://github.com/DynamicsAndNeuralSystems/pyspi/projects) in this repo to open a request.

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
	month = oct,
	year = {2023},
	pages = {883--893},
}
```

</details>

## Other highly comparative toolboxes 🧰
If you are interested in trying other highly comparative toolboxes like _pyspi_, see the below list:

- [_hctsa_](https://github.com/benfulcher/hctsa), the _highly comparative time-series analysis_ toolkit, computes over 7000 time-series features from univariate time series.
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
_pyspi_ is released under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0).

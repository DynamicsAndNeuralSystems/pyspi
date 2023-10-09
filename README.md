# Python Toolkit of Statistics for Pairwise Interactions (_pyspi_)

![](img/pyspi_logo.png)

[![DOI](https://zenodo.org/badge/601919618.svg)](https://zenodo.org/badge/latestdoi/601919618)

_pyspi_ is a comprehensive python library for computing statistics of pairwise interactions (SPIs) from multivariate time-series (MTS) data.

The code provides easy access to hundreds of methods for evaluating the relationship between pairs of time series, from simple statistics (like correlation) to advanced multi-step algorithms (like Granger causality).
The code is licensed under the [GNU GPL v3 license](http://www.gnu.org/licenses/gpl-3.0.html) (or later).

**Feel free to reach out for help with real-world applications.**
Feedback is much appreciated through [issues](https://github.com/DynamicsAndNeuralSystems/pyspi/issues), or [pull requests](https://github.com/DynamicsAndNeuralSystems/pyspi/pulls).

## Acknowledgement :+1:

If you use this software, please read and cite this article:

- &#x1F4D7; O.M. Cliff, A.G. Bryant, J.T. Lizier, N. Tsuchiya, B.D. Fulcher [Unifying pairwise interactions in complex dynamics](https://doi.org/10.1038/s43588-023-00519-x), _Nature Computational Science_ (2023).

Note that [preprint](https://arxiv.org/abs/2201.11941) and [free-to-read](https://rdcu.be/dn3JB) versions of this article are available.

## Getting Started

See the [documentation](https://pyspi-toolkit.readthedocs.io/en/latest/) for installing and setting up _pyspi_.
Once you're done, you can learn how to use the package by checking out the:

- [Simple demo](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/demos/simple_demo.py)
- [Tutorial (finance: stock price time series)](https://github.com/DynamicsAndNeuralSystems/pyspi/blob/main/demos/tutorial.ipynb)
- [Tutorial (neuroimaging: fMRI time series)](https://github.com/anniegbryant/CNS_2022/blob/main/pyspi_tutorial/CNS2022_pyspi_demo.ipynb).

If you have access to a PBS cluster and are processing MTS with many processes (or are analyzing many MTS), then you may find the [_pyspi_ distribute](https://github.com/DynamicsAndNeuralSystems/pyspi-distribute) repository helpful.

If your dataset is large (containing many processes and/or observations), you can use a pre-configured set of reduced statistics or create your own subsets (cf. the [documentation guide](https://pyspi-toolkit.readthedocs.io/en/latest/advanced.html#using-a-reduced-spi-set)).

## Other highly comparative toolboxes

### _hctsa_

[_hctsa_](https://github.com/benfulcher/hctsa), the _highly comparative time-series analysis_ toolkit, computes over 7000 time-series features from univariate time series.

### _hcga_

[_hcga_](https://github.com/barahona-research-group/hcga), a *highly comparative graph analysis* toolkit, computes several thousands of graph features directly from any given network.

## SPI Wishlist

As _pyspi_ is under active development, we are always open to suggestions for new SPIs to be added via the [projects tab](https://github.com/DynamicsAndNeuralSystems/pyspi/projects) in this repo.

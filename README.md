# Python Toolkit of Statistics for Pairwise Interactions (_pyspi_)

![](img/pyspi_logo.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6541022.svg)](https://doi.org/10.5281/zenodo.6541022)

_pyspi_ is a comprehensive python library for computing statistics of pairwise interactions (SPIs) from multivariate time-series (MTS) data.

The code provides easy access to hundreds of methods for evaluating the relationship between pairs of time series, from simple statistics (like correlation) to advanced multi-step algorithms (like Granger causality).
The code is licensed under the [GNU GPL v3 license](http://www.gnu.org/licenses/gpl-3.0.html) (or later).

**Feel free to [email me](mailto:oliver.m.cliff@gmail.com) for help with real-world applications.**
Feedback is much appreciated through email, [issues](https://github.com/olivercliff/pyspi/issues), or [pull requests](https://github.com/olivercliff/pyspi/pulls).

## Acknowledgement

If you use this code, please cite the following preprint:

Oliver M. Cliff, Joseph T. Lizier, Naotsugu Tsuchiya, Ben D. Fulcher, "Unifying Pairwise Interactions in Complex Dynamics," _arXiv_ preprint, [arXiv:2201.11941](https://arxiv.org/abs/2201.11941) (2022).

## Getting Started

See the [documentation](https://pyspi-toolkit.readthedocs.io/en/latest/) for installing and setting up _pyspi_.
Once you're done, you can learn how to use the package by checking out the:

- [Simple demo](https://github.com/olivercliff/pyspi/blob/main/demos/simple_demo.py)
- [Tutorial (finance: stock price time series)](https://github.com/olivercliff/pyspi/blob/main/demos/tutorial.ipynb)
- [Tutorial (neuroimaging: fMRI time series)](https://github.com/anniegbryant/CNS_2022/blob/main/pyspi_tutorial/CNS2022_pyspi_demo.ipynb).

If you have access to a PBS cluster and are processing MTS with many processes (or are analyzing many MTS), then you may find the [_pyspi_ distribute](https://github.com/olivercliff/pyspi-distribute) repository helpful.

If your dataset is large (containing many processes and/or observations), you can use a reduced set of statistics by instantiating the calculator with the `fast=True` parameter (cf. the [simple demo](https://github.com/olivercliff/pyspi/blob/main/demos/simple_demo.py)).

## Other highly comparative toolboxes

### _hctsa_

[_hctsa_](https://github.com/benfulcher/hctsa), the _highly comparative time-series analysis_ toolkit, computes over 7000 time-series features from univariate time series.

### _hcga_

[_hcga_](https://github.com/barahona-research-group/hcga), a *highly comparative graph analysis* toolkit, computes several thousands of graph features directly from any given network.

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
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.8 | 3.9 | 3.10 | 3.11 | 3.12"></a>
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
| [SPI Descriptions](#spi-descriptions-) | A link to the full table of SPIs and detailed descriptions   |
| [Documentation](#documentation)     | A link to our API reference and full documentation on GitBooks |
| [Contributing to _pyspi_](#contributing-to-pyspi-) | A guide for community members willing to contribute to _pyspi_ |
| [Acknowledgement](#acknowledgement-) | A citation for _pyspi_ for scholarly articles                |
| [Our Contributors](#our-contributors-) | A summary of our primary contributors                        |
<hr style="border-top: 3px solid #bbb;">

## Installation 📥
The simplest way to get the _pyspi_ package up and running is to install the package using `pip install`. 
For access to the full library of SPIs, the code requires GNU's [Octave](https://octave.org/download) to be installed on your system.

#### 1. Pre-Install Octave (Optional)
While you can safely install _pyspi_ without first installing `Octave`, you will not have access to the full library of SPIs

#### 2. Create a conda environment (Optional, Recommended)
While you can also install _pyspi_ outside of a conda environment, it depends on a lot of user packages that may make managing dependencies quite difficult. 
So, we would also recommend installing pyspi in a conda environment. Firstly, create a fresh conda environment:
```
conda create -n pyspi python=3.9.0
```
Once you have created the environment, activate it using `conda activate pyspi`.

#### 3. Install with _pip_
Using `pip` for [`pyspi`](https://pypi.org/project/pyspi/):
```
pip install pyspi
```

For a more detailed guide on how to install _pyspi_, as well as how you can use _pyspi_ without first installing Octave, 
please see the [full documentation](https://time-series-features.gitbook.io/pyspi/installation/installing-pyspi).
Additionally, we provide a comprehensive [troubleshooting guide](https://time-series-features.gitbook.io/pyspi/installation/troubleshooting) for users who encounter issues installing _pyspi_ on their system,
as well as [alternative installation options](https://time-series-features.gitbook.io/pyspi/installation/alternative-installation-options). 

## Getting Started 🚀

Once you have installed _pyspi_, you can learn how to apply the package by checking out the [walkthrough tutorials](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials) in our documentation.  Click any of the examples below to access the tutorials in our full documentation:

- [Simple demonstration](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/getting-started-a-simple-demonstration)

- [Finance: stock price time series](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/finance-stock-price-time-series)


- [Neuroimaging: fMRI time series](https://time-series-features.gitbook.io/pyspi/usage/walkthrough-tutorials/neuroimaging-fmri-time-series)

### Advanced Usage
For advanced users, we offer several additional guides in the [full documentation](https://time-series-features.gitbook.io/pyspi/usage/advanced-usage) on how you can distribute your _pyspi_ jobs across PBS clusters, as well as how you can construct your own subsets of SPIs. 

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


## Our Contributors 🌟
We are thankful for the contributions of each and everyone who has helped make this project better. 
Whether you've added a line of code, improved our documentation, or reported an issue, your contributions are greatly appreciated! 
Below are some of the leading contributors to _pyspi_:

<a href="https://github.com/DynamicsAndNeuralSystems/pyspi/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DynamicsAndNeuralSystems/pyspi" />
</a>

## License 🧾
_pyspi_ is released under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0).

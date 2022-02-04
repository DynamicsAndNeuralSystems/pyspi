List of methods
===========================

The full default set of over 250 pairwise interactions are produced by running all of the code files below.
In our default statistics set, most functions are run with multiple input parameters.

Basic statistics
----------------

In this section, we detail SPIs that are traditional techniques for performing statistical inference in bivariate systems.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Method name
     - Description
     - Identifier
   * - :code:`covariance`
     - Covariance matrix
     - :code:`cov`
   * - :code:`precision`
     - Precision matrix
     - :code:`prec`
   * - :code:`spearmanr`
     - Spearman correlation
     - :code:`spearmanr`

Distance similarity
-------------------

In this section, we detail SPIs that we have categorized as distance-based similarity measures in that they
aim to establish statistical similarity or independence based on the pairwise distance between bivariate
observations

Causal indices
--------------

In this section, we detail the 10 SPIs that are derived from causal inference models. 

Information Theory
------------------

The pairwise measures that we employ from information theory are either intended to operate on serially independent observations (e.g., joint entropy and mutual information) or bivariate time series (e.g.,
transfer entropy and stochastic interaction).

Spectral measures
-----------------

Spectral SPIs are computed in the frequency or time-frequency domain, using either Fourier or wavelet
transformations to derive spectral matrices.

Miscellaneous
-------------

A small number of methods do not fit squarely into any category listed above, and so we place them
in a 'miscellaneous' category
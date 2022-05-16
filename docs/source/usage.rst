Usage
=====

Pre-installation
----------------

The code requires GNU's `Octave <https://www.gnu.org/software/octave/index>`_ by default, which is freely available on all popular operating systems.
See the `installation instructions <https://wiki.octave.org/Category:Installation>`_ to find out how to install Octave on your system.

.. note::
   You can safely install `PySPI` without first installing Octave but you will not have access to the `Integrated Information Theory` statistics, see :ref:`Using the toolkit without Octave`.

While you can also install `PySPI` outside of a `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ environment, it depends on a lot of user packages that may make managing dependencies quite difficult.
So, we would also recommend installing `PySPI` in a conda environment.
After `installing conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_, create a new environment for using the toolkit:

.. code-block:: console

   $ conda create -n pyspi python=3.9.0
   $ conda activate pyspi


Installation
------------

Next, download or clone the `latest version <https://github.com/olivercliff/pyspi>`_ from GitHub, unpack and install:

.. code-block:: console

   $ git clone https://github.com/olivercliff/pyspi.git 
   $ cd pyspi
   $ pip install .


Getting Started
---------------

In order to demonstrate the functionality of this software, we will first need a sample multivariate time series (MTS).
We will use data generated from a multivariate Gaussian:

.. code-block::

   import numpy as np
   import random

   random.seed(42)

   M = 5 # 5 processes
   T = 500 # 500 observations

   dataset = np.random.randn(M,T)

Now, given our dataset, we can instantiate the :class:`~pyspi.calculator.Calculator` object:

.. code-block::

   from pyspi.calculator import Calculator

   calc = Calculator(dataset=dataset)

And, using only the :meth:`~pyspi.calculator.Calculator.compute` method, we can compute over 250 statistics for analysing pairwise interactions in the MTS.

.. code-block::

   calc.compute()

.. note::
   While we tried to make the calculator as efficient as possible, computing all statistics can take a while (depending on the size of your dataset).
   You can use a faster set of statistics by instantiating the calculator with :code:`fast=True`, see :class:`~pyspi.calculator.Calculator`.
   Alternatively, design your own subset of the statistics by following the instructions in :ref:`Using a reduced SPI set`.

Once the calculator has computed each of the statistics, you can access all values using the :attr:`~pyspi.calculator.Calculator.table` property:

.. code-block::

   print(calc.table)

Or, extract one matrix of pairwise interactions (MPI) for a given method using their unique `identifier`.
For instance, the following code will extract the covariance matrix computed with the maximum likelihood estimator:

.. code-block::

   print(calc.table['cov_EmpiricalCovariance'])

The identifiers for many of the statistics are outlined in the Supplementary Material of our `preprint <https://arxiv.org/abs/2201.11941>`_, and an up-to-date list of included statistics will be provided in this documentation shortly.
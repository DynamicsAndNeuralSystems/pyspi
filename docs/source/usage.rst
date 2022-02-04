Usage
=====

Pre-installation
----------------

The code requires GNU's `Octave <https://www.gnu.org/software/octave/index>`_ by default, which is freely available on all popular operating systems.
See the `installation instructions <https://wiki.octave.org/Category:Installation>`_ to find out how to install Octave on your system.

.. note::
   You can safely install `PySPI` without first installing Octave but you will not have access to the `Integrated Information Theory` statistics, see :ref:`Using the toolkit without Octave`.

While you can install `PySPI` outside of a `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ environment, it depends on a lot of user packages that may make managing dependencies quite difficult.
So, we would also recommend installing `PySPI` in a conda environment.
After following the `installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for conda, create a new environment for using the toolkit:

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

   dataset = np.random.rand(M,T)

Now, given our dataset, the main functionality of this software can then be accessed using two lines:

.. code-block::

   from pyspi.calculator import Calculator

   calc = Calculator(dataset=dataset)
   calc.compute()



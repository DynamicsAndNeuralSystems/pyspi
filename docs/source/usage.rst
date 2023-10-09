Usage
=====


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
   You can use a faster set of statistics by instantiating the calculator with :code:`subset=fast`, see :class:`~pyspi.calculator.Calculator`.
   We also provide a reduced set of statistics that are useful for many applications, with the option for users to design their own subset of statistics; see :ref:`Using a reduced SPI set`.

Once the calculator has computed each of the statistics, you can access all values using the :attr:`~pyspi.calculator.Calculator.table` property:

.. code-block::

   print(calc.table)

Or, extract one matrix of pairwise interactions (MPI) for a given method using their unique `identifier`.
For instance, the following code will extract the covariance matrix computed with the maximum likelihood estimator:

.. code-block::

   print(calc.table['cov_EmpiricalCovariance'])

The identifiers for many of the statistics are outlined in the Supplementary Material of our `paper <https://doi.org/10.1038/s43588-023-00519-x>`_.

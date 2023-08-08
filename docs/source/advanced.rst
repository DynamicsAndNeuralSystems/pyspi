Advanced
========

The Data object
---------------------

The MTS data is contained within the :class:`~pyspi.data.Data` object, along with preprocessed properties of the MTS that allows us to efficiently compute the methods.
If you want more control over how the MTS are treated upon input, you can directly instantiate a :class:`~pyspi.data.Data` object for inputting to the calculator:

.. code-block::

    from pyspi.data import Data
    from pyspi.calculator import Calculator
    import numpy as np

    M = 10 # Number of processes
    T = 1000 # Number of observations

    z = np.random.rand(M,T)

    # The dim_order argument specifies which dimension is a process (p) and an observation (s).
    # The normalise argument specifies if we should z-score the data.
    dataset = Data(data=z,dim_order='ps',normalise=False)

    calc = Calculator(dataset=dataset)


Using a reduced SPI set
-----------------------

The :class:`~pyspi.calculator.Calculator` object computes 283 SPIs by default, but you may only be interested in a subset of these.
We provide three pre-configured reduced SPI subsets that you may wish to try:

* :code:`fast`: Most SPIs (excluding those that are computationally expensive)
* :code:`sonnet` - 14 SPIs, one from each of the data-driven clusters as described in `Cliff et al. (2023) <https://arxiv.org/abs/2201.11941>`_
* :code:`fabfour` - 4 SPIs that are simple and computationally efficient: Pearson correlation, Spearman correlation, directed information with a Gaussian density estimator, and the power envelope correlation.

You may specify any one of these subsets when you instantiate the :code:`Calculator` object:

.. code-block::

    from pyspi.calculator import Calculator
    calc = Calculator(dataset=dataset, subset='sonnet')

Alternatively, if you would like to use your own bespoke set of SPIs, you can copy a version of the :code:`config.yaml` file to your workspace and edit it to remove any SPIs that you would not like the :code:`Calculator` to compute.
First, copy the :code:`config.yaml` file to your workspace:

.. code-block:: console

    $ cp </path/to/pyspi>/pyspi/config.yaml myconfig.yaml

Once you've got a local version, edit the :code:`myconfig.yaml` file to remove any SPIs you're not interested in.
A minimal configuration file might look like the following if you're only interested in computing a covariance matrix using the maximum likelihood estimator:

.. code-block::

    # Basic statistics
    .statistics.basic:
        # Covariance
        covariance:
            # Maximum likehood estimator
            - estimator: EmpiricalCovariance

Now, when you instantiate the calculator, instead of using the default :code:`config.yaml`, you can input your bespoke configuration file:

.. code-block::

    from pyspi.calculator import Calculator

    calc = Caculator(dataset=dataset,configfile='myconfig.yaml')

Then use the calculator as normal (see :ref:`Usage`).

.. note::
    We have provided a detailed list of many of the statistics included in this toolkit (and the configuration file) in the Supplementary Material of our `preprint <https://arxiv.org/abs/2201.11941>`_, and will include an up-to-date list of statistics in this documentation shortly.
    However, if you have any questions about a particular implementation, do not hesitate to raise an issue in the `github repo <https://github.com/DynamicsAndNeuralSystems/pyspi>`_ for any assistance.

Using the toolkit without Octave
--------------------------------

If you do not wish to first install Octave before using the toolkit, remove the yaml entries for :code:`integrated_information` in the :code:`config.yaml` file (see :ref:`Using a reduced SPI set`).
Advanced
========

The Data object
---------------------

The MTS data is contained within the :code:`Data` object, along with preprocessed properties of the MTS that allows us to efficiently compute the methods.
If you want more control over how the MTS are treated upon input, you can directly instantiate a :code:`Data` object for inputting to the calculator:

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

You can easily use a subset of the SPIs by copying a version of the :code:`config.yaml` file to a local directory and removing those you don't want the calculator to compute.
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
    However, if you have any questions about a particular implementation, do not hesitate to `contact me <mailto:oliver.cliff@sydney.edu.au>`_ for any assistance.

Using the toolkit without Octave
--------------------------------

If you do not wish to first install Octave before using the toolkit, remove the yaml entries for :code:`integrated_information` in the :code:`config.yaml` file (see :ref:`Using a reduced SPI set`).
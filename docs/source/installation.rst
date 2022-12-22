Installation
===================================

Pre-installation
----------------

The code requires GNU's `Octave <https://www.gnu.org/software/octave/index>`_ by default, which is freely available on all popular operating systems.
See the `installation instructions <https://wiki.octave.org/Category:Installation>`_ to find out how to install Octave on your system.

.. note::
   You can safely install `pyspi` without first installing Octave but you will not have access to the `Integrated Information Theory` statistics, see :ref:`Using the toolkit without Octave`.

While you can also install `pyspi` outside of a `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ environment, it depends on a lot of user packages that may make managing dependencies quite difficult.
So, we would also recommend installing `pyspi` in a conda environment.
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

You can confirm that `pyspi` and dependencies installed properly with the following command:

.. code-block:: console

   $ setup.py test
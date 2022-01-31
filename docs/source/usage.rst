Usage
=====

Pre-installation
------------

The code requires GNU's `Octave <https://www.gnu.org/software/octave/index>`_ by default. Install octave using your favourite package manager, e.g.,

.. code-block:: console

   $ apt-get install octave

for Ubuntu;

.. code-block:: console

   $ pacman -S octave

for Arch; and

.. code-block:: console

   $ brew install octave

for MacOS.


Installation
------------

Download or clone the `latest version <https://github.com/olivercliff/pyspi>`_ from GitHub, unpack and run (from the folder containing the ``setup.py`` file):

.. code-block:: console

   $ pip install .

or 


.. code-block:: console

   $ pip install -e .

for editable mode.

We recommend installation in a conda environment, see :doc:`Conda Installation`.

Getting Started
------------

Check out the demo script in ``demos/demo.py``

Conda Installation
------------

.. code-block:: console

   $ git clone https://github.com/olivercliff/pyspi.git 
   $ conda create -n pyspi python=3.9.0
   $ conda activate pyspi
   $ cd pyspi
   $ pip install .
   $ python demos/demo.py

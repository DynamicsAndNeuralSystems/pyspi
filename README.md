# Python Library of Statistics for Pairwise Interactions (pyspi)

*pyspi* provides a comprehensive library for computing pairwise interactions from multivariate time-series data.

The code provides easy access to evaluating hundreds of methods for evaluating the relationship between pairs of time series, from simple statistics (like correlation and coherence) to advanced multi-step algorithms (like convergent cross mapping and transfer entropy).

# Pre-installation

The code requires GNU's [Octave](https://www.gnu.org/software/octave/index) by default. Install octave using your favourite package manager, e.g.,
```
apt-get install octave
```
for Ubuntu;
```
pacman -S octave
```
for Arch; and
```
brew install octave
```

for MacOS.

# Installation

Download or clone the [latest version](https://github.com/olivercliff/pyspi) from GitHub, unpack and run (from the folder containing `pyspi` setup.py file):

```
pip install .
```

or 

```
pip install -e .
```

for editable mode.

We recommend the [installation in a conda environment](#conda-install).

## Getting started

Check out the demo scripts in `demos/demo.py` and `demos/demo.ipynb`

# <a name="conda-install"></a>Conda installation

```
git clone https://github.com/olivercliff/pyspi.git 
conda create -n pyspi python=3.9.0
conda activate pyspi
cd pyspi
pip install .
python demos/demo.py
```

from pyspi.calculator import Calculator
import pytest
import dill
import pyspi
import numpy as np
import yaml, tempfile
from copy import deepcopy
import logging

################### Helper Functions #################

def test_mpi(calc_name, est, mpi_benchmark, mpi_new):
    if not mpi_benchmark.equals(mpi_new):
        diff = abs(mpi_benchmark - mpi_new)
        total_differences = diff.values.size - np.count_nonzero(diff.isna().values)
        min_diff = diff.min().min()
        max_diff = diff.max().max()
        #logging.warning(f"SPI: {est} gave a different output on the {calc_name} dataset!")
        assert False, f"SPI: {est} gave a different output on the {calc_name} dataset!" \
                     f"Min difference: {min_diff}, Max difference: {max_diff}"


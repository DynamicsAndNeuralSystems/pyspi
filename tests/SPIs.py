from pyspi.calculator import Calculator
import pytest
import dill
import pyspi
import numpy as np
import yaml, tempfile
from copy import deepcopy

################### Helper Functions #################

def test_mpi(calc_name, est, mpi_benchmark, mpi_new):
    if not mpi_benchmark.equals(mpi_new):
        diff = abs(mpi_benchmark - mpi_new)
        total_differences = diff.values.size - np.count_nonzero(diff.isna().values)
        min_diff = diff.min().min()
        max_diff = diff.max().max()
        assert False, f"SPI: {est} gave a different output on the {calc_name} dataset!\n" \
                      f"Total differences: {total_differences}, " \
                      f"Min difference: {min_diff}, Max difference: {max_diff}"



# def test_basic_spis(load_benchmark_calcs, compute_new_tables):
#     benchmark_calcs = load_benchmark_calcs
#     new_calcs = compute_new_tables
#     # test each of the basic SPIs on each of the datasets
#     basic_spis = ['Covariance', 'Precision', 'SpearmanR', 
#                 'KendallTau', 'CrossCorrelation']
    
#     for calc_name, benchmark_calc in benchmark_calcs.items():

#         # get all SPI names for calculator
#         spi_dict = benchmark_calc.spis

#         # loop through each calculator and test the basic SPIs
#         print(f'Testing {calc_name}...')
#         # loop through each SPI
#         for spi_name in basic_spis:

#             print(f'Checking {spi_name}...')
#             spi_to_check = getattr(pyspi.statistics.basic, spi_name)
#             spi_estimators = [key for key, value in spi_dict.items() if isinstance(value, spi_to_check)]

#             # now for each spi estimator compare the new table to the reference (benchmark) table
#             for est in spi_estimators:

#                 mpi_benchmark = benchmark_calc.table[est]
#                 mpi_new = new_calcs[calc_name].table[est]

#                 # now compare the two tables
#                 if not mpi_benchmark.equals(mpi_new):
#                     # compute the abs diff
#                     diff = abs(mpi_benchmark - mpi_new)
                    
#                     # get total number of differences
#                     total_differences = diff.values.size - np.count_nonzero(diff.isna().values)
#                     min_diff = diff.min().min()
#                     max_diff = diff.max().max()

#                     assert False, f"SPI: {est} gave a different output on the {calc_name} dataset!\n " \
#                                   f"Total differences: {total_differences}, " \
#                                   f"Min difference: {min_diff}, Max difference: {max_diff}"     

        
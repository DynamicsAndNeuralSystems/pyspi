from pyspi.calculator import Calculator
import pytest
import dill
import pyspi
import numpy as np
from copy import deepcopy


############# Fixtures and helper functions #########

def load_benchmark_calcs():
    benchmark_calcs = dict()
    calcs = ['calc_standard_normal.pkl'] # follow this naming convention -> calc_{name}.pkl
    for calc in calcs:
            # extract the calculator name from the filename
            calc_name = calc[len("calc_"):-len(".pkl")]

            # Load the calculator
            with open(f"{calc}", "rb") as f:
                loaded_calc = dill.load(f)
                benchmark_calcs[calc_name] = loaded_calc
    
    return benchmark_calcs

def load_benchmark_datasets():
    benchmark_datasets = dict()
    dataset_names = ['standard_normal.npy']
    for dname in dataset_names:
        dataset = np.load(f"../pyspi/data/{dname}")
        dataset = dataset.T 
        benchmark_datasets[dname.strip('.npy')] = dataset

    return benchmark_datasets

def compute_new_tables():
    """Compute new tables using the same benchmark dataset(s)."""
    benchmark_datasets = load_benchmark_datasets()
    # Compute new tables on the benchmark datasets
    new_calcs = dict()

    calc_base = Calculator() # create base calculator object

    for dataset in benchmark_datasets.keys():
        calc = deepcopy(calc_base) # make a copy of the base calculator
        calc.load_dataset(dataset=benchmark_datasets[dataset])
        calc.compute()
        new_calcs[dataset] = calc

    return new_calcs

def generate_SPI_test_params():
    """Generate combinations of calculator, dataset and SPI for the fixture."""
    benchmark_calcs = load_benchmark_calcs()
    new_calcs = compute_new_tables()
    params = []
    for calc_name, benchmark_calc in benchmark_calcs.items():
        spi_dict = benchmark_calc.spis
        for spi_est in spi_dict.keys():
            params.append((calc_name, spi_est, benchmark_calc.table[spi_est], new_calcs[calc_name].table[spi_est]))
    
    return params

params = generate_SPI_test_params()
def pytest_generate_tests(metafunc):
    """Create a hook to generate parameter combinations for parameterised test"""
    if "calc_name" in metafunc.fixturenames:
        metafunc.parametrize("calc_name,est,mpi_benchmark,mpi_new", params)

def test_mpi(calc_name, est, mpi_benchmark, mpi_new):
    """Run the test"""
    # check for mismatched NaNs first
    mismatched_nans = (mpi_benchmark.isna() != mpi_new.isna())
    assert not mismatched_nans.any().any(), f"SPI: {est} | Dataset: {calc_name}. Mismatched NaNs."

    epsilon = np.finfo(float).eps
    if not mpi_benchmark.equals(mpi_new):
        diff = abs(mpi_benchmark - mpi_new)
        max_diff = diff.max().max()
        if max_diff > epsilon:
            assert False, f"SPI: {est}. Dataset: {calc_name}. Max difference: {max_diff}"
    


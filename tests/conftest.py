import pytest 
import pyspi
import dill
from pyspi.calculator import Calculator
import numpy as np
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

def load_benchmark_calcs():
    benchmark_calcs = dict()
    calcs = ['calc_standard_normal.pkl'] # follow this naming convention -> calc_{name}.pkl
    for calc in calcs:
            # extract the calculator name from the filename
            calc_name = calc[len("calc_"):-len(".pkl")]

            # Load the calculator
            with open(f"../tests/benchmark_calculators/{calc}", "rb") as f:
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

# def compute_new_tables():
#     benchmark_datasets = load_benchmark_datasets()
#     # Compute new tables on the benchmark datasets
#     #new_calcs = dict()
#     dummy_calc = load_benchmark_calcs()
    
#     # calc_base = Calculator() # create base calculator object
#     # for dataset in benchmark_datasets.keys():
#     #     calc = deepcopy(calc_base) # make a copy of the base calculator
#     #     calc.load_dataset(dataset=benchmark_datasets[dataset])
#     #     calc.compute()
#     #     new_calcs[dataset] = calc

#     return dummy_calc

def compute_new_tables():
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

# def generate_SPI_test_params():
#     benchmark_calcs = load_benchmark_calcs()
#     new_calcs = compute_new_tables()
#     basic_spis = ['Covariance', 'Precision', 'SpearmanR', 'KendallTau', 'CrossCorrelation']
#     params = []
#     for calc_name, benchmark_calc in benchmark_calcs.items():
#         spi_dict = benchmark_calc.spis
#         for spi_name in basic_spis:
#             spi_to_check = getattr(pyspi.statistics.basic, spi_name)
#             spi_estimators = [key for key, value in spi_dict.items() if isinstance(value, spi_to_check)]
#             for est in spi_estimators:
#                 params.append((calc_name, est, benchmark_calc.table[est], new_calcs[calc_name].table[est]))
#     return params

def generate_SPI_test_params():
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
    if "calc_name" in metafunc.fixturenames:
        metafunc.parametrize("calc_name,est,mpi_benchmark,mpi_new", params)



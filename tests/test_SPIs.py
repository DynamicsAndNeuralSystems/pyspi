from pyspi.calculator import Calculator
import pytest
import dill
import pyspi
import numpy as np

############# Fixtures and helper functions #########

def load_benchmark_tables():
    """Function to load the mean and standard deviation tables for each MPI."""
    table_fname = 'CML7_benchmark_tables.pkl'
    with open(f"tests/{table_fname}", "rb") as f:
        loaded_tables = dill.load(f)
    
    return loaded_tables

def load_benchmark_dataset():
    dataset_fname = 'cml7.npy'
    dataset = np.load(f"pyspi/data/{dataset_fname}").T
    return dataset

def compute_new_tables():
    """Compute new tables using the same benchmark dataset(s)."""
    benchmark_dataset = load_benchmark_dataset()
    # Compute new tables on the benchmark dataset
    np.random.seed(42)
    calc = Calculator(dataset=benchmark_dataset, normalise=True, detrend=True)
    calc.compute()
    table_dict = dict()
    for spi in calc.spis:
        table_dict[spi] = calc.table[spi]

    return table_dict

def generate_SPI_test_params():
    """Function to generate combinations of benchmark table, 
    new table for each MPI"""
    benchmark_tables = load_benchmark_tables()
    new_tables = compute_new_tables()
    params = []
    calc = Calculator()
    spis = list(calc.spis.keys())
    spi_ob = list(calc.spis.values())
    for spi_est, spi_ob in zip(spis, spi_ob):
        params.append((spi_est, spi_ob, benchmark_tables[spi_est], new_tables[spi_est].to_numpy()))
    
    return params

params = generate_SPI_test_params()
def pytest_generate_tests(metafunc):
    """Create a hook to generate parameter combinations for parameterised test"""
    if "est" in metafunc.fixturenames:
        metafunc.parametrize("est, est_ob, mpi_benchmark,mpi_new", params)
        

def test_mpi(est, est_ob, mpi_benchmark, mpi_new, spi_warning_logger):
    """Run the benchmarking tests."""
    zscore_threshold = 1 # 2 sigma
    
    # separate the the mean and std. dev tables for the benchmark
    mean_table = mpi_benchmark['mean']
    std_table = mpi_benchmark['std']

    # check std table for zeros and impute with smallest non-zero value
    std_table = np.where(std_table == 0, 1e-10, std_table)
    
    # check that the shapes are equal
    assert mean_table.shape == mpi_new.shape, f"SPI: {est}| Different table shapes. "

    # convert NaNs to zeros before proeceeding - this will take care of diagonal and any null outputs
    mpi_new = np.nan_to_num(mpi_new)
    mpi_mean = np.nan_to_num(mean_table)

    # check if matrix is symmetric (undirected SPI) for num exceed correction
    isSymmetric = "undirected" in est_ob.labels 

    # get the module name for easy reference
    module_name = est_ob.__module__.split(".")[-1]

    if not np.allclose(mpi_new, mpi_mean):
        # tables are not equivalent, quantify the difference by z-scoring.
        diff = abs(mpi_new - mpi_mean)
        zscores = diff/std_table

        idxs_greater_than_thresh = np.argwhere(zscores > zscore_threshold)

        if len(idxs_greater_than_thresh) > 0:
            sigs = zscores[idxs_greater_than_thresh[:, 0], idxs_greater_than_thresh[:, 1]]
            # get the max
            max_z = max(sigs)

            # number of interactions
            num_interactions = mpi_new.size - mpi_new.shape[0]
            # count exceedances
            num_exceed = len(sigs)

            if isSymmetric:
                # number of unique exceedences is half
                num_exceed //= 2
                num_interactions //= 2

            spi_warning_logger(est, module_name, max_z, int(num_exceed), int(num_interactions))

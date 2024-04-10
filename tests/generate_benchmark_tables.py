import numpy as np
import dill
from pyspi.calculator import Calculator

""""Script to generate benchmarking dataset"""
def get_benchmark_tables(calc_list):
    # get the spis from the first calculator
    spis = list(calc_list[1].spis.keys())
    num_procs = calc_list[1].dataset.n_processes
    # create a dict to store the mean and std for each spi
    benchmarks = {key : {'mean': None, 'std': None} for key in spis}
    num_trials = len(calc_list)
    for spi in spis:
        mpi_tensor = np.zeros(shape=(num_trials, num_procs, num_procs))
        for (index, calc) in enumerate(calc_list):
            mpi_tensor[index, :, :] = calc.table[spi].to_numpy()
        mean_matrix = np.mean(mpi_tensor, axis=0) # compute element-wise mean across the first dimension
        std_matrix = np.std(mpi_tensor, axis=0) # compute element-wise std across the first dimension
        benchmarks[spi]['mean'] = mean_matrix
        benchmarks[spi]['std'] = std_matrix
    
    return benchmarks

# load and transpose dataset
dataset = np.load("pyspi/data/cml7.npy").T

# create list to store the calculator objects
store_calcs = list()

for i in range(75):
    np.random.seed(42)
    calc = Calculator(dataset=dataset)
    calc.compute()
    store_calcs.append(calc)

mpi_benchmarks = get_benchmark_tables(store_calcs)

# save data 
with open("tests/CML7_benchmark_tables_new.pkl", "wb") as f:
    dill.dump(mpi_benchmarks, f)

from pyspi.calculator import Calculator
import pytest
import dill
import numpy as numpy
import yaml, tempfile

################### Helper Functions #################
@pytest.fixture
def load_benchmark_calcs():
    benchmark_calcs = list()
    calcs = ['benchmark_calc_forex.pkl', 'benchmark_calc_cml.pkl', 'benchmark_calc_standardnormal.pkl']
    for calc in calcs:
            with open(f"tests/{calc}", "rb") as f:
                loaded_calc = dill.load(f)
                benchmark_calcs.append(loaded_calc)
    return benchmark_calcs

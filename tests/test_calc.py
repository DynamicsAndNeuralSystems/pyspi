from pyspi.calculator import Calculator
import numpy as np 
import os
import pytest

def test_calculator_works():
    calc = Calculator()
    assert isinstance(calc, Calculator), "Calculator failed to instantiate"

@pytest.mark.parametrize("yaml_filename", [
    'fabfour_config', 
    'config', 
    'fast_config', 
    'octaveless_config', 
    'sonnet_config'])
def test_whether_config_files_exist(yaml_filename):
    """Check whether the config, fabfour, fast, octaveless, sonnet_config files exist"""
    expected_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyspi', f'{yaml_filename}.yaml'))
    assert os.path.isfile(expected_file), f"{yaml_filename}.yaml file was not found."

# def test_whether_calculator_instantiates_with_defaults():
#     """Attempt to create an instance of the calculator object with no arguments"""

#     calc = Calculator()

#     assert isinstance(calc, Calculator), "Calculator instance was not created properly"

#     # asssert the initial state of key attributes, given their assumed defaults
#     assert calc.name is None, "Default calculator name should be None"
#     assert calc.labels is None, "Default calculator labels should be None"
#     assert calc.n_spis == 283, "Default calculator instance does not contain 283 SPIs"

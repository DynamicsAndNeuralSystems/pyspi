from pyspi.calculator import Calculator
import numpy as np 
import os
import yaml
import pytest

def test_whether_calculator_loads():
    calc = Calculator()
    assert isinstance(calc, Calculator), "Calculator failed to instantiate"

def test_default_calculator_instantiates_with_correct_num_spis():
    calc = Calculator()
    n_spis_actual = calc.n_spis
    # get expected number of spis based on yaml
    with open('pyspi/config.yaml', 'rb') as y:
        yaml_file = yaml.full_load(y)
    count = 0
    for module in yaml_file.keys():
        for base_spi in yaml_file[module].keys():
            if yaml_file[module][base_spi] == None:
                count += 1
            else:
                count += len(yaml_file[module][base_spi])
    assert count == n_spis_actual, f"Number of SPIs loaded from the calculator ({n_spis_actual}) does not matched expected amount {count}"

@pytest.mark.parametrize("subset", [
    'fabfour',
    'fast',
    'sonnet',
    'octaveless'
])
def test_whether_calculator_instantiates_with_subsets(subset):
    calc = Calculator(subset=subset)
    assert isinstance(calc, Calculator), "Calculator failed to instantiate"

def test_whether_invalid_subset_throws_error():
    with pytest.raises(ValueError) as excinfo:
        calc = Calculator(subset='nviutw')
    assert "Subset 'nviutw' does not exist" in str(excinfo.value), "Subset not found error not displaying."

def test_whether_calculator_compute_fails_with_no_dataset():
    calc = Calculator()
    with pytest.raises(AttributeError) as excinfo:
        calc.compute()
    assert "Dataset not loaded yet" in str(excinfo.value), "Dataset not loaded yet error not displaying."

def test_calculator_name():
    calc = Calculator(name="test name")
    assert calc.name == "test name", "Calculator name property did not return the expected string 'test name'"

def test_calculator_labels():
    test_labels = ['label1', 'label2']
    calc = Calculator(labels = test_labels)
    assert calc.labels == ['label1', 'label2'], f"Calculator labels property did not return the expected list: {test_labels} "

def test_pass_single_integer_as_dataset():
    with pytest.raises(TypeError) as excinfo:
        calc = Calculator(dataset=42)
    assert "Unknown data type" in str(excinfo.value), "Incorrect data type error not displaying for integer dataset."

def test_pass_incorrect_shape_dataset_into_calculator():
    dataset_with_wrong_dim = np.random.randn(3, 5, 10)
    with pytest.raises(RuntimeError) as excinfo:
        calc = Calculator(dataset=dataset_with_wrong_dim)
    assert "Data array dimension (3)" in str(excinfo.value), "Incorrect dimension error message not displaying for incorrect shape dataset."


@pytest.mark.parametrize("nan_loc, expected_output", [
    ([1], "[1]"),
    ([1, 2], "[1 2]"),
    ([0, 2, 3], "[0 2 3]")
    ])
def test_pass_dataset_with_nan_into_calculator(nan_loc, expected_output):
    """Check whether ValueError is raised when a dataset containing a NaN is passed into the calculator object"""
    base_dataset = np.random.randn(5, 100)
    for loc in nan_loc:
        base_dataset[loc, 0] = np.NaN
    with pytest.raises(ValueError) as excinfo:
        calc = Calculator(dataset=base_dataset)
    assert f"non-numerics (NaNs) in processes: {expected_output}" in str(excinfo), "NaNs not detected in dataset when loading into Calculator!"

def test_pass_dataset_with_inf_into_calculator():
    """Check whether ValueError is raised when a dataset containing an inf/-inf value is passed into the calculator object"""
    base_dataset = np.random.randn(5, 100)
    base_dataset[0, 1] = np.inf
    base_dataset[2, 2] = -np.inf
    with pytest.raises(ValueError) as excinfo:
        calc = Calculator(dataset=base_dataset)
    assert f"non-numerics (NaNs) in processes: [0 2]" in str(excinfo), "NaNs not detected in dataset when loading into Calculator!"

@pytest.mark.parametrize("shape, n_procs_expected, n_obs_expected", [
    ((2, 23), 2, 23),
    ((5, 4), 5, 4),
    ((100, 32), 100, 32)
])
def test_data_object_process_and_observations(shape, n_procs_expected, n_obs_expected):
    """Test whether the number of processes and observations for a given dataset is correct"""
    dat = np.random.randn(shape[0], shape[1])
    calc = Calculator(dataset=dat)
    assert calc.dataset.n_observations == n_obs_expected, f"Number of observations returned by Calculator ({calc.dataset.n_observations}) does not match exepected: {n_obs_expected}"
    assert calc.dataset.n_processes == n_procs_expected, f"Number of processes returned by Calculator ({calc.dataset.n_processes}) does not match exepected: {n_procs_expected}"

@pytest.mark.parametrize("yaml_filename", [
    'fabfour_config', 
    'fast_config', 
    'octaveless_config', 
    'sonnet_config'])
def test_whether_config_files_exist(yaml_filename):
    """Check whether the config, fabfour, fast, octaveless, sonnet_config files exist"""
    expected_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyspi', f'{yaml_filename}.yaml'))
    assert os.path.isfile(expected_file), f"{yaml_filename}.yaml file was not found."

@pytest.mark.parametrize("subset, procs, obs", [
    ("all", 2, 100),
    ("all", 5, 100),
    ("fabfour", 8, 100),
    ("fast", 10, 100),
    ("sonnet", 3, 100)
])
def test_whether_table_shape_correct_before_compute(subset, procs, obs):
    dat = np.random.randn(procs, obs)
    calc = Calculator(dataset=dat, subset=subset)
    num_spis = calc.n_spis
    expected_table_shape = (procs, num_spis*procs)
    assert calc.table.shape == expected_table_shape, f"Calculator table ({subset}) shape: ({calc.table.shape}) does not match expected shape: {expected_table_shape}"


from pyspi.utils import filter_spis
import pytest
import yaml
from unittest.mock import mock_open, patch

@pytest.fixture
def mock_yaml_content():
    return {
        "module1": {
            "spi1": {"labels": ["keyword1", "keyword2"], "configs": [1, 2]},
            "spi2": {"labels": ["keyword1"], "configs": [3]},
        },
        "module2": {
            "spi3": {"labels": ["keyword3"], "configs": [1, 2, 3]},
        },
    }

def test_filter_spis_invalid_keywords():
    """Pass in a dataype other than a list for the keywords"""
    with pytest.raises(ValueError) as excinfo:
        filter_spis(keywords="linear", configfile="pyspi/config.yaml")
    assert "Keywords must be provided as a list of strings" in str(excinfo.value)
    # check for passing in an empty list
    with pytest.raises(ValueError) as excinfo:
        filter_spis(keywords=[], configfile="pyspi/config.yaml")
    assert "At least one keyword must be provided" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        filter_spis(keywords=[4], configfile="pyspi/config.yaml")
    assert "All keywords must be strings" in str(excinfo.value)  

def test_filter_spis_with_invalid_config():
    """Pass in an invalid/missing config file"""
    with pytest.raises(FileNotFoundError):
        filter_spis(keywords=["test"], configfile="invalid_config.yaml")

def test_filter_spis_no_matches(mock_yaml_content):
    """Pass in keywords that return no spis and check for ValuError"""
    m = mock_open()
    m().read.return_value = yaml.dump(mock_yaml_content)
    keywords = ["random_keyword"]

    with patch("builtins.open", m), \
        patch("os.path.isfile", return_value=True), \
        patch("yaml.load", return_value=mock_yaml_content):
        with pytest.raises(ValueError) as excinfo:
            filter_spis(keywords=keywords, output_name="mock_filtered_config", configfile="./mock_config.yaml")

    assert "0 SPIs were found" in str(excinfo.value), "Incorrect error message returned when no keywords match found."

def test_filter_spis_normal_operation(mock_yaml_content):
    """Test whether the filter spis function works as expected"""
    m = mock_open()
    m().read_return_value = yaml.dump(mock_yaml_content)
    keywords = ["keyword1", "keyword2"] # filter keys
    expected_output_yaml = {
        "module1": {
            "spi1": {"labels": ["keyword1", "keyword2"], "configs": [1,2]}
        }
    }

    with patch("builtins.open", m), patch("os.path.isfile", return_value=True), \
         patch("yaml.load", return_value=mock_yaml_content), \
         patch("yaml.dump") as mock_dump:
        
        filter_spis(keywords=keywords, output_name="mock_filtered_config", configfile="./mock_config.yaml")

        mock_dump.assert_called_once()
        args, _ = mock_dump.call_args # get call args for dump and intercept
        actual_output = args[0]  # the first argument to yaml.dump should be the yaml

        assert actual_output == expected_output_yaml, "Expected filtered YAML does not match actual filtered YAML."

def test_filter_spis_io_error_on_read():
    # check to see whether io error is raised when trying to access the configfile
    with patch("builtins.open", mock_open(read_data="data")) as mocked_file:
        mocked_file.side_effect = IOError("error")
        with pytest.raises(IOError):
            filter_spis(["keyword"], "output", "config.yaml")
            
def test_filter_spis_saves_with_random_name_if_no_name_provided(mock_yaml_content):
    # mock os.urandom to return a predictable name
    random_bytes = bytes([1, 2, 3, 4])
    expected_random_part = "01020304"

    with patch("builtins.open", mock_open()) as mocked_file, patch("os.path.isfile", return_value=True), \
         patch("yaml.load", return_value=mock_yaml_content), patch("os.urandom", return_value=random_bytes):
        
        # run the filter function without providing an output name
        filter_spis(["keyword1"])

        # construct the expected output name
        expected_file_name_pattern = f"config_{expected_random_part}.yaml"

        # check the mocked open function to see if file with expected name is opened (for writing)
        call_args_list = mocked_file.call_args_list
        found_expected_call = any(
            expected_file_name_pattern in call_args.args[0] and
            ('w' in call_args.args[1] if len(call_args.args) > 1 else 'w' in call_args.kwargs.get('mode', ''))
            for call_args in call_args_list
        )

        assert found_expected_call, f"no file with the expected name {expected_file_name_pattern} was saved."

def test_loads_default_config_if_no_config_specified(mock_yaml_content):
    script_dir = "/fake/script/directory"
    default_config_path = f"{script_dir}/config.yaml"

    with patch("builtins.open", mock_open()) as mocked_open, \
         patch("os.path.isfile", return_value=True), \
         patch("yaml.load", return_value=mock_yaml_content), \
         patch("os.path.dirname", return_value=script_dir), \
         patch("os.path.abspath", return_value=script_dir):
        
        # run filter func without specifying a config file 
        filter_spis(["keyword1"])

        # ensure the mock_open was called with the expected path
        assert any(call.args[0] == default_config_path for call in mocked_open.mock_calls), \
        "Expected default config file to be opened."

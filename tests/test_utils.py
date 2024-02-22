from pyspi.utils import filter_spis
from tempfile import NamedTemporaryFile
import os
import pytest
import yaml

def test_filter_spis_invalid_keywords():
    """Pass in a dataype other than a list for the keywords"""
    with pytest.raises(TypeError) as excinfo:
        filter_spis("pyspi/config.yaml", "linear")
    assert "Keywords must be passed as a list" in str(excinfo.value), "Keywords must be passed as list error not shown."

def test_filter_spis_with_invalid_config():
    """Pass in an invalid/missing config file"""
    with pytest.raises(FileNotFoundError):
        filter_spis("invalid_config.yaml", ["test"])

def test_filter_spis_normal_operation():
    """Test whether the filter spis function works as expected"""
    # create some mock content to filter
    mock_yaml_content = {
        "module1": {
            "spi1": {"labels": ["keyword1", "keyword2"], "configs": [1, 2]},
            "spi2": {"labels": ["keyword1"], "configs": [3]}
        }
    }
    keywords = ["keyword1", "keyword2"]
    expected_output_yaml = {
        "module1": {
            "spi1": {"labels": ["keyword1", "keyword2"], "configs": [1,2]}
        }
    }

    # create temporary YAML to load into the function
    with NamedTemporaryFile('w', delete=False) as tmp_input_yaml:
        yaml.dump(mock_yaml_content, tmp_input_yaml)
        tmp_input_yaml_name = tmp_input_yaml.name # get the temp file location
    
    # create a temporary output YAML name
    tmp_output_yaml_name = tmp_input_yaml_name + "_output"

    filter_spis(f"./{tmp_input_yaml_name[1:]}.yaml", keywords, name=tmp_output_yaml_name)

    # load in the output
    with open(f"{tmp_output_yaml_name}.yaml", "r") as f:
        actual_output = yaml.load(f, Loader=yaml.FullLoader)
    
    assert actual_output == expected_output_yaml, "Expected filtered YAML does not match actual filtered YAML."











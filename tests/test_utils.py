from pyspi.utils import filter_spis
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

def test_filter_spis_no_matches():
    """Pass in keywords that return no spis and check for ValuError"""
    mock_yaml_content = {
        "module1": {
            "spi1": {"labels": ["keyword1", "keyword2"], "configs": [1, 2]},
            "spi2": {"labels": ["keyword1"], "configs": [3]}
        }
    }
    keywords = ["random_keyword"]

    # create temporary YAML to load into the function
    with open("pyspi/mock_config2.yaml", "w") as f:
        yaml.dump(mock_yaml_content, f)
    
    with pytest.raises(ValueError) as excinfo:
        filter_spis("pyspi/mock_config2.yaml", keywords, name="mock_filtered_config")
    assert "0 SPIs were found" in str(excinfo.value), "Incorrect error message returned when no keywords match found."

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
    with open("pyspi/mock_config.yaml", "w") as f:
        yaml.dump(mock_yaml_content, f)

    
    filter_spis("pyspi/mock_config.yaml", keywords, name="mock_filtered_config")

    # load in the output
    with open("pyspi/mock_filtered_config.yaml", "r") as f:
        actual_output = yaml.load(f, Loader=yaml.FullLoader)
    
    assert actual_output == expected_output_yaml, "Expected filtered YAML does not match actual filtered YAML."

"""
Test for phi native functionality using Calculator and direct function calls.
"""

import pytest
import numpy as np
import tempfile
import yaml
from pyspi.calculator import Calculator


def test_phi_native_basic():
    """Test basic phi computation with native implementation using Calculator."""
    # Generate simple test data
    np.random.seed(42)
    dt = np.random.randn(2, 100)  # 2 variables, 100 time points

    # Create inline phi config
    config = {
        '.statistics.infotheory': {
            'IntegratedInformation': {
                'labels': ['undirected', 'nonlinear', 'unsigned', 'bivariate', 'time-dependent'],
                'configs': [
                    {'phitype': 'star'},
                    {'phitype': 'Geo'}
                ]
            }
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Create calculator with phi config
        calc = Calculator(dt, configfile=config_path)

        # Compute phi values
        calc.compute()

        # Basic assertions
        assert calc.table is not None
        assert len(calc.table) > 0

        # Check that phi computations completed without NaN
        for spi_name, spi_table in calc.table.items():
            assert not spi_table.isna().all().all(), f"All values are NaN for {spi_name}"

    finally:
        import os
        os.unlink(config_path)


def test_phi_types():
    """Test different phi types (star, Geo) with and without normalization."""
    np.random.seed(123)
    dt = np.random.randn(2, 50)

    # Create comprehensive phi config with normalization
    config = {
        '.statistics.infotheory': {
            'IntegratedInformation': {
                'labels': ['undirected', 'nonlinear', 'unsigned', 'bivariate', 'time-dependent'],
                'configs': [
                    {'phitype': 'star'},
                    {'phitype': 'star', 'normalization': 1},
                    {'phitype': 'Geo'},
                    {'phitype': 'Geo', 'normalization': 1}
                ]
            }
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        calc = Calculator(dt, configfile=config_path)
        calc.compute()

        # Should have at least one table with multiple phi configurations
        assert len(calc.table) > 0

        # Check that we get finite values
        finite_count = 0
        for spi_name, spi_table in calc.table.items():
            values = spi_table.values
            finite_values = values[np.isfinite(values)]
            if len(finite_values) > 0:
                finite_count += 1

        # At least some configurations should produce finite values
        assert finite_count > 0, "No phi configurations produced finite values"

    finally:
        import os
        os.unlink(config_path)


def test_phi_native_import():
    """Test that phi_native module can be imported and basic functions work."""
    from pyspi.lib.phi_native import phi_comp, logdet, cov_comp

    # Test basic function existence
    assert callable(phi_comp)
    assert callable(logdet)
    assert callable(cov_comp)

    # Test logdet with simple matrix
    test_matrix = np.eye(2) * 2
    result = logdet(test_matrix)
    expected = 2 * np.log(2)  # log det of 2*I
    assert np.isclose(result, expected)


def test_phi_comp_direct():
    """Test phi_comp function directly."""
    from pyspi.lib.phi_native import phi_comp

    np.random.seed(42)
    X = np.random.randn(2, 100)
    Z = np.array([1, 2])  # Partition
    params = {"tau": 1}
    options = {
        "type_of_phi": "star",
        "type_of_dist": "Gauss",
        "normalization": 0
    }

    result = phi_comp(X, Z, params, options)
    assert np.isfinite(result), "phi_comp should return finite value"
    assert isinstance(result, (int, float, np.number)), "phi_comp should return numeric value"


if __name__ == "__main__":
    pytest.main([__file__])
import pytest
import numpy as np
import os
from pyspi.calculator_time import Calculator


def test_calculator_has_timing_after_compute():
    """Test that calculator has timing data after compute."""
    data = np.random.randn(3, 100)
    calc = Calculator(dataset=data, subset="fast")
    calc.compute()

    assert hasattr(calc, 'spi_timings')
    assert isinstance(calc.spi_timings, dict)
    assert len(calc.spi_timings) == calc.n_spis


def test_timing_values_are_numeric():
    """Test that timing values are positive numbers."""
    data = np.random.randn(3, 100)
    calc = Calculator(dataset=data, subset="fast")
    calc.compute()

    for timing in calc.spi_timings.values():
        assert timing >= 0
        assert isinstance(timing, (int, float))


def test_timing_files_created():
    """Test that timing files are created."""
    data = np.random.randn(3, 100)
    calc = Calculator(dataset=data, subset="fast")

    # Remove files if they exist
    for filename in ["time_spi.txt", "time.txt"]:
        if os.path.exists(filename):
            os.remove(filename)

    calc.compute()

    assert os.path.exists("time_spi.txt")
    assert os.path.exists("time.txt")

    # Clean up
    for filename in ["time_spi.txt", "time.txt"]:
        if os.path.exists(filename):
            os.remove(filename)
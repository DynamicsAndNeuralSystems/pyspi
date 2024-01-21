import pytest
from pyspi.calculator import Calculator
import numpy as np 

def test_calculator_works():
    calc = Calculator()

    assert isinstance(calc, Calculator), "Calculator failed to instantiate"
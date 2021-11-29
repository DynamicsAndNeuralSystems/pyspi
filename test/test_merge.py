import numpy as np
import math, pytest, warnings, yaml, os

from pyspi.data import Data
from pyspi.calculator import Calculator
from pyspi.base import undirected

configfilebase = os.path.dirname(__file__)
configfile0 = configfilebase+'/config0.yaml'
configfile1 = configfilebase+'/config1.yaml'

def test_create_yaml():
    myconfig0 = {'.correlation': {'pearsonr': None},
                '.spectral': {'coherence_magnitude': None}}
    myconfig1 = {'.correlation': {'pearsonr': None,
                                    'spearmanr': None},
                    '.spectral': {'coherence_magnitude': None}}

    print(f'Creating yaml file: {configfile0}')
    with open(configfile0,'w') as f:
        yaml.dump(myconfig0,f)

    print(f'Creating yaml file: {configfile1}')
    with open(configfile1,'w') as f:
        yaml.dump(myconfig1,f)

"""
def test_merge_wrong_attr():

    data = Data(np.random.normal(size=(2,100)),name='tester')

    calc0 = Calculator(name='tester0',dataset=data,configfile=configfile0)
    calc1 = Calculator(name='tester1',dataset=data,configfile=configfile1)

    with pytest.raises(TypeError):
        calc0.merge(calc1)

    data = Data(np.random.normal(size=(2,100)),name='tester1')
    calc1 = Calculator(name='tester0',dataset=data,configfile=configfile1)

    with pytest.raises(TypeError):
        calc0.merge(calc1)

def test_merge_output():

    data = Data(np.random.normal(size=(2,100)),name='tester')

    calc0 = Calculator(name='tester',dataset=data,configfile=configfile0)
    calc1 = Calculator(name='tester',dataset=data,configfile=configfile1)

    calc0.compute()
    calc1.compute()

    calc0.merge(calc1)
    calc1.merge(calc0)
"""

def test_destroy_yaml():
    print(f'Removing yaml files.')
    os.remove(configfile0)
    os.remove(configfile1)

if __name__ == '__main__':
    test_create_yaml()
    # test_merge_wrong_attr()
    # test_merge_output()
    test_destroy_yaml()
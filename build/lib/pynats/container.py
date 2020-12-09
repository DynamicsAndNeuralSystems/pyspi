from pynats.calculator import Calculator
from pynats.data import Data
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc
import scipy.spatial as sp
import matplotlib.pyplot as plt
import copy
import yaml

def forall(func):
    def do(self,**kwargs):
        try:
            for i in self._calculators.index:
                calc_ser = self._calculators.loc[i]
                for calc in calc_ser:
                    func(self,calc,**kwargs)
        except AttributeError:
            raise AttributeError(f'No calculators in frame yet. Initialise before calling {func}')
    return do

class CalculatorFrame():

    def __init__(self,datasets=None,names=None,labels=None,calculators=None,**kwargs):
        if calculators is not None:
            self.set_calculator(calculators)

        if datasets is not None:
            if names is None:
                names = [None] * len(datasets)
            if labels is None:
                labels = [None] * len(datasets)
            self.init_from_list(datasets,names,labels,**kwargs)

    def set_calculator(self,calculators,names=None):
        if hasattr(self, '_dataset'):
            Warning('Overwriting dataset without explicitly deleting.')
            del(self._calculators)

        for calc, i in calculators:
            if names is not None:
                self.add_calculator(calc,names[i])
            else:
                self.add_calculator(calc)
    
    def add_calculator(self,calc):

        if not hasattr(self,'_calculators'):
            self._calculators = pd.DataFrame()

        if isinstance(calc,CalculatorFrame):
            self._calculators = pd.concat([self._calculators,calc])
        elif isinstance(calc,Calculator):
            self._calculators = self._calculators.append(pd.Series(data=calc,name=calc.name),ignore_index=True)
        elif isinstance(calc,pd.DataFrame):
            if isinstance(calc.iloc[0],Calculator):
                self._calculators = calc
            else:
                raise TypeError('Received dataframe but it is not in known format.')
        else:
            raise TypeError(f'Unknown data type: {type(calc)}.')

        self.n_calculators = len(self.calculators.index)
    
    def init_from_list(self,datasets,names,labels,**kwargs):
        base_calc = Calculator(**kwargs)
        for i, dataset in enumerate(datasets):
            calc = copy.deepcopy(base_calc)
            calc.load_dataset(dataset)
            calc.name = names[i]
            calc.label = labels[i]
            self.add_calculator(calc)

    def init_from_yaml(self,document,normalise=True,n_processes=None,n_observations=None,**kwargs):
        datasets = []
        names = []
        labels = []
        with open(document) as f:
            yf = yaml.load(f,Loader=yaml.FullLoader)

            for config in yf:
                try:
                    file = config['file']
                    dim_order = config['dim_order']
                    names.append(config['name'])
                    labels.append(config['labels'])
                    datasets.append(Data(data=file,dim_order=dim_order,name=names[-1],normalise=normalise,n_processes=n_processes,n_observations=n_observations))
                except Exception as err:
                    print(f'Loading dataset: {config} failed ({err}).')

        self.init_from_list(datasets,names,labels,**kwargs)

    @property
    def calculators(self):
        """Return data array."""
        try:
            return self._calculators
        except AttributeError:
            return None

    @calculators.setter
    def calculators(self, cs):
        if hasattr(self, 'calculators'):
            raise AttributeError('You can not assign a value to this attribute'
                                 ' directly, use the set_data method instead.')
        else:
            self._calculators = cs

    @calculators.deleter
    def calculators(self):
        print('Overwriting existing calculators.')
        del(self._calculators)


    @forall
    def compute(self,calc):
        calc.compute()

    @forall
    def prune(self,calc,**kwargs):
        calc.prune(**kwargs)
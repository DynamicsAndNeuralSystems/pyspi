from pynats.calculator import Calculator
import numpy as np
import pandas as pd
import seaborn as sns
from sktime.utils import data_container
import pynats.plot as natplt
import scipy.cluster.hierarchy as spc
import scipy.spatial as sp
import matplotlib.pyplot as plt

def forall(func):
    def do(self,**kwargs):
        for i in self._calculators.index:
            calc_ser = self._calculators.loc[i]
            for calc in calc_ser:
                func(self,calc,**kwargs)

    return do


class CalculatorFrame():

    def __init__(self,datasets=None,names=None,labels=None,calculators=None,normalise=True):
        self.normalise = normalise
        if calculators is not None:
            self.set_calculator(calculators)

        if datasets is not None:
            if names is None:
                names = [None] * len(datasets)
            if labels is None:
                labels = [None] * len(datasets)
            for i, dataset in enumerate(datasets):
                calc = Calculator(dataset=dataset,name=names[i],label=labels[i])
                self.add_calculator(calc)

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
            self._calculators = self._calculators.append(pd.Series(data=calc,name=calc.name))
        elif isinstance(calc,pd.DataFrame):
            if isinstance(calc.iloc[0],Calculator):
                self._calculators = calc
            else:
                raise TypeError('Received dataframe but it is not in known format.')
        else:
            raise TypeError(f'Unknown data type: {type(calc)}.')

        self.n_calculators = len(self.calculators.index)

    @property
    def calculators(self):
        """Return data array."""
        return self._calculators

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
    def plot_data(cf,calc,**kwargs):
        natplt.plot_spacetime(calc.dataset,**kwargs)

    @forall
    def compute(self,calc):
        calc.compute()

    @forall
    def prune(self,calc):
        calc.prune()

    @forall
    def clustermap(self,calc,**kwargs):
        natplt.clustermap(calc,**kwargs)

    def clusterall(self,approach='mean',cmap='vlag',**kwargs):
        if approach == 'flatten':
            df = self.flatten(plot=False,**kwargs)
            df.fillna(0,inplace=True)
            corrs = df.corr(method='spearman')
            corrs.fillna(0,inplace=True)
        else:
            df = pd.DataFrame()
            for i in self._calculators.index:
                df2 = natplt.flatten(self._calculators.loc[i][0],plot=False,**kwargs).corr(method='spearman')
                if df.shape[0] > 0:
                    df = pd.concat([df, df2], axis=0, sort=False)
                else:
                    df = df2
            
            corrs = df.groupby('Pairwise measure').mean().reindex(df.keys())
            corrs.fillna(0,inplace=True)
        g = sns.clustermap(corrs, cmap=cmap, center=0.0, xticklabels=1, yticklabels=1)
        ax = g.ax_heatmap
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def flatten(self,**kwargs):
        df = pd.DataFrame()
        for i in self._calculators.index:
            df2 = natplt.flatten(self._calculators.loc[i][0],**kwargs)
            if df.shape[0] > 0:
                df = pd.concat([df, df2], axis=0, sort=False, ignore_index=True)
            else:
                df = df2
        return df
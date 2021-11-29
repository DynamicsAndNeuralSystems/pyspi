# Load some of the packages
import numpy as np
import os
from pyspi.calculator import Calculator
from pyspi.data import load_dataset
import matplotlib.pyplot as plt

import seaborn as sns

configfile = os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'
calc = Calculator(dataset=load_dataset('forex'),configfile=configfile)
calc.compute()

corrmat = calc.flatten().corr(method='spearman').dropna(axis=0,how='all').dropna(axis=1,how='all')
print(f'Number of statistics left after pruning: {corrmat.shape[0]}')

sns.set(font_scale=0.5)
g = sns.clustermap( corrmat.fillna(0), mask=corrmat.isna(),
                    center=0.0,
                    cmap='RdYlBu_r',
                    xticklabels=1, yticklabels=1 )
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.show()
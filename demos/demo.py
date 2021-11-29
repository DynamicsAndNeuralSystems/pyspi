# Load some of the packages
import numpy as np
import os
from pyspi.calculator import Calculator
from pyspi.data import load_dataset
import matplotlib.pyplot as plt

import seaborn as sns

calc = Calculator(dataset=load_dataset('forex'))
calc.compute()
corrmat = calc.table.stack().corr(method='spearman').abs()

sns.set(font_scale=0.5)
g = sns.clustermap( corrmat.fillna(0), mask=corrmat.isna(),
                    center=0.0,
                    cmap='RdYlBu_r',
                    xticklabels=1, yticklabels=1 )
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.show()
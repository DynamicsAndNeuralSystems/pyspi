# Load some of the packages
import numpy as np
import os
from pyspi.calculator import Calculator
from pyspi.data import load_dataset
import matplotlib.pyplot as plt

import seaborn as sns

# Load one of our stored datasets
dataset = load_dataset('forex')

# visualize the dataset as a heat map (also called a temporal raster plot or carpet plot)
plt.pcolormesh(dataset.to_numpy(squeeze=True),cmap='coolwarm',vmin=-2,vmax=2)
plt.show()

# Instantiate the calculator (inputting the dataset)
calc = Calculator(dataset=dataset)

# Compute all SPIs (this may take a while)
calc.compute()

# Now, we can access all of the matrices by calling calc.table.
# This property will be an Nx(SN) pandas dataframe, where N is the number of processes in the dataset and S is the number of SPIs
print(calc.table)

# We can use this to compute the correlation between all of the methods on this dataset...
corrmat = calc.table.stack().corr(method='spearman').abs()

# ...and plot this correlation matrix
sns.set(font_scale=0.5)
g = sns.clustermap( corrmat.fillna(0), mask=corrmat.isna(),
                    center=0.0,
                    cmap='RdYlBu_r',
                    xticklabels=1, yticklabels=1 )
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.show()
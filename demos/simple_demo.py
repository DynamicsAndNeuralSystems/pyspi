""" This simple demo will show you how to quickly get started with evaluating all SPIs.
"""
import numpy as np
from pyspi.calculator import Calculator, CorrelationFrame
import matplotlib.pyplot as plt

dataset = np.random.randn(3,100) # Generate multivariate data with 3 processes and 100 observations

calc = Calculator(dataset=dataset) # Instantiate the calculator, loading the data
calc.compute() # Compute all SPIs

print(f'Obtained results table of shape {calc.table.shape}:')
print(calc.table) # Print the table of results.

R = calc.table['cov_EmpiricalCovariance'] #  Extract the results for an individual SPI (we're using covariance here)
R = calc._get_correlation_df(rmmin=True)

plt.imshow(R)
plt.colorbar()
plt.ylabel('Process')
plt.xlabel('Process')
plt.show()
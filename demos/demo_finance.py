# Import our classes
import os
from pynats.data import Data
from pynats.btsa import btsa

import matplotlib.pyplot as plt
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = ['AAPL', 'MSFT', '^GSPC']

start_date = '2010-01-01'
end_date = '2019-12-31'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

# Just grab the first experiment for now
(S,P) = panel_data['Close'].shape

N = 200

print('Finance data has {} time series with {} observations each.'.format(P,S))

dat = np.zeros((N,P))
for i in range(P):
    ts = panel_data['Close'][tickers[i]].to_numpy()
    dat[:,i] = ts[:N]
    # dat[:,i] = np.diff(np.log(ts))[:N]

data = Data(dat, dim_order='sp')

calc = btsa()

calc.load(data)
calc.compute()

calc.prune()

# savefile = os.path.dirname(__file__) + '/pynats_sktime.pkl'
# print('Saving object to dill database: "{}"'.format(savefile))

# with open(savefile, 'wb') as f:
#     dill.dump(calc, f)

# print('Done.')

# Pretty visualisations
calc.heatmaps(6)
calc.flatten()
calc.clustermap('all')

plt.show()
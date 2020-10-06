# Import our classes
import os
from pynats.data import Data
from pynats.btsa import btsa

import matplotlib.pyplot as plt
import numpy as np
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import sktime
import dill

# tsfile = '../mvts-database/UEA/ArticularyWordRecognition_TEST.ts'
# tsfile = '../mvts-database/UEA/AtrialFibrillation_TEST.ts'
# tsfile = '../mvts-database/UEA/BasicMotions_TEST.ts'
tsfile = '../mvts-database/UEA/EigenWorms_TEST.ts'

train_x, _ = load_from_tsfile_to_dataframe(tsfile)

# Just grab the first experiment for now
S = len(train_x.iloc[0,0])
P = train_x.shape[1]

N = 200

print('sktime file {} has {} time series with {} observations each.'.format(tsfile,P,S))

dat = np.zeros((N,P))
for i in range(P):
    dat[:,i] = train_x.iloc[:,i][0].to_numpy()[:N]

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
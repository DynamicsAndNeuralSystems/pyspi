from scipy import stats as stats
from scipy import signal as sig
import numpy as np
from . import basedep as base

import statsmodels.tsa.stattools as st

class pearsonr(base.symmeas):

    humanname = "Pearson's product-moment correlation coefficient"
    name = "pearsonr"

    def getpwd(self,x,y):
        return stats.pearsonr(x,y)[0]

class spearmanr(base.symmeas):

    humanname = "Spearman's correlation coefficient"
    name = "spearmanr"
    
    def getpwd(self,x,y):
        return stats.spearmanr(x,y).correlation

class kendalltau(base.symmeas):

    humanname = "Kendall's tau"
    name = "kendalltau"

    def getpwd(self,x,y):
        return stats.kendalltau(x,y).correlation

class coint(base.symmeas):
    
    humanname = "Cointegration"
    name = "coint"

    # Return the negative t-statistic (proxy for how co-integrated they are)
    def getpwd(self,x,y):
        return -st.coint(x,y)[0]
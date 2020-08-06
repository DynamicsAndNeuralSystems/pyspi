import statsmodels.tsa.stattools as st
from . import basedep as base

class coint(base.undirected):
    
    humanname = "Cointegration"
    name = "coint"

    # Return the negative t-statistic (proxy for how co-integrated they are)
    def getpwd(self,x,y):
        return -st.coint(x,y)[0]
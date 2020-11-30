from cdt.causality.pairwise import ANM, BivariateFit, CDS, GNN, IGCI, RECI
import numpy as np
import pandas as pd
from pynats.base import directed, undirected, parse, positive, real

class anm(directed,real):

    humanname = "Additive noise model"
    name = 'anm'
    
    def __init__(self):
        pass

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return ANM().predict_proba((z[i], z[j])), data

class gpfit(directed,real):
    
    humanname = 'Gaussian process bivariate fit'
    name = 'gpfit'

    def __init__(self):
        pass

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return BivariateFit().b_fit_score(z[i], z[j]), data

class cds(directed,real):
    
    humanname = 'Conditional distribution similarity statistic'
    name = 'cds'

    def __init__(self):
        pass

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return CDS().cds_score(z[i], z[j]), data

class gnn(undirected,real):

    humanname = 'Shallow generative neural network'
    name = 'gnn'

    def __init__(self):
        pass

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return GNN().predict_proba((z[i],z[j])), data

class igci(directed,real):

    humanname = 'Information-geometric conditional independence'
    name = 'igci'

    def __init__(self):
        pass

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return IGCI().predict_proba((z[i],z[j])), data

class reci(directed,real):

    humanname = 'Neural correlation coefficient'
    name = 'reci'

    def __init__(self):
        pass

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return RECI().b_fit_score(z[i],z[j]), data
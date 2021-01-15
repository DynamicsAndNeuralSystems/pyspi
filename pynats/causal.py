from cdt.causality.pairwise import ANM, BivariateFit, CDS, GNN, IGCI, RECI
from pynats.base import directed, undirected, parse_bivariate, real

class anm(directed,real):

    humanname = "Additive noise model"
    name = 'anm'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return ANM().predict_proba((z[i], z[j]))

class gpfit(directed,real):
    
    humanname = 'Gaussian process bivariate fit'
    name = 'gpfit'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return BivariateFit().b_fit_score(z[i], z[j])

class cds(directed,real):
    
    humanname = 'Conditional distribution similarity statistic'
    name = 'cds'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return CDS().cds_score(z[i], z[j])

class gnn(undirected,real):

    humanname = 'Shallow generative neural network'
    name = 'gnn'

    def __init__(self):
        raise NotImplementedError('Having issues with this one.')
        pass

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return GNN().predict_proba((z[i],z[j]))

class igci(directed,real):

    humanname = 'Information-geometric conditional independence'
    name = 'igci'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return IGCI().predict_proba((z[i],z[j]))

class reci(directed,real):

    humanname = 'Neural correlation coefficient'
    name = 'reci'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return RECI().b_fit_score(z[i],z[j])
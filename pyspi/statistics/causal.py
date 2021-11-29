from cdt.causality.pairwise import ANM, CDS, GNN, IGCI, RECI
import pyEDM

import numpy as np
import pandas as pd

from pyspi.base import directed, undirected, parse_bivariate, parse_multivariate, unsigned, signed

class anm(directed,unsigned):

    humanname = "Additive noise model"
    name = 'anm'
    labels = ['unsigned','causal','unordered','linear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return ANM().anm_score(z[i], z[j])

class cds(directed,unsigned):
    
    humanname = 'Conditional distribution similarity statistic'
    name = 'cds'
    labels = ['unsigned','causal','unordered','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return CDS().cds_score(z[i], z[j])

class reci(directed,unsigned):

    humanname = 'Regression error-based causal inference'
    name = 'reci'
    labels = ['unsigned','causal','unordered','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return RECI().b_fit_score(z[i], z[j])

class igci(directed,unsigned):

    humanname = 'Information-geometric conditional independence'
    name = 'igci'
    labels = ['causal','directed','nonlinear','unsigned','unordered']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return IGCI().predict_proba((z[i],z[j]))

class ccm(directed,signed):

    humanname = "Convergent cross-maping"
    name = "ccm"
    labels = ['causal','directed','nonlinear','temporal','signed']

    def __init__(self,statistic='mean',embedding_dimension=None):
        self._statistic = statistic
        self._E = embedding_dimension

        self.name += f'_E-{embedding_dimension}_{statistic}'

    @property
    def key(self):
        return self._E

    def _from_cache(self,data):
        try:
            ccmf = data.ccm[self.key]
        except (AttributeError,KeyError):
            z = data.to_numpy(squeeze=True)

            M = data.n_processes
            N = data.n_observations
            df = pd.DataFrame(np.concatenate([np.atleast_2d(np.arange(0,N)),z]).T,
                                columns=['index']+[f'proc{p}' for p in range(M)])

            # Get the embedding
            if self._E is None:
                embedding = np.zeros((M,1))

                # Infer optimal embedding from simplex projection
                for _i in range(M):
                    pred = str(10) + ' ' + str(N-10)
                    embed_df = pyEDM.EmbedDimension(dataFrame=df,lib=pred,
                                                    pred=pred,columns=df.columns.values[_i+1],showPlot=False)
                    embedding[_i] = embed_df.max()['E']
            else:
                embedding = np.array([self._E]*M)

            # Compute CCM from the fixed or optimal embedding
            nlibs = 21
            ccmf = np.zeros((M,M,nlibs+1))
            for _i in range(M):
                for _j in range(_i+1,M):
                    try:
                        E = int(max(embedding[[_i,_j]]))
                    except NameError:
                        E = int(self._E)

                    # Get list of library sizes given nlibs and lower/upper bounds based on embedding dimension
                    upperE = int(np.floor((N-E-1)/10)*10)
                    lowerE = int(np.ceil(2*E/10)*10)
                    inc = int((upperE-lowerE) / nlibs)
                    lib_sizes = str(lowerE) + ' ' + str(upperE) + ' ' + str(inc)
                    srcname = df.columns.values[_i+1]
                    targname = df.columns.values[_j+1]
                    ccm_df = pyEDM.CCM(dataFrame=df,E=E,
                                        columns=srcname,target=targname,
                                        libSizes=lib_sizes,sample=100)
                    ccmf[_i,_j] = ccm_df.iloc[:,1].values[:(nlibs+1)]
                    ccmf[_j,_i] = ccm_df.iloc[:,2].values[:(nlibs+1)]

            try:
                data.ccm[self.key] = ccmf
            except AttributeError:
                data.ccm = {self.key: ccmf}
        return ccmf

    @parse_multivariate
    def adjacency(self,data):
        ccmf = self._from_cache(data)

        if self._statistic == 'mean':
            return np.nanmean(ccmf,axis=2)
        elif self._statistic == 'max':
            return np.nanmax(ccmf,axis=2)
        elif self._statistic == 'diff':
            return np.nanmean(ccmf-np.transpose(ccmf,axes=[1,0,2]),axis=2)
        else:
            raise TypeError(f'Unknown statistic: {self._statistic}')
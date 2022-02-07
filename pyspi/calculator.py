# Science/maths/computing tools
import numpy as np
import pandas as pd
import copy, yaml, importlib, time, warnings, os
from tqdm import tqdm
from collections import Counter
from scipy import stats

# From this package
from .data import Data
from .utils import convert_mdf_to_ddf

class Calculator():
    """Compute all pairwise interactions.

    The calculator takes in a multivariate time-series dataset, computes and stores all pairwise interactions for the dataset.
    It uses a YAML configuration file that can be modified in order to compute a reduced set of pairwise methods.

    Example:
        >>> import numpy as np              
        >>> dataset = np.random.randn(5,500)    # create a random multivariate time series (MTS)
        >>> calc = Calculator(dataset=dataset)  # Instantiate the calculator
        >>> calc.compute()                      # Compute all pairwise interactions

    Args:
        dataset (:class:`pyspi.data.Data`, array_like, optional):
            The multivariate time series of M processes and T observations, defaults to None.
        name (str, optional):
            The name of the calculator. Mainly used for printing the results but can be useful if you have multiple instances, defaults to None.
        labels (array_like, optional):
            Any set of strings by which you want to label the calculator. This can be useful later for classification purposes, defaults to None.
        configfile (str, optional):
            The location of the YAML configuration file. See :ref:`Using a reduced SPI set`, defaults to :code:`'</path/to/pyspi>/pyspi/config.yaml'`
    """
    
    def __init__(self,dataset=None,name=None,labels=None,configfile=os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'):
        self._spis = {}
        self._load_yaml(configfile)


        duplicates = [name for name, count in Counter(self._spis.keys()).items() if count > 1]
        if len(duplicates) > 0:
            raise ValueError(f'Duplicate SPI identifiers: {duplicates}.\n Check the config file for duplicates.')

        self._name = name
        self._labels = labels

        print("Number of SPIs: {}".format(len(self.spis)))

        if dataset is not None:
            self.load_dataset(dataset)

    @property
    def spis(self):
        return self._spis

    @spis.setter
    def spis(self,s):
        raise Exception('Do not set this property externally.')

    @property
    def n_spis(self):
        return len(self._spis)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self,d):
        raise Exception('Do not set this property externally. Use the load_dataset() method.')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,n):
        self._name = n

    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self,ls):
        self._labels = ls

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self,a):
        raise Exception('Do not set this property externally. Use the compute() method.')

    @property
    def group(self):
        try:
            return self._group
        except AttributeError as err:
            warnings.warn('Group undefined. Call set_group() method first.')
            raise AttributeError(err)

    @group.setter
    def group(self,g):
        raise Exception('Do not set this property externally. Use the set_group() method.')

    @property
    def group_name(self):
        try:
            return self._group_name
        except AttributeError as err:
            warnings.warn(f'Group name undefined. Call set_group() method first.')
            return None

    @group_name.setter
    def group_name(self,g):
        raise Exception('Do not set this property externally. Use the group() method.')

    def _load_yaml(self,document):
        print("Loading configuration file: {}".format(document))

        with open(document) as f:
            yf = yaml.load(f,Loader=yaml.FullLoader)

            # Instantiate the SPIs
            for module_name in yf:
                print("*** Importing module {}".format(module_name))
                module = importlib.import_module(module_name,__package__)
                for fcn in yf[module_name]:
                    try:
                        for params in yf[module_name][fcn]:
                            print(f'[{self.n_spis}] Adding SPI {module_name}.{fcn}(x,y,{params})...')
                            spi = getattr(module, fcn)(**params)
                            self._spis[spi.identifier] = spi
                            print(f'Succesfully initialised SPI with identifier "{spi.identifier}" and labels {spi.labels}')
                    except TypeError:
                        print(f'[{self.n_spis}] Adding SPI {module_name}.{fcn}(x,y)...')
                        spi = getattr(module, fcn)()
                        self._spis[spi.identifier] = spi
                        print(f'Succesfully initialised SPI with identifier "{spi.identifier}" and labels {spi.labels}')

    def load_dataset(self,dataset):
        if not isinstance(dataset,Data):
            self._dataset = Data(Data.convert_to_numpy(dataset))
        else:
            self._dataset = dataset

        columns = pd.MultiIndex.from_product([self.spis.keys(),self._dataset.procnames],names=['spi','process'])
        self._table = pd.DataFrame(data=np.full((self.dataset.n_processes,self.n_spis*self.dataset.n_processes), np.NaN),
                                    columns=columns,index=self._dataset.procnames)
        self._table.columns.name = 'process'

    def compute(self,replication=None):
        """ Compute the SPIs on the MVTS dataset
        """
        if not hasattr(self,'_dataset'):
            raise AttributeError('Dataset not loaded yet. Please initialise with load_dataset.')

        if replication is None:
            replication = 0

        pbar = tqdm(self.spis.keys())
        for spi in pbar:
            pbar.set_description(f'Processing [{self._name}: {spi}]')
            start_time = time.time()
            try:
                # Get the MPI from the dataset
                S = self._spis[spi].multivariate(self.dataset)

                # Ensure the diagonal is NaN (sometimes set within the functions)
                np.fill_diagonal(S,np.NaN)

                # Save results
                self._table[spi] = S
            except Exception as err:
                warnings.warn(f'Caught {type(err)} for SPI "{spi}": {err}')
                self._table[spi] = np.NaN
        pbar.close()

    def rmmin(self):
        """ Iterate through all spis and remove the minimum (fixes absolute value errors when correlating)
        """
        for mpi, m in zip(self._table,self._spis):
            if not m.issigned():
                mpi -= np.nanmin(mpi)

    def set_group(self,classes):
        self._group = None
        self._group_name = None

        # Ensure this is a list of lists
        for i, cls in enumerate(classes):
            if not isinstance(cls,list):
                classes[i] = [cls]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(set(j_cls)), (f'Class {i_cls} is a subset of class {j_cls}.')

        labset = set(self.labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        if np.count_nonzero(matches) > 1:
            warnings.warn(f'More than one match for classes {classes}')
        else:
            try:
                id = np.where(matches)[0][0]
                self._group = id
                self._group_name = ', '.join(classes[id])
            except (TypeError,IndexError):
                pass

    def merge(self,other):
        """ TODO: Merge two calculators (to include additional SPIs)
        """
        raise NotImplementedError()
        if self.identifier is not other.name:
            raise TypeError(f'Calculator name does do not match. Aborting merge.')
        
        for attr in ['name','n_processes','n_observations']:
            selfattr = getattr(self.dataset,attr)
            otherattr = getattr(other.dataset,attr)
            if selfattr is not otherattr:
                raise TypeError(f'Attribute {attr} does not match between calculators ({selfattr} != {otherattr})')

    def getstatlabels(self):
        return { s.name : s.labels for s in self._spis }

    def get_correlation_df(self,with_labels=False,rmmin=False,which_stat=['spearman']):
        # Sorts out pesky numerical issues in the unsigned spis
        if rmmin:
            self.rmmin()

        # Flatten (get Edge-by-SPI matrix)
        edges = self.table.stack().abs()

        # Correlate the edge matrix (using pearson and/or spearman correlation)
        mdf = pd.DataFrame()
        if 'pearson' in which_stat:
            pmat = edges.corr(method='pearson')
            pmat.index = pd.MultiIndex.from_tuples(['pearson',pmat.index],names=['Type','SPI-1'])
            pmat.columns.name = 'SPI-2'
            mdf = pmat
        if 'spearman' in which_stat:
            spmat = edges.corr(method='spearman')
            spmat.index = pd.MultiIndex.from_product(['spearman',spmat.index],names=['Type','SPI-1'])
            spmat.columns.name = 'SPI-2'
            mdf = mdf.append(spmat)

        if with_labels:
            return mdf, self.getstatlabels()
        else:
            return mdf

""" CalculatorFrame
Container for batch level commands, like computing/pruning/initialising multiple datasets at once
"""
def forall(func):
    def do(self,*args,**kwargs):
        try:
            for i in self._calculators.index:
                calc_ser = self._calculators.loc[i]
                for calc in calc_ser:
                    func(calc,*args,**kwargs)
        except AttributeError:
            raise AttributeError(f'No calculators in frame yet. Initialise before calling {func}')
    return do

class CalculatorFrame():

    def __init__(self,calculators=None,name=None,datasets=None,names=None,labels=None,**kwargs):
        if calculators is not None:
            self.set_calculator(calculators)

        self.identifier = name

        if datasets is not None:
            if names is None:
                names = [None] * len(datasets)
            if labels is None:
                labels = [None] * len(datasets)
            self.init_from_list(datasets,names,labels,**kwargs)

    @property
    def name(self):
        if hasattr(self,'_name') and self._name is not None:
            return self._name
        else:
            return ''

    @name.setter
    def name(self,n):
        self._name = n

    @staticmethod
    def from_calculator(calculator):
        cf = CalculatorFrame()
        cf.add_calculator(calculator)
        return cf

    def set_calculator(self,calculators):
        if hasattr(self, '_dataset'):
            Warning('Overwriting dataset without explicitly deleting.')
            del(self._calculators)

        if isinstance(calculators,Calculator):
            calculators = [calculators]

        for calc in calculators:
            self.add_calculator(calc)
    
    def add_calculator(self,calc):

        if not hasattr(self,'_calculators'):
            self._calculators = pd.DataFrame()

        if isinstance(calc,CalculatorFrame):
            self._calculators = pd.concat([self._calculators,calc])
        elif isinstance(calc,Calculator):
            self._calculators = self._calculators.append(pd.Series(data=calc,name=calc.name),ignore_index=True)
        elif isinstance(calc,pd.DataFrame):
            if isinstance(calc.iloc[0],Calculator):
                self._calculators = calc
            else:
                raise TypeError('Received dataframe but it is not in known format.')
        else:
            raise TypeError(f'Unknown data type: {type(calc)}.')

        self.n_calculators = len(self.calculators.index)
    
    def init_from_list(self,datasets,names,labels,**kwargs):
        base_calc = Calculator(**kwargs)
        for i, dataset in enumerate(datasets):
            calc = copy.deepcopy(base_calc)
            calc.load_dataset(dataset)
            calc.name = names[i]
            calc.labels = labels[i]
            self.add_calculator(calc)

    def init_from_yaml(self,document,normalise=True,n_processes=None,n_observations=None,**kwargs):
        datasets = []
        names = []
        labels = []
        with open(document) as f:
            yf = yaml.load(f,Loader=yaml.FullLoader)

            for config in yf:
                try:
                    file = config['file']
                    dim_order = config['dim_order']
                    names.append(config['name'])
                    labels.append(config['labels'])
                    datasets.append(Data(data=file,dim_order=dim_order,name=names[-1],normalise=normalise,n_processes=n_processes,n_observations=n_observations))
                except Exception as err:
                    warnings.warn(f'Loading dataset: {config} failed ({err}).')

        self.init_from_list(datasets,names,labels,**kwargs)

    @property
    def calculators(self):
        """Return data array."""
        try:
            return self._calculators
        except AttributeError:
            return None

    @calculators.setter
    def calculators(self, cs):
        if hasattr(self, 'calculators'):
            raise AttributeError('You can not assign a value to this attribute'
                                 ' directly, use the set_data method instead.')
        else:
            self._calculators = cs

    @calculators.deleter
    def calculators(self):
        logger.debug('Overwriting existing calculators.')
        del(self._calculators)

    def merge(self,other):
        try:
            self._calculators = self._calculators.append(other._calculators,ignore_index=True)
        except AttributeError:
            self._calculators = other._calculators

    @forall
    def compute(calc):
        calc.compute()

    @forall
    def set_group(calc,*args):
        calc.set_group(*args)

    @forall
    def rmmin(calc):
        calc.rmmin()

    def flattenall(self,**kwargs):
        df = pd.DataFrame()
        for i in self.calculators.index:
            calc = self.calculators.loc[i][0]
            df2 = calc.flatten(**kwargs)
            df = df.append(df2, ignore_index=True)

        df.dropna(axis='index',how='all',inplace=True)
        return df

    def get_correlation_df(self,with_labels=False,flatten_kwargs={},**kwargs):
        if with_labels:
            mlabels = {}
            dlabels = {}

        shapes = pd.DataFrame()
        mdf = pd.DataFrame()
        for calc in [c[0] for c in self.calculators.values]:
            out = calc.get_correlation_df(with_labels=with_labels,flatten_kwargs=flatten_kwargs,**kwargs)

            s = pd.Series(dict(n_processes=calc.dataset.n_processes,n_observations=calc.dataset.n_observations))
            if calc.name is not None:
                s.name = calc.name
                shapes = shapes.append(s)
            else:
                s.name = 'N/A'
                shapes = shapes.append(s)
            if with_labels:
                df = pd.concat({calc.name: out[0]}, names=['Dataset']) 
                try:
                    mlabels = mlabels | out[1]
                except TypeError:
                    mlabels.update(out[1])
                dlabels[calc.name] = calc.labels
            else:
                df = pd.concat({calc.name: out}, names=['Dataset']) 

            # Adds another hierarchical level giving the dataset name
            mdf = mdf.append(df)
        shapes.index.name = 'Dataset'

        if with_labels:
            return mdf, nandf, shapes, mlabels, dlabels
        else:
            return mdf, nandf, shapes

class CorrelationFrame():

    def __init__(self,cf=None,flatten_kwargs={},**kwargs):
        self._slabels = {}
        self._dlabels = {}
        self._mdf = pd.DataFrame()
        self._shapes = pd.DataFrame()
        
        if cf is not None:
            if isinstance(cf,CalculatorFrame) or isinstance(cf,Calculator):
                cf = CalculatorFrame(cf)
                # Store the statistic-focused dataframe, statistic labels, and dataset labels
                self._mdf, self._shapes, self._slabels, self._dlabels = cf.get_correlation_df(with_labels=True,flatten_kwargs=flatten_kwargs,**kwargs)
                self._name = cf.name
            else:
                self.merge(cf)

    @property
    def name(self):
        if not hasattr(self,'_name'):
            return ''
        else:
            return self._name

    @name.setter
    def name(self,n):
        self._name = n

    @property
    def shapes(self):
        return self._shapes

    @property
    def mdf(self):
        return self._mdf

    @property
    def ddf(self):
        if not hasattr(self,'_ddf') or self._ddf.size != self._mdf.size:
            self._ddf = convert_mdf_to_ddf(self.mdf)
        return self._ddf
    
    @property
    def n_datasets(self):
        return self.ddf.shape[1]

    @property
    def n_spis(self):
        return self.mdf.shape[1]

    @property
    def mlabels(self):
        return self._slabels

    @property
    def dlabels(self):
        return self._dlabels

    @mdf.setter
    def mdf(self):
        raise AttributeError('Do not directly set the mdf attribute.')

    @mlabels.setter
    def mlabels(self):
        raise AttributeError('Do not directly set the mlabels attribute.')
        
    @dlabels.setter
    def dlabels(self):
        raise AttributeError('Do not directly set the dlabels attribute.')

    def merge(self,other):
        
        self._ddf = self.ddf.join(other.ddf)
        self._mdf = self._mdf.append(other.mdf,verify_integrity=True)
        self._shapes = self._shapes.append(other.shapes)

        try:
            self._slabels = self._slabels | other.mlabels
            self._dlabels = self._dlabels | other.dlabels
        except TypeError:
            self._slabels.update(other.mlabels)
            self._dlabels.update(other.dlabels)

    def get_pvalues(self):
        if not hasattr(self,'_pvalues'):
            n = self.shapes['n_observations']
            nstats = self.mdf.shape[1]
            ns = np.repeat(n.values,nstats**2).reshape(self.mdf.shape[0],self.mdf.shape[1])
            rsq = self.mdf.values**2
            fval = ns * rsq / (1-rsq)
            self._pvalues = stats.f.sf(fval, 1, ns-1)
        return pd.DataFrame(data=self._pvalues,index=self.mdf.index,columns=self.mdf.columns)

    def compute_significant_values(self):
        pvals = self.get_pvalues()
        nstats = self.mdf.shape[1]
        self._insig_ind = pvals > 0.05 / nstats / (nstats-1) / 2
        
        if not hasattr(self,'_insig_group'):
            pvals = pvals.droplevel(['Dataset','Type'])
            group_pvalue = pd.DataFrame(data=np.full([pvals.columns.size]*2,np.nan),columns=pvals.columns,index=pvals.columns)
            for f1 in pvals.columns:
                print(f'Computing significance for {f1}...')
                for f2 in [f for f in pvals.columns if f is not f1 and np.isnan(group_pvalue[f1][f])]:
                    cp = pvals[f1][f2]
                    group_pvalue[f1][f2] = stats.combine_pvalues(cp[~cp.isna()])[1]
                    group_pvalue[f2][f1] = group_pvalue[f1][f2]
            self._insig_group = group_pvalue > 0.05

    def get_average_correlation(self,thresh=0.2,absolute=True,summary='mean',remove_insig=False):
        mdf = copy.deepcopy(self.mdf)
        # if remove_insig:
        #     mdf[self._insig_ind] = np.nan

        if absolute:
            ss_adj = getattr(mdf.abs().groupby('SPI-1'),summary)()
        else:
            ss_adj = getattr(mdf.groupby('SPI-1'),summary)()
        ss_adj = ss_adj.dropna(thresh=ss_adj.shape[0]*thresh,axis=0).dropna(thresh=ss_adj.shape[1]*thresh,axis=1).sort_index(axis=1)
        if remove_insig:
            ss_adj[self._insig_group.sort_index()] = np.nan

        return ss_adj

    def get_feature_matrix(self,sthresh=0.8,dthresh=0.2):
        fm = self.ddf.drop_duplicates()
            
        # Drop datasets based on NaN threshold
        num_dnans = dthresh*fm.shape[0]
        fm = fm.dropna(axis=1,thresh=num_dnans)

        # Drop measures based on NaN threshold
        num_snans = sthresh*fm.shape[1]
        fm = fm.dropna(axis=0,thresh=num_snans)
        return fm

    @staticmethod
    def _verify_classes(classes):
        # Ensure this is a list of lists
        for i, cls in enumerate(classes):
            if not isinstance(cls,list):
                classes[i] = [cls]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(set(j_cls)), (f'Class {i_cls} is a subset of class {j_cls}.')

    @staticmethod
    def _get_group(labels,classes,instance,verbose=False):
        labset = set(labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        # Iterate through all 
        if np.count_nonzero(matches) > 1:
            if verbose:
                print(f'More than one match in for {instance} whilst searching for {classes} within {labels}). Choosing first one.')
        
        try:
            myid = np.where(matches)[0][0]
            return myid
        except (TypeError,IndexError):
            if verbose:
                print(f'{instance} has no match in {classes}. Options are {labels}')
            return -1

    @staticmethod
    def _set_groups(classes,labels,group_names,group):
        CorrelationFrame._verify_classes(classes)
        for m in labels:
            group[m] = CorrelationFrame._get_group(labels[m],classes,m)

    def set_sgroups(self,classes):
        # Initialise the classes
        self._sgroup_names = { i : ', '.join(c) for i, c in enumerate(classes) }
        self._sgroup_names[-1] = 'N/A'

        self._sgroup_ids = { m : -1 for m in self._slabels }
        CorrelationFrame._set_groups(classes,self._slabels,self._sgroup_names,self._sgroup_ids)
            

    def set_dgroups(self,classes):
        self._dgroup_names = { i : ', '.join(c) for i, c in enumerate(classes) }
        self._dgroup_names[-1] = 'N/A'

        self._dgroup_ids = { d : -1 for d in self._dlabels }
        CorrelationFrame._set_groups(classes,self._dlabels,self._dgroup_names,self._dgroup_ids)

    def get_dgroup_ids(self,names=None):
        if names is None:
            names = self._ddf.columns
        return [self._dgroup_ids[n] for n in names]

    def get_dgroup_names(self,names=None):
        if names is None:
            names = self._ddf.columns
        return [self._dgroup_names[i] for i in self.get_dgroup_ids(names)]

    def get_sgroup_ids(self,names=None):
        if names is None:
            names = self._mdf.columns
        return [self._sgroup_ids[n] for n in names]

    def get_sgroup_names(self,names=None):
        if names is None:
            names = self._mdf.columns
        return [self._sgroup_names[i] for i in self.get_sgroup_ids(names)]

    def relabel_spis(self,names,labels):
        assert len(names) == len(labels), 'Length of spis must equal length of labels.'
        for n, l in zip(names,labels):
            try:
                self._slabels[n] = l
            except AttributeError:
                self._slabels = {n: l}

    def relabel_data(self,names,labels):
        assert len(names) == len(labels), 'Length of datasets must equal length of labels.'
        for n, l in zip(names,labels):
            self._dlabels[n] = l
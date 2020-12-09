# Science/maths/computing tools
import numpy as np
import pandas as pd
from scipy import stats
import yaml
import importlib
import math
import time
import multiprocessing
import warnings
import dill
import os

# Plotting tools
from tqdm import tqdm
from tqdm import trange
from collections import Counter

# From this package
from pynats.data import Data
from pynats import utils

class Calculator():
    """Calculator for one multivariate time-series dataset
    """
    
    # Initializer / Instance Attributes
    def __init__(self,dataset=None,name=None,label=None,configfile=os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'):

        self._load_yaml(configfile)

        duplicates = [name for name, count in Counter(self._measure_names).items() if count > 1]
        if len(duplicates) > 0:
            raise ValueError(f'Duplicate measure identifiers: {duplicates}.\n Check the config file for duplicates.')

        self._nmeasures = len(self._measures)
        self._nclasses = len(self._classes)
        self._proctimes = np.empty(self._nmeasures)
        self._name = name
        self._label = label

        print("Number of pairwise measures: {}".format(self._nmeasures))

        if dataset is not None:
            self.load_dataset(dataset)

    @property
    def n_measures(self):
        return self._nmeasures

    @n_measures.setter
    def n_measures(self,n):
        raise Exception('Do not set this property externally.')

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self,d):
        raise Exception('Do not set this property externally. Use the load_dataset method.')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,n):
        self._name = n

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self,l):
        self._label = l

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self,a):
        raise Exception('Do not set this property externally. Use the compute method to obtain property.')

    def _load_yaml(self,document):
        print("Loading configuration file: {}".format(document))
        self._classes = []
        self._class_names = []

        self._measures = []
        self._measure_names = []

        with open(document) as f:
            yf = yaml.load(f,Loader=yaml.FullLoader)

            # Instantiate the calc classes 
            for module_name in yf:
                print("*** Importing module {}".format(module_name))
                classes = yf[module_name]
                module = importlib.import_module(module_name,__package__)

                for class_name in classes:
                    paramlist = classes[class_name]

                    self._classes.append(getattr(module, class_name))
                    self._class_names.append(class_name)
                    if paramlist is not None:
                        for params in paramlist:
                            print(f'[{len(self._measures)}] Adding measure {module_name}.{class_name}(x,y,{params})...')
                            self._measures.append(self._classes[-1](**params))
                            self._measure_names.append(self._measures[-1].name)
                            print('Succesfully initialised with identifier "{}"'.format(self._measures[-1].name))
                    else:
                        print(f'[{len(self._measures)}] Adding measure {module_name}.{class_name}(x,y)...')
                        self._measures.append(self._classes[-1]())
                        self._measure_names.append(self._measures[-1].name)
                        print('Succesfully initialised with identifier "{}"'.format(self._measures[-1].name))

    def load_dataset(self,dataset):
        if not isinstance(dataset,Data):
            self._dataset = Data.convert_to_numpy(dataset)
        else:
            self._dataset = dataset
        self._adjacency = np.empty((self._nmeasures,
                                    self.dataset.n_processes,
                                    self.dataset.n_processes))

    def compute(self,replication=None):
        """ Compute the dependency measures for all target processes for a given replication
        """
        if replication is None:
            replication = 0

        pbar = tqdm(range(self._nmeasures))
        for m in pbar:
            pbar.set_description(f'Processing [{self._name}: {self._measure_names[m]}]')
            start_time = time.time()
            try:
                self._adjacency[m], self._dataset = self._measures[m].adjacency(self.dataset)
            except Exception as err:
                warnings.warn(f'Caught exception for measure "{self._measure_names[m]}": {err}')
                self._adjacency[m] = np.NaN
            self._proctimes[m] = time.time() - start_time
        pbar.close()

    def prune(self,meas_nans=0.0,proc_nans=0.9):
        """Prune the bad processes/measures
        """
        print(f'Pruning:\n\t- Measures with more than {100*meas_nans}% bad values'
                f', and\n\t- Processes with more than {100*proc_nans}% bad values')

        # First, iterate through the time-series and remove any that have NaN's > ts_nans
        M = self._nmeasures * (2*(self._dataset.n_processes-1))
        threshold = M * proc_nans
        rm_list = []
        for proc in range(self._dataset.n_processes):

            other_procs = [i for i in range(self._dataset.n_processes) if i != proc]

            flat_adj = self._adjacency[:,other_procs,proc].reshape((M//2,1))
            flat_adj = np.concatenate((flat_adj,self._adjacency[:,proc,other_procs].reshape((M//2,1))))

            nzs = np.count_nonzero(np.isnan(flat_adj))
            if nzs > threshold:
                # print(f'Removing process {proc} with {nzs} ({100*nzs/M.1f}%) special characters.')
                print('Removing process {} with {} ({}.1f%) special characters.'.format(proc,nzs,100*nzs/M))
                rm_list.append(proc)

        # Remove from the dataset object
        self._dataset.remove_process(rm_list)

        # Remove from the adjacency matrix (should probs move this to an attribute that cannot be set)
        self._adjacency = np.delete(self._adjacency,rm_list,axis=1)
        self._adjacency = np.delete(self._adjacency,rm_list,axis=2)

        # Then, iterate through the measures and remove any that have NaN's > meas_nans
        M = self._dataset.n_processes ** 2 - self._dataset.n_processes
        threshold = M * meas_nans
        il = np.tril_indices(self._dataset.n_processes,-1)

        rm_list = []
        for meas in range(self._nmeasures):

            flat_adj = self._adjacency[meas,il[1],il[0]].reshape((M//2,1))
            flat_adj = np.concatenate((flat_adj,
                                        self._adjacency[meas,il[0],il[1]].reshape((M//2,1))))

            # Ensure normalisation, etc., can happen
            if not np.isfinite(flat_adj.sum()):
                rm_list.append(meas)
                print(f'Measure "[{meas}] {self._measure_names[meas]}" has non-finite sum. Removing.')
                continue

            nzs = np.size(flat_adj) - np.count_nonzero(np.isfinite(flat_adj))
            if nzs > threshold:
                rm_list.append(meas)
                print('Removing measure "[{}] {}" with {} ({:.1f}%) '
                        'NaNs (max is {} [{}%])'.format(meas, self._measure_names[meas],
                                                        nzs,100*nzs/M, threshold, 100*meas_nans))

        # Remove the measure from the adjacency and process times matrix
        self._adjacency = np.delete(self._adjacency,rm_list,axis=0)
        self._proctimes = np.delete(self._proctimes,rm_list,axis=0)

        # Remove from the measure lists (move to a method and protect measure)
        for meas in sorted(rm_list,reverse=True):
            del self._measures[meas]
            del self._measure_names[meas]

        self._nmeasures = len(self._measures)
        print('Number of pairwise measures after pruning: {}'.format(self._nmeasures))

    # TODO - merge two calculators (e.g., to include missing/decentralised data or measures)
    def merge(self,calc2):
        if not isinstance(calc2,Calculator):
            raise TypeError('Input must be of type pynats.Calculator')

    def save(self,filename):
        print('Saving object to dill database: "{filename}"')
        with open(filename, 'wb') as f:
            dill.dump(self, f)
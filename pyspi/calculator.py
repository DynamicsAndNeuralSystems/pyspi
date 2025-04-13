# Science/maths/computing tools
import numpy as np
import pandas as pd
import copy, yaml, importlib, time, warnings, os
from tqdm import tqdm
from collections import Counter
from scipy import stats
from colorama import init, Fore
init(autoreset=True)

# From this package
from .data import Data
from .utils import convert_mdf_to_ddf, check_optional_deps, inspect_calc_results


class Calculator:
    """Compute all pairwise interactions.

    The calculator takes in a multivariate time-series dataset, computes and stores all pairwise interactions for the dataset.
    It uses a YAML configuration file that can be modified in order to compute a reduced set of pairwise methods.

    Example:
        >>> import numpy as np
        >>> dataset = np.random.randn(5,500)    # create a random multivariate time series (MTS)
        >>> calc = Calculator(dataset=dataset)  # Instantiate the calculator
        >>> calc.compute()                      # Compute all pairwise interactions

    Args:
        dataset (:class:`~pyspi.data.Data`, array_like, optional):
            The multivariate time series of M processes and T observations, defaults to None.
        name (str, optional):
            The name of the calculator. Mainly used for printing the results but can be useful if you have multiple instances, defaults to None.
        labels (array_like, optional):
            Any set of strings by which you want to label the calculator. This can be useful later for classification purposes, defaults to None.
        subset (str, optional):
            A pre-configured subset of SPIs to use. Options are "all", "fast", "sonnet", or "fabfour", defaults to "all".
        configfile (str, optional):
            The location of the YAML configuration file for a user-defined subset. See :ref:`Using a reduced SPI set`, defaults to :code:`'</path/to/pyspi>/pyspi/config.yaml'`
        detrend (bool, optional):
            If True, detrend the dataset along the time axis before normalising (if enabled), defaults to True.
        normalise (bool, optional):
            If True, z-score normalise the dataset along the time axis before computing SPIs, defaults to True.
            Detrending (if enabled) is always applied before normalisation.
    """
    _optional_dependencies = None

    def __init__(
        self, dataset=None, name=None, labels=None, subset="all", configfile=None,
        detrend=True, normalise=True
    ):
        self._spis = {}
        self._excluded_spis = list()
        self._normalise = normalise
        self._detrend = detrend

        # Define configfile by subset if it was not specified
        if configfile is None:
            if subset == "fast":
                configfile = (
                    os.path.dirname(os.path.abspath(__file__)) + "/fast_config.yaml"
                )
            elif subset == "sonnet":
                configfile = (
                    os.path.dirname(os.path.abspath(__file__)) + "/sonnet_config.yaml"
                )
            elif subset == "fabfour":
                configfile = (
                    os.path.dirname(os.path.abspath(__file__)) + "/fabfour_config.yaml"
                )
            # If no configfile was provided but the subset was not one of the above (or the default 'all'), raise an error
            elif subset != "all":
                raise ValueError(
                    f"Subset '{subset}' does not exist. Try 'all' (default), 'fast', 'sonnet', or 'fabfour'."
                )
            else:
                configfile = os.path.dirname(os.path.abspath(__file__)) + "/config.yaml"

        # add dependency checks here if the calculator is being instantiated for the first time
        if not Calculator._optional_dependencies:
            # check if optional dependencies exist
            print("Checking if optional dependencies exist...")
            Calculator._optional_dependencies = check_optional_deps()

        self._load_yaml(configfile)

        duplicates = [
            name for name, count in Counter(self._spis.keys()).items() if count > 1
        ]
        if len(duplicates) > 0:
            raise ValueError(
                f"Duplicate SPI identifiers: {duplicates}.\n Check the config file for duplicates."
            )

        self._name = name
        self._labels = labels

        print(f"="*100)
        print(Fore.GREEN + f"{len(self.spis)} SPI(s) were successfully initialised.\n")
        if len(self._excluded_spis) > 0:
            missing_deps = [dep for dep, is_met in self._optional_dependencies.items() if not is_met]
            print(Fore.YELLOW + "**** SPI Initialisation Warning ****")
            print(Fore.YELLOW + "\nSome dependencies were not detected, which has led to the exclusion of certain SPIs:")
            print("\nMissing Dependencies:")

            for dep in missing_deps:
                print(f"- {dep}")

            print(f"\nAs a result, a total of {len(self._excluded_spis)} SPI(s) have been excluded:\n")

            dependency_groups = {}
            for spi in self._excluded_spis:
                for dep in spi[1]: 
                    if dep not in dependency_groups:
                        dependency_groups[dep] = []
                    dependency_groups[dep].append(spi[0])

            for dep, spis in dependency_groups.items():
                print(f"\nDependency - {dep} - affects {len(spis)} SPI(s)")
                print("Excluded SPIs:")
                for spi in spis:
                    print(f"  - {spi}")

            print(f"\n" + "="*100)
            print(Fore.YELLOW + "\nOPTIONS TO PROCEED:\n")
            print(f"  1) Install the following dependencies to access all SPIs: [{', '.join(missing_deps)}]")
            callable_name = "{Calculator/CalculatorFrame}"
            print(f"  2) Continue with a reduced set of {self.n_spis} SPIs by calling {callable_name}.compute(). \n")
            print(f"="*100 + "\n")

        if dataset is not None:
            self.load_dataset(dataset)

    @property
    def spis(self):
        """Dict of SPIs.

        Keys are the SPI identifier and values are their objects.
        """
        return self._spis

    @spis.setter
    def spis(self, s):
        raise Exception("Do not set this property externally.")

    @property
    def n_spis(self):
        """Number of SPIs in the calculator."""
        return len(self._spis)

    @property
    def dataset(self):
        """Dataset as a data object."""
        return self._dataset

    @dataset.setter
    def dataset(self, d):
        raise Exception(
            "Do not set this property externally. Use the load_dataset() method."
        )

    @property
    def name(self):
        """Name of the calculator."""
        return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def labels(self):
        """List of calculator labels."""
        return self._labels

    @labels.setter
    def labels(self, ls):
        self._labels = ls

    @property
    def table(self):
        """Results table for all pairwise interactions."""
        return self._table

    @table.setter
    def table(self, a):
        raise Exception(
            "Do not set this property externally. Use the compute() method."
        )

    @property
    def group(self):
        """The numerical group assigned during :meth:`~pyspi.Calculator.calculator.set_group`."""
        try:
            return self._group
        except AttributeError as err:
            warnings.warn("Group undefined. Call set_group() method first.")
            raise AttributeError(err)

    @group.setter
    def group(self, g):
        raise Exception(
            "Do not set this property externally. Use the set_group() method."
        )

    @property
    def group_name(self):
        """The group name assigned during :meth:`~pyspi.Calculator.calculator.set_group`."""
        try:
            return self._group_name
        except AttributeError as err:
            warnings.warn(f"Group name undefined. Call set_group() method first.")
            return None

    @group_name.setter
    def group_name(self, g):
        raise Exception("Do not set this property externally. Use the group() method.")

    def _load_yaml(self, document):
        print("Loading configuration file: {}".format(document))

        with open(document) as f:
            yf = yaml.load(f, Loader=yaml.FullLoader)

            # Instantiate the SPIs
            for module_name in yf:
                print("*** Importing module {}".format(module_name))
                module = importlib.import_module(module_name, __package__)
                for fcn in yf[module_name]:
                    deps = yf[module_name][fcn].get('dependencies')
                    if deps is not None:
                        all_deps_met = all(Calculator._optional_dependencies.get(dep, False) for dep in deps)
                        if not all_deps_met:
                            current_base_spi = yf[module_name][fcn]
                            print(f"Optional dependencies: {deps} not met. Skipping {len(current_base_spi.get('configs'))} SPI(s):")
                            for params in current_base_spi.get('configs'):
                                print(f"*SKIPPING SPI: {module_name}.{fcn}(x,y,{params})...")
                                self._excluded_spis.append([f"{fcn}(x,y,{params})", deps])
                            continue
                    try:
                        for params in yf[module_name][fcn].get('configs'):
                            print(
                                f"[{self.n_spis}] Adding SPI {module_name}.{fcn}(x,y,{params})"
                            )
                            spi = getattr(module, fcn)(**params)
                            self._spis[spi.identifier] = spi
                            print(
                                f'Succesfully initialised SPI with identifier "{spi.identifier}" and labels {spi.labels}'
                            )
                    except TypeError:
                        print(f"[{self.n_spis}] Adding SPI {module_name}.{fcn}(x,y)...")
                        spi = getattr(module, fcn)()
                        self._spis[spi.identifier] = spi
                        print(
                            f'Succesfully initialised SPI with identifier "{spi.identifier}" and labels {spi.labels}'
                        )

    def load_dataset(self, dataset):
        """Load new dataset into existing instance.

        Args:
            dataset (:class:`~pyspi.data.Data`, array_list):
                New dataset to attach to calculator.
        """
        if not isinstance(dataset, Data):
            self._dataset = Data(Data.convert_to_numpy(dataset), normalise=self._normalise, detrend=self._detrend)
        else:
            self._dataset = dataset

        columns = pd.MultiIndex.from_product(
            [self.spis.keys(), self._dataset.procnames], names=["spi", "process"]
        )
        self._table = pd.DataFrame(
            data=np.full(
                (self.dataset.n_processes, self.n_spis * self.dataset.n_processes),
                np.nan,
            ),
            columns=columns,
            index=self._dataset.procnames,
        )
        self._table.columns.name = "process"

    def compute(self):
        """Compute the SPIs on the MVTS dataset."""
        if not hasattr(self, "_dataset"):
            raise AttributeError(
                "Dataset not loaded yet. Please initialise with load_dataset."
            )

        pbar = tqdm(self.spis.keys())
        for spi in pbar:
            pbar.set_description(f"Processing [{self._name}: {spi}]")
            start_time = time.time()
            try:
                # Get the MPI from the dataset
                S = self._spis[spi].multivariate(self.dataset)

                # Ensure the diagonal is NaN (sometimes set within the functions)
                np.fill_diagonal(S, np.nan)

                # Save results
                self._table[spi] = S
            except Exception as err:
                warnings.warn(f'Caught {type(err)} for SPI "{spi}": {err}')
                self._table[spi] = np.nan
        pbar.close()
        print(Fore.GREEN + f"\nCalculation complete. Time taken: {pbar.format_dict['elapsed']:.4f}s")
        inspect_calc_results(self)
        
    def _rmmin(self):
        """Iterate through all spis and remove the minimum (fixes absolute value errors when correlating)"""
        for spi in self.spis:
            mpi = self.table[spi]
            if not self.spis[spi].issigned():
                self.table[spi] = mpi - np.nanmin(mpi)

    def set_group(self, classes):
        """Assigns a numeric value to this instance based on list of classes.

        Args:
            classes (list):
                If any of the labels in this instance matches one in the class list, then we assign the index
                value to this class.
        """
        self._group = None
        self._group_name = None

        # Ensure this is a list of lists
        for i, c in enumerate(classes):
            if not isinstance(c, list):
                classes[i] = [c]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(
                    set(j_cls)
                ), f"Class {i_cls} is a subset of class {j_cls}."

        labset = set(self.labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        if np.count_nonzero(matches) > 1:
            warnings.warn(f"More than one match for classes {classes}")
        else:
            try:
                id = np.where(matches)[0][0]
                self._group = id
                self._group_name = ", ".join(classes[id])
            except (TypeError, IndexError):
                pass

    def _merge(self, other):
        """TODO: Merge two calculators (to include additional SPIs)"""
        raise NotImplementedError()
        if self.identifier is not other.name:
            raise TypeError(f"Calculator name does do not match. Aborting merge.")

        for attr in ["name", "n_processes", "n_observations"]:
            selfattr = getattr(self.dataset, attr)
            otherattr = getattr(other.dataset, attr)
            if selfattr is not otherattr:
                raise TypeError(
                    f"Attribute {attr} does not match between calculators ({selfattr} != {otherattr})"
                )

    def get_stat_labels(self):
        """Get the labels for each statistic.

        Returns:
            stat_labels (dict): dictionary of
        """
        return {k: v.labels for k, v in zip(self._spis.keys(), self._spis.values())}

    def _get_correlation_df(self, with_labels=False, rmmin=False):
        # Sorts out pesky numerical issues in the unsigned spis
        if rmmin:
            self._rmmin()

        # Flatten (get Edge-by-SPI matrix)
        edges = self.table.stack()

        # Correlate the edge matrix (using pearson and/or spearman correlation)
        cf = pd.DataFrame(
            index=[c for c in edges.columns], columns=[c for c in edges.columns]
        )
        # Need to iterate through each pair to handle unsigned/signed statistics
        for i, s0 in enumerate(edges.columns):
            for j, s1 in enumerate(edges.columns[i + 1 :]):

                if self.spis[s0].issigned() and self.spis[s1].issigned():
                    # When they're both signed, just take the correlation
                    cf.iloc[[i, i + j + 1], [i, i + j + 1]] = edges[[s0, s1]].corr(
                        method="spearman"
                    )
                else:
                    # Otherwise, take the absolute value to make sure we compare like-for-like
                    cf.iloc[[i, i + j + 1], [i, i + j + 1]] = (
                        edges[[s0, s1]].abs().corr(method="spearman")
                    )

        cf.index.name = "SPI-1"
        cf.columns.name = "SPI-2"

        if with_labels:
            return cf, self.getstatlabels()
        else:
            return cf


def forall(func):
    def do(self, *args, **kwargs):
        try:
            for i in self._calculators.index:
                calc_ser = self._calculators.loc[i]
                for calc in calc_ser:
                    func(calc, *args, **kwargs)
        except AttributeError:
            raise AttributeError(
                f"No calculators in frame yet. Initialise before calling {func}"
            )

    return do


class CalculatorFrame:
    """ CalculatorFrame
        Container for batch level commands, like computing/pruning/initialising multiple datasets at once
    """
    def __init__(
        self,
        calculators=None,
        name=None,
        datasets=None,
        names=None,
        labels=None,
        **kwargs,
    ):
        if calculators is not None:
            self.set_calculator(calculators)

        self.name = name

        if datasets is not None:
            if names is None:
                names = [None] * len(datasets)
            if labels is None:
                labels = [None] * len(datasets)
            self.init_from_list(datasets, names, labels, **kwargs)

    @property
    def name(self):
        if hasattr(self, "_name") and self._name is not None:
            return self._name
        else:
            return ""

    @name.setter
    def name(self, n):
        self._name = n

    @staticmethod
    def from_calculator(calculator):
        cf = CalculatorFrame()
        cf.add_calculator(calculator)
        return cf

    def set_calculator(self, calculators):
        if hasattr(self, "_dataset"):
            Warning("Overwriting dataset without explicitly deleting.")
            del self._calculators

        if isinstance(calculators, Calculator):
            calculators = [calculators]

        if isinstance(calculators, CalculatorFrame):
            self.add_calculator(calculators)
        else:
            for calc in calculators:
                self.add_calculator(calc)

    def add_calculator(self, calc):

        if not hasattr(self, "_calculators"):
            self._calculators = pd.DataFrame()

        if isinstance(calc, CalculatorFrame):
            self._calculators = pd.concat([self._calculators.values, calc])
        elif isinstance(calc, Calculator):
            self._calculators = pd.concat(
                [self._calculators, pd.Series(data=calc, name=calc.name)],
                ignore_index=True,
            )
        elif isinstance(calc, pd.DataFrame):
            if isinstance(calc.iloc[0], Calculator):
                self._calculators = calc
            else:
                raise TypeError("Received dataframe but it is not in known format.")
        else:
            raise TypeError(f"Unknown data type: {type(calc)}.")

        self.n_calculators = len(self.calculators.index)

    def init_from_list(self, datasets, names, labels, **kwargs):
        base_calc = Calculator(**kwargs)
        for i, dataset in enumerate(datasets):
            calc = copy.deepcopy(base_calc)
            calc.load_dataset(dataset)
            calc.name = names[i]
            calc.labels = labels[i]
            self.add_calculator(calc)

    def init_from_yaml(
        self, document, detrend=True, normalise=True, n_processes=None, n_observations=None, **kwargs
    ):
        datasets = []
        names = []
        labels = []
        with open(document) as f:
            yf = yaml.load(f, Loader=yaml.FullLoader)

            for config in yf:
                try:
                    file = config["file"]
                    dim_order = config["dim_order"]
                    names.append(config["name"])
                    labels.append(config["labels"])
                    datasets.append(
                        Data(
                            data=file,
                            dim_order=dim_order,
                            name=names[-1],
                            detrend=detrend,
                            normalise=normalise,
                            n_processes=n_processes,
                            n_observations=n_observations,
                        )
                    )
                except Exception as err:
                    warnings.warn(f"Loading dataset: {config} failed ({err}).")

        self.init_from_list(datasets, names, labels, **kwargs)

    @property
    def calculators(self):
        """Return data array."""
        try:
            return self._calculators
        except AttributeError:
            return None

    @calculators.setter
    def calculators(self, cs):
        if hasattr(self, "calculators"):
            raise AttributeError(
                "You can not assign a value to this attribute"
                " directly, use the set_data method instead."
            )
        else:
            self._calculators = cs

    @calculators.deleter
    def calculators(self):
        warnings.warn("Overwriting existing calculators.")
        del self._calculators

    def merge(self, other):
        try:
            self._calculators = pd.concat(
                [self._calculators, other._calculators], ignore_index=True
            )
        except AttributeError:
            self._calculators = other._calculators

    @forall
    def compute(calc):
        calc.compute()

    @property
    def groups(self):
        groups = []
        for i in self._calculators.index:
            calc_ser = self._calculators.loc[i]
            for calc in calc_ser:
                groups.append(calc.group)
        return groups

    @forall
    def set_group(calc, *args):
        calc.set_group(*args)

    @forall
    def _rmmin(calc):
        calc._rmmin()

    def get_correlation_df(self, with_labels=False, **kwargs):
        if with_labels:
            mlabels = {}
            dlabels = {}

        shapes = pd.DataFrame()
        mdf = pd.DataFrame()
        for calc in [c[0] for c in self.calculators.values]:
            out = calc._get_correlation_df(with_labels=with_labels, **kwargs)

            s = pd.Series(
                dict(
                    n_processes=calc.dataset.n_processes,
                    n_observations=calc.dataset.n_observations,
                )
            )
            if calc.name is not None:
                s.name = calc.name
                shapes = pd.concat([shapes, pd.DataFrame(s).T])
            else:
                s.name = "N/A"
                shapes = pd.concat([shapes, pd.DataFrame(s).T])
            if with_labels:
                df = pd.concat({calc.name: out[0]}, names=["Dataset"])
                try:
                    mlabels = mlabels | out[1]
                except TypeError:
                    mlabels.update(out[1])
                dlabels[calc.name] = calc.labels
            else:
                df = pd.concat({calc.name: out}, names=["Dataset"])

            # Adds another hierarchical level giving the dataset name
            mdf = pd.concat([mdf, df])
        shapes.index.name = "Dataset"

        if with_labels:
            return mdf, shapes, mlabels, dlabels
        else:
            return mdf, shapes


class CorrelationFrame:
    def __init__(self, cf=None, **kwargs):
        self._slabels = {}
        self._dlabels = {}
        self._mdf = pd.DataFrame()
        self._shapes = pd.DataFrame()

        if cf is not None:
            if isinstance(cf, Calculator):
                cf = CalculatorFrame(cf)

            if isinstance(cf, CalculatorFrame):
                # Store the statistic-focused dataframe, statistic labels, and dataset labels
                (
                    self._mdf,
                    self._shapes,
                    self._slabels,
                    self._dlabels,
                ) = cf.get_correlation_df(with_labels=True, **kwargs)
                self._name = cf.name
            else:
                self.merge(cf)

    @property
    def name(self):
        if not hasattr(self, "_name"):
            return ""
        else:
            return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def shapes(self):
        return self._shapes

    @property
    def mdf(self):
        return self._mdf

    @property
    def ddf(self):
        if not hasattr(self, "_ddf") or self._ddf.size != self._mdf.size:
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
        raise AttributeError("Do not directly set the mdf attribute.")

    @mlabels.setter
    def mlabels(self):
        raise AttributeError("Do not directly set the mlabels attribute.")

    @dlabels.setter
    def dlabels(self):
        raise AttributeError("Do not directly set the dlabels attribute.")

    def merge(self, other):
        if not all(isinstance(i[0], str) for i in self._mdf.index):
            raise TypeError(
                f"This operation only works with named calculators (set each calc.name property)."
            )

        try:
            self._ddf = self.ddf.join(other.ddf)
            self._mdf = pd.concat([self._mdf, other.mdf], verify_integrity=True)
            self._shapes = pd.concat(
                [self._shapes, other.shapes], verify_integrity=True
            )
        except KeyError:
            self._ddf = copy.deepcopy(other.ddf)
            self._mdf = copy.deepcopy(other.mdf)
            self._shapes = copy.deepcopy(other.shapes)

        try:
            self._slabels = self._slabels | other.mlabels
            self._dlabels = self._dlabels | other.dlabels
        except TypeError:
            self._slabels.update(other.mlabels)
            self._dlabels.update(other.dlabels)

    def get_pvalues(self):
        if not hasattr(self, "_pvalues"):
            n = self.shapes["n_observations"]
            nstats = self.mdf.shape[1]
            ns = np.repeat(n.values, nstats**2).reshape(
                self.mdf.shape[0], self.mdf.shape[1]
            )
            rsq = self.mdf.values**2
            fval = ns * rsq / (1 - rsq)
            self._pvalues = stats.f.sf(fval, 1, ns - 1)
        return pd.DataFrame(
            data=self._pvalues, index=self.mdf.index, columns=self.mdf.columns
        )

    def compute_significant_values(self):
        pvals = self.get_pvalues()
        nstats = self.mdf.shape[1]
        self._insig_ind = pvals > 0.05 / nstats / (nstats - 1) / 2

        if not hasattr(self, "_insig_group"):
            pvals = pvals.droplevel(["Dataset", "Type"])
            group_pvalue = pd.DataFrame(
                data=np.full([pvals.columns.size] * 2, np.nan),
                columns=pvals.columns,
                index=pvals.columns,
            )
            for f1 in pvals.columns:
                print(f"Computing significance for {f1}...")
                for f2 in [
                    f
                    for f in pvals.columns
                    if f is not f1 and np.isnan(group_pvalue[f1][f])
                ]:
                    cp = pvals[f1][f2]
                    group_pvalue[f1][f2] = stats.combine_pvalues(cp[~cp.isna()])[1]
                    group_pvalue[f2][f1] = group_pvalue[f1][f2]
            self._insig_group = group_pvalue > 0.05

    def get_average_correlation(
        self, thresh=0.2, absolute=True, summary="mean", remove_insig=False
    ):
        mdf = copy.deepcopy(self.mdf)

        if absolute:
            ss_adj = getattr(mdf.abs().groupby("SPI-1"), summary)()
        else:
            ss_adj = getattr(mdf.groupby("SPI-1"), summary)()
        ss_adj = (
            ss_adj.dropna(thresh=ss_adj.shape[0] * thresh, axis=0)
            .dropna(thresh=ss_adj.shape[1] * thresh, axis=1)
            .sort_index(axis=1)
        )
        if remove_insig:
            ss_adj[self._insig_group.sort_index()] = np.nan

        return ss_adj

    def get_feature_matrix(self, sthresh=0.8, dthresh=0.2, dropduplicates=True):

        fm = self.ddf
        if dropduplicates:
            fm = fm.drop_duplicates()

        # Drop datasets based on NaN threshold
        num_dnans = dthresh * fm.shape[0]
        fm = fm.dropna(axis=1, thresh=num_dnans)

        # Drop measures based on NaN threshold
        num_snans = sthresh * fm.shape[1]
        fm = fm.dropna(axis=0, thresh=num_snans)
        return fm

    @staticmethod
    def _verify_classes(classes):
        # Ensure this is a list of lists
        for i, cls in enumerate(classes):
            if not isinstance(cls, list):
                classes[i] = [cls]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(
                    set(j_cls)
                ), f"Class {i_cls} is a subset of class {j_cls}."

    @staticmethod
    def _get_group(labels, classes, instance, verbose=False):
        labset = set(labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        # Iterate through all
        if np.count_nonzero(matches) > 1:
            if verbose:
                print(
                    f"More than one match in for {instance} whilst searching for {classes} within {labels}). Choosing first one."
                )

        try:
            myid = np.where(matches)[0][0]
            return myid
        except (TypeError, IndexError):
            if verbose:
                print(f"{instance} has no match in {classes}. Options are {labels}")
            return -1

    @staticmethod
    def _set_groups(classes, labels, group_names, group):
        CorrelationFrame._verify_classes(classes)
        for m in labels:
            group[m] = CorrelationFrame._get_group(labels[m], classes, m)

    def set_sgroups(self, classes):
        # Initialise the classes
        self._sgroup_names = {i: ", ".join(c) for i, c in enumerate(classes)}
        self._sgroup_names[-1] = "N/A"

        self._sgroup_ids = {m: -1 for m in self._slabels}
        CorrelationFrame._set_groups(
            classes, self._slabels, self._sgroup_names, self._sgroup_ids
        )

    def set_dgroups(self, classes):
        self._dgroup_names = {i: ", ".join(c) for i, c in enumerate(classes)}
        self._dgroup_names[-1] = "N/A"

        self._dgroup_ids = {d: -1 for d in self._dlabels}
        CorrelationFrame._set_groups(
            classes, self._dlabels, self._dgroup_names, self._dgroup_ids
        )

    def get_dgroup_ids(self, names=None):
        if names is None:
            names = self._ddf.columns

        return [self._dgroup_ids[n] for n in names]

    def get_dgroup_names(self, names=None):
        if names is None:
            names = self._ddf.columns

        return [self._dgroup_names[i] for i in self.get_dgroup_ids(names)]

    def get_sgroup_ids(self, names=None):
        if names is None:
            names = self._mdf.columns

        return [self._sgroup_ids[n] for n in names]

    def get_sgroup_names(self, names=None):
        if names is None:
            names = self._mdf.columns

        return [self._sgroup_names[i] for i in self.get_sgroup_ids(names)]

    def relabel_spis(self, names, labels):
        assert len(names) == len(labels), "Length of spis must equal length of labels."

        for n, l in zip(names, labels):
            try:
                self._slabels[n] = l
            except AttributeError:
                self._slabels = {n: l}

    def relabel_data(self, names, labels):
        assert len(names) == len(
            labels
        ), "Length of datasets must equal length of labels."

        for n, l in zip(names, labels):
            self._dlabels[n] = l

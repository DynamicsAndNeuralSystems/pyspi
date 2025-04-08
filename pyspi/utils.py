import numpy as np
from scipy.stats import zscore
import warnings
import pandas as pd
import os
import yaml 
from colorama import Fore, init
init(autoreset=True)

def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # If the check cannot be properly performed we fallback to omitting
        # nan values and raising a warning. This can happen when attempting to
        # sum things that are not numbers (e.g. as in the function `mode`).
        contains_nan = False
        nan_policy = 'omit'
        warnings.warn("The input array could not be properly checked for nan "
                      "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)


def strshort(instr,mlength):
    """Shorten a string using ellipsis
    """
    if isinstance(instr,list):
        outstr = []
        for i in range(len(instr)):
            cstr = instr[i]
            outstr.append((cstr[:mlength-6] + '...' + cstr[-3:]) if len(cstr) > mlength else cstr)
    else:
        outstr = (instr[:mlength-6] + '...' + instr[-3:]) if len(instr) > mlength else instr
    return outstr

def acf(x,mode='positive'):
    """Return the autocorrelation function
    """
    if x.ndim > 1:
        x = np.squeeze(x)

    x = zscore(x)
    acf = np.correlate(x,x,mode='full')
    acf = acf / acf[acf.size//2]

    if mode == 'positive':
        return acf[acf.size//2:]
    else:
        return acf

def swap_chars(s, i_1, i_2):
    """Swap to characters in a string.

    Example:
        >>> print(swap_chars('heLlotHere', 2, 6))
        'heHlotLere'
    """
    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    return ''.join([s[0:i_1], s[i_2], s[i_1+1:i_2], s[i_1], s[i_2+1:]])

def normalise(a, axis=0, nan_policy='propogate'):

    contains_nan, nan_policy = _contains_nan(a, nan_policy)

    if contains_nan and nan_policy == 'omit':
        return (a - np.nanmin(a,axis=axis)) / (np.nanmax(a,axis=axis) - np.nanmin(a,axis=axis))
    else:
        return (a - np.min(a,axis=axis)) / (np.max(a,axis=axis) - np.min(a,axis=axis))

def standardise(a, dimension=0, df=1):
    """Z-standardise a numpy array along a given dimension.

    Standardise array along the axis defined in dimension using the denominator
    (N - df) for the calculation of the standard deviation.

    Args:
        a : numpy array
            data to be standardised
        dimension : int [optional]
            dimension along which array should be standardised
        df : int [optional]
            degrees of freedom for the denominator of the standard derivation

    Returns:
        numpy array
            standardised data
    """
    # Avoid division by standard deviation if the process is constant.
    a_sd = a.std(axis=dimension, ddof=df)

    if np.isclose(a_sd, 0):
        return a - a.mean(axis=dimension)
    else:
        return (a - a.mean(axis=dimension)) / a_sd

def convert_mdf_to_ddf(df):
    ddf = pd.pivot_table(data=df.stack(dropna=False).reset_index(),index='Dataset',columns=['SPI-1', 'SPI-2'],dropna=False).T.droplevel(0)
    return ddf

def is_jpype_jvm_available():
    """Check whether a JVM is accessible via Jpype"""
    try:
        import jpype as jp
        if not jp.isJVMStarted():
            jarloc = (os.path.dirname(os.path.abspath(__file__)) + "/lib/jidt/infodynamics.jar")
            # if JVM not started, start a session
            print(f"Starting JVM with java class {jarloc}.")
            jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)
        return True
    except Exception as e:
        print(f"Jpype JVM not available: {e}")
        return False

def is_octave_available():
    """Check whether octave is available"""
    try:
        from oct2py import Oct2Py
        oc = Oct2Py()
        oc.exit()
        return True
    except Exception as e:
        print(f"Octave not available: {e}")
        return False

def check_optional_deps():
    """Bundle all of the optional
    dependency checks together."""
    isAvailable = {}
    isAvailable['octave'] = is_octave_available()
    isAvailable['java'] = is_jpype_jvm_available()

    return isAvailable

def filter_spis(keywords, output_name = None, configfile= None):
    """
    Filter a YAML using a list of keywords, and save the reduced set as a new
    YAML with a user-specified name (or a random one if not provided) in the
    current directory.

    Args:
        keywords (list): A list of keywords (as strings) to filter the YAML.
        output_name (str, optional): The desired name for the output file. Defaults to a random name. 
        configfile (str, optional): The path to the input YAML file. Defaults to the `config.yaml' in the pyspi dir. 

    Raises:
        ValueError: If `keywords` is not a list or if no SPIs match the keywords.
        FileNotFoundError: If the specified `configfile` or the default `config.yaml` is not found.
        IOError: If there's an error reading the YAML file.
    """
    # handle invalid keyword input
    if not keywords:
        raise ValueError("At least one keyword must be provided.")
    if not all(isinstance(keyword, str) for keyword in keywords):
        raise ValueError("All keywords must be strings.")
    if not isinstance(keywords, list):
        raise ValueError("Keywords must be provided as a list of strings.")

    # if no configfile and no keywords are provided, use the default 'config.yaml' in pyspi location
    if configfile is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_config = os.path.join(script_dir, 'config.yaml')
        if not os.path.isfile(default_config):
            raise FileNotFoundError(f"Default 'config.yaml' file not found in {script_dir}.")
        configfile = default_config
        source_file_info = f"Default 'config.yaml' file from {script_dir} was used as the source file."
    else:
        source_file_info = f"User-specified config file '{configfile}' was used as the source file."

    # load in user-specified yaml
    try:
        with open(configfile) as f:
            yf = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{configfile}' not found.")
    except Exception as e:
        # handle all other exceptions
        raise IOError(f"An error occurred while trying to read '{configfile}': {e}")

    # new dictionary to be converted to final YAML
    filtered_subset = {}
    spis_found = 0
    
    for module in yf:
        module_spis = {}
        for spi in yf[module]:
            spi_labels = yf[module][spi].get('labels')
            if all(keyword in spi_labels for keyword in keywords):
                module_spis[spi] = yf[module][spi]
                if yf[module][spi].get('configs'):
                    spis_found += len(yf[module][spi].get('configs'))
                else:
                    spis_found += 1
    
        if module_spis:
            filtered_subset[module] = module_spis
    
    # check that > 0 SPIs found
    if spis_found == 0:
        raise ValueError(f"0 SPIs were found with the specific keywords: {keywords}.")
    
    # construct output file path
    if output_name is None:
        # use a unique name
        output_name = "config_" + os.urandom(4).hex()

    output_file = os.path.join(os.getcwd(), f"{output_name}.yaml")
    
    # write to YAML
    with open(output_file, "w") as outfile:
        yaml.dump(filtered_subset, outfile, default_flow_style=False, sort_keys=False)

    # output relevant information
    print(f"""\nOperation Summary:
-----------------
- {source_file_info}
- Total SPIs Matched: {spis_found} SPI(s) were found with the specific keywords: {keywords}.
- New File Created: A YAML file named `{output_name}.yaml` has been saved in the current directory: `{output_file}'
- Next Steps: To utilise the filtered set of SPIs, please initialise a new Calculator instance with the following command:
`Calculator(configfile='{output_file}')`
""")

def inspect_calc_results(calc):
    """
    Display a summary of the computed SPI results, including counts of successful computations, 
    outputs with NaNs, and partially computed results.
    """
    total_num_spis = calc.n_spis
    num_procs = calc.dataset.n_processes
    spi_results = dict({'Successful': list(), 'NaNs': list(), 'Partial NaNs': list()})
    for key in calc.spis.keys():
        if calc.table[key].isna().all().all():
            spi_results['NaNs'].append(key)
        elif calc.table[key].isnull().values.sum() > num_procs:
            # off-diagonal NaNs
            spi_results['Partial NaNs'].append(key)
        else:
            # returned numeric values (i.e., not NaN)
            spi_results['Successful'].append(key)
    
    # print summary
    double_line_60 = "="*60
    single_line_60 = "-"*60
    print("\nSPI Computation Results Summary")
    print(double_line_60)
    print(f"\nTotal number of SPIs attempted: {total_num_spis}")
    print(f"Number of SPIs successfully computed: {len(spi_results['Successful'])} ({len(spi_results['Successful']) / total_num_spis * 100:.2f}%)")
    print(single_line_60)
    print("Category       | Count | Percentage")
    print(single_line_60)
    for category, spis in spi_results.items():
        count = len(spis)
        percentage = (count / total_num_spis) * 100
        print(f"{category:14} | {count:5} | {percentage:6.2f}%")
    print(single_line_60)

    if spi_results['NaNs']:
        print(f"\n[{len(spi_results['NaNs'])}] SPI(s) produced NaN outputs:")
        print(single_line_60)
        for i, spi in enumerate(spi_results['NaNs']):
            print(f"{i+1}. {spi}")
        print(single_line_60 + "\n")
    if spi_results['Partial NaNs']:
        print(f"\n[{len(spi_results['Partial NaNs'])}] SPIs which produced partial NaN outputs:")
        print(single_line_60)
        for i, spi in enumerate(spi_results['Partial NaNs']):
            print(f"{i+1}. {spi}")
        print(single_line_60 + "\n")
    
"""pyspi utility functions."""
import numpy as np
from scipy.stats import zscore
import warnings
import pandas as pd
import os
import yaml 

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

def spi_filter(configfile, keywords, name="filtered_config"):
    """Filter a YAML using a list of keywords, and save the reduced
    set as a new YAML with a user-specified name in the current
    directory."""
    
    # check that keywords is a list
    if not isinstance(keywords, list):
        raise TypeError("Keywords must be passed as a list.")
    # load in the original YAML
    with open(configfile) as f:
        yf = yaml.load(f, Loader=yaml.FullLoader)
    
    # new dictonary to be converted to final YAML
    filtered_subset = {}
    spis_found = 0
    
    for module in yf:
        module_spis = {}
        for spi in yf[module]:
            spi_labels = yf[module][spi].get('labels')
            if all(keyword in spi_labels for keyword in keywords):
                module_spis[spi] = yf[module][spi]
                spis_found += len(yf[module][spi].get('configs'))
        if module_spis:
            filtered_subset[module] = module_spis
    
    # check that > 0 SPIs found
    if spis_found == 0:
        raise ValueError(f"0 SPIs were found with the specific keywords: {keywords}.")
    
    # write to YAML
    with open(f"pyspi/{name}.yaml", "w") as outfile:
        yaml.dump(filtered_subset, outfile, default_flow_style=False, sort_keys=False)

    # output relevant information
      # output relevant information
    print(f"""\nOperation Summary:
-----------------
- Total SPIs Matched: {spis_found} SPI(s) were found with the specific keywords: {keywords}.
- New File Created: A YAML file named `{name}.yaml` has been saved in the current directory: `pyspi/{name}.yaml'
- Next Steps: To utilise the filtered set of SPIs, please initialise a new Calculator instance with the following command:
`Calculator(configfile='pyspi/{name}.yaml')`
""")
    
    
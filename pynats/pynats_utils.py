"""Provide _nats_ utility functions."""
import pprint
import numpy as np
from scipy import stats

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
    x = stats.zscore(x)
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
    # Don't divide by standard devitation if process is constant.
    a_sd = a.std(axis=dimension, ddof=df)
    if np.isclose(a_sd, 0):
        return a - a.mean(axis=dimension)
    else:
        return (a - a.mean(axis=dimension)) / a_sd
"""Provide _mtsda_ utility functions."""
import pprint
import numpy as np

def swap_chars(s, i_1, i_2):
    """Swap to characters in a string.

    Adapted from IDTxL

    Example:
        >>> print(swap_chars('heLlotHere', 2, 6))
        'heHlotLere'
    """
    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    return ''.join([s[0:i_1], s[i_2], s[i_1+1:i_2], s[i_1], s[i_2+1:]])

def standardise(a, dimension=0, df=1):
    """Z-standardise a numpy array along a given dimension.

    Adapted from IDTxL.

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
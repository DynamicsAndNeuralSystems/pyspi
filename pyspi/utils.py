"""Helpers for parameter checking, identifiers and result summaries."""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore
import yaml


def require_int(name, value, minimum=1):
    """
    Check that an SPI parameter is an integer of at least ``minimum``.

    Used for every parameter that counts samples (histories, delays,
    neighbour counts, lag windows). Unlike ``int(value)``, it refuses a
    ``bool``, a float such as 2.7 (rather than truncating it), and NaN or Inf
    with a message about the parameter.

    Parameters
    ----------
    name : str
        Name of the parameter, for the error message.
    value : int
        The value to check: a Python or numpy integer, not a ``bool``.
    minimum : int, optional
        Smallest allowed value. Default: 1.

    Returns
    -------
    int
        ``value`` as a Python ``int``. A ``TypeError`` is raised for a
        non-integer and a ``ValueError`` for a value below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer >= {minimum}, got {value!r} "
            f"({type(value).__name__})."
        )
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return int(value)


def require_positive_float(name, value):
    """
    Check that an SPI parameter is a strictly positive, finite real number.

    The companion of :func:`require_int` for continuous parameters such as
    kernel bandwidths. It refuses a ``bool``, a string, NaN and Inf.

    Parameters
    ----------
    name : str
        Name of the parameter, for the error message.
    value : float
        The value to check.

    Returns
    -------
    float
        ``value`` as a ``float``. A ``TypeError`` is raised for a value that
        is not a real number and a ``ValueError`` for one that is not finite
        and positive.
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a positive real number, got {value!r} "
            f"({type(value).__name__})."
        )
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be a positive real number, got {value!r} "
            f"({type(value).__name__})."
        ) from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return value


# Relative tolerance at the Nyquist edge, as in `spectral._closed_band`, so a
# band edge equal to ``fs / 2`` up to floating-point round-off is accepted.
_NYQUIST_RTOL = 1e-9


def require_band(fs, fmin, fmax, strict=False):
    """
    Check a frequency band ``[fmin, fmax]`` given in the units of ``fs``.

    The rule the multitaper, spectral GC and wavelet SPIs apply at
    construction: ``fs`` is a positive finite number, and
    ``0 <= fmin <= fmax <= fs / 2``, with ``fmin < fmax`` when ``strict``.
    The Nyquist edge is compared with a relative tolerance of 1e-9. At the
    default ``fs = 1`` the band is in cycles per sample.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    fmin : float
        Lower edge of the band.
    fmax : float
        Upper edge of the band.
    strict : bool, optional
        Require ``fmin < fmax``, for an estimator that needs a band of
        positive width. Default: False.

    Returns
    -------
    tuple
        ``(fs, fmin, fmax)`` as given, so that identifiers keep their
        formatting. A ``TypeError`` is raised for a value that is not a real
        number and a ``ValueError`` for one outside the rule.
    """
    require_positive_float("fs", fs)
    nyquist = fs / 2
    for name, value in (("fmin", fmin), ("fmax", fmax)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"{name} must be a real number, got {value!r} "
                f"({type(value).__name__}).")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        if value < 0:
            raise ValueError(
                f"{name} must be in [0, fs / 2] = [0, {fmt_param(nyquist)}], "
                f"got {value!r}.")
        if value > nyquist and not np.isclose(value, nyquist,
                                              rtol=_NYQUIST_RTOL, atol=0):
            raise ValueError(
                f"{name} must be in [0, fs / 2] = [0, {fmt_param(nyquist)}], "
                f"got {value!r}; the band is in the units of fs, which "
                f"defaults to 1 (cycles per sample), so a band in Hz needs fs "
                f"set to the sampling rate.")
    if fmin > fmax or (strict and fmin == fmax):
        relation = "<" if strict else "<="
        raise ValueError(
            f"fmin must be {relation} fmax, got fmin={fmin!r} and fmax={fmax!r}.")
    return fs, fmin, fmax


def empty_band_error(owner, fmin, fmax, detail):
    """
    The ``NotAppropriate`` for a band that holds no frequency of a series' grid.

    The grid depends on the series' length, so the band cannot support an
    estimate for this input; a band that is invalid for any series is refused
    at construction instead (``require_band``).

    Parameters
    ----------
    owner : str
        Name of the SPI class.
    fmin, fmax : float
        Edges of the band.
    detail : str
        What the band holds and why, e.g. ``"no frequency of the series'
        grid (grid step 0.001)"``.

    Returns
    -------
    pyspi.base.NotAppropriate
        The error, for the caller to raise.
    """
    # Imported here: pyspi.base imports pyspi.data, which imports this module.
    from pyspi.base import NotAppropriate
    return NotAppropriate(
        f"{owner}: the band [{fmin}, {fmax}] holds {detail}; widen it.")


# Two series are treated as linearly dependent when the smallest eigenvalue of
# their correlation matrix is at most this fraction of the largest, which is
# |r| within about 2e-12 of 1: the tolerance the VAR SPIs apply to a residual
# covariance.
DEPENDENT_PAIR_RCOND = 1e-12


def dependent_pair_error(owner, i, j, cov, consequence,
                         of="the correlation matrix of the two series"):
    """
    The ``NotAppropriate`` for a pair of linearly dependent series, or None.

    The criterion is the ratio of the smallest to the largest eigenvalue of
    the correlation matrix implied by the pair's ``2 x 2`` covariance,
    ``(1 - |r|) / (1 + |r|)``: the pair is refused when it is at most
    ``DEPENDENT_PAIR_RCOND``. The ratio does not depend on the scale of either
    series. A series with no variance makes the ratio 0. A covariance that is
    not finite is not refused here.

    Parameters
    ----------
    owner : str
        Name of the SPI class, or of the method, for the message.
    i, j : int
        The processes of the pair, for the message.
    cov : array_like
        The pair's ``2 x 2`` covariance matrix.
    consequence : str
        What cannot be computed for such a pair, completing the message, e.g.
        ``"the covariance the VAR fit inverts is singular"``.
    of : str, optional
        What ``cov`` is the covariance of, for the message.
        Default: ``"the correlation matrix of the two series"``.

    Returns
    -------
    pyspi.base.NotAppropriate or None
        The error, for the caller to raise, when the pair is linearly
        dependent; otherwise None.
    """
    cov = np.asarray(cov, dtype=float)
    a, b, c = cov[0, 0], cov[1, 1], cov[0, 1]
    if not np.isfinite([a, b, c]).all():
        return None
    if a > 0 and b > 0:
        r = min(abs(c) / np.sqrt(a * b), 1.0)
        ratio = (1.0 - r) / (1.0 + r)
    else:
        ratio = 0.0
    if ratio > DEPENDENT_PAIR_RCOND:
        return None
    # Imported here: pyspi.base imports pyspi.data, which imports this module.
    from pyspi.base import NotAppropriate
    return NotAppropriate(
        f"{owner}: processes {i} and {j} are linearly dependent (smallest/"
        f"largest eigenvalue of {of} {ratio:.2g}, at most "
        f"{DEPENDENT_PAIR_RCOND:g}), so {consequence}.")


def acf(x, mode='positive'):
    """
    Sample autocorrelation function of a series, computed by FFT.

    The series is demeaned and divided by its standard deviation, and the
    autocorrelation is normalised to 1 at lag 0 (the biased estimator).

    Parameters
    ----------
    x : numpy.ndarray
        The series; singleton dimensions are squeezed.
    mode : str, optional
        ``'positive'`` for lags ``0, ..., N - 1``; any other value for the
        symmetric function at lags ``-(N - 1), ..., N - 1``.
        Default: ``'positive'``.

    Returns
    -------
    numpy.ndarray
        The autocorrelation at each lag, of length ``N`` or ``2N - 1``; all
        zeros for a constant series.
    """
    if x.ndim > 1:
        x = np.squeeze(x)

    x = x - x.mean()
    s = x.std()
    if s == 0:
        n = len(x)
        return np.zeros(n) if mode == 'positive' else np.zeros(2 * n - 1)
    x = x / s

    n = len(x)
    fft_size = 2 * n
    X = np.fft.rfft(x, n=fft_size)
    acf_full = np.fft.irfft(X * np.conj(X), n=fft_size)[:n]
    acf_full = acf_full / acf_full[0]  # acf[0] = 1

    if mode == 'positive':
        return acf_full
    # full symmetric ACF
    return np.concatenate([acf_full[::-1], acf_full[1:]])

def swap_chars(s, i_1, i_2):
    """
    Swap two characters of a string.

    Parameters
    ----------
    s : str
        The string.
    i_1, i_2 : int
        Positions of the two characters.

    Returns
    -------
    str
        ``s`` with the characters at ``i_1`` and ``i_2`` exchanged.

    Examples
    --------
    >>> swap_chars('heLlotHere', 2, 6)
    'heHlotLere'
    """
    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    return ''.join([s[0:i_1], s[i_2], s[i_1+1:i_2], s[i_1], s[i_2+1:]])

def convert_mdf_to_ddf(df):
    """
    Reshape SPI-SPI correlations from one block per dataset to one column per dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The correlations with a ``(Dataset, SPI-1)`` row index and ``SPI-2``
        columns, as ``CalculatorFrame.get_correlation_df`` returns them.
        Dataset names must be unique.

    Returns
    -------
    pandas.DataFrame
        One row per ``(SPI-1, SPI-2)`` pair and one column per dataset.
    """
    # pivot_table would average two datasets that share a name.
    dup = df.index[df.index.duplicated()]
    if len(dup):
        names = sorted({str(k[0]) for k in dup})
        raise ValueError(
            f"Dataset names must be unique; duplicated: {', '.join(names)}. "
            "Give each dataset its own name (e.g. via `names=` or `calc.name`)."
        )
    ddf = pd.pivot_table(data=df.stack(future_stack=True).reset_index(),index='Dataset',columns=['SPI-1', 'SPI-2'],dropna=False).T.droplevel(0)
    return ddf

def filter_spis(keywords, output_name = None, configfile= None):
    """
    Write a config of the SPIs that carry every one of the given labels.

    Each SPI of the source config is instantiated and kept if its labels
    (after the config's own labels are merged in) include all ``keywords``.
    The reduced config is written as ``<output_name>.yaml`` in the current
    directory, and a summary is printed.

    Parameters
    ----------
    keywords : list of str
        Labels an SPI must all carry, such as ``['linear', 'undirected']``.
    output_name : str or None, optional
        Name of the output file, without ``.yaml``. Default: None, a random
        name ``config_<hex>``.
    configfile : str or None, optional
        The source config: a bundled config name or a path to a YAML file.
        Default: None, the bundled ``full`` config.

    Returns
    -------
    None
        A ``ValueError`` is raised if ``keywords`` is not a non-empty list of
        strings, includes ``'unsigned'`` (not a label) or no SPI matches,
        and a ``FileNotFoundError`` if the config is not found.
    """
    # handle invalid keyword input
    if not keywords:
        raise ValueError("At least one keyword must be provided.")
    if not all(isinstance(keyword, str) for keyword in keywords):
        raise ValueError("All keywords must be strings.")
    if not isinstance(keywords, list):
        raise ValueError("Keywords must be provided as a list of strings.")
    if "unsigned" in keywords:
        raise ValueError(
            "'unsigned' is not a label. An SPI is labelled 'signed' when zero "
            "means no relationship and the sign says which of two opposite "
            "relationships holds. For the one-ended strengths of dependence, "
            "filter by 'strength' and leave out the SPIs labelled 'signed'.")

    # Default to the full bundled config; otherwise accept a bundled name or path.
    from pyspi.calculator import resolve_config, load_config_yaml
    if configfile is None:
        configfile = resolve_config("full")
        source_file_info = f"Default bundled config '{configfile}' was used as the source file."
    else:
        configfile = resolve_config(configfile)
        source_file_info = f"User-specified config file '{configfile}' was used as the source file."

    # load in user-specified yaml
    try:
        yf = load_config_yaml(configfile)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{configfile}' not found.")
    except Exception as e:
        # handle all other exceptions
        raise IOError(f"An error occurred while trying to read '{configfile}': {e}")

    # Filter on the labels of each instantiated SPI, not those in the YAML:
    # some traits are set per variant in __init__ (such as 'directed' for
    # cointegration's aeg method).
    import importlib
    from pyspi.calculator import (
        _merge_spi_labels,
        _split_config_params,
        _expand_lagged_correlation_configs,
    )

    filtered_subset = {}
    spis_found = 0
    keywords = set(keywords)

    for module_name in yf:
        module = importlib.import_module(module_name, "pyspi")
        module_spis = {}
        for spi_name, entry in (yf[module_name] or {}).items():
            entry = dict(entry or {})
            family_labels = entry.get("labels")
            configs = entry.get("configs")
            # Same expansion the loader applies, so max_tau reaches the
            # constructor as the tau values it stands for.
            if spi_name == "LaggedCorrelation" and configs is not None:
                configs = _expand_lagged_correlation_configs(configs)

            if configs is None:
                spi = getattr(module, spi_name)()
                _merge_spi_labels(spi, family_labels)
                if keywords <= set(spi.labels or []):
                    module_spis[spi_name] = entry
                    spis_found += 1
                continue

            kept = []
            for params in configs:
                clean, config_labels = _split_config_params(params)
                spi = getattr(module, spi_name)(**clean)
                _merge_spi_labels(spi, family_labels, config_labels)
                if keywords <= set(spi.labels or []):
                    kept.append(params)
            if kept:
                kept_entry = dict(entry)
                kept_entry["configs"] = kept
                module_spis[spi_name] = kept_entry
                spis_found += len(kept)

        if module_spis:
            filtered_subset[module_name] = module_spis

    # check that > 0 SPIs found
    if spis_found == 0:
        raise ValueError(f"0 SPIs were found with the specific keywords: {keywords}.")

    # construct output file path
    if output_name is None:
        # use a unique name
        output_name = "config_" + os.urandom(4).hex()

    output_file = os.path.join(os.getcwd(), f"{output_name}.yaml")

    # write to YAML
    with open(output_file, "w", encoding="utf-8") as outfile:
        yaml.dump(filtered_subset, outfile, default_flow_style=False, sort_keys=False)

    # output relevant information
    print(f"""\nOperation Summary:
-----------------
- {source_file_info}
- Total SPIs Matched: {spis_found} SPI(s) were found with the specific keywords: {keywords}.
- New File Created: A YAML file named `{output_name}.yaml` has been saved in the current directory: `{output_file}'
- Next Steps: To utilise the filtered set of SPIs, please initialise a new Calculator instance with the following command:
`Calculator(config='{output_file}')`
""")

def _print_timing_summary(calc, n_slowest=5):
    """Print the total compute time and the slowest SPIs of ``calc``."""
    timings = {k: v for k, v in calc.timings.items() if v}
    if not timings:
        # Every SPI was restored from a checkpoint, so nothing was timed.
        return
    total = sum(timings.values())
    print(f"Total compute time: {total:.2f}s across {len(timings)} timed SPI(s)")
    slowest = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)[:n_slowest]
    width = max(len(k) for k, _ in slowest)
    print(f"Slowest {len(slowest)}:")
    for key, secs in slowest:
        print(f"  {key:<{width}}  {secs:7.2f}s  ({secs / total * 100:5.1f}%)")
    print("-" * 60)


def inspect_calc_results(calc):
    """
    Print a summary of a computed Calculator's results.

    SPIs are counted as successful, "NaNs" (no good off-diagonal value) or
    "Partial NaNs" (some), from ``calc.quality``. Each SPI listed under the
    last two shows what its other cells are (error, NaN, Inf, -Inf, complex),
    with counts. The total time and the slowest SPIs are printed too.

    Parameters
    ----------
    calc : Calculator
        A Calculator after :meth:`~pyspi.calculator.Calculator.compute`.

    Returns
    -------
    None
    """
    from pyspi.quality import Quality, quality_from_values, quality_summary

    total_num_spis = calc.n_spis
    num_procs = calc.dataset.n_processes
    off_diagonal = ~np.eye(num_procs, dtype=bool)
    quality = getattr(calc, 'quality', None)
    spi_results = dict({'Successful': list(), 'NaNs': list(), 'Partial NaNs': list()})
    breakdown = {}
    all_cells = []
    for key in calc.spis.keys():
        if quality is not None:
            q = quality[key].to_numpy()[off_diagonal]
        else:
            q = quality_from_values(calc.table[key].to_numpy())[1][off_diagonal]
        all_cells.append(q)
        bad = q[q != Quality.GOOD]
        if bad.size:
            breakdown[key] = ", ".join(f"{n} {label}"
                                       for label, n in quality_summary(bad).items())
        elif not q.size:
            # One process: there is no pair, so no value to compute.
            breakdown[key] = "no off-diagonal values"
        if bad.size == q.size:
            spi_results['NaNs'].append(key)
        elif bad.size:
            spi_results['Partial NaNs'].append(key)
        else:
            spi_results['Successful'].append(key)

    # print summary
    double_line_60 = "="*60
    single_line_60 = "-"*60
    print("\nSPI Computation Results Summary")
    print(double_line_60)
    print(f"\nTotal number of SPIs attempted: {total_num_spis}")
    print(f"Number of SPIs successfully computed: {len(spi_results['Successful'])} ({len(spi_results['Successful']) / total_num_spis * 100:.2f}%)")
    if all_cells:
        cells = quality_summary(np.concatenate(all_cells))
        print("Off-diagonal values: " + ", ".join(f"{n} {label}" for label, n in cells.items()))
    print(single_line_60)
    _print_timing_summary(calc)
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
            print(f"{i+1}. {spi} ({breakdown[spi]})")
        print(single_line_60 + "\n")
    if spi_results['Partial NaNs']:
        print(f"\n[{len(spi_results['Partial NaNs'])}] SPIs which produced partial NaN outputs:")
        print(single_line_60)
        for i, spi in enumerate(spi_results['Partial NaNs']):
            print(f"{i+1}. {spi} ({breakdown[spi]})")
        print(single_line_60 + "\n")


def fmt_param(x):
    """
    Format a parameter value for an SPI identifier, losslessly.

    Identifiers name table columns and checkpoint files, so two different
    values must never format alike, and two equal ones must always format
    alike. A numpy scalar is formatted as the equal Python number, so
    ``np.float64(0.25)`` gives ``0.25`` and ``np.True_`` gives ``True``. An
    integral float is formatted as the integer, so ``1.0`` gives ``1``, as
    ``1`` does. Any other float is formatted with ``repr``, the shortest
    string that round-trips; anything else with ``str``.

    Parameters
    ----------
    x : object
        The parameter value.

    Returns
    -------
    str
        The formatted value.
    """
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, np.integer):
        x = int(x)
    elif isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return str(x)
        if x.is_integer():
            return str(int(x))
        return repr(x)
    return str(x)

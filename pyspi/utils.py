import numpy as np
from scipy.stats import zscore
import warnings
import pandas as pd
import os
import yaml

def require_int(name, value, minimum=1):
    """Return ``value`` as an ``int``, or say precisely why it is not one.

    Strict on three things that a bare ``int(value) < minimum`` check lets past:

    * ``bool`` is a subclass of ``int``, so ``k_history=True`` would arrive as 1.
    * ``int(2.7)`` truncates, so a parameter the caller plainly meant as
      something else is silently changed rather than rejected.
    * ``float('nan')`` and ``float('inf')`` raise from ``int()`` with a message
      about the conversion rather than about the parameter.

    Shared by every public numeric parameter that indexes samples -- histories,
    delays, neighbour counts, search bounds, lag windows -- so the rule is one
    rule rather than a per-class habit.
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
    """Return ``value`` as a strictly positive finite ``float``.

    The companion to :func:`require_int` for the genuinely continuous
    parameters -- kernel bandwidths, band fractions. Rejects ``bool`` (which
    ``float()`` happily turns into 1.0), NaN and the infinities, and anything
    that is not a real number at all.
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


def acf(x, mode='positive'):
    """Return the autocorrelation function using FFT-based computation.

    O(N log N) via FFT, replacing the original O(N^2) np.correlate approach.
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
    acf_full = acf_full / acf_full[0]  # normalize so acf[0] = 1

    if mode == 'positive':
        return acf_full
    # full symmetric ACF
    return np.concatenate([acf_full[::-1], acf_full[1:]])

def swap_chars(s, i_1, i_2):
    """Swap to characters in a string.

    Example:
        >>> print(swap_chars('heLlotHere', 2, 6))
        'heHlotLere'
    """
    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    return ''.join([s[0:i_1], s[i_2], s[i_1+1:i_2], s[i_1], s[i_2+1:]])

def convert_mdf_to_ddf(df):
    ddf = pd.pivot_table(data=df.stack(future_stack=True).reset_index(),index='Dataset',columns=['SPI-1', 'SPI-2'],dropna=False).T.droplevel(0)
    return ddf

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

    # Default to the full bundled config; otherwise accept a bundled name or path.
    from pyspi.calculator import resolve_config
    if configfile is None:
        configfile = resolve_config("full")
        source_file_info = f"Default bundled config '{configfile}' was used as the source file."
    else:
        configfile = resolve_config(configfile)
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

    # Filter on the labels each SPI *actually* carries once instantiated, not on
    # the family labels written in the YAML. Several traits are set per variant
    # in __init__ -- 'antisymmetric' for phase measures whose band statistic is
    # the mean, 'directed' for cointegration's aeg method -- and are invisible
    # in the raw file. Matching families also selected every config in them,
    # so a family could not be filtered down to the variants that qualified.
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
    with open(output_file, "w") as outfile:
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
    """Print total compute time and the slowest SPIs.

    Per-SPI timings are always available as ``calc.timings``; this surfaces the
    part that actually informs a decision -- which SPIs dominate the run, and
    are therefore what a cheaper ``config=`` drops.
    """
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
            print(f"{i+1}. {spi}")
        print(single_line_60 + "\n")
    if spi_results['Partial NaNs']:
        print(f"\n[{len(spi_results['Partial NaNs'])}] SPIs which produced partial NaN outputs:")
        print(single_line_60)
        for i, spi in enumerate(spi_results['Partial NaNs']):
            print(f"{i+1}. {spi}")
        print(single_line_60 + "\n")


def fmt_param(x):
    """Format a parameter value for use inside an SPI identifier, losslessly.

    Identifiers are the package's primary key: they name table columns, name
    checkpoint files, and are compared when resuming. Formatting floats with
    ``.3g``/``.4g`` made them lossy, so two genuinely different
    parameterisations could render to the same identifier and silently overwrite
    one another at dict insertion.

    ``repr`` of a Python float is the shortest string that round-trips, so it is
    both lossless and stable across platforms. Values that already print
    identically under ``.3g`` are unaffected, which is why no bundled config's
    identifiers change.
    """
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return str(x)
        return repr(x)
    return str(x)

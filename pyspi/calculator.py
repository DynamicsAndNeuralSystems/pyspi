"""The Calculator, which computes every SPI of a config on a dataset, and its results."""
import copy
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
from typing import NamedTuple, Optional
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from . import _parallel
from ._compat import resolve_config_arg, resolve_zscore, translate_calculator_kwargs
from ._logging import configure as _configure_logging, get_logger
from .base import DIRECTEDNESS_TRAITS, ONE_VALUE_PER_PAIR
from .data import Data
from .quality import Quality
from .utils import convert_mdf_to_ddf, inspect_calc_results, require_int

logger = get_logger("pyspi.calculator")


# Labels of the SPIs in a config.

def _as_label_list(labels):
    if labels is None:
        return []
    if isinstance(labels, str):
        return [labels]
    return list(labels)


def _is_module_label(label):
    return (
        isinstance(label, str)
        and len(label) == 3
        and label[0] == "M"
        and (label[1:].isdigit() or label[1:] == "XX")
    )


_CONTRADICTORY_LABELS = (
    ("linear", "nonlinear"),
    ("directed", "undirected"),
    ("signed", "unsigned"),
)


def _merge_spi_labels(spi, family_labels=None, config_labels=None):
    """Set ``spi.labels`` to its own labels plus the config's, without contradictions."""
    labels = list(getattr(spi, "labels", []))
    family_labels = _as_label_list(family_labels)
    config_labels = _as_label_list(config_labels)

    if any(_is_module_label(label) for label in config_labels):
        labels = [label for label in labels if not _is_module_label(label)]
        family_labels = [
            label for label in family_labels if not _is_module_label(label)
        ]

    merged = []
    for label in labels + family_labels + config_labels:
        if label not in merged:
            merged.append(label)

    # The SPI's own structural trait (set by its class and __init__, before
    # the config's labels are merged) overrides any the config declares, so
    # exactly one of DIRECTEDNESS_TRAITS survives and `filter_spis` gives one
    # answer.
    own = [label for label in DIRECTEDNESS_TRAITS if label in labels]
    if own:
        trait = own[0]
        merged = [
            label for label in merged
            if label not in DIRECTEDNESS_TRAITS or label == trait
        ]

    # Likewise for linear/nonlinear: if the class commits to one, the other
    # is dropped, so a config cannot add `linear` next to a class's `nonlinear`.
    own_lin = [label for label in ("linear", "nonlinear") if label in labels]
    if len(own_lin) == 1:
        merged = [
            label for label in merged
            if label not in ("linear", "nonlinear") or label == own_lin[0]
        ]

    # Likewise the SPI's own issigned() decides signed/unsigned. It is not just
    # a label: `Calculator._rmmin` shifts unsigned SPIs and the SPI-SPI
    # correlations take their abs(), which would destroy a lead/lag sign.
    issigned = getattr(spi, "issigned", None)
    if issigned is not None:
        actual = "signed" if issigned() else "unsigned"
        merged = [label for label in merged
                  if label not in ("signed", "unsigned")] + [actual]

    # Whatever the class did not settle must not be left contradictory.
    for pair in _CONTRADICTORY_LABELS:
        if all(label in merged for label in pair):
            raise ValueError(
                f"{type(spi).__name__} ends up with contradictory labels "
                f"{pair[0]!r} and {pair[1]!r}: {merged}"
            )
    spi.labels = merged


def _split_config_params(params):
    """``(constructor parameters, labels)`` of one entry of a config's ``configs`` list."""
    params = dict(params or {})
    config_labels = params.pop("labels", None)
    return params, config_labels

CONFIG_DIR = Path(__file__).parent / "configs"


def bundled_configs():
    """
    Names of the bundled configs.

    Returns
    -------
    list of str
        The values of ``config`` that name a bundled config, sorted.
    """
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def resolve_config(config):
    """
    Resolve a config name or path to the path of its YAML file.

    Parameters
    ----------
    config : str or path-like
        The name of a bundled config (``"full"``, ``"fast"``,
        ``"benchmarked_p90"``, ...) or the path of a YAML file. A value with a
        directory component or a ``.yaml`` or ``.yml`` suffix is a path;
        anything else is looked up in :data:`CONFIG_DIR`.

    Returns
    -------
    str
        Path of the config file. A missing file raises ``FileNotFoundError``
        and an unknown name ``ValueError``.
    """
    text = str(config)
    looks_like_path = (
        os.sep in text
        or (os.altsep is not None and os.altsep in text)
        or text.endswith((".yaml", ".yml"))
    )
    if looks_like_path:
        path = Path(text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return str(path)

    path = CONFIG_DIR / f"{text}.yaml"
    if text == "all":
        raise ValueError(
            "Unknown config 'all'. The config of every SPI is called 'full': "
            "pass config='full'.")
    if not path.is_file():
        raise ValueError(
            f"Unknown config '{text}'. Bundled configs are: "
            f"{', '.join(bundled_configs())}. "
            f"To use your own, pass a path to a .yaml file."
        )
    return str(path)


@functools.lru_cache(maxsize=1)
def _full_config_cache_buckets():
    """``{cache_bucket: {identifiers}}`` for the ``full`` config, built once per session."""
    from collections import defaultdict

    available = defaultdict(set)
    for key, spi in load_spis_from_yaml(resolve_config("full"), quiet=True).items():
        bucket = _parallel.cache_bucket(spi)
        if bucket is not None:
            available[bucket].add(key)
    return available


def warn_partial_cache_buckets(spis):
    """
    Warn when a config keeps only part of an expensive shared-cache group.

    Some SPI families build one expensive cached intermediate on the Data
    object and derive cheap variants from it (``ccm``, ``barycenter``). The
    cache is built as soon as any member runs, so the other members are close
    to free, and keeping only some of them pays the full cost for fewer SPIs.

    Parameters
    ----------
    spis : dict
        ``{identifier: SPI instance}``, as :func:`load_spis_from_yaml`
        returns.
    """
    # Only caches expensive enough for the advice to matter.
    EXPENSIVE = {"ccm", "barycenter"}

    from collections import defaultdict

    # The definition the scheduler buckets on.
    bucket = _parallel.cache_bucket

    # Keyed by identifier, not class: variants of one class share the cache.
    kept = defaultdict(set)
    for key, spi in spis.items():
        b = bucket(spi)
        if b is not None:
            kept[b].add(key)
    if not any(bkey[0] in EXPENSIVE for bkey in kept):
        # Return before building the `full` config, which imports every SPI.
        return
    try:
        available = _full_config_cache_buckets()
    except Exception:  # never let an advisory check break a run
        return
    for bkey, have in sorted(kept.items(), key=lambda kv: str(kv[0])):
        ns = bkey[0]
        if ns not in EXPENSIVE:
            continue
        missing = sorted(available.get(bkey, set()) - have)
        if not missing:
            continue
        shown = ", ".join(missing[:3]) + (f", +{len(missing) - 3} more" if len(missing) > 3 else "")
        logger.warning(
            "Config keeps %d of %d SPIs sharing the '%s' cache. That cache is "
            "built regardless, so the other %d (%s) are close to free -- "
            "keeping a strict subset pays the full cache cost for fewer SPIs.",
            len(have), len(have) + len(missing), ns, len(missing), shown,
        )


# Fields of `run_spec` that say where the run came from rather than what it
# computes, kept out of the content hash `run_digest`: the same config and
# data digest the same from any location. `COMPUTATION_VERSION` is hashed
# separately, so an upgrade that changes no computed value still resumes.
_DIGEST_EXCLUDED_SPEC_FIELDS = frozenset({
    "config", "configfile", "dataset_name",
    "computation_version", "pyspi_version", "dependency_versions",
})


@functools.cache
def _installed_versions():
    """
    pyspi's version and the installed version of each runtime dependency.

    The dependencies are pyspi's declared requirements, extras excluded.
    ``None`` marks a package that is not installed or carries no metadata.
    """
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        requires = importlib.metadata.requires("pyspi") or []
    except importlib.metadata.PackageNotFoundError:
        requires = []
    deps = sorted({
        re.match(r"[A-Za-z0-9_.\-]+", req).group(0)
        for req in requires if "extra ==" not in req
    }, key=str.lower)
    return version("pyspi"), {name: version(name) for name in deps}


# Version of the .npz layout; 0 is a file without the field. Schema 2 adds
# `quality`, the per-cell codes, `pair_errors`, and `directedness`, each SPI's
# symmetry trait; a schema-1 file still loads, with its codes derived from its
# values and `errors` and its traits rebuilt from the config it names.
_NPZ_SCHEMA = 2


class Results(NamedTuple):
    """
    The contents of a results ``.npz``, as :func:`load_results` returns them.

    A field the file does not record is ``None``.

    Attributes
    ----------
    table : pandas.DataFrame
        The values, as :func:`load_table` returns them.
    quality : pandas.DataFrame
        A ``uint8`` DataFrame with the index and columns of ``table``, one
        :class:`pyspi.quality.Quality` code per cell (``Calculator.quality``).
    errors : dict or None
        ``{identifier: message}`` for the SPIs that failed
        (``Calculator.errors``).
    run_spec : dict or None
        The resolved specification of the run (``Calculator.run_spec``).
    pair_errors : dict or None
        The messages of failed pairs (``Calculator.pair_errors``).
    """
    table: pd.DataFrame
    quality: pd.DataFrame
    errors: Optional[dict]
    run_spec: Optional[dict]
    pair_errors: Optional[dict]


def _frame_from_matrices(matrices, spis, procs):
    """
    ``(n_spis, M, M)`` -> the wide ``M x (n_spis * M)`` frame :attr:`Calculator.table` uses.

    The one builder for :attr:`Calculator.table`, :attr:`Calculator.quality`
    and what :func:`load_table` / :func:`load_results` return, so their index
    and columns cannot drift apart. The dtype of ``matrices`` is kept.
    """
    spis, procs = list(spis), list(procs)
    matrices = np.asarray(matrices)
    n, M = len(spis), len(procs)
    if matrices.shape != (n, M, M):
        raise ValueError(f"matrices have shape {matrices.shape}, expected ({n}, {M}, {M})")
    # Row r is [matrices[0][r], matrices[1][r], ...]: the matrices side by side.
    data = matrices.transpose(1, 0, 2).reshape(M, n * M).copy()
    frame = pd.DataFrame(
        data=data,
        columns=pd.MultiIndex.from_product([spis, procs], names=["spi", "process"]),
        index=procs,
    )
    frame.columns.name = "process"
    return frame


def _read_npz(path):
    """Read and validate a ``.npz`` results file; return its parts as a dict."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(
            f"Can only load '.npz' (got '{path.suffix}'). CSV export is one-way; "
            f"re-run the calculation or save as .npz."
        )
    # allow_pickle=False: loading a results table must never execute code. A
    # file that stores names as object arrays fails here.
    with np.load(path, allow_pickle=False) as f:
        missing = {"values", "spis", "processes"} - set(f.files)
        if missing:
            raise ValueError(
                f"{path} is not a pyspi results table (missing {sorted(missing)})."
            )
        schema = int(f["schema"]) if "schema" in f.files else 0
        if schema > _NPZ_SCHEMA:
            raise ValueError(
                f"{path} was written with schema {schema}, but this pyspi "
                f"understands up to {_NPZ_SCHEMA}. Upgrade pyspi."
            )
        values = f["values"]
        quality = f["quality"] if "quality" in f.files else None
        spis = [str(s) for s in f["spis"]]
        procs = [str(p) for p in f["processes"]]
        directedness = (
            [str(t) for t in f["directedness"]] if "directedness" in f.files else None
        )
        # Provenance: which SPIs and pairs failed, and what produced the file.
        meta = {name: str(f[name])
                for name in ("run_spec", "run_digest", "errors", "pair_errors")
                if name in f.files}

    # Check the whole shape, so a malformed file fails here and says so.
    if values.shape != (len(spis), len(procs), len(procs)):
        raise ValueError(
            f"{path} is malformed: values has shape {values.shape}, expected "
            f"({len(spis)}, {len(procs)}, {len(procs)})."
        )
    if quality is not None and quality.shape != values.shape:
        raise ValueError(
            f"{path} is malformed: quality has shape {quality.shape}, expected "
            f"{values.shape}."
        )
    if directedness is not None:
        unknown = sorted(set(directedness) - set(DIRECTEDNESS_TRAITS))
        if len(directedness) != len(spis) or unknown:
            raise ValueError(
                f"{path} is malformed: directedness holds {len(directedness)} "
                f"entries for {len(spis)} SPIs"
                + (f", and unknown traits {unknown}" if unknown else "") + "."
            )
        directedness = dict(zip(spis, directedness))
    parsed = {}
    for name in ("run_spec", "errors", "pair_errors"):
        if name in meta:
            try:
                parsed[name] = json.loads(meta[name])
            except json.JSONDecodeError:
                parsed[name] = meta[name]
    return dict(schema=schema, values=values, quality=quality, spis=spis,
                procs=procs, run_digest=meta.get("run_digest"),
                directedness=directedness, **parsed)


def load_table(path):
    """
    Load the results table that :meth:`Calculator.save` wrote to ``.npz``.

    Only ``.npz`` round-trips; ``.csv`` is a one-way export. For the per-cell
    quality codes, use :func:`load_results`.

    Parameters
    ----------
    path : str or path-like
        Path of the ``.npz`` file.

    Returns
    -------
    pandas.DataFrame
        The table as ``Calculator.table`` holds it: rows are processes,
        columns a ``(spi, process)`` MultiIndex. ``table.attrs`` holds the
        file's ``schema``, ``run_spec``, ``run_digest`` and ``errors``, and
        ``directedness``, ``{identifier: trait}`` with each SPI's symmetry
        trait, which :func:`feature_vector` uses.
    """
    return _table_from_parts(_read_npz(path))


def _table_from_parts(parts):
    """The results table of a read ``.npz``, with its small metadata in ``attrs``."""
    table = _frame_from_matrices(parts["values"], parts["spis"], parts["procs"])
    # Only small values go in `attrs`: pandas deep-copies them on every derived
    # frame and compares them in `concat`, so the quality codes stay out.
    table.attrs["schema"] = parts["schema"]
    for name in ("run_spec", "errors"):
        if name in parts:
            table.attrs[name] = parts[name]
    if parts["run_digest"] is not None:
        table.attrs["run_digest"] = parts["run_digest"]
    if parts["directedness"] is not None:
        table.attrs["directedness"] = parts["directedness"]
    return table


def load_results(path):
    """
    Load everything :meth:`Calculator.save` wrote to ``.npz``.

    A file without quality codes (schema < 2) has them derived from its values
    and ``errors``: a cell is GOOD, NaN or +/-Inf by its value, the diagonal
    is NaN, and every non-finite off-diagonal cell of an SPI recorded as
    failed is ERROR. Such a file holds no pair failures, so those cells read
    as NaN and ``pair_errors`` is ``None``.

    Parameters
    ----------
    path : str or path-like
        Path of the ``.npz`` file.

    Returns
    -------
    Results
        ``Results(table, quality, errors, run_spec, pair_errors)``: ``table``
        as :func:`load_table` returns it, ``quality`` the matching ``uint8``
        DataFrame of :class:`pyspi.quality.Quality` codes, and the rest as
        described in :class:`Results`.
    """
    parts = _read_npz(path)
    table = _table_from_parts(parts)
    as_dict = lambda x: x if isinstance(x, dict) else None
    errors = as_dict(parts.get("errors"))
    quality = parts["quality"]
    if quality is None:
        quality = np.empty(parts["values"].shape, dtype=np.uint8)
        for n, (key, values) in enumerate(zip(parts["spis"], parts["values"])):
            quality[n] = _parallel.derive_quality(values, (errors or {}).get(key))
    quality = _frame_from_matrices(quality.astype(np.uint8, copy=False),
                                   parts["spis"], parts["procs"])
    pair_errors = as_dict(parts.get("pair_errors"))
    if pair_errors is not None:
        pair_errors = {k: [tuple(p) for p in v] for k, v in pair_errors.items()}
    return Results(table=table, quality=quality, errors=errors,
                   run_spec=as_dict(parts.get("run_spec")), pair_errors=pair_errors)


def feature_vector(table, directedness=None, include_self=False):
    """
    Flatten a results table to one value per independent pairwise feature.

    :attr:`Calculator.table` stores every SPI as a full ``M x M`` matrix. For
    an ``undirected`` SPI the lower triangle is a copy of the upper one and
    for an ``antisymmetric`` SPI it is its negation, so each holds
    ``M(M-1)/2`` independent values. For these two traits the vector keeps the
    pairs ``(i, j)`` with ``i < j`` (in process order); ``directed`` and
    ``asymmetric`` SPIs keep both orientations. Rows are sources, so a
    positive antisymmetric value at ``(i, j)`` means process ``i`` leads
    process ``j``.

    Parameters
    ----------
    table : pandas.DataFrame
        :attr:`Calculator.table`, or the frame :func:`load_table` returns.
    directedness : dict, optional
        ``{identifier: trait}``, with the trait one of ``'undirected'``,
        ``'antisymmetric'``, ``'directed'`` or ``'asymmetric'``. If None, the
        traits :meth:`Calculator.save` recorded are used, which
        :func:`load_table` puts in ``table.attrs["directedness"]``; for a
        file that does not record them, they are rebuilt from the config
        named in ``table.attrs["run_spec"]``, using each SPI's
        :meth:`~pyspi.base.Directed.directedness`. An SPI of the table with
        no trait raises ``ValueError``. Default: None.
    include_self : bool, optional
        Also keep the self-pairs ``(i, i)``, one per process and SPI. The
        diagonal is NaN for pairwise SPIs and a constant for the few
        multivariate SPIs that fill it. Default: False.

    Returns
    -------
    pandas.Series
        Named ``"value"`` and indexed by ``(spi, source, target)``, in table
        column order and then row-major pair order, so vectors from datasets
        with the same processes and config line up element for element. NaNs
        are kept so the length is fixed; a feature is NaN wherever its cell in
        :attr:`Calculator.quality` is not ``GOOD``, and that code says why.
    """
    source = ""
    if directedness is None:
        directedness = getattr(table, "attrs", {}).get("directedness")
    if directedness is None:
        directedness = _directedness_from_table(table)
        source = (f" The file records no traits, and the config it names, "
                  f"{table.attrs['run_spec']['config']!r}, does not hold them.")
    spis = list(table.columns.unique(level="spi"))
    missing = [spi for spi in spis if spi not in directedness]
    if missing:
        shown = ", ".join(missing[:5]) + (f", +{len(missing) - 5} more"
                                          if len(missing) > 5 else "")
        raise ValueError(
            f"No symmetry trait for {len(missing)} SPI(s) of the table "
            f"({shown}), so it is unknown whether each holds one value per "
            f"pair or two.{source} Pass directedness={{identifier: trait}} "
            f"with a trait for every SPI."
        )
    procs = list(table.index)
    M = len(procs)
    if list(table.columns.unique(level="process")) != procs:
        raise ValueError(
            "table columns and rows name different processes; "
            "feature_vector needs the square (spi, process) layout "
            "Calculator.table and load_table produce."
        )
    upper = np.triu(np.ones((M, M), dtype=bool), k=1)
    offdiag = ~np.eye(M, dtype=bool)
    diag = np.eye(M, dtype=bool)

    keys, sources, targets, values = [], [], [], []
    src_idx, tgt_idx = np.indices((M, M))
    for spi in spis:
        trait = directedness[spi]
        if trait not in DIRECTEDNESS_TRAITS:
            raise ValueError(
                f"Unknown directedness {trait!r} for SPI {spi!r}; expected one "
                f"of {list(DIRECTEDNESS_TRAITS)}."
            )
        mask = upper if trait in ONE_VALUE_PER_PAIR else offdiag
        if include_self:
            mask = mask | diag
        A = table[spi].to_numpy(dtype=float)
        if A.shape != (M, M):
            raise ValueError(
                f"SPI {spi!r} occupies {A.shape[1]} columns, expected {M}."
            )
        n = int(mask.sum())
        keys.extend([spi] * n)
        sources.extend(np.asarray(procs, dtype=object)[src_idx[mask]])
        targets.extend(np.asarray(procs, dtype=object)[tgt_idx[mask]])
        values.append(A[mask])

    index = pd.MultiIndex.from_arrays(
        [keys, sources, targets], names=["spi", "source", "target"]
    )
    data = np.concatenate(values) if values else np.empty(0)
    return pd.Series(data, index=index, name="value")


def _directedness_from_table(table):
    """
    Recover ``{identifier: trait}`` for a table from a file that does not record it.

    ``table.attrs["run_spec"]["config"]`` names the bundled config or YAML
    path the SPIs were built from; they are instantiated again to read their
    traits. Raises if the config is missing or does not resolve, rather than
    falling back to both orientations for every SPI.
    """
    spec = table.attrs.get("run_spec") if hasattr(table, "attrs") else None
    config = spec.get("config") if isinstance(spec, dict) else None
    if config is None:
        raise ValueError(
            "This table does not record the config it was computed with, so "
            "the symmetry of each SPI is unknown. Pass directedness= "
            "{identifier: trait}, e.g. {k: s.directedness() for k, s in "
            "Calculator(config=...).spis.items()}."
        )
    try:
        spis = load_spis_from_yaml(resolve_config(config), quiet=True)
    except Exception as err:
        raise ValueError(
            f"Could not rebuild the SPIs from config {config!r} recorded in "
            f"the table ({type(err).__name__}: {err}). Pass directedness= "
            f"explicitly."
        ) from err
    return {k: spi.directedness() for k, spi in spis.items()}


class _ConfigLoader(yaml.FullLoader):
    """
    FullLoader that refuses a key repeated within one mapping.

    PyYAML keeps the last of two equal keys, so a module named twice in a
    config would silently lose its first block of SPIs.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_CONFIG_ENTRY_KEYS = ("labels", "configs", "dependencies")


def load_config_yaml(configfile):
    """
    Read an SPI config, refusing malformed ones.

    A key other than ``labels``, ``configs`` or ``dependencies`` in a class
    entry is refused rather than ignored, since a misspelt ``config:`` would
    build the class with its default parameters. A key repeated within one
    mapping is refused too. A bare ``Class:`` line is an entry with no keys.

    Parameters
    ----------
    configfile : str or path-like
        Path of the YAML file.

    Returns
    -------
    dict
        ``{module: {class name: entry}}``, each entry a dict.
    """
    with open(configfile) as f:
        yf = yaml.load(f, Loader=_ConfigLoader)
    if not isinstance(yf, dict):
        raise ValueError(f"{configfile}: expected a mapping of modules to SPI classes.")
    out = {}
    for module_name, module_spis in yf.items():
        module_spis = module_spis or {}
        if not isinstance(module_spis, dict):
            raise ValueError(f"{configfile}: {module_name} must map class names to entries.")
        out[module_name] = {}
        for fcn, entry in module_spis.items():
            entry = entry or {}
            if not isinstance(entry, dict):
                raise ValueError(f"{configfile}: {module_name}.{fcn} must be a mapping.")
            unknown = sorted(set(entry) - set(_CONFIG_ENTRY_KEYS))
            if unknown:
                raise ValueError(
                    f"{configfile}: {module_name}.{fcn} has unknown key(s) "
                    f"{unknown}; allowed: {list(_CONFIG_ENTRY_KEYS)}.")
            out[module_name][fcn] = entry
    return out


def load_spis_from_yaml(configfile, quiet=False):
    """
    Instantiate every SPI of a config.

    Used by :class:`Calculator` and by the parallel workers. Progress is
    logged by the ``pyspi.calculator`` logger at INFO level. Two SPIs whose
    identifiers are equal, or equal ignoring letter case, raise ``ValueError``.

    Parameters
    ----------
    configfile : str or path-like
        Path of the YAML file.
    quiet : bool, optional
        Log nothing. Default: False.

    Returns
    -------
    dict
        ``{identifier: SPI instance}``, in config order.
    """
    spis = {}
    log = (lambda *a, **k: None) if quiet else logger.info
    log("Loading configuration file: %s", configfile)
    yf = load_config_yaml(configfile)
    for module_name, module_spis in yf.items():
        log("Importing module %s", module_name)
        module = importlib.import_module(module_name, __package__)
        for fcn, entry in module_spis.items():
            family_labels = entry.get("labels")
            configs = entry.get("configs")
            if fcn == "LaggedCorrelation" and configs is not None:
                configs = _expand_lagged_correlation_configs(configs)
            if configs is None:
                spi = getattr(module, fcn)()
                _merge_spi_labels(spi, family_labels)
                _insert_spi(spis, spi, module_name, fcn, None)
                log('[%d] %s.%s(x,y) -> "%s"', len(spis), module_name, fcn, spi.identifier)
                continue
            for params in configs:
                params, config_labels = _split_config_params(params)
                spi = getattr(module, fcn)(**params)
                _merge_spi_labels(spi, family_labels, config_labels)
                _insert_spi(spis, spi, module_name, fcn, params)
                log('[%d] %s.%s(x,y,%s) -> "%s"', len(spis), module_name, fcn, params, spi.identifier)
    return spis


def _insert_spi(spis, spi, module_name, fcn, params):
    """
    Insert an SPI, rejecting an identifier already in ``spis``, or one that
    differs from an identifier in ``spis`` only in letter case.

    The identifier names the table column and the checkpoint files, and those
    files must stay distinct on case-insensitive file systems (the macOS and
    Windows defaults).
    """
    source = f"{module_name}.{fcn}{f'({params})' if params else ''}"
    existing = spis.get(spi.identifier)
    if existing is not None:
        raise ValueError(
            f"Duplicate SPI identifier {spi.identifier!r}: "
            f"{type(existing).__name__} and {source} both produce it. "
            "Two configs of the same class must differ in a parameter that "
            "reaches the identifier."
        )
    folded = spi.identifier.casefold()
    for other in spis:
        if other.casefold() == folded:
            raise ValueError(
                f"SPI identifiers {other!r} ({type(spis[other]).__name__}) and "
                f"{spi.identifier!r} ({source}) differ only in letter case. "
                "Each SPI's checkpoint files are named after its identifier, "
                "and on a case-insensitive file system (the macOS and Windows "
                "defaults) the two would share one file."
            )
    spis[spi.identifier] = spi


def _expand_lagged_correlation_configs(configs):
    """Expand each ``max_tau: K`` entry of LaggedCorrelation into ``tau = 1, ..., K``."""
    expanded = []
    for params in configs or []:
        if "max_tau" in params:
            if "tau" in params:
                raise ValueError("LaggedCorrelation config cannot set both tau and max_tau.")
            max_tau = require_int("max_tau", params["max_tau"], minimum=1)
            base = {key: value for key, value in params.items() if key != "max_tau"}
            for tau in range(1, max_tau + 1):
                entry = dict(base)
                entry["tau"] = tau
                expanded.append(entry)
        else:
            expanded.append(dict(params))
    return expanded


class Calculator:
    """
    Compute every SPI of a config on a multivariate time series.

    The SPIs are instantiated from the config when the Calculator is built;
    :meth:`compute` fills :attr:`table`, one ``M x M`` matrix per SPI with
    row = source and column = target, and :attr:`quality`, which says why any
    value is NaN. :meth:`save` writes both to a file.

    Every argument after ``dataset`` is keyword-only.

    Parameters
    ----------
    dataset : Data, array-like or None, optional
        The ``M`` processes of ``T`` observations, as a :class:`~pyspi.data.Data`
        (used as it is preprocessed) or anything ``Data`` reads (an array, a
        DataFrame or a file path, laid out as ``dim_order`` says), which is
        detrended and z-scored as set here. Default: None; load one later
        with :meth:`load_dataset`.
    name : str or None, optional
        Name of the calculator, shown in progress output and used to tell
        datasets apart in a :class:`CalculatorFrame`. Default: None.
    labels : list or None, optional
        Labels of the dataset, for grouping it with :meth:`set_group`.
        Default: None.
    config : str or path-like or None, optional
        Which SPIs to compute: the name of a bundled config or the path of a
        YAML file. Default: None, which means ``"full"``. The bundled configs
        are:

        - ``"full"``: every SPI.
        - ``"fast"``: all but the slowest SPIs.
        - ``"sonnet"``: 14 representative SPIs, one per module (M01-M14);
          M07 has no bundled SPI, so M14 has two.
        - ``"fabfour"``: 4 SPIs: covariance, Spearman's correlation, directed
          information and power-envelope correlation.
        - ``"benchmarked_p80"``, ``"_p90"``, ``"_p95"``, ``"_p99"``: the
          SPIs below the 80th, 90th, 95th or 99th percentile of compute cost
          among those timed in pyspi's compute-cost benchmark; SPIs not
          timed there are in none of them.
    detrend : bool, optional
        Remove the least-squares line from each process. Default: False.
    zscore : bool or None, optional
        z-score each process along time. This is per process, not over the
        whole dataset, so it removes each process's units without letting
        the other processes influence any pairwise statistic.
        Default: None, which means True.
    verbose : bool, optional
        Log progress at INFO level and print a summary after :meth:`compute`.
        Default: True.
    dim_order : str or None, optional
        How ``dataset`` is laid out, as for :class:`~pyspi.data.Data`:
        ``'ps'`` for an ``(M, T)`` array, one row per process, or ``'sp'``
        for a ``(T, M)`` array or a DataFrame with one column per process. A
        DataFrame's row (``'ps'``) or column (``'sp'``) labels name the
        processes. None reads ``'ps'`` and warns if that gives more processes
        than observations; an explicit value never warns. Ignored for a
        ``Data``. Default: None.
    subset, configfile, normalise : optional
        Deprecated spellings of ``config`` (``subset="all"`` is
        ``config="full"``), ``config`` (a path) and ``zscore``, removed in
        pyspi 4.0. Each emits a ``DeprecationWarning`` and cannot be combined
        with the argument that replaces it. Default: None.

    Examples
    --------
    >>> import numpy as np
    >>> from pyspi import Calculator
    >>> calc = Calculator(dataset=np.random.randn(5, 500), config="fast")
    >>> calc.compute()
    >>> calc.table  # one 5 x 5 block of columns per SPI
    """
    def __init__(
        self, dataset=None, *, name=None, labels=None, config=None,
        detrend=False, zscore=None, verbose=True, dim_order=None,
        subset=None, configfile=None, normalise=None,
    ):
        # Deprecated spellings (subset, configfile, normalise).
        config = resolve_config_arg(config, subset, configfile)
        zscore = resolve_zscore(zscore, normalise)
        self._spis = {}
        self._zscore = zscore
        self._detrend = detrend
        self._dim_order = dim_order
        self._timings = {}
        self._errors = {}
        self._pair_errors = {}
        self._verbose = verbose

        # verbose maps to a process-global pyspi logger level (INFO vs WARNING).
        _configure_logging(verbose)

        configfile = resolve_config(config)

        self._configfile = configfile
        # The bytes the SPIs are built from, in case the file changes later.
        try:
            self._config_bytes = Path(configfile).read_bytes()
        except OSError:
            self._config_bytes = b"<configfile unreadable>"
        self._config = config
        self._spis = load_spis_from_yaml(configfile)

        self._name = name
        self._labels = labels

        logger.info("%d SPI(s) were successfully initialised.", len(self.spis))
        # Bundled configs are chosen deliberately, so only user configs are
        # checked.
        if Path(configfile).parent != CONFIG_DIR:
            warn_partial_cache_buckets(self._spis)

        if dataset is not None:
            self.load_dataset(dataset)

    @property
    def spis(self):
        """``{identifier: SPI instance}`` for every SPI of the config."""
        return self._spis

    @spis.setter
    def spis(self, s):
        raise Exception("Do not set this property externally.")

    @property
    def errors(self):
        """
        ``{identifier: "ExcType: message"}`` for every SPI that failed.

        A failed SPI still occupies its column in :attr:`table`, filled with
        NaN. Consult this before interpreting a table: a NaN column is
        otherwise indistinguishable from a legitimately undefined statistic.
        :attr:`quality` says the same per cell.

        An SPI whose every off-diagonal value is undefined (NaN, Inf, ...) is
        listed too, with ``"ValueError: SPI returned no finite off-diagonal
        values"``, although its cells keep the codes of what it returned.
        Failures of single pairs are in :attr:`pair_errors`, not here.
        """
        return dict(self._errors)

    @property
    def pair_errors(self):
        """
        ``{identifier: [(source, target, "ExcType: message"), ...]}`` for failed pairs.

        A pair whose computation raised inside an SPI that otherwise succeeded
        (see :func:`pyspi.base.fill_pairwise`) is NaN in :attr:`table` and
        ``Quality.ERROR`` in :attr:`quality`; this keeps its message.
        ``source`` and ``target`` are process names: the row and column of the
        cell. SPIs with no failed pair are absent.
        """
        return {k: list(v) for k, v in self._pair_errors.items()}

    @property
    def run_digest(self):
        """
        Content hash of the resolved run: spec plus the input data itself.

        This is what a checkpoint is bound to: a checkpoint resumes only into a
        run with the same dataset, preprocessing, config, process order and
        computation version, not merely the same SPI identifiers and width.

        The dataset bytes are hashed, not just its shape and name: two datasets
        of the same width with the same name are exactly the case that needs
        separating.
        """
        # Location-only fields are excluded; the config's contents are hashed.
        spec = {k: v for k, v in self.run_spec.items()
                if k not in _DIGEST_EXCLUDED_SPEC_FIELDS}
        h = hashlib.sha256()
        h.update(json.dumps(spec, sort_keys=True, default=str).encode())
        # The algorithm, not only its inputs: COMPUTATION_VERSION changes
        # whenever a change alters computed values.
        h.update(_parallel.COMPUTATION_VERSION.encode())
        # The config's contents, so an edited config gives a different digest.
        h.update(self._config_bytes)
        dataset = getattr(self, "_dataset", None)
        if dataset is not None:
            arr = np.ascontiguousarray(dataset.to_numpy())
            # Native dtype: coercing to float64 first would make distinct
            # large-integer datasets collide.
            h.update(f"{arr.dtype.str}{arr.shape}".encode())
            h.update(arr.tobytes())
        return h.hexdigest()

    @property
    def run_spec(self):
        """
        The resolved specification of what this Calculator computes.

        The config it resolved to, the preprocessing applied, the dataset's
        shape and process names, the SPIs instantiated, and the versions of
        pyspi, its computations and its dependencies. Saved with the results,
        to tie them to the run that produced them.
        """
        dataset = getattr(self, "_dataset", None)
        return {
            "config": str(self._config),
            "configfile": str(self._configfile),
            # The dataset's own flags: a Data passed in keeps its preprocessing.
            "zscore": bool(dataset.zscore) if dataset is not None else bool(self._zscore),
            "detrend": bool(dataset.detrend) if dataset is not None else bool(self._detrend),
            "n_processes": int(dataset.n_processes) if dataset is not None else None,
            "n_observations": (
                int(dataset.n_observations) if dataset is not None else None
            ),
            "procnames": list(dataset.procnames) if dataset is not None else None,
            "dataset_name": dataset.name if dataset is not None else None,
            "spi_identifiers": sorted(self._spis),
            "computation_version": _parallel.COMPUTATION_VERSION,
            "pyspi_version": _installed_versions()[0],
            "dependency_versions": dict(_installed_versions()[1]),
        }

    @property
    def n_spis(self):
        """Number of SPIs in the calculator."""
        return len(self._spis)

    @property
    def dataset(self):
        """The loaded dataset, a :class:`~pyspi.data.Data`."""
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
        """Labels of the dataset, as used by :meth:`set_group`."""
        return self._labels

    @labels.setter
    def labels(self, ls):
        self._labels = ls

    @property
    def timings(self):
        """Per-SPI wall-clock times (seconds) from the last compute() call."""
        return dict(self._timings)

    @property
    def table(self):
        """
        The results table.

        A DataFrame with processes as rows and a ``(spi, process)`` column
        MultiIndex, so ``table[identifier]`` is that SPI's ``M x M`` matrix,
        row = source and column = target. NaN until :meth:`compute` has run.
        :meth:`feature_vector` flattens it to one value per independent pair.
        """
        return self._table

    @table.setter
    def table(self, a):
        raise Exception(
            "Do not set this property externally. Use the compute() method."
        )

    @property
    def quality(self):
        """
        Per-cell quality codes: a ``uint8`` DataFrame shaped like :attr:`table`.

        Each cell holds a :class:`pyspi.quality.Quality` code. ``GOOD`` (0)
        means the value in :attr:`table` is a real, finite number; every other
        code means the value is NaN and says why: ``ERROR`` (the SPI, or that
        pair, raised), ``NAN`` (the SPI returned NaN: undefined or not
        appropriate here; also every diagonal cell), ``INF`` / ``NEG_INF``,
        ``COMPLEX``, or ``NOT_COMPUTED`` (255) before :meth:`compute` reaches
        the SPI. ``pyspi.quality.quality_labels`` turns the codes into text.
        """
        return self._quality

    @quality.setter
    def quality(self, q):
        raise Exception(
            "Do not set this property externally. Use the compute() method."
        )

    def __repr__(self):
        ds = getattr(self, "_dataset", None)
        shape = f"{ds.n_processes}x{ds.n_observations}" if ds is not None else "no dataset"
        done = len(self._timings)
        state = "not computed" if not done else f"{done}/{self.n_spis} computed"
        failed = f", {len(self._errors)} failed" if self._errors else ""
        return f"<Calculator config={self._config!r} {shape} {state}{failed}>"

    def _repr_html_(self):
        """A table of :meth:`summary`, shown by Jupyter."""
        s = self.summary()
        rows = [
            ("config", s["config"]),
            ("dataset", f'{s["n_processes"]} x {s["n_observations"]}'),
            ("SPIs", f'{s["n_computed"]}/{s["n_spis"]} computed'),
            ("failed", ", ".join(s["failed"]) if s["failed"] else "none"),
            ("time", f'{s["total_seconds"]}s'),
            ("slowest", ", ".join(f"{k} ({v}s)" for k, v in s["slowest"][:3]) or "-"),
        ]
        body = "".join(
            f'<tr><td style="text-align:left;padding-right:1em;'
            f'color:#666">{k}</td><td style="text-align:left">{v}</td></tr>'
            for k, v in rows
        )
        return f'<table><tbody>{body}</tbody></table>'

    def to_frame(self, dropna=True):
        """
        The results in long form, one row per ``(spi, source, target)``.

        The same values as :attr:`table`, in the shape suited to plotting,
        ``groupby`` and joins. Self-pairs are dropped.

        Both orientations of every pair are listed, so an undirected SPI
        appears twice per pair; :meth:`feature_vector` gives one value per
        independent pair.

        Parameters
        ----------
        dropna : bool, optional
            Drop the rows whose value is NaN. Default: True.

        Returns
        -------
        pandas.DataFrame
            Columns ``spi``, ``source``, ``target`` and ``value``.
        """
        long = (
            self.table.rename_axis(index="source")
            .stack(level=["spi", "process"], future_stack=True)
            .rename("value")
            .reset_index()
            .rename(columns={"process": "target"})
        )
        long = long[long["source"] != long["target"]]
        if dropna:
            long = long.dropna(subset=["value"])
        return long[["spi", "source", "target", "value"]].reset_index(drop=True)

    def feature_vector(self, include_self=False):
        """
        One value per independent pairwise feature, as a ``pandas.Series``.

        Turns :attr:`table` into features for classification or regression
        across datasets. For ``undirected`` and ``antisymmetric`` SPIs only the
        pairs ``(i, j)`` with ``i < j`` are kept (the other triangle is a copy,
        or a copy with the sign flipped; a positive value means ``i`` leads
        ``j``). ``directed`` and ``asymmetric`` SPIs keep both orientations.
        The trait comes from each SPI's
        :meth:`~pyspi.base.Directed.directedness`.

        :func:`pyspi.feature_vector` does the same for a table loaded with
        :func:`pyspi.load_table`.

        Parameters
        ----------
        include_self : bool, optional
            Also keep the self-pairs ``(i, i)``. Default: False.

        Returns
        -------
        pandas.Series
            Indexed by ``(spi, source, target)``, as in :meth:`to_frame`.
            NaNs are kept, so the vector has the same length and order for
            every dataset computed with this config and number of processes.
        """
        directedness = {k: spi.directedness() for k, spi in self.spis.items()}
        return feature_vector(self.table, directedness, include_self=include_self)

    def summary(self):
        """
        An overview of the last :meth:`compute` call, for printing.

        Returns
        -------
        dict
            The dataset's name and shape, the config, the numbers of SPIs,
            computed SPIs and failed SPIs, the failed identifiers, the total
            time in seconds and the five slowest SPIs with their times.
        """
        timings = self._timings
        total = sum(timings.values())
        slowest = sorted(timings.items(), key=lambda kv: -kv[1])[:5]
        return {
            "dataset": self.dataset.name if hasattr(self, "_dataset") else None,
            "n_processes": self.dataset.n_processes if hasattr(self, "_dataset") else None,
            "n_observations": self.dataset.n_observations if hasattr(self, "_dataset") else None,
            "config": str(self._config),
            "n_spis": self.n_spis,
            "n_computed": len(timings),
            "n_failed": len(self._errors),
            "failed": sorted(self._errors),
            "total_seconds": round(total, 3),
            "slowest": [(k, round(v, 3)) for k, v in slowest],
        }

    @property
    def group(self):
        """The index of the group assigned by :meth:`set_group`."""
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
        """The name of the group assigned by :meth:`set_group`."""
        try:
            return self._group_name
        except AttributeError as err:
            warnings.warn(f"Group name undefined. Call set_group() method first.")
            return None

    @group_name.setter
    def group_name(self, g):
        raise Exception("Do not set this property externally. Use the group() method.")

    def load_dataset(self, dataset, dim_order=None):
        """
        Load a dataset, replacing any earlier one and its results.

        Parameters
        ----------
        dataset : Data or array-like
            A :class:`~pyspi.data.Data`, copied and used as it is
            preprocessed, or anything ``Data`` reads, which is detrended and
            z-scored as set for this Calculator.
        dim_order : str or None, optional
            How ``dataset`` is laid out: ``'ps'`` (one row per process) or
            ``'sp'`` (one column per process), as for
            :class:`~pyspi.data.Data`; None takes the ``dim_order`` this
            Calculator was built with, and if that is None too, reads
            ``'ps'`` and warns if that gives more processes than
            observations. Ignored for a ``Data``. Default: None.
        """
        if not isinstance(dataset, Data):
            self._dataset = Data(
                dataset,
                dim_order=self._dim_order if dim_order is None else dim_order,
                zscore=self._zscore,
                detrend=self._detrend,
            )
        else:
            # A copy, so changing the caller's Data cannot change this run,
            # with the preprocessing that Data carries.
            self._dataset = Data._from_prepared_array(
                np.array(dataset.to_numpy(), copy=True),
                procnames=dataset.procnames,
                name=dataset.name,
            )
            self._dataset.zscore = dataset.zscore
            self._dataset.detrend = dataset.detrend

        # Results belong to the dataset that produced them.
        self._errors = {}
        self._pair_errors = {}
        self._timings = {}

        shape = (self.n_spis, self.dataset.n_processes, self.dataset.n_processes)
        procs = self._dataset.procnames
        self._table = _frame_from_matrices(np.full(shape, np.nan), self.spis, procs)
        self._quality = _frame_from_matrices(
            np.full(shape, Quality.NOT_COMPUTED, dtype=np.uint8), self.spis, procs)

    def save(self, path):
        """
        Write the results to a file, in the format its suffix names.

        ``.npz`` (recommended) stores the values as an ``(n_spis, M, M)`` array
        with the SPI and process names, and round-trips exactly through
        :func:`pyspi.load_table`. It also stores :attr:`quality`,
        :attr:`errors`, :attr:`pair_errors`, the run's provenance, which
        :func:`pyspi.load_results` reads back, and each SPI's symmetry trait
        (:meth:`~pyspi.base.Directed.directedness`), which
        :func:`pyspi.feature_vector` uses. ``.csv`` holds the values only,
        for looking at small results.

        Parameters
        ----------
        path : str or path-like
            Path of the file, ending in ``.npz`` or ``.csv``.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        path = Path(path)
        M = self.dataset.n_processes
        if path.suffix == ".csv":
            self.table.to_csv(path)
        elif path.suffix == ".npz":
            keys = list(self.spis)
            values = np.stack([self.table[k].to_numpy(dtype=float) for k in keys])
            quality = np.stack([self.quality[k].to_numpy(dtype=np.uint8) for k in keys])
            np.savez_compressed(
                path, values=values, quality=quality,
                # dtype='U', not object, which would need allow_pickle to load.
                spis=np.array(keys, dtype="U"),
                processes=np.array(self.dataset.procnames, dtype="U"),
                directedness=np.array(
                    [self.spis[k].directedness() for k in keys], dtype="U"
                ),
                schema=np.array(_NPZ_SCHEMA),
                run_spec=np.array(
                    json.dumps(self.run_spec, sort_keys=True, default=str), dtype="U"
                ),
                run_digest=np.array(self.run_digest, dtype="U"),
                errors=np.array(
                    json.dumps(self.errors, sort_keys=True, default=str), dtype="U"
                ),
                pair_errors=np.array(
                    json.dumps(self.pair_errors, sort_keys=True, default=str), dtype="U"
                ),
            )
        else:
            raise ValueError(
                f"Unsupported suffix '{path.suffix}'. Use '.npz' (recommended) "
                f"or '.csv'."
            )
        logger.info("Wrote %d SPI(s) x %dx%d -> %s", self.n_spis, M, M, path)
        return path

    def compute(
        self,
        n_jobs=None,
        checkpoint_dir=None,
        resume=True,
        retry_failed=True,
        mp_context=None,
        progress=True,
    ):
        """
        Compute every SPI on the loaded dataset.

        Fills :attr:`table` and :attr:`quality`, and records failures in
        :attr:`errors` and :attr:`pair_errors` and times in :attr:`timings`.
        A failed SPI does not stop the run.

        With ``n_jobs > 1`` each worker pins its BLAS and OpenMP pools to one
        thread so the workers do not oversubscribe the cores. macOS's
        Accelerate BLAS cannot be pinned this way, so there ``n_jobs > 1`` can
        oversubscribe BLAS-heavy SPIs; prefer ``n_jobs=1`` for a single
        dataset. ``n_jobs=1`` leaves the libraries free to use several threads.

        Parameters
        ----------
        n_jobs : int or None, optional
            Number of worker processes: ``1`` runs serially in this process,
            ``-1`` uses every core available to it and ``-2`` all but one;
            ``0`` raises ``ValueError``. Default: None, the ``PYSPI_N_JOBS``
            environment variable, or 1 if it is unset.
        checkpoint_dir : str or path-like or None, optional
            Directory to which each finished SPI is written as
            ``<identifier>.npy``, so an interrupted run can resume. A
            directory that belongs to a different run raises ``ValueError``.
            Default: None.
        resume : bool, optional
            Load and skip the SPIs already in ``checkpoint_dir``.
            Default: True.
        retry_failed : bool, optional
            Recompute, rather than resume, checkpoints of SPIs that failed.
            Default: True.
        mp_context : str or None, optional
            Multiprocessing start method when ``n_jobs > 1``: ``'fork'``,
            ``'spawn'`` or ``'forkserver'``. Default: None, ``'fork'`` on
            Linux (workers inherit the imported modules) and ``'spawn'``
            elsewhere (fork is unsafe on macOS and absent on Windows).
        progress : bool, optional
            Show a progress bar. Default: True.
        """
        if not hasattr(self, "_dataset"):
            raise AttributeError(
                "Dataset not loaded yet. Please initialise with load_dataset."
            )

        n_jobs = _parallel.resolve_n_jobs(n_jobs)
        _parallel.guard_oversubscription(n_jobs)

        spi_keys = list(self.spis.keys())
        M = self.dataset.n_processes
        cp_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self._resume_rejected = False
        if cp_dir is not None:
            cp_dir.mkdir(parents=True, exist_ok=True)
            digest = self.run_digest
            owned, reason = _parallel.checkpoint_owner_matches(cp_dir, digest)
            if not owned:
                # Refuse another run's results, and do not delete them.
                self._resume_rejected = True
                raise ValueError(
                    f"Checkpoint directory {cp_dir} belongs to a different run "
                    f"({reason}). Use a separate directory per run, or remove "
                    f"it yourself if the old results are no longer wanted."
                )
            _parallel.write_manifest(cp_dir, digest, self.run_spec)

        # Resume: skip SPIs whose checkpoint exists.
        if cp_dir is not None and resume:
            done, spi_keys = _parallel.load_checkpoints(
                cp_dir, spi_keys, M, retry_failed=retry_failed
            )
            for key, res in done.items():
                self._record(key, res)
            if done:
                logger.info("Resumed %d SPI(s) from %s", len(done), cp_dir)

        if not spi_keys:
            logger.info("All SPIs already cached; nothing to compute.")
            if self._verbose:
                inspect_calc_results(self)
            return

        t_start = time.perf_counter()

        if n_jobs <= 1:
            self._compute_serial(spi_keys, M, cp_dir, progress)
        else:
            n_workers = min(int(n_jobs), len(spi_keys))
            ctx = mp_context or _parallel.default_mp_context()
            logger.info("Parallel compute: %d SPI(s) via %d workers (mp=%s)",
                        len(spi_keys), n_workers, ctx)
            results = _parallel.run_parallel(
                self._spis, self._dataset, spi_keys,
                n_jobs=n_workers, mp_context=ctx,
                checkpoint_dir=cp_dir, progress=progress,
                configfile=self._configfile,
                config_bytes=self._config_bytes,
            )
            for key, res in results.items():
                self._record(key, res)

        elapsed = time.perf_counter() - t_start
        logger.info("Calculation complete. Time taken: %.4fs", elapsed)
        if self._verbose:
            inspect_calc_results(self)

    def _compute_serial(self, spi_keys, M, cp_dir, progress):
        iterable = tqdm(spi_keys) if progress else spi_keys
        for key in iterable:
            if progress:
                iterable.set_description(f"Processing [{self._name}: {key}]")
            # The primitive the workers use, so both paths agree on failures.
            res = _parallel.execute_spi(self._spis[key], self.dataset, key, M)
            self._record(key, res)
            _parallel.write_checkpoint(cp_dir, key, res.values, res.err,
                                       res.quality, res.pair_errors)

    def _record(self, key, res):
        """Record one ``_parallel.SPIResult``: values, quality codes, errors and warnings."""
        S, err, warns = res.values, res.err, res.warns
        self._table[key] = S
        self._quality[key] = np.asarray(res.quality, dtype=np.uint8)
        self._timings[key] = res.elapsed
        procs = list(self._dataset.procnames)
        if res.pair_errors:
            self._pair_errors[key] = [(procs[i], procs[j], msg)
                                      for i, j, msg in res.pair_errors]
        else:
            self._pair_errors.pop(key, None)
        for w in warns:
            warnings.warn(f'SPI "{key}": {w}')
        if err is not None:
            self._errors[key] = err
            warnings.warn(f'Caught error for SPI "{key}": {err}')
        else:
            # A successful recomputation clears an earlier failure.
            self._errors.pop(key, None)

    def _rmmin(self):
        """Shift each unsigned SPI in :attr:`table` to a minimum of 0."""
        self._table = self._min_shifted_table()

    def _min_shifted_table(self):
        """A copy of the table with each unsigned SPI shifted to a minimum of 0."""
        table = self.table.copy()
        for spi in self.spis:
            if not self.spis[spi].issigned():
                mpi = table[spi]
                table[spi] = mpi - np.nanmin(mpi)
        return table

    def set_group(self, classes):
        """
        Assign this dataset to the class whose labels it carries.

        Sets :attr:`group` to the index of the one class in ``classes`` all of
        whose labels are among :attr:`labels`, and :attr:`group_name` to its
        labels joined by commas. With no match, or more than one (which
        warns), the group is None.

        Parameters
        ----------
        classes : list
            One entry per class: a label or a list of labels. No class may be
            a subset of another.
        """
        self._group = None
        self._group_name = None

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
        """Merge two calculators (not implemented)."""
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
        """
        The labels of each SPI.

        Returns
        -------
        dict
            ``{identifier: list of labels}``.
        """
        return {k: v.labels for k, v in zip(self._spis.keys(), self._spis.values())}

    def _get_correlation_df(self, with_labels=False, rmmin=False):
        """
        Spearman correlation between every pair of SPIs, over the pairs of processes.

        A pair of SPIs is correlated on absolute values unless both are
        signed. With ``rmmin`` the unsigned SPIs are first shifted to a
        minimum of 0, on a copy of the table.
        Returns the SPI-by-SPI DataFrame, and the SPI labels if
        ``with_labels``.
        """
        table = self._min_shifted_table() if rmmin else self.table

        # Edge-by-SPI matrix; dropna removes the self-pairs, which are NaN.
        edges = table.stack(future_stack=True).dropna(how="all")

        # Spearman correlation of every SPI pair, on pairwise complete values:
        # two signed SPIs as they are, a pair with an unsigned SPI through abs().
        signed = np.array([self.spis[s].issigned() for s in edges.columns])
        cf = edges.abs().corr(method="spearman")
        if signed.any():
            cols = edges.columns[signed]
            cf.loc[cols, cols] = edges[cols].corr(method="spearman")

        cf.index.name = "SPI-1"
        cf.columns.name = "SPI-2"

        if with_labels:
            return cf, self.get_stat_labels()
        else:
            return cf


def forall(func):
    """Make a CalculatorFrame method that calls ``func(calc, ...)`` on each calculator."""
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
    """
    A set of Calculators, one per dataset, run and compared together.

    Parameters
    ----------
    calculators : Calculator, CalculatorFrame, list or None, optional
        Calculators to start with. Default: None.
    name : str or None, optional
        Name of the frame. Default: None.
    datasets : list or None, optional
        Datasets to build one Calculator each for, all with the same
        ``**kwargs``. Default: None.
    names : list or None, optional
        One name per dataset. Default: None (unnamed).
    labels : list or None, optional
        One list of labels per dataset. Default: None.
    **kwargs
        Arguments of :class:`Calculator`, such as ``config``, ``detrend`` and
        ``zscore``, for the Calculators built from ``datasets``.
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
        kwargs = translate_calculator_kwargs(kwargs)
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
        """
        A CalculatorFrame holding ``calculator``.

        Parameters
        ----------
        calculator : Calculator
            The calculator.

        Returns
        -------
        CalculatorFrame
        """
        cf = CalculatorFrame()
        cf.add_calculator(calculator)
        return cf

    def set_calculator(self, calculators):
        """
        Replace the calculators of the frame (with a warning if it had any).

        Parameters
        ----------
        calculators : Calculator, CalculatorFrame or iterable of Calculator
            The new calculators.
        """
        if hasattr(self, "_calculators"):
            warnings.warn("Overwriting existing calculators without explicitly deleting.")
            del self._calculators

        if isinstance(calculators, Calculator):
            calculators = [calculators]

        if isinstance(calculators, CalculatorFrame):
            self.add_calculator(calculators)
        else:
            for calc in calculators:
                self.add_calculator(calc)

    def add_calculator(self, calc):
        """
        Add calculators to the frame.

        Parameters
        ----------
        calc : Calculator, CalculatorFrame or pandas.DataFrame
            A calculator, the calculators of another frame, or a DataFrame of
            calculators, which replaces those of the frame.
        """
        if not hasattr(self, "_calculators"):
            self._calculators = pd.DataFrame()

        if isinstance(calc, CalculatorFrame):
            self._calculators = pd.concat(
                [self._calculators, calc._calculators], ignore_index=True
            )
        elif isinstance(calc, Calculator):
            # One Calculator per row, in column 0 under every pandas version.
            self._calculators = pd.concat(
                [self._calculators, pd.DataFrame({0: [calc]})],
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
        """
        Add one Calculator per dataset, all built with the same arguments.

        Parameters
        ----------
        datasets : list
            The datasets, each a :class:`~pyspi.data.Data` or array-like.
        names : list
            One name per dataset.
        labels : list
            One list of labels per dataset.
        **kwargs
            Arguments of :class:`Calculator`.
        """
        kwargs = translate_calculator_kwargs(kwargs)
        base_calc = Calculator(**kwargs)
        for i, dataset in enumerate(datasets):
            calc = copy.deepcopy(base_calc)
            calc.load_dataset(dataset)
            calc.name = names[i]
            calc.labels = labels[i]
            self.add_calculator(calc)

    def init_from_yaml(
        self, document, detrend=False, zscore=None, n_processes=None, n_observations=None,
        normalise=None, **kwargs
    ):
        """
        Add one Calculator per dataset listed in a YAML file.

        A dataset that fails to load is skipped with a warning.

        Parameters
        ----------
        document : str or path-like
            A YAML list with one entry per dataset, each with the keys
            ``file`` (a path ``Data`` reads), ``dim_order``, ``name`` and
            ``labels``.
        detrend : bool, optional
            Remove the least-squares line from each process. Default: False.
        zscore : bool or None, optional
            z-score each process along time. Default: None, which means True.
        n_processes : int or None, optional
            Keep only the first ``n_processes`` processes of each dataset.
            Default: None (all).
        n_observations : int or None, optional
            Keep only the first ``n_observations`` observations of each
            dataset. Default: None (all).
        normalise : bool or None, optional
            Deprecated spelling of ``zscore``, removed in pyspi 4.0.
            Default: None.
        **kwargs
            Arguments of :class:`Calculator`.
        """
        zscore = resolve_zscore(zscore, normalise)
        kwargs = translate_calculator_kwargs(kwargs)
        datasets = []
        names = []
        labels = []
        with open(document) as f:
            yf = yaml.load(f, Loader=yaml.FullLoader)

            for config in yf:
                try:
                    dataset = Data(
                        data=config["file"],
                        dim_order=config["dim_order"],
                        name=config["name"],
                        detrend=detrend,
                        zscore=zscore,
                        n_processes=n_processes,
                        n_observations=n_observations,
                    )
                    labels_i = config["labels"]
                except Exception as err:
                    warnings.warn(f"Loading dataset: {config} failed ({err}).")
                    continue
                # Only once the dataset loads, so names and labels stay aligned.
                datasets.append(dataset)
                names.append(config["name"])
                labels.append(labels_i)

        self.init_from_list(datasets, names, labels, **kwargs)

    @property
    def calculators(self):
        """The calculators, a DataFrame with one per row, or None."""
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
        """
        Add the calculators of another CalculatorFrame.

        Parameters
        ----------
        other : CalculatorFrame
            The frame whose calculators are added.
        """
        try:
            self._calculators = pd.concat(
                [self._calculators, other._calculators], ignore_index=True
            )
        except AttributeError:
            self._calculators = other._calculators

    def compute(self, **kwargs):
        """
        Compute every calculator in the frame, one dataset after another.

        For many datasets on many cores, one dataset per process (such as a
        scheduler array job) is faster than ``n_jobs > 1`` here, because SPIs
        that share a cache run in one worker. See "Running at scale" in the
        README.

        Parameters
        ----------
        **kwargs
            Arguments of :meth:`Calculator.compute`, so ``n_jobs=4``
            parallelises within each dataset.
        """
        if not hasattr(self, "_calculators"):
            raise AttributeError("No calculators in frame yet. Initialise before computing.")
        for i in self._calculators.index:
            for calc in self._calculators.loc[i]:
                calc.compute(**kwargs)

    @property
    def groups(self):
        """The group index of each calculator, as set by :meth:`set_group`."""
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
        """
        Spearman correlations between SPIs, over pairs of processes, per dataset.

        For each dataset, every pair of SPIs is correlated across the pairs of
        processes (on absolute values unless both SPIs are signed). An unnamed
        dataset is named ``dataset_<i>`` by its position.

        Parameters
        ----------
        with_labels : bool, optional
            Also return the SPI and dataset labels. Default: False.
        **kwargs
            ``rmmin=True`` shifts each unsigned SPI to a minimum of 0 first.

        Returns
        -------
        mdf : pandas.DataFrame
            The correlations, with a ``(Dataset, SPI-1)`` row index and
            ``SPI-2`` columns.
        shapes : pandas.DataFrame
            ``n_processes`` and ``n_observations`` of each dataset.
        mlabels : dict
            ``{identifier: labels}``, only with ``with_labels``.
        dlabels : dict
            ``{dataset name: labels}``, only with ``with_labels``.
        """
        if with_labels:
            mlabels = {}
            dlabels = {}

        shapes = pd.DataFrame()
        mdf = pd.DataFrame()
        for i, calc in enumerate(c[0] for c in self.calculators.values):
            out = calc._get_correlation_df(with_labels=with_labels, **kwargs)
            # By position, so unnamed datasets stay apart.
            name = calc.name if calc.name not in (None, "") else f"dataset_{i}"

            s = pd.Series(
                dict(
                    n_processes=calc.dataset.n_processes,
                    n_observations=calc.dataset.n_observations,
                )
            )
            s.name = name
            shapes = pd.concat([shapes, pd.DataFrame(s).T])
            if with_labels:
                df = pd.concat({name: out[0]}, names=["Dataset"])
                try:
                    mlabels = mlabels | out[1]
                except TypeError:
                    mlabels.update(out[1])
                dlabels[name] = calc.labels
            else:
                df = pd.concat({name: out}, names=["Dataset"])

            mdf = pd.concat([mdf, df])
        shapes.index.name = "Dataset"

        if with_labels:
            return mdf, shapes, mlabels, dlabels
        else:
            return mdf, shapes


class CorrelationFrame:
    """
    Correlations between SPIs, over pairs of processes, across datasets.

    Holds the SPI-by-SPI Spearman correlations of each dataset
    (:meth:`CalculatorFrame.get_correlation_df`), for comparing how SPIs
    relate across datasets.

    Parameters
    ----------
    cf : Calculator, CalculatorFrame, CorrelationFrame or None, optional
        The computed calculators, or a CorrelationFrame to copy in.
        Default: None.
    **kwargs
        Arguments of :meth:`CalculatorFrame.get_correlation_df`, such as
        ``rmmin``.
    """

    def __init__(self, cf=None, **kwargs):
        self._slabels = {}
        self._dlabels = {}
        self._mdf = pd.DataFrame()
        self._shapes = pd.DataFrame()

        if cf is not None:
            if isinstance(cf, Calculator):
                cf = CalculatorFrame(cf)

            if isinstance(cf, CalculatorFrame):
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
        """``n_processes`` and ``n_observations`` of each dataset."""
        return self._shapes

    @property
    def mdf(self):
        """The correlations, with a ``(Dataset, SPI-1)`` row index and ``SPI-2`` columns."""
        return self._mdf

    @property
    def ddf(self):
        """The correlations with one row per ``(SPI-1, SPI-2)`` and one column per dataset."""
        if not hasattr(self, "_ddf") or self._ddf.size != self._mdf.size:
            self._ddf = convert_mdf_to_ddf(self.mdf)
        return self._ddf

    @property
    def n_datasets(self):
        """Number of datasets."""
        return self.ddf.shape[1]

    @property
    def n_spis(self):
        """Number of SPIs."""
        return self.mdf.shape[1]

    @property
    def mlabels(self):
        """``{identifier: labels}`` of the SPIs."""
        return self._slabels

    @property
    def dlabels(self):
        """``{dataset name: labels}`` of the datasets."""
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
        """
        Add the datasets of another CorrelationFrame; datasets must be named.

        Parameters
        ----------
        other : CorrelationFrame
            The frame whose correlations and labels are added.
        """
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
        """Not available: the pairs of processes are not independent samples."""
        raise NotImplementedError(
            "CorrelationFrame.get_pvalues() is disabled: its observations "
            "are process edges, not independent time samples. The previous "
            "F test incorrectly used time-series length T as its sample size; "
            "using the edge count would still ignore dependence between edges "
            "that share nodes. Use an independently validated "
            "network-preserving permutation/QAP analysis instead."
        )

    def compute_significant_values(self):
        """Not available: the pairs of processes are not independent samples."""
        raise NotImplementedError(
            "CorrelationFrame.compute_significant_values() is disabled "
            "because it depends on invalid edge-correlation p-values. Edges "
            "sharing nodes are dependent, so neither time-series length nor "
            "raw edge count is a valid sample size. Use an independently "
            "validated network-preserving permutation/QAP analysis instead."
        )

    def get_average_correlation(
        self, thresh=0.2, absolute=True, summary="mean", remove_insig=False
    ):
        """
        The SPI-by-SPI correlation matrix summarised over datasets.

        Parameters
        ----------
        thresh : float, optional
            Drop the SPIs (rows, then columns) with fewer than this fraction
            of non-NaN entries. Default: 0.2.
        absolute : bool, optional
            Summarise the absolute correlations. Default: True.
        summary : str, optional
            The pandas groupby reduction, such as ``'mean'`` or ``'median'``.
            Default: ``'mean'``.
        remove_insig : bool, optional
            Not available; True raises ``NotImplementedError``.
            Default: False.

        Returns
        -------
        pandas.DataFrame
            ``SPI-1`` by ``SPI-2``, columns sorted.
        """
        if remove_insig:
            raise NotImplementedError(
                "CorrelationFrame.get_average_correlation(remove_insig=True) "
                "is disabled because it depends on invalid edge-correlation "
                "p-values. Edges sharing nodes are dependent, so neither "
                "time-series length nor raw edge count is a valid sample "
                "size. Use an independently validated network-preserving "
                "permutation/QAP analysis instead."
            )
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
        return ss_adj

    def get_feature_matrix(self, sthresh=0.8, dthresh=0.2, dropduplicates=True):
        """
        The SPI-pair-by-dataset correlations, with sparse rows and columns dropped.

        Parameters
        ----------
        sthresh : float, optional
            Drop the SPI pairs (rows) with fewer than this fraction of
            non-NaN entries. Default: 0.8.
        dthresh : float, optional
            Drop the datasets (columns) with fewer than this fraction of
            non-NaN entries; applied first. Default: 0.2.
        dropduplicates : bool, optional
            Drop duplicated rows first. Default: True.

        Returns
        -------
        pandas.DataFrame
            One row per ``(SPI-1, SPI-2)`` and one column per dataset.
        """
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
        """Make each class a list of labels, and refuse a class that is a subset of another."""
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
        """Index of the first class whose labels are all in ``labels``, or -1."""
        labset = set(labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        if np.count_nonzero(matches) > 1:
            if verbose:
                logger.warning(
                    "More than one match for %s whilst searching for %s within "
                    "%s. Choosing the first.", instance, classes, labels,
                )

        try:
            myid = np.where(matches)[0][0]
            return myid
        except (TypeError, IndexError):
            if verbose:
                logger.warning("%s has no match in %s. Options are %s",
                               instance, classes, labels)
            return -1

    @staticmethod
    def _set_groups(classes, labels, group_names, group):
        """Fill ``group`` with the class index of each key of ``labels``."""
        CorrelationFrame._verify_classes(classes)
        for m in labels:
            group[m] = CorrelationFrame._get_group(labels[m], classes, m)

    def set_sgroups(self, classes):
        """
        Assign each SPI to the first class whose labels it carries.

        Parameters
        ----------
        classes : list of list
            One list of labels per class. No class may be a subset of another.
            An SPI that matches no class gets group -1, named ``"N/A"``.
        """
        self._sgroup_names = {i: ", ".join(c) for i, c in enumerate(classes)}
        self._sgroup_names[-1] = "N/A"

        self._sgroup_ids = {m: -1 for m in self._slabels}
        CorrelationFrame._set_groups(
            classes, self._slabels, self._sgroup_names, self._sgroup_ids
        )

    def set_dgroups(self, classes):
        """
        Assign each dataset to the first class whose labels it carries.

        Parameters
        ----------
        classes : list
            As for :meth:`set_sgroups`.
        """
        self._dgroup_names = {i: ", ".join(c) for i, c in enumerate(classes)}
        self._dgroup_names[-1] = "N/A"

        self._dgroup_ids = {d: -1 for d in self._dlabels}
        CorrelationFrame._set_groups(
            classes, self._dlabels, self._dgroup_names, self._dgroup_ids
        )

    def get_dgroup_ids(self, names=None):
        """
        Group indices of datasets, from :meth:`set_dgroups`.

        Parameters
        ----------
        names : list or None, optional
            Dataset names. Default: None, every dataset.

        Returns
        -------
        list of int
        """
        if names is None:
            names = self._ddf.columns

        return [self._dgroup_ids[n] for n in names]

    def get_dgroup_names(self, names=None):
        """
        Group names of datasets, from :meth:`set_dgroups`.

        Parameters
        ----------
        names : list or None, optional
            Dataset names. Default: None, every dataset.

        Returns
        -------
        list of str
        """
        if names is None:
            names = self._ddf.columns

        return [self._dgroup_names[i] for i in self.get_dgroup_ids(names)]

    def get_sgroup_ids(self, names=None):
        """
        Group indices of SPIs, from :meth:`set_sgroups`.

        Parameters
        ----------
        names : list or None, optional
            SPI identifiers. Default: None, every SPI.

        Returns
        -------
        list of int
        """
        if names is None:
            names = self._mdf.columns

        return [self._sgroup_ids[n] for n in names]

    def get_sgroup_names(self, names=None):
        """
        Group names of SPIs, from :meth:`set_sgroups`.

        Parameters
        ----------
        names : list or None, optional
            SPI identifiers. Default: None, every SPI.

        Returns
        -------
        list of str
        """
        if names is None:
            names = self._mdf.columns

        return [self._sgroup_names[i] for i in self.get_sgroup_ids(names)]

    def relabel_spis(self, names, labels):
        """
        Replace the labels of SPIs.

        Parameters
        ----------
        names : list
            SPI identifiers.
        labels : list
            One list of labels per SPI.
        """
        assert len(names) == len(labels), "Length of spis must equal length of labels."

        for n, l in zip(names, labels):
            try:
                self._slabels[n] = l
            except AttributeError:
                self._slabels = {n: l}

    def relabel_data(self, names, labels):
        """
        Replace the labels of datasets.

        Parameters
        ----------
        names : list
            Dataset names.
        labels : list
            One list of labels per dataset.
        """
        assert len(names) == len(
            labels
        ), "Length of datasets must equal length of labels."

        for n, l in zip(names, labels):
            self._dlabels[n] = l

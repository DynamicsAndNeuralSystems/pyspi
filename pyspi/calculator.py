# Science/maths/computing tools
import numpy as np
import pandas as pd
import copy, yaml, importlib, time, warnings, os
import functools
import hashlib, json
from pathlib import Path
from tqdm import tqdm

# From this package
from .data import Data
from .utils import convert_mdf_to_ddf, inspect_calc_results, require_int
from . import _parallel
from ._logging import get_logger, configure as _configure_logging

logger = get_logger("pyspi.calculator")


# ---------------------------------------------------------------------------
# LaggedCorrelation config expansion: max_tau -> tau=1..max_tau
# ---------------------------------------------------------------------------

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


def _merge_spi_labels(spi, family_labels=None, config_labels=None):
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

    # The SPI's own `issigned()` is authoritative over any declared
    # signed/unsigned label. It is not metadata: `Calculator._rmmin` subtracts
    # the minimum from every SPI reporting unsigned, and `set_group` correlates
    # unsigned SPIs through `abs()`. A config that declares `unsigned` over an
    # antisymmetric measure (which the shipped configs do for `phase`, `pli`,
    # `wpli`, `psi`, `gd` and `ccm_*_diff`) does not merely mislabel it -- it
    # asks for a transform that destroys the lead/lag its sign carries. The
    # label follows the implementation, not the other way round.
    # The SPI's *own* structural trait is authoritative over any structural
    # trait the config declares. `labels` here is the instance's list, before the
    # family and per-config labels are folded in, so it is what the class and
    # `__init__` decided from the shape of the matrix they produce.
    #
    # Exactly one of the four traits survives. Without this, `gd_*` carried the
    # class's `antisymmetric` and the config's `directed` at once, and
    # `gd_*_rvalue` -- symmetric by construction, since it stores |r| -- came
    # back both `undirected` and `directed` when loaded through YAML, so
    # `filter_spis` answered either way for the same SPI.
    _DIRECTEDNESS = ("antisymmetric", "asymmetric", "undirected", "directed")
    own = [label for label in _DIRECTEDNESS if label in labels]
    if own:
        trait = own[0]
        merged = [
            label for label in merged
            if label not in _DIRECTEDNESS or label == trait
        ]

    issigned = getattr(spi, "issigned", None)
    if issigned is not None:
        actual = "signed" if issigned() else "unsigned"
        merged = [label for label in merged
                  if label not in ("signed", "unsigned")] + [actual]
    spi.labels = merged


def _split_config_params(params):
    params = dict(params or {})
    config_labels = params.pop("labels", None)
    return params, config_labels

CONFIG_DIR = Path(__file__).parent / "configs"


def bundled_configs():
    """Names of the bundled configs, i.e. the valid non-path values of ``config``."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def resolve_config(config):
    """Resolve ``config`` to a config yaml path.

    Accepts either the name of a bundled config (``"full"``, ``"fast"``,
    ``"benchmarked_p90"``, ...) or a path to a user-written yaml. A value is
    treated as a path if it carries a directory component or a ``.yaml``/
    ``.yml`` suffix; otherwise it is looked up in :data:`CONFIG_DIR`.
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
    if not path.is_file():
        raise ValueError(
            f"Unknown config '{text}'. Bundled configs are: "
            f"{', '.join(bundled_configs())}. "
            f"To use your own, pass a path to a .yaml file."
        )
    return str(path)


@functools.lru_cache(maxsize=1)
def _full_config_cache_buckets():
    """``{cache_bucket: {identifiers}}`` for the `full` config, built once.

    Cached because the caller is an advisory check that runs on every
    `Calculator` construction, and the answer is a property of the shipped
    config rather than of the run.
    """
    from collections import defaultdict

    available = defaultdict(set)
    for key, spi in load_spis_from_yaml(resolve_config("full"), quiet=True).items():
        bucket = _parallel.cache_bucket(spi)
        if bucket is not None:
            available[bucket].add(key)
    return available


def warn_partial_cache_buckets(spis):
    """Warn when a config keeps only part of a shared-cache group.

    Several SPI families build one expensive cached intermediate on the Data
    object and then derive cheap variants from it (``ccm``, ``spectral_mv``,
    ``barycenter``, ...). The cache is built as soon as *any* member runs, so
    keeping a strict subset pays the full cache cost for fewer SPIs -- the
    remaining members are close to free. ``bench/cut_config.py`` already snaps
    generated configs up to whole buckets; this gives a hand-written config
    the same nudge.
    """
    # Only caches expensive enough for the advice to matter. Amortized cost per
    # SPI at the M=16, T=800 anchor cell (bench/results/analysis/report.md):
    #   ccm 292.7s | barycenter 11.1s | spectral_bv 1.6s | spectral_mv, coint,
    #   covariance all <0.3s.
    # Warning on the cheap ones is noise: fabfour deliberately keeps 1 of 32
    # covariance SPIs, and the whole covariance cache is 0.6s.
    EXPENSIVE = {"ccm", "barycenter"}

    from collections import defaultdict

    # The same definition the scheduler buckets on; see _parallel.cache_bucket.
    bucket = _parallel.cache_bucket

    # Keyed by identifier, not class: a bucket is a set of *variants* (ccm is
    # one class with nine configs), and it is the variants that share the cache.
    kept = defaultdict(set)
    for key, spi in spis.items():
        b = bucket(spi)
        if b is not None:
            kept[b].add(key)
    if not any(bkey[0] in EXPENSIVE for bkey in kept):
        # Nothing this advisory could say anything about. Checked before the
        # `full` config is touched: building it instantiates 322 SPIs and pulls
        # a heavy dependency tree, which is real import time for a check that
        # only ever comments on `ccm` and `barycenter`.
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


# Fields of `run_spec` that say *where* the run came from rather than *what* it
# computes. They are recorded for provenance and deliberately kept out of
# `run_digest`, which is a content hash: the same config contents and the same
# data must digest the same from a checkout, a wheel or a temporary directory.
_DIGEST_EXCLUDED_SPEC_FIELDS = frozenset({"config", "configfile", "dataset_name"})


# Bumped whenever the .npz layout changes incompatibly. Schema 0 means "written
# before the field existed", i.e. a pre-3.0.0 pickled file.
_NPZ_SCHEMA = 1


def load_table(path):
    """Load a results table written by :meth:`Calculator.save`.

    Returns the same DataFrame ``Calculator.table`` returns: rows are
    processes, columns are a ``(spi, process)`` MultiIndex. Only ``.npz``
    round-trips exactly -- ``.csv`` is a one-way human-readable export.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(
            f"Can only load '.npz' (got '{path.suffix}'). CSV export is one-way; "
            f"re-run the calculation or save as .npz."
        )
    # allow_pickle=False: loading a results table must never be able to execute
    # code. Files written by pyspi < 3.0.0 stored names as object arrays and
    # will fail here; re-save them from a Calculator.
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
        spis = [str(s) for s in f["spis"]]
        procs = [str(p) for p in f["processes"]]
        # Provenance. `save()` has always written these three and `load_table`
        # has never read them, so a loaded table could not be asked which SPIs
        # failed, what produced it, or whether it matched a rerun.
        meta = {name: str(f[name]) for name in ("run_spec", "run_digest", "errors")
                if name in f.files}

    # The whole shape, not just `ndim` and axis 0. A file whose matrices were
    # the wrong width reached `MultiIndex.from_product` and failed there, with
    # a reshape error rather than a statement about the file.
    if values.shape != (len(spis), len(procs), len(procs)):
        raise ValueError(
            f"{path} is malformed: values has shape {values.shape}, expected "
            f"({len(spis)}, {len(procs)}, {len(procs)})."
        )
    table = pd.DataFrame(
        data=np.concatenate(list(values), axis=1),
        columns=pd.MultiIndex.from_product([spis, procs], names=["spi", "process"]),
        index=procs,
    )
    table.columns.name = "process"
    # `DataFrame.attrs` rather than a wrapper type: it is pandas' documented
    # place for exactly this, and it keeps `load_table` returning the same
    # object `Calculator.table` does.
    table.attrs["schema"] = schema
    for name in ("run_spec", "errors"):
        if name in meta:
            try:
                table.attrs[name] = json.loads(meta[name])
            except json.JSONDecodeError:
                table.attrs[name] = meta[name]
    if "run_digest" in meta:
        table.attrs["run_digest"] = meta["run_digest"]
    return table


def load_spis_from_yaml(configfile, quiet=False):
    """Instantiate all SPIs from a configfile.

    Returns a dict mapping identifier to SPI instance.

    Shared between :class:`Calculator` and the parallel worker initializer
    (see :func:`pyspi._parallel._worker_init`) so workers don't need to
    instantiate a throwaway Calculator just to rebuild ``_spis``. Progress is
    emitted via the ``pyspi.calculator`` logger at INFO level.
    """
    spis = {}
    log = (lambda *a, **k: None) if quiet else logger.info
    log("Loading configuration file: %s", configfile)
    with open(configfile) as f:
        yf = yaml.load(f, Loader=yaml.FullLoader)
    for module_name, module_spis in yf.items():
        log("Importing module %s", module_name)
        module = importlib.import_module(module_name, __package__)
        for fcn, entry in (module_spis or {}).items():
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
    """Insert an SPI, rejecting identifier collisions.

    The identifier is the primary key: it names the table column and the
    checkpoint file. Detecting duplicates *after* building the dict could never
    work, because dict insertion has already discarded the loser -- the old
    ``Counter(self._spis.keys())`` check could not return a count above 1.
    """
    existing = spis.get(spi.identifier)
    if existing is not None:
        raise ValueError(
            f"Duplicate SPI identifier {spi.identifier!r}: "
            f"{type(existing).__name__} and {module_name}.{fcn}"
            f"{f'({params})' if params else ''} both produce it. "
            "Two configs of the same class must differ in a parameter that "
            "reaches the identifier."
        )
    spis[spi.identifier] = spi


def _expand_lagged_correlation_configs(configs):
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
    """Compute all pairwise interactions.

    The calculator takes in a multivariate time-series dataset (MTS), computes and stores all pairwise interactions for the dataset.
    It uses a YAML configuration file that can be modified in order to compute a reduced set of pairwise methods.

    Example:
        >>> import numpy as np
        >>> dataset = np.random.randn(5,500)    # create a random multivariate time series (MTS)
        >>> calc = Calculator(dataset=dataset)  # Instantiate the calculator
        >>> calc.compute()                      # Compute all pairwise interactions

    Args:
        dataset (:class:`~pyspi.data.Data`, array_like, optional):
            The multivariate time series of M processes and T observations, default=None.
        name (str, optional):
            The name of the calculator. Mainly used for printing the results but can be useful if you have multiple instances, default=None.
        labels (array_like, optional):
            Any set of strings by which you want to label the calculator. This can be useful later for classification purposes, default=None.
        config (str, optional):
            Which SPIs to compute. Either the name of a bundled config or a path
            to your own YAML file, default="full". Bundled configs are:

            - ``"full"`` -- every SPI (322).
            - ``"fast"`` -- drops the slowest SPIs.
            - ``"sonnet"`` -- 14 representative SPIs, one per module (M01-M14).
            - ``"fabfour"`` -- 4 SPIs: covariance, Spearman, directed information,
              power-envelope correlation.
            - ``"benchmarked_p80"`` / ``"_p90"`` / ``"_p95"`` / ``"_p99"`` -- keep
              the fastest N% of SPIs by measured amortized compute cost, so
              ``benchmarked_p80`` is the cheapest and ``benchmarked_p99`` the most
              complete. See ``bench/README.md`` for how these were derived.
        detrend (bool, optional):
            If True, detrend each time series in the MTS dataset individually along the time axis, default=False.
        zscore (bool, optional):
            If True, z-score each time series in the MTS dataset individually along
            the time axis, default=True. Per-process (rather than whole-dataset)
            standardisation is deliberate: it removes each process's arbitrary
            gain/units without letting the choice of the other processes in the
            dataset influence any pairwise statistic.
    """
    def __init__(
        self, dataset=None, name=None, labels=None, config="full",
        detrend=False, zscore=True, verbose=True,
    ):
        self._spis = {}
        self._zscore = zscore
        self._detrend = detrend
        self._timings = {}
        self._errors = {}
        self._verbose = verbose

        # verbose maps to a process-global pyspi logger level (INFO vs WARNING).
        _configure_logging(verbose)

        configfile = resolve_config(config)

        self._configfile = configfile  # stored so parallel workers can re-instantiate SPIs
        # Snapshot at construction: the SPIs were built from *these* bytes, so
        # the digest must reflect them even if the file changes afterwards.
        try:
            self._config_bytes = Path(configfile).read_bytes()
        except OSError:
            self._config_bytes = b"<configfile unreadable>"
        self._config = config
        # Duplicates are rejected at insertion inside load_spis_from_yaml; a
        # post-hoc Counter over dict keys can never see a count above 1.
        self._spis = load_spis_from_yaml(configfile)

        self._name = name
        self._labels = labels

        logger.info("%d SPI(s) were successfully initialised.", len(self.spis))
        # Bundled configs are curated deliberately -- sonnet, for instance, is
        # one representative SPI per module, not a cost-optimised set -- so the
        # advice only applies to configs the user wrote.
        if Path(configfile).parent != CONFIG_DIR:
            warn_partial_cache_buckets(self._spis)

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
    def errors(self):
        """``{identifier: "ExcType: message"}`` for every SPI that failed.

        A failed SPI still occupies its column in :attr:`table`, filled with
        NaN. Consult this before interpreting a table: a NaN column is
        otherwise indistinguishable from a legitimately undefined statistic.
        """
        return dict(self._errors)

    @property
    def run_digest(self):
        """Content hash of the resolved run: spec plus the input data itself.

        This is what a checkpoint is bound to. Resume previously validated only
        the SPI identifier and an ``(M, M)`` shape, so any other run of the same
        width silently inherited the earlier run's numbers -- a different
        dataset, different preprocessing, a different config, or a permuted
        process order all resumed clean.

        The dataset bytes are hashed, not just its shape and name: two datasets
        of the same width with the same name are exactly the case that needs
        separating.
        """
        # Location-only fields are excluded. `config` and `configfile` are the
        # name and the *absolute resolved path*, so hashing the spec verbatim
        # made an identical config and dataset digest differently in a source
        # checkout and in an installed wheel -- and differently again in a
        # temporary directory. That is a false negative rather than an unsafe
        # reuse (a checkpoint is refused when it should have been accepted), but
        # it defeats the point of a content hash. The path stays in `run_spec`
        # as provenance; the *contents* are hashed below.
        spec = {k: v for k, v in self.run_spec.items()
                if k not in _DIGEST_EXCLUDED_SPEC_FIELDS}
        h = hashlib.sha256()
        h.update(json.dumps(spec, sort_keys=True, default=str).encode())
        # The algorithm, not only its inputs. Identical data and an identical
        # config computed by two different estimator implementations are two
        # different results, and a digest that cannot tell them apart lets a
        # checkpoint from one be resumed by the other. `COMPUTATION_VERSION` is
        # bumped whenever a change alters computed values.
        h.update(_parallel.COMPUTATION_VERSION.encode())
        # Hash the config *contents*, not just its path and the identifiers it
        # produces. A config edited in place otherwise produced an identical
        # digest and silently resumed the previous parameterisation's results.
        h.update(self._config_bytes)
        dataset = getattr(self, "_dataset", None)
        if dataset is not None:
            arr = np.ascontiguousarray(dataset.to_numpy())
            # Native dtype: coercing to float64 first made distinct large-integer
            # datasets collide.
            h.update(f"{arr.dtype.str}{arr.shape}".encode())
            h.update(arr.tobytes())
        return h.hexdigest()

    @property
    def run_spec(self):
        """The resolved specification of what this Calculator computes.

        One canonical description of the run — the config actually resolved to,
        the preprocessing actually applied, the dataset shape and process names
        actually loaded, and the SPI set actually instantiated. Recorded so a
        result can be tied back to the run that produced it rather than being
        identified by SPI name and matrix width alone.
        """
        dataset = getattr(self, "_dataset", None)
        return {
            "config": str(self._config),
            "configfile": str(self._configfile),
            # The dataset's own flags: a prepared Data supplied by the caller
            # carries its own preprocessing, and the Calculator's flags were
            # never applied to it.
            "zscore": bool(dataset.zscore) if dataset is not None else bool(self._zscore),
            "detrend": bool(dataset.detrend) if dataset is not None else bool(self._detrend),
            "n_processes": int(dataset.n_processes) if dataset is not None else None,
            "n_observations": (
                int(dataset.n_observations) if dataset is not None else None
            ),
            "procnames": list(dataset.procnames) if dataset is not None else None,
            "dataset_name": dataset.name if dataset is not None else None,
            "spi_identifiers": sorted(self._spis),
        }

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
    def timings(self):
        """Per-SPI wall-clock times (seconds) from the last compute() call."""
        return dict(self._timings)

    @property
    def table(self):
        """Results table for all pairwise interactions."""
        return self._table

    @table.setter
    def table(self, a):
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
        """Rendered by Jupyter in place of the default object repr.

        The bare `<Calculator at 0x...>` told you nothing about whether the run
        had happened or how it went, which is the first thing you want after
        calling compute().
        """
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
        """Results in long form: one row per ``(spi, source, target)``.

        :attr:`table` is wide -- an ``M x (n_spis * M)`` frame with a
        ``(spi, process)`` column MultiIndex -- which is the right shape for
        storage but awkward for plotting, ``groupby``, or joining against SPI
        labels. This is the same data with one value per row.

        Self-pairs are dropped (their diagonal is NaN by construction); with
        ``dropna=False`` failed SPIs keep their NaN rows.
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

    def summary(self):
        """One-line-per-fact overview of the last :meth:`compute` call.

        Human-facing convenience over :attr:`timings`, :attr:`errors` and
        :attr:`n_spis`, which stay as they are: those are the programmatic
        handles you filter and assert on, this is the thing you print.
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

    def load_dataset(self, dataset):
        """Load new dataset into existing instance.

        Args:
            dataset (:class:`~pyspi.data.Data`, array_list):
                New dataset to attach to calculator.
        """
        if not isinstance(dataset, Data):
            self._dataset = Data(
                Data.convert_to_numpy(dataset),
                zscore=self._zscore,
                detrend=self._detrend,
            )
        else:
            # Snapshot rather than alias. A caller-owned Data could be mutated
            # after construction -- changing its width left the table at the old
            # shape and computation failed; a same-width change silently kept
            # stale process labels. The snapshot also records the preprocessing
            # the data *actually* carries, not the Calculator flags that were
            # bypassed when a prepared Data was supplied.
            self._dataset = Data._from_prepared_array(
                np.array(dataset.to_numpy(), copy=True),
                procnames=dataset.procnames,
                name=dataset.name,
            )
            self._dataset.zscore = dataset.zscore
            self._dataset.detrend = dataset.detrend

        # Results belong to the dataset that produced them.
        self._errors = {}
        self._timings = {}

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

    def save(self, path):
        """Write the results table to ``path``; the format follows the suffix.

        ``.npz`` (recommended) stores the results in their natural shape -- an
        ``(n_spis, M, M)`` float array plus the SPI and process names -- and
        round-trips exactly through :func:`pyspi.load_table`. ``.csv`` is for
        eyeballing small results; it is impractical for the full SPI set.
        """
        path = Path(path)
        M = self.dataset.n_processes
        if path.suffix == ".csv":
            self.table.to_csv(path)
        elif path.suffix == ".npz":
            keys = list(self.spis)
            values = np.stack([self.table[k].to_numpy(dtype=float) for k in keys])
            np.savez_compressed(
                path, values=values,
                # dtype='U', not object: object arrays are only loadable with
                # allow_pickle=True, which reintroduces arbitrary code execution
                # on load -- the exact hazard that motivated dropping pickle.
                spis=np.array(keys, dtype="U"),
                processes=np.array(self.dataset.procnames, dtype="U"),
                schema=np.array(_NPZ_SCHEMA),
                run_spec=np.array(
                    json.dumps(self.run_spec, sort_keys=True, default=str), dtype="U"
                ),
                run_digest=np.array(self.run_digest, dtype="U"),
                errors=np.array(
                    json.dumps(self.errors, sort_keys=True, default=str), dtype="U"
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
        """Compute every SPI on the loaded dataset.

        Args:
            n_jobs (int, optional): Number of worker processes. ``None`` (default)
                falls back to the ``PYSPI_N_JOBS`` environment variable, or 1 if
                unset. ``1`` runs serially in this process.
            checkpoint_dir (str | Path, optional): If set, each finished SPI is
                written to ``<dir>/<identifier>.npy`` atomically. Enables resume.
            resume (bool): If True (default) and ``checkpoint_dir`` contains
                results from a prior run, those SPIs are loaded and skipped.
            retry_failed (bool): If True (default), checkpoints carrying an
                ``.error`` sidecar are recomputed rather than resumed. Set
                False to inherit a prior run's failures as-is.
            mp_context (str, optional): Multiprocessing start method when
                ``n_jobs>1``. Default (``None``): ``"fork"`` on Linux (workers
                inherit imported state via copy-on-write — ~2x faster startup),
                ``"spawn"`` on macOS/Windows (fork is unsafe/absent there).
            progress (bool): Show a tqdm progress bar (default True).

        Backend threading: when ``n_jobs>1`` each worker pins its nested pools
        (OpenMP/OpenBLAS/MKL, pyEDM) to one thread/process so the
        workers don't oversubscribe the cores. macOS is an exception — its
        Accelerate BLAS cannot be thread-pinned by threadpoolctl, so on macOS
        ``n_jobs>1`` can oversubscribe BLAS-heavy SPIs; prefer ``n_jobs=1``
        there for a single dataset. ``n_jobs=1`` always leaves the backends
        free to self-parallelise.
        """
        if not hasattr(self, "_dataset"):
            raise AttributeError(
                "Dataset not loaded yet. Please initialise with load_dataset."
            )

        if n_jobs is None:
            n_jobs = int(os.getenv("PYSPI_N_JOBS", "1"))
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
                # Refuse to inherit another run's results. Resuming here is how
                # a different dataset of the same width silently returned the
                # previous run's numbers.
                self._resume_rejected = True
                # Refuse rather than delete. Deleting another run's results to
                # make room is destructive and, if interrupted midway, relabels
                # whatever survives as this run.
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
            for key, (S, err, warns, _t) in done.items():
                self._record(key, S, err, warns, 0.0)
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
            for key, (S, err, warns, elapsed) in results.items():
                self._record(key, S, err, warns, elapsed)

        elapsed = time.perf_counter() - t_start
        logger.info("Calculation complete. Time taken: %.4fs", elapsed)
        if self._verbose:
            inspect_calc_results(self)

    def _compute_serial(self, spi_keys, M, cp_dir, progress):
        iterable = tqdm(spi_keys) if progress else spi_keys
        for key in iterable:
            if progress:
                iterable.set_description(f"Processing [{self._name}: {key}]")
            # Same primitive the workers use, so both paths agree on what
            # counts as a failure and on what the caller is shown.
            S, err, warns, elapsed = _parallel.run_spi(
                self._spis[key], self.dataset, key, M
            )
            self._record(key, S, err, warns, elapsed)
            _parallel.write_checkpoint(cp_dir, key, S, err)

    def _record(self, key, S, err, warns, elapsed):
        """Commit one SPI result, its error, and its warnings."""
        self._table[key] = S
        self._timings[key] = elapsed
        for w in warns:
            warnings.warn(f'SPI "{key}": {w}')
        if err is not None:
            self._errors[key] = err
            warnings.warn(f'Caught error for SPI "{key}": {err}')
        else:
            # A recomputation that succeeds clears the previous failure. Without
            # this, `compute(retry_failed=True)` on a resumed run left the old
            # entry in `calc.errors` next to the good column it had just
            # written, and `save()` froze that contradiction into the file.
            self._errors.pop(key, None)

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

        # Flatten (get Edge-by-SPI matrix). future_stack=True is the pandas 3
        # behaviour; unlike the legacy default it keeps all-NaN rows, so the
        # explicit dropna reproduces the old semantics (self-pairs are all-NaN
        # because the SPI diagonals are NaN). Verified equivalent on pandas 2.3.
        edges = self.table.stack(future_stack=True).dropna(how="all")

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
            return cf, self.get_stat_labels()
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

        if not hasattr(self, "_calculators"):
            self._calculators = pd.DataFrame()

        if isinstance(calc, CalculatorFrame):
            self._calculators = pd.concat(
                [self._calculators, calc._calculators], ignore_index=True
            )
        elif isinstance(calc, Calculator):
            # Keep one Calculator per row in a stable, explicitly named column.
            # Concatenating a named Series into an empty DataFrame happened to
            # produce column 0 in pandas 2, but pandas 3 uses the Series name as
            # the column label and successive calculators form a sparse frame.
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
        base_calc = Calculator(**kwargs)
        for i, dataset in enumerate(datasets):
            calc = copy.deepcopy(base_calc)
            calc.load_dataset(dataset)
            calc.name = names[i]
            calc.labels = labels[i]
            self.add_calculator(calc)

    def init_from_yaml(
        self, document, detrend=False, zscore=True, n_processes=None, n_observations=None, **kwargs
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
                            zscore=zscore,
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

    def compute(self, **kwargs):
        """Compute every calculator in the frame, one dataset after another.

        Keyword arguments are forwarded to :meth:`Calculator.compute`, so
        ``frame.compute(n_jobs=4)`` parallelises *within* each dataset.

        Note that for many datasets on many cores, running one dataset per
        process (e.g. a scheduler array job) beats ``n_jobs>1`` here: SPIs
        sharing a cache run serially inside a single worker, which floors the
        achievable speedup at roughly 2-4x irrespective of ``n_jobs``. See
        "Running at scale" in the README.
        """
        if not hasattr(self, "_calculators"):
            raise AttributeError("No calculators in frame yet. Initialise before computing.")
        for i in self._calculators.index:
            for calc in self._calculators.loc[i]:
                calc.compute(**kwargs)

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
        raise NotImplementedError(
            "CorrelationFrame.get_pvalues() is disabled: its observations "
            "are process edges, not independent time samples. The previous "
            "F test incorrectly used time-series length T as its sample size; "
            "using the edge count would still ignore dependence between edges "
            "that share nodes. Use an independently validated "
            "network-preserving permutation/QAP analysis instead."
        )

    def compute_significant_values(self):
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

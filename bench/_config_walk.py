"""Shared config walker for the bench scripts.

``cut_config``, ``forecast_cell`` and ``analyse_cells`` all need to enumerate the
SPIs a config yaml would instantiate. This mirrors
:func:`pyspi.calculator.load_spis_from_yaml` — including the ``LaggedCorrelation``
``max_tau`` expansion and the stripping of config-level ``labels:`` annotations
that are not constructor kwargs — and is the single place that mirroring lives.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

# cache_bucket is re-exported, not redefined: the benchmark tooling, the
# scheduler and the config advisory must not drift apart on what "shares a
# cache" means.
from pyspi._parallel import cache_bucket  # noqa: F401
from pyspi.calculator import _expand_lagged_correlation_configs, _split_config_params


def walk_spis(configfile):
    """Yield ``(module_name, class_name, params, identifier, spi)`` for every SPI.

    ``params`` is the *raw* config variant (still carrying any ``labels:``
    annotation) so that callers re-emitting YAML can round-trip it verbatim; the
    SPI itself is constructed from the constructor-only subset.
    """
    source = yaml.safe_load(Path(configfile).read_text())
    for module_name, module_spis in source.items():
        module = importlib.import_module(module_name, "pyspi")
        for class_name, entry in (module_spis or {}).items():
            if entry is None:
                continue
            configs = entry.get("configs")
            if class_name == "LaggedCorrelation" and configs is not None:
                configs = _expand_lagged_correlation_configs(configs)
            cls = getattr(module, class_name)
            for params in ([None] if configs is None else configs):
                if params is None:
                    spi = cls()
                else:
                    ctor_params, _ = _split_config_params(params)
                    spi = cls(**ctor_params)
                yield module_name, class_name, params, spi.identifier, spi

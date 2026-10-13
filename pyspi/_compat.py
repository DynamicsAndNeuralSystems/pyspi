"""Deprecation shims for the 2.x keyword arguments (removed in 4.0).

2.x spelled the preprocessing flag ``normalise`` and chose SPIs with
``subset`` / ``configfile``; 3.0 uses ``zscore`` and a single ``config``.
Each helper maps the old spelling onto the new one with a
``DeprecationWarning`` and refuses a call that mixes both spellings.

``stacklevel`` is the depth of the warning's target counted from the public
function that calls the helper: the default of 2 points at the caller of that
public function, i.e. the user's line.
"""

from __future__ import annotations

import warnings


def resolve_zscore(zscore, normalise, stacklevel=2):
    """Return the effective ``zscore`` (default True) given a legacy ``normalise``."""
    if normalise is None:
        return True if zscore is None else zscore
    if zscore is not None:
        raise TypeError("Pass either 'zscore' or the deprecated 'normalise', not both.")
    warnings.warn(
        "'normalise' is deprecated and will be removed in pyspi 4.0; use 'zscore'.",
        DeprecationWarning, stacklevel=stacklevel + 1,
    )
    return normalise


def resolve_config_arg(config, subset, configfile, stacklevel=2):
    """Return the effective ``config`` (default "full") given legacy ``subset``/``configfile``."""
    if subset is None and configfile is None:
        return "full" if config is None else config
    if config is not None:
        raise TypeError(
            "Pass either 'config' or the deprecated 'subset'/'configfile', not both."
        )
    if subset is not None and configfile is not None:
        raise TypeError(
            "'subset' and 'configfile' are both deprecated and mutually exclusive; "
            "use a single 'config'."
        )
    if configfile is not None:
        warnings.warn(
            "'configfile' is deprecated and will be removed in pyspi 4.0; "
            "use config=<path to yaml>.",
            DeprecationWarning, stacklevel=stacklevel + 1,
        )
        return configfile
    mapped = "full" if subset == "all" else subset
    warnings.warn(
        f"'subset' is deprecated and will be removed in pyspi 4.0; use config={mapped!r}.",
        DeprecationWarning, stacklevel=stacklevel + 1,
    )
    return mapped


def translate_calculator_kwargs(kwargs, stacklevel=2):
    """Rewrite legacy keys in a ``Calculator(**kwargs)`` dict (used by CalculatorFrame)."""
    if not any(k in kwargs for k in ("subset", "configfile", "normalise")):
        return kwargs
    kwargs = dict(kwargs)
    legacy_config = {k: kwargs.pop(k, None) for k in ("subset", "configfile")}
    config = kwargs.pop("config", None)
    normalise = kwargs.pop("normalise", None)
    zscore = kwargs.pop("zscore", None)
    if any(v is not None for v in legacy_config.values()):
        kwargs["config"] = resolve_config_arg(
            config, stacklevel=stacklevel + 1, **legacy_config)
    elif config is not None:
        kwargs["config"] = config
    if normalise is not None:
        kwargs["zscore"] = resolve_zscore(zscore, normalise, stacklevel=stacklevel + 1)
    elif zscore is not None:
        kwargs["zscore"] = zscore
    return kwargs

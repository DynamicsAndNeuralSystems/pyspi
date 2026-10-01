"""Logging setup for pyspi.

pyspi emits informational output (config loading, per-SPI init, compute
progress) through the standard ``logging`` module under the ``pyspi`` logger.

Behaviour:
- ``pyspi/__init__.py`` attaches a ``NullHandler`` so importing pyspi never
  prints anything on its own (standard library practice).
- ``Calculator(verbose=...)`` calls :func:`configure` as a convenience for
  interactive/script users: ``verbose=True`` (default) attaches a colored
  ``StreamHandler`` at INFO; ``verbose=False`` raises the level to WARNING so
  only problems surface.
- Power users who attach their own handler before constructing a Calculator
  are respected — :func:`configure` will not add a second StreamHandler, and
  ``verbose`` then only adjusts the level.

Note: ``verbose`` maps to a *process-global* logger level. Two Calculators
with different ``verbose`` in the same process share one level (last wins).
For per-call control of the parallel progress bar use ``compute(progress=...)``.
"""

from __future__ import annotations

import logging

from colorama import Fore, Style
from colorama import init as _colorama_init

_colorama_init(autoreset=True)

_LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Style.BRIGHT,
}

_LOGGER_NAME = "pyspi"


class ColorFormatter(logging.Formatter):
    """Formatter that tints the whole record by level (via colorama)."""

    def format(self, record):
        msg = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{msg}{Style.RESET_ALL}" if color else msg


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a pyspi logger. ``name`` may be a dotted sub-name (e.g. 'pyspi.calculator')."""
    return logging.getLogger(name)


def configure(verbose: bool = True) -> None:
    """Idempotently attach a colored StreamHandler to the ``pyspi`` logger.

    Args:
        verbose: True -> level INFO (chatty); False -> level WARNING (problems only).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in logger.handlers
    )
    if not has_stream:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

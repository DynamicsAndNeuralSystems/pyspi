"""Logging setup for pyspi.

pyspi emits informational output (config loading, per-SPI init, compute
progress) through the standard ``logging`` module under the ``pyspi`` logger.

Behaviour:
- ``pyspi/__init__.py`` attaches a ``NullHandler`` so importing pyspi never
  prints anything on its own (standard library practice). Import also leaves
  ``sys.stdout``/``sys.stderr`` and the root logger untouched.
- ``Calculator(verbose=True)`` (the default) calls :func:`configure`, which
  attaches one colored ``StreamHandler`` to the ``pyspi`` logger and sets it to
  INFO -- once, however many Calculators are built. ``propagate`` is not
  touched, so a host application's root handlers still receive the records.
- ``Calculator(verbose=False)`` does nothing to logging at all: no handler, no
  level. Records then follow the application's own configuration (or Python's
  last-resort handler, which shows WARNING and above).
- Power users who attach their own handler to the ``pyspi`` logger are
  respected: :func:`configure` will not add a second one.

Note: ``verbose=True`` sets a *process-global* logger level, and it is never
lowered again by a later ``verbose=False``. For per-call control of the
parallel progress bar use ``compute(progress=...)``.
"""

from __future__ import annotations

import logging

from colorama import Fore, Style

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
    """Opt in to pyspi's console logging; idempotent.

    Args:
        verbose: True -> attach a colored StreamHandler (once) and log at INFO.
            False -> leave logging entirely to the application.
    """
    if not verbose:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in logger.handlers
    )
    if not has_stream:
        # Translate ANSI colors on legacy Windows consoles; a no-op elsewhere.
        # Called here, on opt-in, rather than at import: colorama.init() swaps
        # out sys.stdout/sys.stderr, which a library must never do.
        try:
            from colorama import just_fix_windows_console
            just_fix_windows_console()
        except ImportError:  # colorama < 0.4.6
            pass
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("%(message)s"))
        logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)

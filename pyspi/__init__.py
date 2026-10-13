import logging
import numpy as np

# No OMP_NUM_THREADS default here. A setdefault at this point runs after numpy
# (imported above, and by the caller) has already initialised its BLAS, so it
# never pinned anything in-process; it only mutated the environment of every
# program that imports pyspi. Thread pinning lives in pyspi._parallel: workers
# pin BLAS in _worker_init, and Calculator.compute warns about (or, on a
# single-core allocation, pins) an oversubscribed setup.

# Standard library practice: a library should not emit log output unless the
# application configures a handler. Calculator(verbose=...) attaches a real
# handler on demand via pyspi._logging.configure().
logging.getLogger("pyspi").addHandler(logging.NullHandler())

# NumPy 2 removed np.NaN; some legacy code paths still reference it.
if not hasattr(np, "NaN"):
    np.NaN = np.nan


from .calculator import Calculator, load_table  # noqa: E402

__all__ = ["Calculator", "load_table"]

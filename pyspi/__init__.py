# BLAS threading: default to single-threaded BLAS so that parallel workers
# (see Calculator.compute) don't oversubscribe (n_jobs workers x full BLAS).
# Workers pin BLAS to 1 thread in _parallel._worker_init; this line ensures
# the main process (and any serial run) does the same by default.
import os
import logging
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')

# Standard library practice: a library should not emit log output unless the
# application configures a handler. Calculator(verbose=...) attaches a real
# handler on demand via pyspi._logging.configure().
logging.getLogger("pyspi").addHandler(logging.NullHandler())

# NumPy 2 removed np.NaN; some legacy code paths still reference it.
if not hasattr(np, "NaN"):
    np.NaN = np.nan


from .calculator import Calculator, load_table  # noqa: E402

__all__ = ["Calculator", "load_table"]

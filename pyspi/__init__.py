"""pyspi: statistics of pairwise interactions for multivariate time series."""
import logging

import numpy as np

# No OMP_NUM_THREADS default here: set after numpy has loaded its BLAS, it
# would pin nothing and only change the environment of the importing program.
# Thread pinning is in pyspi._parallel.

# A library logs nothing unless the application adds a handler;
# Calculator(verbose=...) adds one through pyspi._logging.configure().
logging.getLogger("pyspi").addHandler(logging.NullHandler())

# NumPy 2 has no np.NaN; the alias is kept for code that still uses it.
if not hasattr(np, "NaN"):
    np.NaN = np.nan


# The package imports come after the np.NaN alias above.
from .calculator import Calculator, Results, feature_vector, load_results, load_table  # noqa: E402
# Per-cell quality codes (shared with pyhctsa); see Calculator.quality.
from .quality import Quality  # noqa: E402
# Changed whenever a change can alter a valid SPI output. Record it with saved
# results, and do not pool results computed under different values.
from ._parallel import COMPUTATION_VERSION  # noqa: E402

__all__ = ["Calculator", "load_table", "load_results", "Results", "Quality",
           "feature_vector", "COMPUTATION_VERSION"]

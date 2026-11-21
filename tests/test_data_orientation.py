"""Which axis of the input is time, and the process names a DataFrame carries.

``(M, T)``, one row per process, is the default (``dim_order="ps"``);
``dim_order="sp"`` reads ``(T, M)`` arrays and DataFrames with one column per
process. Data read as more processes than observations warn, since that is
the usual sign of the other layout.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from pyspi import Calculator
from pyspi.data import Data

X = np.random.default_rng(0).standard_normal((4, 50))   # (M, T)
NAMES = ["a", "b", "c", "d"]


def _orientation_warnings(record):
    return [w for w in record if issubclass(w.category, UserWarning)
            and "processes of" in str(w.message)]


def test_time_along_rows_warns_once_and_names_the_fix():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        d = Data(X.T)
    (w,) = _orientation_warnings(record)
    assert (d.n_processes, d.n_observations) == (50, 4)
    assert "shape (50, 4)" in str(w.message) and "dim_order='sp'" in str(w.message)
    # The warning points at the caller's line, not into pyspi.
    assert w.filename == __file__


def test_no_warning_when_processes_do_not_outnumber_observations():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Data(X)
        Data(X.T, dim_order="sp")
        Data(np.random.default_rng(1).standard_normal((5, 5)))
    assert not _orientation_warnings(record)


def test_sp_reads_the_transposed_array_identically():
    np.testing.assert_array_equal(Data(X.T, dim_order="sp").to_numpy(),
                                  Data(X).to_numpy())


@pytest.mark.parametrize("dim_order, frame", [
    ("ps", pd.DataFrame(X, index=NAMES)),
    ("sp", pd.DataFrame(X.T, columns=NAMES)),
])
def test_dataframe_labels_name_the_processes(dim_order, frame):
    d = Data(frame, dim_order=dim_order)
    assert d.procnames == NAMES
    np.testing.assert_array_equal(d.to_numpy(), Data(X).to_numpy())


def test_dataframe_labels_are_coerced_to_str_and_truncated():
    frame = pd.DataFrame(X.T, columns=[10, 20, 30, 40])
    assert Data(frame, dim_order="sp").procnames == ["10", "20", "30", "40"]
    assert Data(frame, dim_order="sp", n_processes=2).procnames == ["10", "20"]


def test_range_index_and_explicit_procnames():
    # A RangeIndex holds positions, not names.
    assert Data(pd.DataFrame(X)).procnames == [f"proc-{i}" for i in range(4)]
    # Names passed explicitly win over the frame's labels.
    frame = pd.DataFrame(X, index=NAMES)
    assert Data(frame, procnames=list("wxyz")).procnames == list("wxyz")


def test_duplicate_dataframe_labels_are_refused():
    with pytest.raises(ValueError, match="unique"):
        Data(pd.DataFrame(X, index=["a", "a", "b", "c"]))


def test_calculator_passes_dim_order_and_keeps_labels():
    calc = Calculator(dataset=pd.DataFrame(X.T, columns=NAMES), dim_order="sp",
                      config="fabfour", verbose=False)
    assert calc.dataset.procnames == NAMES
    assert list(calc.table.index) == NAMES
    np.testing.assert_array_equal(calc.dataset.to_numpy(), Data(X).to_numpy())


def test_load_dataset_dim_order_overrides_the_calculator_default():
    calc = Calculator(config="fabfour", verbose=False)
    calc.load_dataset(X.T, dim_order="sp")
    assert (calc.dataset.n_processes, calc.dataset.n_observations) == (4, 50)

"""Which SPI columns of two calculator tables differ, and by how much."""
import numpy as np


def table_differences(table, ref):
    """
    One line per SPI whose block in ``table`` is not identical to ``ref``'s.

    Each line names the SPI, the number of cells that differ (a cell that is
    NaN in one table only counts), and the largest absolute and relative
    difference over the cells finite in both. The relative difference is taken
    against the larger of the two magnitudes. An empty list means the tables
    agree bit for bit, NaN for NaN.
    """
    lines = []
    for key in ref.columns.get_level_values(0).unique():
        x = table[key].to_numpy(float)
        y = ref[key].to_numpy(float)
        if np.array_equal(x, y, equal_nan=True):
            continue
        both = np.isfinite(x) & np.isfinite(y)
        nan_mismatch = np.isnan(x) != np.isnan(y)
        differs = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        abs_diff = np.abs(x[both] - y[both])
        scale = np.maximum(np.abs(x[both]), np.abs(y[both]))
        rel_diff = np.divide(abs_diff, scale, out=np.zeros_like(abs_diff),
                             where=scale > 0)
        max_abs = abs_diff.max() if abs_diff.size else 0.0
        max_rel = rel_diff.max() if rel_diff.size else 0.0
        lines.append(f"{key}: {int(differs.sum())} of {x.size} cells differ "
                     f"({int(nan_mismatch.sum())} NaN in one table only), "
                     f"max abs {max_abs:.3g}, max rel {max_rel:.3g}")
    return lines

"""Which SPI columns of two calculator tables differ, and by how much."""
import numpy as np

# SPIs compared within (atol, rtol) where two runs of one build are expected
# to agree. `anm` solves a Gaussian-process system whose kernel matrix is
# ill-conditioned on the short fixtures (cond ~ 6e11), so a last-bit
# difference between two BLAS calls is amplified to about 1e-4 relative;
# measured between two runs in one process on macOS: 1.0e-4. The tolerance is
# the one `test_baseline_drift.LOOSE_SPIS` gives `anm` across platforms.
REPEAT_TOLERANCES = {"anm": (1e-6, 1e-3)}


def table_differences(table, ref, tolerances=None):
    """
    One line per SPI whose block in ``table`` does not match ``ref``'s.

    An SPI matches when its two blocks are identical, NaN for NaN. An SPI
    named in ``tolerances`` (identifier -> ``(atol, rtol)``) also matches when
    its finite cells agree within that tolerance and its NaNs are in the same
    cells. Each line names the SPI, the number of cells that differ (a cell
    that is NaN in one table only counts), and the largest absolute and
    relative difference over the cells finite in both. The relative difference
    is taken against the larger of the two magnitudes. An empty list means
    every SPI matches.
    """
    tolerances = tolerances or {}
    lines = []
    for key in ref.columns.get_level_values(0).unique():
        x = table[key].to_numpy(float)
        y = ref[key].to_numpy(float)
        if np.array_equal(x, y, equal_nan=True):
            continue
        if key in tolerances:
            atol, rtol = tolerances[key]
            if np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True):
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

"""Quality labels for computed values.

Every computed value comes with a quality code, stored in a ``uint8`` array of the same shape
as the values. Code 0 means the value is a real, finite number. Every other code means the
value is stored as NaN, and the code says why. The value array is therefore always safe to use
directly (for example in scikit-learn), and the quality array says what each NaN means.

Codes 0-7 are those of hctsa's ``TS_Quality`` (``TS_CalculateFeatureVector.m``), so results
from hctsa, pyhctsa and pyspi can be compared code for code. Codes 8 and above are additions.

This module is shared, unchanged, by pyhctsa and pyspi. Keep the two copies identical.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

__all__ = ["Quality", "classify", "quality_from_values", "quality_labels", "quality_summary"]


class Quality(IntEnum):
    """What a computed value is. Only ``GOOD`` values are kept; all others are stored as NaN."""

    GOOD = 0
    """A real, finite number."""
    ERROR = 1
    """The computation raised an exception (the message is recorded separately)."""
    NAN = 2
    """The computation returned NaN: the quantity is undefined or not appropriate for this input."""
    INF = 3
    """The computation returned +Inf."""
    NEG_INF = 4
    """The computation returned -Inf."""
    COMPLEX = 5
    """The computation returned a complex number with a non-zero imaginary part (a bug)."""
    EMPTY = 6
    """The computation returned nothing (``None`` or an empty array)."""
    LINK_ERROR = 7
    """An expected output was missing from the computation's result (as in hctsa)."""
    NON_SCALAR = 8
    """The computation returned something that is not a single number (a bug)."""
    REJECTED = 9
    """The input was rejected before any computation (e.g. constant, or containing NaN or Inf)."""
    NOT_COMPUTED = 255
    """Not computed (yet)."""


_LABELS = {
    Quality.GOOD: "good",
    Quality.ERROR: "error",
    Quality.NAN: "NaN",
    Quality.INF: "Inf",
    Quality.NEG_INF: "-Inf",
    Quality.COMPLEX: "complex",
    Quality.EMPTY: "empty",
    Quality.LINK_ERROR: "link error",
    Quality.NON_SCALAR: "non-scalar",
    Quality.REJECTED: "rejected input",
    Quality.NOT_COMPUTED: "not computed",
}


def classify(value) -> tuple[float, Quality]:
    """Return ``(stored_value, quality)`` for one raw output.

    ``stored_value`` is ``float(value)`` when the quality is ``GOOD`` and NaN otherwise. Booleans
    and integers are numbers; a complex number with zero imaginary part is its real part; a
    one-element array is its element.
    """
    if value is None:
        return np.nan, Quality.EMPTY
    if isinstance(value, str):
        return np.nan, Quality.NON_SCALAR
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan, Quality.EMPTY
        if value.size != 1:
            return np.nan, Quality.NON_SCALAR
        value = value.reshape(-1)[0]
    if isinstance(value, (complex, np.complexfloating)):
        if value.imag != 0:
            return np.nan, Quality.COMPLEX
        value = value.real
    if not isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return np.nan, Quality.NON_SCALAR
    x = float(value)
    if np.isnan(x):
        return np.nan, Quality.NAN
    if np.isinf(x):
        return np.nan, Quality.INF if x > 0 else Quality.NEG_INF
    return x, Quality.GOOD


def quality_from_values(values) -> tuple[np.ndarray, np.ndarray]:
    """Classify a numeric array element-wise; return ``(stored_values, quality)``.

    Use this for array-valued results (e.g. a matrix of pairwise values). The input may be real
    or complex; the returned values are float64 with every non-``GOOD`` entry NaN.
    """
    a = np.asarray(values)
    if a.dtype == object:  # mixed Python objects: classify each element
        x = np.full(a.shape, np.nan)
        q = np.zeros(a.shape, dtype=np.uint8)
        for idx, v in np.ndenumerate(a):
            x[idx], q[idx] = classify(v)
        return x, q
    q = np.zeros(a.shape, dtype=np.uint8)
    if np.iscomplexobj(a):
        cplx = a.imag != 0
        q[cplx] = Quality.COMPLEX
        a = a.real
    x = np.array(a, dtype=np.float64, copy=True)
    ok = q == Quality.GOOD
    q[ok & np.isnan(x)] = Quality.NAN
    q[ok & np.isposinf(x)] = Quality.INF
    q[ok & np.isneginf(x)] = Quality.NEG_INF
    x[q != Quality.GOOD] = np.nan
    return x, q


def quality_labels(quality):
    """The text label of each code, in an array (or DataFrame) of the same shape."""
    lookup = {int(k): v for k, v in _LABELS.items()}
    try:
        import pandas as pd
        if isinstance(quality, pd.DataFrame):
            return quality.apply(lambda col: col.map(lambda c: lookup.get(int(c), f"code {int(c)}")))
    except ImportError:  # pragma: no cover
        pass
    q = np.asarray(quality)
    return np.vectorize(lambda c: lookup.get(int(c), f"code {int(c)}"), otypes=[object])(q)


def quality_summary(quality) -> dict[str, int]:
    """Counts of each quality label, most common first."""
    codes, counts = np.unique(np.asarray(quality), return_counts=True)
    out = {}
    for c, n in zip(codes.tolist(), counts.tolist()):
        known = c in Quality._value2member_map_
        out[_LABELS[Quality(c)] if known else f"code {c}"] = n
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

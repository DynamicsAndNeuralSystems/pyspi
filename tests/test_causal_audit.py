"""Audit checks for `pyspi.statistics.causal`, against published definitions.

Each test has an independent oracle: a closed-form relation, a direct
re-implementation from the paper, or a synthetic system whose causal direction
is known by construction. Convergent cross-mapping is checked on the coupled
logistic maps of Sugihara et al. (2012), Science 338:496, where the result to
reproduce is the paper's own: if X drives Y and not the reverse, the shadow
manifold of Y cross-maps X with converging skill, and the shadow manifold of X
cannot cross-map Y.
"""
import numpy as np
import pytest

from pyspi.data import Data


def _coupled_logistic(T, beta_xy=0.0, beta_yx=0.1, rx=3.8, ry=3.5, burn=300):
    """Sugihara et al. (2012), Eq. (1) of the SOM.

    X(t+1) = X(t) [rx - rx X(t) - beta_xy Y(t)]
    Y(t+1) = Y(t) [ry - ry Y(t) - beta_yx X(t)]

    With beta_xy = 0, X is autonomous and drives Y: the true direction is X -> Y.
    """
    x, y = np.empty(T + burn), np.empty(T + burn)
    x[0], y[0] = 0.4, 0.2
    for t in range(T + burn - 1):
        x[t + 1] = x[t] * (rx - rx * x[t] - beta_xy * y[t])
        y[t + 1] = y[t] * (ry - ry * y[t] - beta_yx * x[t])
    return np.vstack([x[burn:], y[burn:]])


@pytest.fixture(scope="module")
def x_drives_y():
    return Data(_coupled_logistic(300), dim_order="ps")


# ---------------------------------------------------------------------------
# Convergent cross-mapping
# ---------------------------------------------------------------------------

def test_ccm_rows_are_sources_on_sugiharas_coupled_logistic_map(x_drives_y):
    """X drives Y, so A[X, Y] must carry the evidence: row = source.

    Sugihara et al. (2012): "if X causes Y, then ... the historical record of Y
    can be used to estimate states of X", i.e. *Y xmap X* is the skill that
    converges. pyEDM names that column ``Y:X`` (the library variable first),
    and pyspi's directed convention puts the cause on the row, so A[X, Y] must
    be Y-xmap-X: high. A[Y, X] is X-xmap-Y, which on this system is near zero.
    """
    from pyspi.statistics.causal import ConvergentCrossMapping

    X, Y = 0, 1
    for statistic in ("mean", "max"):
        A = np.asarray(ConvergentCrossMapping(
            statistic=statistic, embedding_dimension=2).multivariate(x_drives_y))
        assert A[X, Y] > 0.5 > A[Y, X], (
            f"ccm {statistic}: X drives Y, so A[X,Y] (Y xmap X) must be the "
            f"converging skill. Got A[X,Y]={A[X, Y]:.3f}, A[Y,X]={A[Y, X]:.3f}: "
            f"the matrix is transposed relative to row=source, column=target.")

    D = np.asarray(ConvergentCrossMapping(
        statistic="diff", embedding_dimension=2).multivariate(x_drives_y))
    assert D[X, Y] > 0.5 and D[Y, X] == -D[X, Y], (
        f"ccm diff must be positive from the driver: got D[X,Y]={D[X, Y]:.3f}")

"""
Test for phi native functionality using Calculator and direct function calls.
"""

import pytest
import numpy as np
import os
import tempfile
import yaml
from pyspi.calculator import Calculator


def test_phi_native_basic():
    """Test basic phi computation with native implementation using Calculator."""
    # Generate simple test data
    np.random.seed(42)
    dt = np.random.randn(2, 100)  # 2 variables, 100 time points

    # Create inline phi config
    config = {
        '.statistics.infotheory': {
            'IntegratedInformation': {
                'labels': ['undirected', 'nonlinear', 'unsigned', 'bivariate', 'time-dependent'],
                'configs': [
                    {'phitype': 'star'},
                    {'phitype': 'Geo'}
                ]
            }
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Create calculator with phi config
        calc = Calculator(dt, config=config_path)

        # Compute phi values
        calc.compute()

        # Basic assertions
        assert calc.table is not None
        assert len(calc.table) > 0

        # Check that phi computations completed without NaN
        for spi_name, spi_table in calc.table.items():
            assert not spi_table.isna().all().all(), f"All values are NaN for {spi_name}"

    finally:
        import os
        os.unlink(config_path)


def test_phi_types():
    """Test different phi types (star, Geo) with and without normalization."""
    np.random.seed(123)
    dt = np.random.randn(2, 50)

    # Create comprehensive phi config with normalization
    config = {
        '.statistics.infotheory': {
            'IntegratedInformation': {
                'labels': ['undirected', 'nonlinear', 'unsigned', 'bivariate', 'time-dependent'],
                'configs': [
                    {'phitype': 'star'},
                    {'phitype': 'star', 'normalization': 1},
                    {'phitype': 'Geo'},
                    {'phitype': 'Geo', 'normalization': 1}
                ]
            }
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        calc = Calculator(dt, config=config_path)
        calc.compute()

        # Should have at least one table with multiple phi configurations
        assert len(calc.table) > 0

        # Check that we get finite values
        finite_count = 0
        for spi_name, spi_table in calc.table.items():
            values = spi_table.values
            finite_values = values[np.isfinite(values)]
            if len(finite_values) > 0:
                finite_count += 1

        # At least some configurations should produce finite values
        assert finite_count > 0, "No phi configurations produced finite values"

    finally:
        import os
        os.unlink(config_path)


def test_phi_native_import():
    """Test that phi_native module can be imported and basic functions work."""
    from pyspi.lib.phi_native import phi_comp, logdet, cov_comp

    # Test basic function existence
    assert callable(phi_comp)
    assert callable(logdet)
    assert callable(cov_comp)

    # Test logdet with simple matrix
    test_matrix = np.eye(2) * 2
    result = logdet(test_matrix)
    expected = 2 * np.log(2)  # log det of 2*I
    assert np.isclose(result, expected)


def test_phi_comp_direct():
    """Test phi_comp function directly."""
    from pyspi.lib.phi_native import phi_comp

    np.random.seed(42)
    X = np.random.randn(2, 100)
    Z = np.array([1, 2])  # Partition
    params = {"tau": 1}
    options = {
        "type_of_phi": "star",
        "type_of_dist": "Gauss",
        "normalization": 0
    }

    result = phi_comp(X, Z, params, options)
    assert np.isfinite(result), "phi_comp should return finite value"
    assert isinstance(result, (int, float, np.number)), "phi_comp should return numeric value"


# ---------------------------------------------------------------------------
# Integrated information against its definitions
# ---------------------------------------------------------------------------
#
# The references below share no code with pyspi.lib.phi_native. Both work from
# the covariances of the past state X = x[:, :T-tau] and the present state
# Y = x[:, tau:], as the practical PhiToolbox's Cov_comp does, and both use
# the atomic bipartition of a pair -- the only partition IntegratedInformation
# ever passes.

_FIXTURES = os.path.join(os.path.dirname(__file__), "data", "fixtures")


def _past_present_covariances(X, tau=1):
    T = X.shape[1]
    past = X[:, :T - tau] - X[:, :T - tau].mean(axis=1, keepdims=True)
    present = X[:, tau:] - X[:, tau:].mean(axis=1, keepdims=True)
    n = T - tau - 1
    return past @ past.T / n, past @ present.T / n, present @ present.T / n


def _logdet(M):
    sign, value = np.linalg.slogdet(M)
    assert sign > 0
    return value


def _phi_g_by_direct_minimisation(X, tau=1):
    """phi_G from its definition (Oizumi, Tsuchiya & Amari 2016, PNAS).

    The disconnected model keeps each unit's own autoregression and cuts every
    cross-unit coefficient, Y = A' X + E' with A' block-diagonal; for a given
    A' the optimal noise covariance is Sigma_E + (A - A') Sigma_X (A - A')^T,
    so phi_G = min over block-diagonal A' of 0.5 * log(|that| / |Sigma_E|).
    For a pair with univariate parts A' is diagonal: two free numbers,
    minimised here by BFGS with the analytic gradient.
    """
    from scipy.optimize import minimize

    S_X, S_XY, S_Y = _past_present_covariances(X, tau)
    A = np.linalg.solve(S_X, S_XY).T
    S_E = S_Y - S_XY.T @ np.linalg.solve(S_X, S_XY)

    def objective(a):
        D = A - np.diag(a)
        S = S_E + D @ S_X @ D.T
        grad = -np.diag(np.linalg.solve(S, D @ S_X))
        return 0.5 * (_logdet(S) - _logdet(S_E)), grad

    return min(minimize(objective, x0, jac=True, method="BFGS",
                        options={"gtol": 1e-14}).fun
               for x0 in (np.diag(A).copy(), np.zeros(2)))


def _phi_star_by_mismatched_decoding(X, tau=1):
    """phi* = I(X;Y) - max_beta I*(beta) (Oizumi et al. 2016, PLoS Comput Biol).

    Written from the definition of the mismatched-decoding information with the
    disconnected model q(Y|X) = prod_i p(Y_i | X_i) = N(Y; A_D X, C_D):

        I*(beta) = -E_Y log E_X q(Y|X)^beta + E_XY log q(Y|X)^beta.

    For beta > 0, q^beta is proportional to N(Y; A_D X, C_D / beta), so the
    X-average is a Gaussian convolution, N(Y; 0, S_beta) with
    S_beta = A_D Sigma_X A_D^T + C_D / beta, and

        I*(beta) = 0.5 log|beta S_beta| - 0.5 log|C_D|
                   + 0.5 tr(S_beta^-1 Sigma_Y) - beta N / 2.

    This route never forms the Q / R matrices phi_native uses.
    """
    from scipy.optimize import minimize_scalar

    S_X, S_XY, S_Y = _past_present_covariances(X, tau)
    S_YX = S_XY.T
    N = S_X.shape[0]
    mi = 0.5 * (_logdet(S_Y) - _logdet(S_Y - S_YX @ np.linalg.solve(S_X, S_XY)))
    A_D = np.diag(np.diag(S_YX) / np.diag(S_X))
    C_D = np.diag(np.diag(S_Y) - np.diag(S_YX) ** 2 / np.diag(S_X))

    def mismatched(beta):
        S = A_D @ S_X @ A_D.T + C_D / beta
        return (0.5 * _logdet(beta * S) - 0.5 * _logdet(C_D)
                + 0.5 * np.trace(np.linalg.solve(S, S_Y)) - beta * N / 2)

    best = minimize_scalar(lambda log_beta: -mismatched(np.exp(log_beta)),
                           bracket=(-3.0, 0.0, 3.0), tol=1e-12)
    return mi + best.fun


def _ar1_pair(seed, T=1000, a=(0.98, 0.95)):
    """Two independent, strongly autocorrelated AR(1) series, standardised."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((2, T))
    X = np.zeros((2, T))
    for t in range(1, T):
        X[:, t] = np.asarray(a) * X[:, t - 1] + noise[:, t]
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)


def _kuramoto_pair(i, j):
    from pyspi.data import Data
    Z = Data(os.path.join(_FIXTURES, "kuramoto_M7_T100.npy"),
             dim_order="sp").to_numpy(squeeze=True)
    return Z[[i, j]]


def _phi(X, phitype, normalization=0):
    from pyspi.lib.phi_native import phi_comp
    return phi_comp(X, np.array([1, 2]), {"tau": 1},
                    {"type_of_phi": phitype, "type_of_dist": "Gauss",
                     "normalization": normalization})


_PHI_CASES = [
    pytest.param(lambda: _kuramoto_pair(1, 5), id="kuramoto-1-5"),
    pytest.param(lambda: _kuramoto_pair(2, 3), id="kuramoto-2-3"),
] + [pytest.param(lambda s=s: _ar1_pair(s), id=f"ar1-seed{s}")
     for s in (0, 1, 45, 47, 49)]


@pytest.mark.parametrize("make_pair", _PHI_CASES)
def test_phi_g_is_the_constrained_minimum_of_its_definition(make_pair):
    """phi_G must be evaluated at a disconnected model, and at its minimum.

    The cases include weakly coupled, strongly autocorrelated pairs (kuramoto
    (1, 5), AR(1) seeds 45/47/49), where a solver that stops on a small change
    in log|Sigma_E'| before the cross-unit coefficients reach zero is far off.
    """
    X = make_pair()
    expected = _phi_g_by_direct_minimisation(X)
    assert _phi(X, "Geo") == pytest.approx(expected, rel=1e-8, abs=1e-13)


@pytest.mark.parametrize("make_pair", _PHI_CASES)
def test_phi_star_matches_the_mismatched_decoding_definition(make_pair):
    """abs=1e-9 is the beta search's own resolution, not slack: phi_native
    maximises I*(beta) with L-BFGS-B at its default ftol (2.2e-9 absolute
    for |I*| < 1), and on the weakly coupled ar1 pairs that leaves 3e-11."""
    X = make_pair()
    expected = _phi_star_by_mismatched_decoding(X)
    assert _phi(X, "star") == pytest.approx(expected, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("phitype", ["star", "Geo"])
def test_phi_normalisation_divides_by_the_smaller_part_entropy(phitype):
    """normalization=1 is phi / min_i H(X_i past), PhiToolbox's definition."""
    X = _kuramoto_pair(0, 6)
    S_X, _, _ = _past_present_covariances(X)
    H_parts = 0.5 * np.log(2 * np.pi * np.e * np.diag(S_X))
    assert _phi(X, phitype, 1) == pytest.approx(
        _phi(X, phitype, 0) / H_parts.min(), rel=1e-12)


@pytest.mark.parametrize("phitype", ["star", "Geo"])
def test_phi_is_symmetric_in_the_pair(phitype):
    """IntegratedInformation is declared undirected; the atomic bipartition of
    (i, j) and of (j, i) is the same partition."""
    X = _kuramoto_pair(0, 4)
    assert _phi(X, phitype) == pytest.approx(_phi(X[::-1].copy(), phitype),
                                             rel=1e-8, abs=1e-13)

"""
Native Python implementation of integrated information (phi) calculation.

This module provides a pure Python implementation of the phi computation
algorithms from the PhiToolbox, eliminating the need for MATLAB/Octave emulation.

Based on:
- Oizumi et al., 2016, PLoS Comp Biol
- Original PhiToolbox MATLAB implementation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve, inv, det, LinAlgError
import warnings


def logdet(X):
    """
    Compute log determinant in a numerically stable way.

    Args:
        X: Square matrix

    Returns:
        Log determinant of X
    """
    n = X.shape[0]

    try:
        # Use Cholesky decomposition for positive definite matrices
        L = np.linalg.cholesky(X)
        return 2 * np.sum(np.log(np.diag(L)))
    except LinAlgError:
        # Fallback to eigenvalue decomposition for general case
        eigenvals = np.linalg.eigvals(X)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Filter out numerical zeros
        if len(eigenvals) == 0:
            return -np.inf
        return np.sum(np.log(eigenvals))


def cov_comp(X, tau, isjoint=True):
    """
    Compute covariance matrices from time series data.

    Args:
        X: Time series data (units x time)
        tau: Time lag between past and present states
        isjoint: Whether to compute joint covariance matrices

    Returns:
        Dictionary containing covariance matrices
    """
    T = X.shape[1]

    if isjoint:
        t_range1 = np.arange(T - tau)
        t_range2 = np.arange(tau, T)

        X1 = X[:, t_range1]
        X1 = X1 - X1.mean(axis=1, keepdims=True)  # subtract mean
        X2 = X[:, t_range2]
        X2 = X2 - X2.mean(axis=1, keepdims=True)  # subtract mean

        Cov_X = X1 @ X1.T / (T - tau - 1)  # equal-time covariance at "PAST"
        Cov_Y = X2 @ X2.T / (T - tau - 1)  # equal-time covariance at "PRESENT"
        Cov_XY = X1 @ X2.T / (T - tau - 1)  # time-lagged covariance

        return {
            'Cov_X': Cov_X,
            'Cov_Y': Cov_Y,
            'Cov_XY': Cov_XY
        }
    else:
        X_centered = X - X.mean(axis=1, keepdims=True)
        Cov_X = X_centered @ X_centered.T / (T - 1)
        return {'Cov_X': Cov_X}


def cov_cond(Cov_X, Cov_XY, Cov_Y):
    """
    Compute conditional covariance of X given Y.

    Args:
        Cov_X: Covariance of X
        Cov_XY: Cross-covariance of X and Y
        Cov_Y: Covariance of Y

    Returns:
        Conditional covariance of X given Y
    """
    try:
        return Cov_X - Cov_XY @ solve(Cov_Y, Cov_XY.T)
    except LinAlgError:
        # Fallback for singular matrices
        Cov_Y_inv = np.linalg.pinv(Cov_Y)
        return Cov_X - Cov_XY @ Cov_Y_inv @ Cov_XY.T


def h_gauss(Cov_X):
    """
    Calculate entropy under Gaussian assumption.

    Args:
        Cov_X: Covariance matrix

    Returns:
        Gaussian entropy
    """
    n = Cov_X.shape[0]
    return 0.5 * logdet(Cov_X) + 0.5 * n * np.log(2 * np.pi * np.e)


def i_s_objective(beta, C_D_beta1_inv, Cov_X_inv, Cov_X, Cov_Y, C_D_cond, S_left, S_right, I_s_d_Const_part):
    """
    Objective function for beta optimization in phi_star computation.

    Args:
        beta: Optimization parameter
        Various precomputed matrices and constants

    Returns:
        Negative I_s (for minimization)
    """
    C_D_beta_inv = beta * C_D_beta1_inv
    Q_inv = Cov_X_inv + C_D_beta_inv

    try:
        norm_t = 0.5 * logdet(Q_inv) + 0.5 * logdet(Cov_X)

        # Compute R term
        C_D_cond_inv = inv(C_D_cond)
        Q_inv_S_right = solve(Q_inv, S_right)
        R = beta * C_D_cond_inv - beta**2 * S_left @ Q_inv_S_right

        trace_t = 0.5 * np.trace(Cov_Y @ R)
        I_s = norm_t + trace_t - beta * I_s_d_Const_part

        return -I_s  # Return negative for minimization
    except LinAlgError:
        return np.inf


def phi_star_gauss(Cov_X, Cov_XY, Cov_Y, Z, beta_init=1.0, normalization=0):
    """
    Calculate integrated information phi_star based on mismatched decoding.

    Args:
        Cov_X: Covariance of past data
        Cov_XY: Cross-covariance between past and present
        Cov_Y: Covariance of present data
        Z: Partition array indicating group membership
        beta_init: Initial value for optimization
        normalization: Whether to normalize by entropy

    Returns:
        Tuple of (phi_star, mutual_information, optimal_beta)
    """
    N = Cov_X.shape[0]

    # Compute conditional covariance and entropy
    Cov_Y_X = cov_cond(Cov_Y, Cov_XY.T, Cov_X)
    H_cond = h_gauss(Cov_Y_X)

    if np.isinf(H_cond):
        warnings.warn("Infinite entropy detected")
        return np.nan, np.nan, np.nan

    if np.iscomplex(H_cond):
        warnings.warn("Complex entropy detected")
        return np.nan, np.nan, np.nan

    H = h_gauss(Cov_Y)
    I = H - H_cond  # mutual information

    # Setup partition matrices
    N_c = int(np.max(Z))
    M_cells = []
    for i in range(1, N_c + 1):
        M_cells.append(np.where(Z == i)[0])

    # Initialize decomposed matrices
    X_D = np.zeros((N, N))
    YX_D = np.zeros((N, N))
    C_D_cond = np.zeros((N, N))

    for i, M in enumerate(M_cells):
        Cov_X_p = Cov_X[np.ix_(M, M)]
        Cov_Y_p = Cov_Y[np.ix_(M, M)]
        Cov_YX_p = Cov_XY[np.ix_(M, M)].T
        Cov_Y_X_p = cov_cond(Cov_Y_p, Cov_YX_p, Cov_X_p)

        X_D[np.ix_(M, M)] = Cov_X_p
        YX_D[np.ix_(M, M)] = Cov_YX_p
        C_D_cond[np.ix_(M, M)] = Cov_Y_X_p

    # Compute optimization matrices
    try:
        Cov_X_inv = inv(Cov_X)

        # Precompute terms for optimization
        X_D_inv_YX_D_T = solve(X_D, YX_D.T)
        C_D_cond_inv_YX_D = solve(C_D_cond, YX_D)
        X_D_inv = solve(X_D, np.eye(N))

        C_D_beta1_inv = X_D_inv_YX_D_T @ C_D_cond_inv_YX_D @ X_D_inv
        S_left = solve(C_D_cond.T, YX_D) @ X_D_inv
        S_right = X_D_inv_YX_D_T @ solve(C_D_cond, np.eye(N))

        I_s_d_Const_part = 0.5 * N

        # Optimize beta
        result = minimize(
            i_s_objective,
            beta_init,
            args=(C_D_beta1_inv, Cov_X_inv, Cov_X, Cov_Y, C_D_cond, S_left, S_right, I_s_d_Const_part),
            method='L-BFGS-B'
        )

        beta_opt = result.x[0] if hasattr(result.x, '__len__') else result.x
        I_s = -result.fun

    except (LinAlgError, np.linalg.LinAlgError):
        warnings.warn("Matrix inversion failed in phi_star computation")
        return np.nan, I, np.nan

    # Compute phi_star
    phi_star = I - I_s

    # Apply normalization if requested
    if normalization == 1:
        H_p = np.zeros(N_c)
        for i, M in enumerate(M_cells):
            Cov_X_p = Cov_X[np.ix_(M, M)]
            H_p[i] = h_gauss(Cov_X_p)

        if N_c == 1:
            phi_star = phi_star / H_p[0]
        else:
            phi_star = phi_star / ((N_c - 1) * np.min(H_p))

    return phi_star, I, beta_opt


def phi_g_gauss_al(Cov_X, Cov_E, A, Z, normalization=0, maxiter=100000, error=1e-10):
    """
    Calculate phi_G using Augmented Lagrangian method.

    Args:
        Cov_X: Equal time covariance of X (past)
        Cov_E: Covariance of noise E
        A: Connectivity (autoregressive) matrix
        Z: Partition array
        normalization: Whether to normalize by entropy
        maxiter: Maximum iterations
        error: Convergence threshold

    Returns:
        Tuple of (phi_G, Cov_E_p, A_p)
    """
    mu = 2.0
    alpha = 0.9
    gamma = 1.01

    n = Cov_X.shape[0]
    N_c = int(np.max(Z))

    # Create partition cells
    M_cells = []
    for i in range(1, N_c + 1):
        M_cells.append(np.where(Z == i)[0])

    # Initialize disconnected connectivity matrix
    A_p = np.zeros((n, n))
    for M in M_cells:
        A_p[np.ix_(M, M)] = A[np.ix_(M, M)]

    B = A_p.copy()
    Lambda = np.zeros((n, n))

    # Eigendecomposition of Cov_X
    D_Cov_X, Q = np.linalg.eigh(Cov_X)

    Cov_E_p = Cov_E.copy()
    val_constraint_past = 0

    for iter_count in range(maxiter):
        Cov_E_p_past = Cov_E_p.copy()

        # Update covariance
        A_diff = A - A_p
        Cov_E_p = Cov_E + A_diff @ Cov_X @ A_diff.T

        # Eigendecomposition of Cov_E_p
        try:
            D_Cov_E_p, P = np.linalg.eigh(Cov_E_p)

            # Ensure positive eigenvalues
            D_Cov_E_p = np.maximum(D_Cov_E_p, 1e-12)

            # Update A_p using the Augmented Lagrangian formula
            term1 = P.T @ (B + Lambda / mu) @ Q
            term2 = np.linalg.solve(np.diag(D_Cov_E_p), P.T @ A @ Q @ np.diag(D_Cov_X) / mu)

            # Element-wise division with broadcasting
            denominator = 1 + D_Cov_X[np.newaxis, :] / (D_Cov_E_p[:, np.newaxis] * mu)
            numerator = term1 + term2

            A_p = P @ (numerator / denominator) @ Q.T

        except np.linalg.LinAlgError:
            warnings.warn("Linear algebra error in phi_G computation")
            return np.nan, Cov_E_p, A_p

        # Update B with partition constraints
        B = A_p / 2
        for M in M_cells:
            B[np.ix_(M, M)] *= 2

        # Update Lagrangian multiplier
        Lambda = Lambda - mu * (A_p - B)

        # Check constraint satisfaction
        val_constraint = np.sum((A_p - B)**2)
        if iter_count > 0:
            if val_constraint > alpha * val_constraint_past:
                mu = gamma * mu
        val_constraint_past = val_constraint

        # Check convergence
        phi_update = logdet(Cov_E_p) - logdet(Cov_E_p_past)
        if abs(phi_update) < error:
            break

    # Calculate phi_G
    phi_G = (logdet(Cov_E_p) - logdet(Cov_E)) / 2

    # Apply normalization if requested
    if normalization == 1:
        H_p = np.zeros(N_c)
        for i, M in enumerate(M_cells):
            Cov_X_p = Cov_X[np.ix_(M, M)]
            H_p[i] = h_gauss(Cov_X_p)

        if N_c == 1:
            phi_G = phi_G / H_p[0]
        else:
            phi_G = phi_G / ((N_c - 1) * np.min(H_p))

    return phi_G, Cov_E_p, A_p


def phi_g_gauss(Cov_X, Cov_XY, Cov_Y, Z, method=None, normalization=0):
    """
    Calculate integrated information phi_G based on information geometry.

    Args:
        Cov_X: Covariance of past data
        Cov_XY: Cross-covariance between past and present
        Cov_Y: Covariance of present data
        Z: Partition array
        method: Optimization method ('AL' for Augmented Lagrangian, 'LI' for LBFGS+Iterative)
        normalization: Whether to normalize by entropy

    Returns:
        Tuple of (phi_G, Cov_E_p, A_p)
    """
    # Transform covariance matrices to AR model parameters: Y = AX + E
    try:
        A = solve(Cov_X, Cov_XY).T  # A = Cov_XY' / Cov_X
        Cov_E = Cov_Y - Cov_XY.T @ solve(Cov_X, Cov_XY)  # Cov_E = Cov_Y - Cov_XY'/Cov_X*Cov_XY
    except np.linalg.LinAlgError:
        warnings.warn("Singular covariance matrix in phi_G computation")
        return np.nan, None, None

    # Choose method based on number of groups if not specified
    if method is None:
        K = len(np.unique(Z))
        method = 'AL' if K <= 3 else 'LI'

    # For now, only implement AL method
    if method == 'AL':
        return phi_g_gauss_al(Cov_X, Cov_E, A, Z, normalization)
    else:
        # Fall back to AL method for now
        warnings.warn(f"Method '{method}' not implemented, using 'AL' instead")
        return phi_g_gauss_al(Cov_X, Cov_E, A, Z, normalization)


def phi_comp(X, Z, params, options):
    """
    Main function to compute phi from time series data.
    This provides the same interface as the MATLAB phi_comp function.

    Args:
        X: Time series data (units x time)
        Z: Partition array
        params: Dictionary with parameters (must contain 'tau')
        options: Dictionary with options (must contain 'type_of_phi', 'type_of_dist', 'normalization')

    Returns:
        Phi value
    """
    # Extract parameters
    tau = params.get('tau', 1)
    phi_type = options.get('type_of_phi', 'star')
    dist_type = options.get('type_of_dist', 'Gauss')
    normalization = options.get('normalization', 0)
    phi_g_method = options.get('phi_G_OptimMethod', None)

    # Only support Gaussian distributions for now
    if dist_type != 'Gauss':
        raise NotImplementedError("Only Gaussian distributions are currently supported")

    # Compute probability distributions (covariances for Gaussian case)
    isjoint = phi_type != 'MI1'
    probs = cov_comp(X, tau, isjoint)

    # Compute phi based on type
    if phi_type == 'star':
        phi, _, _ = phi_star_gauss(
            probs['Cov_X'],
            probs['Cov_XY'],
            probs['Cov_Y'],
            Z,
            beta_init=1.0,
            normalization=normalization
        )
        return phi
    elif phi_type == 'Geo':
        phi, _, _ = phi_g_gauss(
            probs['Cov_X'],
            probs['Cov_XY'],
            probs['Cov_Y'],
            Z,
            method=phi_g_method,
            normalization=normalization
        )
        return phi
    else:
        raise NotImplementedError(f"Phi type '{phi_type}' is not yet implemented")
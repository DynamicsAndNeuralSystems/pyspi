from .numpy_dependence import compute_IDS_numpy

def compute_IDS(X, Y=None, num_terms=6, p_norm='max', 
                p_val=False, num_tests=100, bandwidth_term=1/2):
    """Compute IDS between all pairs of variables in X (or between X and Y).
    
    Taken from the implementation in: https://github.com/aradha/interdependence_scores
    

    Parameters:
        X: np.ndarray or torch.Tensor
        Y: np.ndarray or torch.Tensor (optional)
        num_terms: Number of terms for Taylor series approximation (optional)
        p_norm: String 'max' if using IDS-max.  1 or 2 for IDS-1, IDS-2, respectively.  (optional)
        p_val: Boolean.  Indicates whether to compute p-values using permutation tests
        num_tests: Number of permutation tests if p_val=True
        bandwidth_term: Constant term in Gaussian kernel
    Returns:
        IDS matrix, p-value matrix (if p_val=True)
    """
    return compute_IDS_numpy(X, Y=Y, num_terms=num_terms, p_norm=p_norm, 
                                 p_val=p_val, num_tests=num_tests, bandwidth_term=bandwidth_term)
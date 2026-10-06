import numpy as np

from pyspi.base_new import DirectedNewSPI

class TransferEntropyNewSPI(DirectedNewSPI):
    """
    Transfer Entropy implementation for the new PySPI architecture.
    
    Transfer Entropy (TE) is an information-theoretic measure of directed
    information transfer between two random processes. It quantifies how much knowing
    the past of one process (source) improves the prediction of another process,
    beyond what the target's own past provides.
    
    parameters
    ----------
    lag_source : int, default=1
        The lag to be used for the source process.
    lag_target : int, default=1
        The lag to be used for the target process.
    estimator : str, default="binning"
        Method to estimate probability distributions.
    n_bins : int, default=10
        Number of bins to use for the binning estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from pyspu.data import Data
    >>> from pyspi.base_new import TransferEntropyNewSPI
    >>> data = Data(np.random.rand(100, 2))  # Two processes
    >>> te = TransferEntropyNewSPI(lag_source=1, lag_target=1, estimator="binning", n_bins=10)
    >>> result = te.spi_mat(data)
    >>> print(result.shape) # (2, 2)
    """

    _tags = {
        "capability-directed": True,
        "capability-signed": False,
        "capability-multivariate": True,
        "identifier": "transfer_entropy",
        "name": "TransferEntropyNewSPI",
        "python_dependencies": ["numpy", "scipy"]
    }

    def __init__(self, lag_source=1, lag_target=1, estimator="binning", n_bins=10, **kwargs):
        self.lag_source = lag_source
        self.lag_target = lag_target
        self.estimator = estimator
        self.n_bins = n_bins
        super().__init__(**kwargs)

    def _spi(self, data, i=0):
        """
        Compute unvariate transfer entropy for a single process.
        Returns 0.0, since TE requires 2 processes
        
        Parameters
        ----------
        data : Data
            The input data used for computation
        i : int
            Index of the process
        
        Returns
        -------
        float 
            Always returns 0.0 as TE is inherently bivariate.
        """
        return 0.0

    def _compute_pair(self, data, data2, i, j):
        """
        Compute transfer entropy from process i to process j.
        
        Parameters
        ----------
        data : Data
            The input data used for computation
        data2 : Data
            The second input data used for computation
        i : int
            Index of the source process
        j : int
            Index of the target process
        
        Returns
        -------
        float
            The transfer entropy from process i to process j.
        """
        if data is data2 and i==j:
            return 0.0

        x_src = data.processes[i]
        y_tgt = data.processes[j]

        return self._get_transfer_entropy(x_src, y_tgt)

    def _get_transfer_entropy(self, x_src, y_tgt):
        """Compute Transfer Entropy from source to target time series.
        
        Parameters
        ----------
        x_src : np.ndarray
            Source time series.
        y_tgt : np.ndarray
            Target time series.

        Returns
        -------
        float 
            Tranfer Entropy value
        """
        x = x_src.flatten() if x_src.ndim > 1 else x_src
        y = y_tgt.flatten() if y_tgt.ndim > 1 else y_tgt

        min_length = max(self.lag_source, self.lag_target) + 1
        if len(x) < min_length or len(y) < min_length:
            raise ValueError(f"Input time series must be at least {min_length} samples long.")
        
        # Create lagged versions
        start_idx = max(self.lag_source, self.lag_target)
        x_lagged = np.array([x[i - self.lag_source] for i in range(start_idx, len(x))])
        y_lagged = np.array([y[i - self.lag_target] for i in range(start_idx, len(y))])
        y_future = np.array([y[i] for i in range(start_idx, len(y))])

        if len(x_lagged) == 0 or len(y_lagged) == 0 or len(y_future) == 0:
            return np.nan
        
        if self.estimator == "binning":
            # Discretize the data
            x_bins = np.histogram_bin_edges(x, bins=self.n_bins)
            y_bins = np.histogram_bin_edges(y, bins=self.n_bins)
            
            x_lagged_binned = np.digitize(x_lagged, x_bins)
            y_lagged_binned = np.digitize(y_lagged, y_bins) 
            y_future_binned = np.digitize(y_future, y_bins)
            
            # Compute probability distributions
            p_yfuture_ylag = self._joint_prob(y_future_binned, y_lagged_binned)
            p_yfuture_ylag_xlag = self._joint_prob_3d(y_future_binned, y_lagged_binned, x_lagged_binned)
            
            p_ylag = self._marginal_prob(y_lagged_binned)
            p_ylag_xlag = self._joint_prob(y_lagged_binned, x_lagged_binned)
            
            # Compute conditional entropies
            H1 = self._conditional_entropy(p_yfuture_ylag, p_ylag)
            H2 = self._conditional_entropy(p_yfuture_ylag_xlag, p_ylag_xlag)
            
            te_value = H1 - H2
            return max(0.0, te_value)  # TE should be non-negative
        else:
            raise NotImplementedError(f"Estimator {self.estimator} not implemented yet.")

    def _joint_prob(self, x, y):
        """Estimate 2D joint probability distribution."""
        if len(x) == 0 or len(y) == 0:
            return np.array([[1e-10]])
            
        bins = (max(1, int(np.max(x)) + 1), max(1, int(np.max(y)) + 1))
        hist, _, _ = np.histogram2d(x, y, bins=bins)
        return hist / np.sum(hist) if np.sum(hist) > 0 else hist + 1e-10
    
    def _joint_prob_3d(self, x, y, z):
        """Estimate 3D joint probability distribution."""
        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            return np.array([[[1e-10]]])
            
        xyz = np.vstack((x, y, z)).T
        unique, counts = np.unique(xyz, axis=0, return_counts=True)
        
        max_x = max(1, int(np.max(x)) + 1)
        max_y = max(1, int(np.max(y)) + 1) 
        max_z = max(1, int(np.max(z)) + 1)
        
        hist = np.zeros((max_x, max_y, max_z))
        for (i, j, k), c in zip(unique, counts):
            hist[int(i), int(j), int(k)] = c
        return hist / np.sum(hist) if np.sum(hist) > 0 else hist + 1e-10
    
    def _marginal_prob(self, x):
        """Estimate 1D marginal probability distribution."""
        if len(x) == 0:
            return np.array([1e-10])
            
        bins = max(1, int(np.max(x)) + 1)
        hist, _ = np.histogram(x, bins=bins)
        return hist / np.sum(hist) if np.sum(hist) > 0 else hist + 1e-10
    
    def _conditional_entropy(self, joint, marginal):
        """Calculate conditional entropy H(Y|X) = H(X,Y) - H(X)."""
        eps = 1e-10
        
        if np.sum(joint) == 0 or np.sum(marginal) == 0:
            return 0.0
            
        joint = np.clip(joint, eps, 1.0)
        marginal = np.clip(marginal, eps, 1.0)
        
        H_joint = -np.sum(joint * np.log(joint + eps))
        H_marginal = -np.sum(marginal * np.log(marginal + eps))
        
        return H_joint - H_marginal



        


        

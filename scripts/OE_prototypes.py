import numpy as np

def RtoS(R, s):
    """
    Convert correlation matrix (R) to covariance matrix (S)

    Parameters
    ----------
    R : ndarray
        a 2-dimensional ndarray, shaped (n,n), containing the correlation matrix.
    s : ndarray
        a 1-dimensional ndarray containing the per variable uncertainty (e.g.,
        the square root of the diagonal), shaped (n,)
    Returns
    -------
    S : ndarray
        a 2-dimensional, square ndarray, shaped (n,n), with the covariance matrix

    """
    S = np.outer(s, s) * R
    return S

def StoR(S):
    """
    convert covariance matrix (S) to correlation matrix (R)

    Parameters
    ----------
    S : ndarray
        a 2-dimensional, square ndarray, shaped (n,n), with the covariance matrix

    Returns
    -------
    R : ndarray
        a 2-dimensional ndarray, shaped (n,n), containing the correlation matrix.
    s : ndarray
        a 1-dimensional ndarray containing the per variable uncertainty (e.g.,
        the square root of the diagonal), shaped (n,)
    """
    s = np.sqrt(np.diag(S))
    S_inv = np.diag(1/s)
    R = S_inv @ S @ S_inv
    return R, s

def level_exp_cov_matrix(levels, s, corr_length):
    """
    compute a covariance matrix (S) given an input level array, standard
    deviation and correlation length per level. This implements an analytic
    expression for the correlation length (see below), that follows a
    negative exponential.
    The input correlation length is assumed to be in the same units as the
    levels array. This could be pressure, or altitude in km.

    The correlation matrix entry R[i,j], will be equal to:
    exp( -|z_i - z_j| / corr_length )
    where z_i and z_j are the values of the levels input at i, j, and the
    corr_length is equal to the minimum correlation length at i, j.

    Parameters
    ----------
    levels : ndarray
        1D array, shaped (n,) with the level positions. Typically this would be
        level pressures or level geometric altitudes.
    s : ndarray
        1D array, same shape as levels. This contains the per-level standard 
        deviation; the diagonal of the returned S array will contain s^2.
    corr_length : ndarray
        1D array, same shape as levels. This contains the per-level correlation
        length, and is assumed to be in the same units as levels.
    """
    nlevels = levels.shape[0]
    R = np.zeros((nlevels, nlevels))

    for i, j in np.ndindex(R.shape):
        A = -1.0 / min(corr_length[i],corr_length[j])
        R[i,j] = np.exp(A * np.abs(levels[i]-levels[j]))

    S = RtoS(R, s)

    return S, R


def compute_A(K, Se, Sa, method=1, diagonal_Se=False):
    """
    compute the averaging kernel matrix.

    Parameters
    ----------
    K : ndarray
        Jacobian, the derivative of the measurement space quantity with respect
        to the state space quantity. Shape is (m,n), where m is the number of
        measurement variables (typically, the number of spectral points)
        and n is the number of state variables.
    Se : ndarray
        Measurement noise array. This method does not assume a diagonal form
        for Se; this is a square array shaped (m,m)
    Sa : ndarray:
        A priori state covariance. Shaped (n,n)
    diagonal_Se : bool
        Set to true if the Se matrix is diagonal. In this case, the Se inverse
        calculation can be done with a much faster shortcut calculation.

    Returns
    -------
    A : ndarray
        The averaging kernel matrix, shaped (n,n)

    Notes
    -----
    The assumed units in K, Se, and Sa, must all be in agreement.
    The DOFS can be computed by taking the trace of A.
    """
    if method == 1:
        if diagonal_Se:
            SeI = np.diag(1.0/np.diag(Se))
        else:
            SeI = np.linalg.inv(Se)
        KSeIK = K.T @ SeI @ K
        inv_hatS = KSeIK + np.linalg.inv(Sa)
        hatS = np.linalg.inv(inv_hatS)
        A = hatS @ KSeIK
    elif method == 2:
        SaK = Sa @ K.T
        tmp = np.linalg.inv(K @ SaK + Se)
        A = SaK @ tmp @ K
    else:
        raise ValueError('unknown method')
    
    return A


def random_correlated(M, S, K=100):
    """
    Create N-dimensional normally distributed, correlated
    random variables according to an input mean M and covariance S.

    Parameters:
    -----------
    M : ndarray
        a 1-D array with the variable mean. This is shaped (N) or (N,1)
    S : ndarray
        a 2-D array with the covariance, with shape (N, N). This should be
        a valid covariance (e.g., symmetric and square).
    K : int
        optional, the number of realizations. default is 100.

    Returns:
    --------
    x : ndarray
        The K correlated random variables. Each single realization is
        really a 1D array (a vector) with shape (N,); the output array x
        is shaped (N,K) to contain the K realizations.
    """

    M = np.asarray(M)
    S = np.asarray(S)

    if M.ndim == 1:
        M = M[:, np.newaxis]
    N = M.shape[0]

    if (S.shape[0] != N) or (S.shape[1] != N):
        raise ValueError("Shape input mismatch between M and S")

    U, sigma, V = np.linalg.svd(S)
    W = U @ np.diag(np.sqrt(sigma))

    x_uncorr = np.random.normal(loc=0.0, scale=1.0, size = (N, K))
    x_corr =  W @ x_uncorr + M

    return x_corr

import numpy as np
from process_model import *

def measurement_model(x):
    z = np.array([0, 0, 0, 0, 0, 0])
    z[0:3] = x[0:3]
    z[3:6] = x[6:9]
    return z

def getSigmaPoints(x, P, n):
    """
    Function to compute sigma points used in unscented transform
    
    Args:
        x (numpy.ndarray): State vector of size 15x1.
        P (numpy.ndarray): Covariance matrix of size 15x15.
        n (int): Dimensionality of the state vector : 15
    
    Returns:
        X (numpy.ndarray): Sigma point states
        wm (numpy.ndarray): weight for mean
        wc (numpy.ndarray): weight for variance
    """

    # Calculate the square root of P
    A = np.linalg.cholesky(P).T

    # Set the scale factor
    kappa = 3 - n

    # Calculate the sigma points
    X = np.zeros((n, 2*n+1))
    wm = np.zeros(2*n+1)
    wc = np.zeros(2*n+1)

    # Set the first sigma point
    X[:,0] = x.flatten()

    # Set the weights
    wm[0] = kappa / (n + kappa)
    wc[0] = kappa / (n + kappa) + (1 - x[0]**2 + x[2]**2 + x[4]**2 + x[6]**2 + x[8]**2 + x[10]**2 + x[12]**2 + x[14]**2) / 2

    for i in range(1, n+1):
        X[:, i] = (x.squeeze() + A[:, i-1]).reshape(-1)
        X[:,n+i] = (x.squeeze() - A[:,i-1]).reshape(-1)

        wm[i] = 1 / (2 * (n + kappa))
        wc[i] = 1 / (2 * (n + kappa))

        wm[n+i] = 1 / (2 * (n + kappa))
        wc[n+i] = 1 / (2 * (n + kappa))

    return X, wm, wc
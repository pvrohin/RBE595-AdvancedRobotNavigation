import numpy as np
import scipy.io
from extract_pose import estimate_pose

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

# Function to estimate observation model covariance
def estimate_covariances(filename):
    # Load data
    data = load_data(filename)
    
    # Initialize variables
    n = len(data['data'])  # Number of time steps
    R_sum = np.zeros((6, 6))  # Initialize sum of outer products
    
    # Iterate over each time step
    for i in range(n):
        # Extract ground truth orientation data from vicon field
        orientation = data['vicon'][i][3:6]
        
        # Rearrange observation model equation to solve for noise term nu_t
        z = np.concatenate((data['vicon'][i][:3], orientation))
        z_hat = np.concatenate((estimate_pose(data['data'][i])[0], orientation))
        nu_t = z - z_hat
        
        # Compute outer product of nu_t
        R_sum += np.outer(nu_t, nu_t)
    
    # Compute sample covariance matrix
    R = R_sum / (n - 1)
    
    return R

# Call the function with the filename of the .mat file containing the data
R = estimate_covariances('data/studentdata0.mat')
print("Estimated Observation Model Covariance (R):\n", R)

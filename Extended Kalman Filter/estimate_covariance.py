import numpy as np
import scipy.io
from extract_pose_venky import estimate_pose, world_corners

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

# Function to estimate observation model covariance
def estimate_covariances(data):
    
    # Initialize variables
    n = len(data['data'])  # Number of time steps
    R_sum = np.zeros((6, 6))  # Initialize sum of outer products

    tag_coordinates = world_corners()
    
    # Iterate over each time step
    for i in range(n):
        # Extract ground truth orientation data from vicon field
        position = data['vicon'][i][:3]
        orientation = data['vicon'][i][3:6]

        if len(data['data'][i]['id']) == 0:
            continue
        
        # Rearrange observation model equation to solve for noise term nu_t
        z = np.concatenate((position, orientation))
        z_hat = estimate_pose(data['data'][i], tag_coordinates)
        z_hat = np.array(z_hat)
        #Reshape z_hat to be (6,)
        z_hat = z_hat.reshape(6,)

        nu_t = z - z_hat
        
        # Compute outer product of nu_t
        R_sum += np.outer(nu_t, nu_t)
    
    # Compute sample covariance matrix
    R = R_sum / (n - 1)
    
    return R

def main():
    # Call the function with the filename of the .mat file containing the data
    # Load data
    filename = 'data/studentdata0.mat'
    data = load_data(filename)

    #Loop through the data and print the tag IDs
    for i in range(len(data['data'])):
        # If the tag id is an integer, convert it to a list
        if isinstance(data['data'][i]['id'], int):
            data['data'][i]['id'] = [data['data'][i]['id']]
        
        # Check if p1, p2, p3, p4 are 1D and convert them to 2D if they are
        for point in ['p1', 'p2', 'p3', 'p4']:
            if len(data['data'][i][point].shape) == 1:
                data['data'][i][point] = data['data'][i][point].reshape(1, -1)

    data['vicon'] = np.array(data['vicon'])
    #  Transpose it
    data['vicon'] = data['vicon'].T

    R = estimate_covariances(data)
    print("Estimated Observation Model Covariance (R):\n", R)
    eigen_values, eigen_vectors = np.linalg.eig(R)
    print("Eigen Values")
    print(eigen_values)
    print("Eigen Vectors")
    print(eigen_vectors)

if __name__ == "__main__":
    main()

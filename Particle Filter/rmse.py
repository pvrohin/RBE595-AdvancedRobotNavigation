from particle_filter import ParticleFilter, run
import numpy as np
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter, run
import scipy.io
import argparse

def rmse(estimated, actual):
    return np.sqrt(np.mean((estimated - actual) ** 2))

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

def main():
    # Call the function with the filename of the .mat file containing the data
    # Load data
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the dataset number. The dataset number should be between 0 and 7.")
    parser.add_argument('dataset_number', type=int, help='The dataset number to process (0-7)')

    # Parse the arguments
    args = parser.parse_args()

    # Use the dataset number in your filename
    filename = f'data/studentdata{args.dataset_number}.mat'
    data = load_data(filename)

    # Loop through the data and print the tag IDs
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

    # Initialize lists to store ground truth position and orientation
    ground_truth_positions = []
    ground_truth_orientations = []

    # Loop through data and store ground truth position and orientation from data['vicon'] and data['time']
    for i in range(len(data['vicon'])):
        # Extract ground truth position and orientation from data
        ground_truth_position = data['vicon'][i][:3]
        ground_truth_orientation = data['vicon'][i][3:6]
        
        # Append ground truth data to lists
        ground_truth_positions.append(ground_truth_position)
        ground_truth_orientations.append(ground_truth_orientation)

    # Create a particle filter object
    pf = ParticleFilter(2000, filename)

    # Run the particle filter
    filtered_positions, estimated_positions = pf.run(data, args.dataset_number, 'highest_weighted')

    # Run the Extended Kalman Filter
    ekf = ExtendedKalmanFilter()
    ekf_estimated_positions = ekf.run(data)

    filtered_positions = np.array(filtered_positions)
    estimated_positions = np.array(estimated_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    ground_truth_orientations = np.array(ground_truth_orientations)

    # Calculate the RMSE
    filtered_rmse = rmse(filtered_positions, ground_truth_positions)
    estimated_rmse = rmse(estimated_positions, ground_truth_positions)

    ekf_rmse = rmse(ekf_estimated_positions, ground_truth_positions)

    print(f'Filtered RMSE: {filtered_rmse}')
    print(f'Estimated RMSE: {estimated_rmse}')

    print(f'EKF RMSE: {ekf_rmse}')

if __name__ == '__main__':
    main()
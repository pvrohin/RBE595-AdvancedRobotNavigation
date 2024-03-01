import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from extract_pose import estimate_pose,tag_coordinates

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

# Function to visualize trajectory and orientation
def visualize_trajectory(filename):
    # Load data
    data = load_data(filename)
    
    # Initialize lists to store estimated and ground truth positions and orientations
    estimated_positions = []
    ground_truth_positions = []
    estimated_orientations = []
    ground_truth_orientations = []
    
    # Iterate over each time step
    for i in range(len(data['data'])):
        # Estimate pose using Task 1 function
        position, orientation = estimate_pose(data['data'][i], tag_coordinates)
        
        # Extract ground truth position and orientation from data
        # Check if i is within the range of data['vicon']
        
        # Extract ground truth position and orientation from data
        ground_truth_position = data['vicon'][i][:3]
        ground_truth_orientation = data['vicon'][i][3:6]
        
        # Append estimated and ground truth data to lists
        estimated_positions.append(position)
        ground_truth_positions.append(ground_truth_position)
        estimated_orientations.append(orientation)
        ground_truth_orientations.append(ground_truth_orientation)
    
    # Convert lists to numpy arrays
    estimated_positions = np.array(estimated_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    estimated_orientations = np.array(estimated_orientations)
    ground_truth_orientations = np.array(ground_truth_orientations)
    
    # Plot trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory')
    ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Trajectory')
    ax.legend()
    
    # Plot orientation
    # fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    # axs[0].plot(estimated_orientations[:, 0], label='Estimated Roll')
    # axs[0].plot(ground_truth_orientations[:, 0], label='Ground Truth Roll')
    # axs[0].set_ylabel('Roll (rad)')
    # axs[1].plot(estimated_orientations[:, 1], label='Estimated Pitch')
    # axs[1].plot(ground_truth_orientations[:, 1], label='Ground Truth Pitch')
    # axs[1].set_ylabel('Pitch (rad)')
    # axs[2].plot(estimated_orientations[:, 2], label='Estimated Yaw')
    # axs[2].plot(ground_truth_orientations[:, 2], label='Ground Truth Yaw')
    # axs[2].set_ylabel('Yaw (rad)')
    # for ax in axs:
    #     ax.legend()
    # plt.tight_layout()
    # plt.show()

# Call the function with the filename of the .mat file containing the data
visualize_trajectory('data/studentdata0.mat')

import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

def estimate_pose(data, tag_coordinates):
    
    # Extract 3D coordinates of AprilTag corners from the map layout
    map_corners_3d = []
    for data_id in data['id']:
        for id, corners in tag_coordinates:
            if data_id == id:
                for corner in corners:
                    corner_3d = np.array([corner[0], corner[1], 0])
                    map_corners_3d.append(corner_3d)
    
    map_corners_3d = np.array(map_corners_3d)
                
    # Extract 2D projections of AprilTag corners from the image data
    # the format of the p1 through p4 arrays are not[ [x1, y1], [x2, y2], [x3, y3]] like I presumed but its actually [ [x1, x2, x3], [y1, y2, y3]]
    data['p1'] = np.array(data['p1'])

    if len(data['p1']) == 1:
        data['p1'] = data['p1'].reshape(-1, 1)
    
    data['p1'] = data['p1'].T

    data['p2'] = np.array(data['p2'])

    if len(data['p2']) == 1:
        data['p2'] = data['p2'].reshape(-1, 1)

    data['p2'] = data['p2'].T

    data['p3'] = np.array(data['p3'])

    if len(data['p3']) == 1:
        data['p3'] = data['p3'].reshape(-1, 1)
    
    data['p3'] = data['p3'].T

    data['p4'] = np.array(data['p4'])

    if len(data['p4']) == 1:
        data['p4'] = data['p4'].reshape(-1, 1)

    data['p4'] = data['p4'].T

    image_corners_2d = []
    
    for i in range(len(data['p1'])):
        image_corners_2d.append(data['p4'][i])
        image_corners_2d.append(data['p3'][i])
        image_corners_2d.append(data['p2'][i])
        image_corners_2d.append(data['p1'][i])

    image_corners_2d = np.array(image_corners_2d)

    # Camera intrinsic parameters and distortion coefficients (from parameters.txt)
    camera_matrix = np.array([[314.1779, 0, 199.4848],
                          [0, 314.2218, 113.7838],
                          [0, 0, 1]])  # Replace with actual camera matrix
    dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])     # Replace with actual distortion coefficients
    
    # Solve the PnP problem
    success, rvec, tvec = cv2.solvePnP(map_corners_3d, image_corners_2d, camera_matrix, dist_coeffs)

    if not success:
        raise RuntimeError("PnP solver failed to converge")
    
    # Write a rotation matrix for rotation about z-axis
    Rz = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0], 
                  [np.sin(np.pi/4), np.cos(np.pi/4), 0], 
                  [0, 0, 1],
                  ])
    
    # Write a rotation matrix for rotation about x-axis by 180 degrees
    Rx = np.array([[1, 0, 0], 
                  [0, -1, 0], 
                  [0, 0, -1],
                  ])
    
    rotation_matrix = np.dot(Rz, Rx)

    # Combine rotation matrix and translation vector into a single pose matrix
    cam_to_drone = np.array(
        [[rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], -0.04],
            [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], 0],
            [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], -0.03],
            [0, 0, 0, 1]]
    )

    orientation = cv2.Rodrigues(rvec)[0]

    cam_to_world = np.array([np.concatenate((orientation[0],tvec[0])),
                             np.concatenate((orientation[1],tvec[1])),
                            np.concatenate((orientation[2],tvec[2])),
                            [0, 0, 0, 1]])
    
    drone_to_world = np.dot(np.linalg.inv(cam_to_world), cam_to_drone)

    position = drone_to_world[:3, 3]

    orientation = rotation_matrix_to_euler_angles(drone_to_world[:3, :3])

    return position, orientation

def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    rotation_matrix_to_euler_angles converts a 3x3 rotation matrix to
    a tuple of Euler angles in XZY rotation order.
    """
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]

    yaw = np.arctan(-r12 / r22)
    roll = np.arctan(r32 * np.cos(yaw) / r22)
    pitch = np.arctan(-r31 / r33)

    return yaw, pitch, roll

def world_corners():
    tag_size = 0.152  # meters
    tag_spacing = 0.152  # meters
    special_spacing = 0.178  # meters (for columns 3-4 and 6-7)

    # Define the grid size
    num_rows = 9
    num_cols = 12

    # Initialize list to store tag coordinates
    tag_coordinates = []

    # Iterate over each tag ID
    for row in range(num_rows):
        for col in range(num_cols):
            tag_id = row * num_cols + col
            
            # Determine the top-left corner of the tag
            x = col * (tag_size + tag_spacing)
            y = row * (tag_size + tag_spacing)
            
            # Adjust for special spacing
            if row >= 3:
                y += special_spacing
            if row >= 7:
                y += special_spacing
            
            # Compute coordinates of the other corners
            corner1 = (x, y)
            corner2 = (x , y+tag_size)
            corner3 = (x + tag_size, y + tag_size)
            corner4 = (x+ tag_size, y )
            
            # Store tag ID and its coordinates
            tag_coordinates.append((tag_id, [corner1, corner2, corner3, corner4]))

    return tag_coordinates

# Load data from .mat file
data = load_data('data/studentdata0.mat')

print(len(data['data']))

# Loop through the data and print the tag IDs
for i in range(len(data['data'])):
    # If the tag id is an integer, convert it to a list
    if isinstance(data['data'][i]['id'], int):
        data['data'][i]['id'] = [data['data'][i]['id']]
    
    # Check if p1, p2, p3, p4 are 1D and convert them to 2D if they are
    for point in ['p1', 'p2', 'p3', 'p4']:
        if len(data['data'][i][point].shape) == 1:
            data['data'][i][point] = data['data'][i][point].reshape(1, -1)

tag_coordinates = world_corners()

estimated_positions = []
estimated_orientations = []
ground_truth_positions = []
ground_truth_orientations = []

#Loop through each data point and estimate the pose
for i in range(len(data['data'])):
    #If the tag ID has no elements in the array, skip it
    if len(data['data'][i]['id']) == 0:
        continue
    position, orientation = estimate_pose(data['data'][i], tag_coordinates)
    #Store the position and orientation for visualization or further processing
    # Extract ground truth position and orientation from data
        
    # Append estimated and ground truth data to lists
    estimated_positions.append(position)
    
    estimated_orientations.append(orientation)

print(len(estimated_positions))

#Print a sample element of data['vicon']
print(len(data['vicon'][0]))

data['vicon'] = np.array(data['vicon'])
#Transpose it
data['vicon'] = data['vicon'].T
    
# Loop through data and store ground truth position and orientation from data['vicon'] and data['time']
for i in range(len(data['vicon'])):
    # Extract ground truth position and orientation from data
    ground_truth_position = data['vicon'][i][:3]
    ground_truth_orientation = data['vicon'][i][3:6]
    
    # Append ground truth data to lists
    ground_truth_positions.append(ground_truth_position)
    ground_truth_orientations.append(ground_truth_orientation)

# Convert lists to numpy arrays
estimated_positions = np.array(estimated_positions)
ground_truth_positions = np.array(ground_truth_positions)
estimated_orientations = np.array(estimated_orientations)
ground_truth_orientations = np.array(ground_truth_orientations)

# # Plot trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory')
ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Drone Trajectory')
ax.legend()
    
#Plot the estimated and ground truth orientations with the same length of data
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(estimated_orientations[:, 0], label='Estimated Roll')
ax[0].plot(ground_truth_orientations[:, 0], label='Ground Truth Roll')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Roll')
ax[0].set_title('Estimated vs. Ground Truth Roll')
ax[0].legend()
ax[1].plot(estimated_orientations[:, 1], label='Estimated Pitch')
ax[1].plot(ground_truth_orientations[:, 1], label='Ground Truth Pitch')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Pitch')
ax[1].legend()
ax[2].plot(estimated_orientations[:, 2], label='Estimated Yaw')
ax[2].plot(ground_truth_orientations[:, 2], label='Ground Truth Yaw')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Yaw')
ax[2].legend()

#Plot orientation
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
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Read data from file
data = np.loadtxt("kalman_filter_data_mocap.txt", delimiter=",")

# Extract time step
delta_t = data[1, 0] - data[0, 0]

# Define mass of the drone
mass = 0.027  # 27 grams

# Define initial covariance matrix
initial_covariance = np.eye(6) * 1e-6  # Small initial covariance

# Define process noise covariance matrix
process_noise_std = 1e-4  # Experimentally determined
process_noise_covariance = np.eye(6) * process_noise_std

# Define measurement noise covariance matrix for position
position_measurement_noise_std = 0.05  # Low noise for position
position_measurement_noise_covariance = np.eye(3) * position_measurement_noise_std**2

# Define measurement noise covariance matrix for velocity
velocity_measurement_noise_std = 0.05  # Low noise for velocity
velocity_measurement_noise_covariance = np.eye(3) * velocity_measurement_noise_std**2

# Initialize state vector
initial_position = data[0, 4:7]
initial_velocity = np.zeros(3)
initial_state = np.hstack((initial_position, initial_velocity))

# Initialize Kalman filter matrices
# A = np.block([
#     [np.zeros((3, 3)), np.eye(3)],
#     [np.zeros((3, 3)), np.zeros((3, 3))]
# ])

A = np.block([
    [np.eye(3), np.eye(3)*delta_t],
    [np.zeros((3, 3)), np.eye(3)]
])

B = np.block([
    [np.zeros((3, 3))],
    [np.eye(3) * delta_t / mass]
])

H_position = np.block([
    [np.eye(3), np.zeros((3, 3))]
])

H_velocity = np.block([
    [np.zeros((3, 3)), np.eye(3)]
])

# Initialize state and covariance
x_hat = initial_state
P = initial_covariance

# Kalman Filter Loop
estimated_positions = []
for i in range(len(data)):
    # Prediction step
    x_hat_minus = A @ x_hat
    P_minus = A @ P @ A.T + process_noise_covariance

    # Update step based on measurement type
    #if i < len(data):
    measurement = data[i, 4:7]
    H = H_position
    R = position_measurement_noise_covariance
    # else:
    #     measurement = data[i, 1:4]
    #     H = H_velocity
    #     R = velocity_measurement_noise_covariance

    y = measurement - H @ x_hat_minus
    S = H @ P_minus @ H.T + R
    K = P_minus @ H.T @ np.linalg.inv(S)
    x_hat = x_hat_minus + K @ y
    P = (np.eye(6) - K @ H) @ P_minus

    estimated_positions.append(x_hat[:3])

# Convert estimated positions to numpy array for plotting
estimated_positions = np.array(estimated_positions)

#Plot estimated positions and actual positions in the same 3D plot

# Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Position',color='blue', s=20)
# # Plot actual positions
# #ax.plot(data[:, 4], data[:, 5], data[:, 6], label='Actual Position', color='red')
# ax.set_title('Kalman Filter Estimated Position')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.show()

# Plotting actual positions
fig_actual = plt.figure()
ax_actual = fig_actual.add_subplot(111, projection='3d')
ax_actual.plot(data[:, 4], data[:, 5], data[:, 6], label='Actual Position', color='red')
ax_actual.set_xlabel('X')
ax_actual.set_ylabel('Y')
ax_actual.set_zlabel('Z')
ax_actual.legend()
ax_actual.set_title('Actual Position')

# Plotting estimated positions
fig_estimated = plt.figure()
ax_estimated = fig_estimated.add_subplot(111, projection='3d')
ax_estimated.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Position', color='blue')
ax_estimated.set_xlabel('X')
ax_estimated.set_ylabel('Y')
ax_estimated.set_zlabel('Z')
ax_estimated.legend()
ax_estimated.set_title('Estimated Position')

plt.show()

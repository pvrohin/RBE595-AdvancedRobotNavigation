import numpy as np
from scipy.linalg import block_diag
import scipy.io
from estimate_covariance import estimate_covariances
from extract_pose import estimate_pose
import matplotlib.pyplot as plt

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

# Extended Kalman Filter (EKF) class
class EKF:
    def __init__(self, Q, R):
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Observation noise covariance matrix

    def predict(self, x, P, dt, u):
        # Predict step
        F = self.compute_process_model_jacobian(x, dt, u)
        x_pred = self.process_model(x, dt, u)
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        # Update step
        H = self.compute_observation_model_jacobian(x_pred)
        y = z - self.observation_model(x_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_updated = x_pred + K @ y
        P_updated = (np.eye(len(x_pred)) - K @ H) @ P_pred
        return x_updated, P_updated

    def process_model(self, x, dt, u):
        # Process model function
        p, q, p_dot, bg, ba = x[:3], x[3:6], x[6:9], x[9:12], x[12:]
        wx, wy, wz = u
        
        # Calculate orientation matrices
        G = np.array([
            [np.cos(q[1]), 0, -np.cos(q[0]) * np.sin(q[1])],
            [0, 1, np.sin(q[0])],
            [np.sin(q[1]), 0, np.cos(q[0]) * np.cos(q[1])]
        ])
        
        # Compute process model
        p_pred = p + dt * (G.T @ p_dot)
        q_pred = q + dt * np.array([wx, wy, wz]) - dt * bg
        p_dot_pred = p_dot - dt * ba
        bg_pred = bg
        ba_pred = ba

        return np.concatenate((p_pred, q_pred, p_dot_pred, bg_pred, ba_pred))

    def compute_process_model_jacobian(self, x, dt, u):
        # Compute the Jacobian of the process model
        p, q, p_dot, bg, ba = x[:3], x[3:6], x[6:9], x[9:12], x[12:]
        wx, wy, wz = u
        
        # Compute orientation matrices
        G = np.array([
            [np.cos(q[1]), 0, -np.cos(q[0]) * np.sin(q[1])],
            [0, 1, np.sin(q[0])],
            [np.sin(q[1]), 0, np.cos(q[0]) * np.cos(q[1])]
        ])
        G_dot = np.array([
            [0, -np.sin(q[1]), -np.cos(q[0]) * np.cos(q[1])],
            [0, 0, np.cos(q[0])],
            [0, -np.cos(q[1]), np.sin(q[0]) * np.sin(q[1])]
        ])
        
        # Compute Jacobian
        F = np.eye(15)
        F[:3, 3:6] = dt * G.T
        F[3:6, 3:6] = np.eye(3) - dt * np.eye(3)
        F[3:6, 9:12] = -dt * np.eye(3)
        F[3:6, 12:] = -dt * np.eye(3)
        F[6:9, 9:12] = -dt * np.eye(3)
        F[9:12, 12:] = -dt * np.eye(3)

        return F

    def observation_model(self, x):
        # Observation model function
        p, q = x[:3], x[3:6]
        return np.concatenate((p, q))

    def compute_observation_model_jacobian(self, x):
        # Compute the Jacobian of the observation model
        H = np.zeros((6, 15))
        H[:3, :3] = np.eye(3)
        H[3:, 3:6] = np.eye(3)
        return H

# Load data
filename = 'data/studentdata0.mat'
data = load_data(filename)

# Estimate observation model covariance
R = estimate_covariances(filename)

# Initialize EKF
Q = np.eye(15) * 0.01  # Process noise covariance matrix (adjust as needed)
ekf = EKF(Q, R)

# Initial state estimate and covariance
x = np.zeros(15)  # Initialize state vector (adjust as needed)
P = np.eye(15) * 0.01  # Initialize covariance matrix (adjust as needed)

# Time step
dt = 0.01  # Adjust as needed

estimated_positions = []
ground_truth_positions = []

# Iterate over each time step
for i in range(len(data['data'])):
    # Predict step
    x_pred, P_pred = ekf.predict(x, P, dt, u)
    
    # Update step
    z = data['vicon'][i][:6]  # Get observation (ground truth) from motion capture
    x_updated, P_updated = ekf.update(x_pred, P_pred, z)
    
    # Update state and covariance for next iteration
    x = x_updated
    P = P_updated

    # Visualize or store results as needed
    # Store estimated and ground truth positions
    estimated_positions.append(x[:3])  # Extract position from state estimate
    ground_truth_positions.append(z[:3])  # Extract position from ground truth

# Convert lists to arrays for plotting
estimated_positions = np.array(estimated_positions)
ground_truth_positions = np.array(ground_truth_positions)

# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory')
ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Estimated vs Ground Truth Trajectory')
ax.legend()
plt.show()
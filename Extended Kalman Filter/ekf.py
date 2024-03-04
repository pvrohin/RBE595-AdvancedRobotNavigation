import numpy as np
from scipy.linalg import block_diag
import scipy.io
from estimate_covariance import estimate_covariances
from extract_pose_venky import estimate_pose, world_corners
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Matrix

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
        wx, wy, wz, vx, vy, vz = u
        
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
        wx, wy, wz, vx, vy, vz = u
        
        # Compute orientation matrices
        G = np.array([
            [np.cos(q[1]), 0, -np.cos(q[0]) * np.sin(q[1])],
            [0, 1, np.sin(q[0])],
            [np.sin(q[1]), 0, np.cos(q[0]) * np.cos(q[1])]
        ])

        phi = q[0] #Phi
        theta = q[1] #Theta
        psi = q[2] #Psi

        # Calculate the cosine and sine values of the angles
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        R_q = np.array([ cos_psi*cos_theta - sin_phi*sin_psi*sin_theta, -cos_phi*sin_psi, cos_psi*sin_theta + cos_theta*sin_phi*sin_psi,
                         cos_theta*sin_psi + cos_psi*sin_phi*sin_theta, cos_phi*cos_psi, sin_psi*sin_theta - cos_psi*cos_theta*sin_phi,
                         -cos_phi*sin_theta, sin_phi, cos_phi*cos_theta])
        R_q = R_q.reshape(3, 3)

        #Compute inverse of G
        G = np.linalg.inv(G)
        
        # Declare F as a (15,) dimensional vector
        F = np.zeros(15)
        #First 3 rows are pdot
        F[:3] = p_dot
        #Next 3 rows are Ginverse * wx, wy, wz
        F[3:6] = G @ np.array([wx, wy, wz])
        # g is the gravity vector
        g = np.array([0, 0, 9.81])
        #Next 3 rows are g + R_q * vx, vy, vz
        F[6:9] = g + R_q @ np.array([vx, vy, vz])
        #Next 3 rows are 0
        F[9:12] = 0
        #Last 3 rows are 0
        F[12:] = 0

        #Multiply F by dt
        F = F * dt

        #Add the current state to F

        print(F.shape)
        print(x.shape)

        F = F + x

        print(F.shape)

        # Reshape F to be a column vector
        F_column = F.reshape(-1, 1)

        print(F_column.shape)

        # Define symbolic variables
        # Define symbolic variables
        p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3, wx, wy, wz, vx, vy, vz = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3 wx wy wz vx vy vz')
        symbols = [p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3, wx, wy, wz, vx, vy, vz]

        # Convert F_column to a SymPy Matrix
        F_matrix = sp.Matrix(F_column)

        # Compute the Jacobian of the process model
        Jacobian_J = F_matrix.jacobian(symbols)

        # Convert the Jacobian to a numpy array
        Jacobian_J_np = np.array(Jacobian_J)

        print(Jacobian_J_np.shape) 

        return Jacobian_J_np

    def observation_model(self, x):
        # Observation model function
        #tag_coordinates = world_corners()
        p, q = x[:3], x[3:6]
        return np.concatenate((p, q))

    def compute_observation_model_jacobian(self, x):
        # Compute the Jacobian of the observation model
        H = np.zeros((6, 15))
        H[:3, :3] = np.eye(3)
        H[3:, 3:6] = np.eye(3)
        return H

# Load data
filename = 'data/studentdata2.mat'
data = load_data(filename)

#Print a sample element of data
print(data['data'][40])

# Loop through the data and print the tag IDs
for i in range(len(data['data'])):
    # If the tag id is an integer, convert it to a list
    if isinstance(data['data'][i]['id'], int):
        data['data'][i]['id'] = [data['data'][i]['id']]
    
    # Check if p1, p2, p3, p4 are 1D and convert them to 2D if they are
    for point in ['p1', 'p2', 'p3', 'p4']:
        if len(data['data'][i][point].shape) == 1:
            data['data'][i][point] = data['data'][i][point].reshape(1, -1)

# Estimate observation model covariance
R = estimate_covariances(data)

# Initialize EKF
Q = np.eye(15) * 0.01  # Process noise covariance matrix (adjust as needed)
ekf = EKF(Q, R)

# Initial state estimate and covariance
x = np.zeros(15)  # Initialize state vector (adjust as needed)
P = np.eye(15) * 0.01  # Initialize covariance matrix (adjust as needed)

estimated_positions = []
ground_truth_positions = []

# data['vicon'] = np.array(data['vicon'])
# #  Transpose it
# data['vicon'] = data['vicon'].T

# Iterate over each time step
for i in range(len(data['data'])-1):
    if len(data['data'][i]['id']) == 0:
        continue
    # Predict step
    dt = data['data'][i+1]['t'] - data['data'][i]['t']  # Time step
    # IMU data is present in data[drpy][i] and data[acc][i], combine them to get control input
    u = np.concatenate((data['data'][i]['omg'], data['data'][i]['acc']))
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
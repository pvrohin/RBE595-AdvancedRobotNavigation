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

    def update(self, x_pred, P_pred, z, estimated_pose):
        # Update step
        H = self.compute_observation_model_jacobian(x_pred)
        y = z - estimated_pose
        S = H @ P_pred @ H.T + self.R
        S_float = S.astype(np.float64)
        K = P_pred @ H.T @ np.linalg.inv(S_float)
        x_updated = x_pred + K @ y
        P_updated = (np.eye(len(x_pred)) - K @ H) @ P_pred
        return x_updated, P_updated
    
    def calculate_symbolic(self):
        dt = sp.symbols('dt')

        p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

        # Define the matrix elements
        G_q = sp.Matrix([
            [sp.cos(q2), 0, -sp.cos(q1)*sp.sin(q2)],
            [0, 1, sp.sin(q1)],
            [sp.sin(q2), 0, sp.cos(q1)*sp.cos(q2)]
        ])

        # Compute the inverse of G_q
        G_q_inv = G_q.inv()

        # Write R_q as a 3x3 matrix just like G_q
        R_q = sp.Matrix([
                        [ sp.cos(q3)*sp.cos(q2) - sp.sin(q1)*sp.sin(q2)*sp.sin(q3), -sp.cos(q1)*sp.sin(q3), sp.cos(q3)*sp.sin(q2) + sp.cos(q2)*sp.sin(q1)*sp.sin(q3)],
                        [ sp.cos(q3)*sp.sin(q1)*sp.sin(q2) + sp.cos(q2)*sp.sin(q3), sp.cos(q1)*sp.cos(q3), sp.sin(q3)*sp.sin(q2) - sp.cos(q3)*sp.cos(q2)*sp.sin(q1)],
                        [ -sp.cos(q1)*sp.sin(q2), sp.sin(q1), sp.cos(q1)*sp.cos(q2)]
                        ])

        # Define the state vector x = [p, q, p_dot, bg, ba]
        x = sp.Matrix([p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3])

        # Create a new matrix including only p_dot1, p_dot2, p_dot3
        p_dot = sp.Matrix([p_dot1, p_dot2, p_dot3])

        # Define the input vector u = [wx, wy, wz, vx, vy, vz]
        wx, wy, wz, vx, vy, vz = sp.symbols('wx wy wz vx vy vz')
        u = sp.Matrix([wx, wy, wz, vx, vy, vz])

        uw = sp.Matrix([wx, wy, wz])
        ua = sp.Matrix([vx, vy, vz])

        # Define the gravity vector
        g = sp.Matrix([0, 0, -9.81])

        nbg = sp.Matrix([0, 0, 0])
        nba = sp.Matrix([0, 0, 0])

        # Define the x_dot equation x_dot = f(x, u) = [p_dot, G_q_inv * u, g + R_q * u, 0, 0]
        x_dot = sp.Matrix([p_dot, G_q_inv * uw, g + R_q * ua, nbg, nba])

        F = x + x_dot*dt

        # Compute the Jacobian of the process model
        Jacobian_J = F.jacobian(x)

        return F, Jacobian_J

    def process_model(self, x, delta_t, u):
        # Process model function
        dt = sp.symbols('dt')

        p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

        wx, wy, wz, vx, vy, vz = sp.symbols('wx wy wz vx vy vz')
        
        F,_ = self.calculate_symbolic()

        # Substitue the values of x and u into F    
        F = F.subs({p1: x[0], p2: x[1], p3: x[2], q1: x[3], q2: x[4], q3: x[5], p_dot1: x[6], p_dot2: x[7], p_dot3: x[8], bg1: x[9], bg2: x[10], bg3: x[11], ba1: x[12], ba2: x[13], ba3: x[14], wx: u[0], wy: u[1], wz: u[2], vx: u[3], vy: u[4], vz: u[5], dt: delta_t})

        # Convert F to a numpy array of shape (15,)
        F_np = np.array(F)

        # Convert F to a numpy array of shape (15,)
        F_np = F_np.reshape(15,)

        # Return the process model
        return F_np
        
    def compute_process_model_jacobian(self, x, delta_t, u):

        dt = sp.symbols('dt')

        p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

        wx, wy, wz, vx, vy, vz = sp.symbols('wx wy wz vx vy vz')
        
        _, Jacobian_J = self.calculate_symbolic()

        # Substitute the values of x and u into Jacobian_J
        Jacobian_J = Jacobian_J.subs({p1: x[0], p2: x[1], p3: x[2], q1: x[3], q2: x[4], q3: x[5], p_dot1: x[6], p_dot2: x[7], p_dot3: x[8], bg1: x[9], bg2: x[10], bg3: x[11], ba1: x[12], ba2: x[13], ba3: x[14], wx: u[0], wy: u[1], wz: u[2], vx: u[3], vy: u[4], vz: u[5], dt: delta_t})

        # Convert Jacobian_J to a numpy array
        Jacobian_J_np = np.array(Jacobian_J)

        return Jacobian_J_np
    
    # Write an observation model function that uses extract_pose to get the position and orientation of the drone
    def observation_model(self, x, estimated_pose):
        # Observation model function
        p, q = x[:3], x[3:6]
        return np.concatenate((p, q))
        #return estimated_pose

    def compute_observation_model_jacobian(self, x):
        # Compute the Jacobian of the observation model
        H = np.zeros((6, 15))
        H[:3, :3] = np.eye(3)
        H[3:, 3:6] = np.eye(3)
        return H

# Load data
filename = 'data/studentdata0.mat'
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

# Estimate observation model covariance and get estimated poses from extract_pose
R, estimated_poses = estimate_covariances(data)

print(len(estimated_poses))
print(len(data['vicon']))
print(len(data['data']))

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

tag_coordinates = world_corners()

# Loop through data and store ground truth position and orientation from data['vicon'] and data['time']
for i in range(len(data['vicon'])):
    # Extract ground truth position and orientation from data
    ground_truth_position = data['vicon'][i][:3]
    ground_truth_orientation = data['vicon'][i][3:6]
    
    # Append ground truth data to lists
    ground_truth_positions.append(ground_truth_position)
    #ground_truth_orientations.append(ground_truth_orientation)

# Iterate over each time step
for i in range(len(estimated_poses) - 1):
    if len(data['data'][i]['id']) == 0:
        continue
    
    # Predict step
    dt = data['data'][i+1]['t'] - data['data'][i]['t']  # Time step
    
    # IMU data is present in data[drpy][i] and data[acc][i], combine them to get control input
    u = np.concatenate((data['data'][i]['drpy'], data['data'][i]['acc']))
    x_pred, P_pred = ekf.predict(x, P, dt, u)
    
    print("itr: ", i)
    
    #position, orientation = estimate_pose(data['data'][i], tag_coordinates)
    # Update step
    # Get observation (ground truth) from motion capture
    z = np.concatenate((data['vicon'][i][:3], data['vicon'][i][3:6]))
    
    x_updated, P_updated = ekf.update(x_pred, P_pred, z, estimated_poses[i])
    
    # Update state and covariance for next iteration
    x = x_updated
    P = P_updated

    # Visualize or store results as needed
    # Store estimated and ground truth positions
    estimated_positions.append(x[:3])  # Extract position from state estimate

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
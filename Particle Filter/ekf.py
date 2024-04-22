import cv2
import numpy as np
import argparse
from scipy.linalg import block_diag
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from estimate_covariance import estimate_covariances
from extract_pose_final import estimate_pose, world_corners
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
        H = self.compute_observation_model_jacobian()
        y = estimated_pose - H @ x_pred
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

    def compute_observation_model_jacobian(self):
        # Compute the Jacobian of the observation model
        H = np.zeros((6, 15))
        H[:3, :3] = np.eye(3)
        H[3:, 3:6] = np.eye(3)
        return H
    
def run():

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

    #Average Covariance
    R = np.array([
        [7.09701409e-03, 2.66809900e-05, 1.73906943e-03, 4.49014777e-04, 3.66195490e-03, 8.76154421e-04],
        [2.66809900e-05, 4.70388499e-03, -1.33432420e-03, -3.46505064e-03, 1.07454548e-03, -1.69184839e-04],
        [1.73906943e-03, -1.33432420e-03, 9.00885499e-03, 1.80220246e-03, 3.27846190e-03, -1.11786368e-03],
        [4.49014777e-04, -3.46505064e-03, 1.80220246e-03, 5.27060654e-03, 1.01361187e-03, -5.86487142e-04],
        [3.66195490e-03, 1.07454548e-03, 3.27846190e-03, 1.01361187e-03, 7.24994152e-03, -1.36454993e-03],
        [8.76154421e-04, -1.69184839e-04, -1.11786368e-03, -5.86487142e-04, -1.36454993e-03, 1.21162646e-03]
    ])

    # Initialize EKF
    Q = np.eye(15) * 1e-6  # Process noise covariance matrix (adjust as needed)
    ekf = EKF(Q, R)

    # Initial state estimate and covariance
    x = np.zeros(15)  # Initialize state vector (adjust as needed)
    x[:3] = data['vicon'][0][:3]
    x[3:6] = data['vicon'][0][3:6]
    x[6:9] = data['vicon'][0][6:9]
    x[9:] = 0
    P = np.eye(15) * 0.01  # Initialize covariance matrix (adjust as needed)

    estimated_positions = []
    ground_truth_positions = []
    ground_truth_orientations = []

    tag_coordinates = world_corners()

    obs_model_positions = []
    obs_model_orientations = []

    # Loop through data and store ground truth position and orientation from data['vicon'] and data['time']
    for i in range(len(data['vicon'])):
        # Extract ground truth position and orientation from data
        ground_truth_position = data['vicon'][i][:3]
        ground_truth_orientation = data['vicon'][i][3:6]
        
        # Append ground truth data to lists
        ground_truth_positions.append(ground_truth_position)
        ground_truth_orientations.append(ground_truth_orientation)

    # Iterate over each time step
    for i in range(len(data['data'])-1):
        if len(data['data'][i]['id']) == 0:
            # Predict step
            dt = data['data'][i+1]['t'] - data['data'][i]['t']  # Time step

            print("itr: ", i)
            
            # IMU data is present in data[drpy][i] and data[acc][i], combine them to get control input
            if args.dataset_number == 0:
                u = np.concatenate((data['data'][i]['drpy'], data['data'][i]['acc']))
            else:
                u = np.concatenate((data['data'][i]['omg'], data['data'][i]['acc']))
            #u = np.concatenate((data['vicon'][i][6:9], data['vicon'][i][9:12]))
            x_pred, P_pred = ekf.predict(x, P, dt, u)

            # Update state and covariance for next iteration
            x = x_pred
            P = P_pred

        else:
            # Predict step
            dt = data['data'][i+1]['t'] - data['data'][i]['t']  # Time step
            
            # IMU data is present in data[drpy][i] and data[acc][i], combine them to get control input
            if args.dataset_number == 0:
                u = np.concatenate((data['data'][i]['drpy'], data['data'][i]['acc']))
            else:
                u = np.concatenate((data['data'][i]['omg'], data['data'][i]['acc']))
            x_pred, P_pred = ekf.predict(x, P, dt, u)
            
            print("itr: ", i)
            
            # Update step
            # Define z by concatenating the position and orientation from x_pred
            z = x_pred[:6]

            position, orientation = estimate_pose(data['data'][i], tag_coordinates)
            obs_model_positions.append(position)
            obs_model_orientations.append(orientation)
            # put position and orientation in a single array of shape (6,)
            estimated_pose = np.concatenate((position, orientation))
            
            x_updated, P_updated = ekf.update(x_pred, P_pred, z, estimated_pose)
            
            # Update state and covariance for next iteration
            x = x_updated
            P = P_updated

        # Visualize or store results as needed
        # Store estimated and ground truth positions
        estimated_positions.append(x[:3])  # Extract position from state estimate

    # Convert lists to arrays for plotting
    estimated_positions = np.array(estimated_positions)
    ground_truth_positions = np.array(ground_truth_positions)

    obs_model_positions = np.array(obs_model_positions)
    obs_model_orientations = np.array(obs_model_orientations)

    # Plot the estimated trajectory as dots and the ground truth trajectory as a line
    # Plot 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory')
    #ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory', c='r', marker='o')
    ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth Trajectory')
    ax.plot(obs_model_positions[:, 0], obs_model_positions[:, 1], obs_model_positions[:, 2], label='Obs Model Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated vs Ground Truth Trajectory vs Obs Model Trajectory')
    ax.legend()
    plt.show()

    return estimated_positions

if __name__ == '__main__':
    estimated_positions = run()
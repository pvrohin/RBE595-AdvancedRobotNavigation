import cv2
import numpy as np
from scipy.linalg import block_diag
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from estimate_covariance import estimate_covariances
from extract_pose_venky import estimate_pose, world_corners
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
        #y = z - estimated_pose
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

# Call the function with the filename of the .mat file containing the data
# Load data
filename = 'data/studentdata1.mat'
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

#Estimate observation model covariance and get estimated poses from extract_pose
# R = np.array([[ 0.64810354 , 0.00507018 , 0.03933114  ,0.00416992, -0.03576452 , 0.0299308 ],
#  [ 0.00507018 , 0.3001099 , -0.02684017, -0.03552516 , 0.01032146 ,-0.00271169],
#  [ 0.03933114, -0.02684017,  0.0479052 , -0.01483446, 0.00762338, -0.01163921],
#  [ 0.00416992, -0.03552516, -0.01483446,  0.06967535, -0.01529277, 0.00273384],
#  [-0.03576452 , 0.01032146,  0.00762338, -0.01529277 , 0.01710786 ,-0.00967626],
#  [ 0.0299308 , -0.00271169, -0.01163921 , 0.00273384 ,-0.00967626 , 0.01276012]])

#Dataset 1
# R = np.array([
#     [4.01277933e-03, 1.18288647e-03, -2.00353030e-03, -5.74159527e-04, 2.58453957e-03, 3.31938031e-04],
#     [1.18288647e-03, 5.43967539e-03, -4.66980820e-03, -3.87141261e-03, 5.64105295e-04, 5.92553237e-04],
#     [2.00353030e-03, -4.66980820e-03, 1.24291009e-02, 2.44736338e-03, -2.65707665e-04, -1.24055364e-03],
#     [5.74159527e-04, -3.87141261e-03, 2.44736338e-03, 3.53193690e-03, -3.22706410e-04, -3.36813351e-04],
#     [2.58453957e-03, 5.64105295e-04, -2.65707665e-04, -3.22706410e-04, 2.63705969e-03, 4.27377172e-05],
#     [3.31938031e-04, 5.92553237e-04, -1.24055364e-03, -3.36813351e-04, 4.27377172e-05, 1.89300698e-04]
# ])

#Dataset 2
# R = np.array([
#     [4.86409531e-03, -2.94125078e-04, 2.32462827e-03, 1.87162183e-03, 3.41062843e-03, -8.91656484e-05],
#     [-2.94125078e-04, 3.75937724e-03, 3.06819200e-04, -1.96590128e-03, 1.44735836e-03, -1.99711265e-04],
#     [2.32462827e-03, 3.06819200e-04, 1.58949347e-02, 4.23769157e-03, 3.55140819e-03, -1.94017055e-03],
#     [1.87162183e-03, -1.96590128e-03, 4.23769157e-03, 3.41537425e-03, 9.04672800e-04, -4.90091508e-04],
#     [3.41062843e-03, 1.44735836e-03, 3.55140819e-03, 9.04672800e-04, 3.91137920e-03, -3.76095525e-04],
#     [-8.91656484e-05, -1.99711265e-04, -1.94017055e-03, -4.90091508e-04, -3.76095525e-04, 3.21519940e-04]
# ])

#Dataset 3
# R = np.array([
#     [3.52547293e-03, -4.72620065e-04, 3.07029719e-03, 6.22102878e-05, 3.67552366e-03, -9.46301282e-05],
#     [4.72620065e-04, 3.88425041e-03, -1.96856899e-03, -3.17453865e-03, 9.18529526e-05, 1.74223365e-04],
#     [3.07029719e-03, -1.96856899e-03, 9.63698086e-03, 1.04535253e-03, 3.89692176e-03, -7.68074907e-04],
#     [6.22102878e-05, -3.17453865e-03, 1.04535253e-03, 3.43191599e-03, -3.66936788e-04, -1.29199561e-04],
#     [3.67552366e-03, 9.18529526e-05, 3.89692176e-03, -3.66936788e-04, 4.64044464e-03, -1.70354010e-04],
#     [9.46301282e-05, 1.74223365e-04, -7.68074907e-04, -1.29199561e-04, -1.70354010e-04, 1.34914826e-04]
# ])

#Dataset 4
# R = np.array([
#     [5.80720160e-03, 1.14209446e-03, -3.32687103e-03, -4.51481879e-03, -2.22035532e-03, 5.58421955e-04],
#     [1.14209446e-03, 3.76246524e-03, -1.65576059e-03, -1.17842357e-03, -2.34043302e-03, 8.15228628e-05],
#     [-3.32687103e-03, -1.65576059e-03, 1.20429769e-02, -2.09502222e-03, 3.12967094e-03, -5.49304255e-04],
#     [-4.51481879e-03, -1.17842357e-03, -2.09502222e-03, 9.28640486e-03, 1.87405911e-03, -5.15425799e-04],
#     [-2.22035532e-03, -2.34043302e-03, 3.12967094e-03, 1.87405911e-03, 5.40791117e-03, -2.87277973e-04],
#     [5.58421955e-04, 8.15228628e-05, -5.49304255e-04, -5.15425799e-04, -2.87277973e-04, 1.84442372e-04],
# ])

#Dataset 5
# R = np.array([
#     [1.18828159e-02, -1.57135877e-04, 6.84362005e-03, 2.75507433e-03, 1.11343543e-02, -5.18876334e-04],
#     [-1.57135877e-04, 5.11853661e-03, -7.00718409e-04, -4.64546311e-03, 1.01445288e-03, 1.04231491e-04],
#     [6.84362005e-03, -7.00718409e-04, 7.22308207e-03, 2.18075250e-03, 6.07548527e-03, -6.58825680e-04],
#     [2.75507433e-03, -4.64546311e-03, 2.18075250e-03, 4.87070199e-03, 1.51707136e-03, -1.91024090e-04],
#     [1.11343543e-02, 1.01445288e-03, 6.07548527e-03, 1.51707136e-03, 1.08072792e-02, -4.44675170e-04],
#     [-5.18876334e-04, 1.04231491e-04, -6.58825680e-04, -1.91024090e-04, -4.44675170e-04, 8.23956082e-05]
# ])

#Dataset 6
# R = np.array([
#     [7.56080877e-03, 2.58498816e-04, 4.26804240e-03, 2.78682407e-03, 7.68447496e-03, -4.19928996e-04],
#     [2.58498816e-04, 5.38572490e-03, -4.66333273e-04, -5.30269266e-03, 2.11255935e-03, 3.10509562e-05],
#     [4.26804240e-03, -4.66333273e-04, 6.33646109e-03, 2.23178635e-03, 3.96139946e-03, -7.22476457e-04],
#     [2.78682407e-03, -5.30269266e-03, 2.23178635e-03, 6.59630424e-03, 1.00365272e-03, -1.90238025e-04],
#     [7.68447496e-03, 2.11255935e-03, 3.96139946e-03, 1.00365272e-03, 8.61378814e-03, -3.98425477e-04],
#     [-4.19928996e-04, 3.10509562e-05, -7.22476457e-04, -1.90238025e-04, -3.98425477e-04, 1.07436905e-04]
# ])

#Dataset 7
# R = np.array([
#     [0.00787652, 0.0002269, 0.00462655, 0.00362929, 0.00697831, -0.00029106],
#     [0.0002269, 0.00483996, -0.00177252, -0.00401243, 0.0024168, 0.00014734],
#     [0.00462655, -0.00177252, 0.00731506, 0.00361734, 0.00315442, -0.00074367],
#     [0.00362929, -0.00401243, 0.00361734, 0.00530187, 0.00134563, -0.00023165],
#     [0.00697831, 0.0024168, 0.00315442, 0.00134563, 0.0072551, -0.00016959],
#     [-0.00029106, 0.00014734, -0.00074367, -0.00023165, -0.00016959, 0.00010396]
# ])

# Average Covariance
R = np.array([
    [7.09701409e-03, 2.66809900e-05, 1.73906943e-03, 4.49014777e-04, 3.66195490e-03, 8.76154421e-04],
    [2.66809900e-05, 4.70388499e-03, -1.33432420e-03, -3.46505064e-03, 1.07454548e-03, -1.69184839e-04],
    [1.73906943e-03, -1.33432420e-03, 9.00885499e-03, 1.80220246e-03, 3.27846190e-03, -1.11786368e-03],
    [4.49014777e-04, -3.46505064e-03, 1.80220246e-03, 5.27060654e-03, 1.01361187e-03, -5.86487142e-04],
    [3.66195490e-03, 1.07454548e-03, 3.27846190e-03, 1.01361187e-03, 7.24994152e-03, -1.36454993e-03],
    [8.76154421e-04, -1.69184839e-04, -1.11786368e-03, -5.86487142e-04, -1.36454993e-03, 1.21162646e-03]
])

# Initialize EKF
Q = np.eye(15) * 0.01  # Process noise covariance matrix (adjust as needed)
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
        u = np.concatenate((data['data'][i]['omg'], data['data'][i]['acc']))
        #u = np.concatenate((data['vicon'][i][6:9], data['vicon'][i][9:12]))
        x_pred, P_pred = ekf.predict(x, P, dt, u)
        
        print("itr: ", i)
        
        #position, orientation = estimate_pose(data['data'][i], tag_coordinates)
        # Update step
        # Get observation (ground truth) from motion capture
        #z = np.concatenate((data['vicon'][i][:3], data['vicon'][i][3:6]))
        # Define z by concatenating the position and orientation from x_pred
        z = x_pred[:6]

        position, orientation = estimate_pose(data['data'][i], tag_coordinates)
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

#if any values in estimated positions are greater than 10 and less than -10, make it 0
# for i in range(len(estimated_positions)):
#     if estimated_positions[i][0] > 10 or estimated_positions[i][0] < -10:
#         estimated_positions[i][0] = 0
#     if estimated_positions[i][1] > 10 or estimated_positions[i][1] < -10:
#         estimated_positions[i][1] = 0
#     if estimated_positions[i][2] > 10 or estimated_positions[i][2] < -10:
#         estimated_positions[i][2] = 0

# Plot the estimated trajectory as dots and the ground truth trajectory as a line
# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory')
#ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory', c='r', marker='o')
ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Estimated vs Ground Truth Trajectory')
ax.legend()
plt.show()
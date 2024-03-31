import numpy as np
import argparse
from scipy.linalg import block_diag
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from extract_pose_final import estimate_pose, world_corners
import sympy as sp

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

# Particle Filter class
class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_noise_covariance, observation_noise_covariance):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance
        self.particles = np.zeros((num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt, u):
        # Predict step for each particle
        for i in range(self.num_particles):
            # Sample from process noise distribution
            noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.process_noise_covariance)
            # Update particle state based on process model
            self.particles[i] = process_model(self.particles[i], dt, u) + noise

    def update(self, z):
        # Update step for each particle
        for i in range(self.num_particles):
            # Compute observation likelihood
            residual = z - observation_model(self.particles[i]) 
            likelihood = self.calculate_likelihood(residual)
            # Update particle weight
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Resampling step
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate_state(self):
        # Estimate state based on particle distribution
        state_estimate = np.average(self.particles, axis=0, weights=self.weights)
        return state_estimate

    def calculate_likelihood(self, residual):
        # Calculate observation likelihood based on observation noise covariance
        normalization_constant = np.sqrt((2 * np.pi) ** self.state_dim * np.linalg.det(self.observation_noise_covariance))
        exponent = -0.5 * np.dot(np.dot(residual.T, np.linalg.inv(self.observation_noise_covariance)), residual)
        likelihood = np.exp(exponent) / normalization_constant
        return likelihood
    
def calculate_symbolic():
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

def process_model(x, delta_t, u):
    # Process model function
    dt = sp.symbols('dt')

    p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

    wx, wy, wz, vx, vy, vz = sp.symbols('wx wy wz vx vy vz')
        
    F,_ = calculate_symbolic()

    # Substitue the values of x and u into F    
    F = F.subs({p1: x[0], p2: x[1], p3: x[2], q1: x[3], q2: x[4], q3: x[5], p_dot1: x[6], p_dot2: x[7], p_dot3: x[8], bg1: x[9], bg2: x[10], bg3: x[11], ba1: x[12], ba2: x[13], ba3: x[14], wx: u[0], wy: u[1], wz: u[2], vx: u[3], vy: u[4], vz: u[5], dt: delta_t})

    # Convert F to a numpy array of shape (15,)
    F_np = np.array(F)

    # Convert F to a numpy array of shape (15,)
    F_np = F_np.reshape(15,)

    # Return the process model
    return F_np

# Observation model function
def observation_model(x):
    # Implement your observation model here
    # This function should return the expected observation based on the current state
    pass

# Create the parser
parser = argparse.ArgumentParser(description="Process the dataset number. The dataset number should be between 0 and 7.")
parser.add_argument('dataset_number', type=int, help='The dataset number to process (0-7)')

# Parse the arguments
args = parser.parse_args()

# Use the dataset number in your filename
filename = f'data/studentdata{args.dataset_number}.mat'
data = load_data(filename)

# Define process noise covariance and observation noise covariance
process_noise_covariance = np.eye(15) * 1e-6  # Adjust as needed
observation_noise_covariance = np.array([
    [7.09701409e-03, 2.66809900e-05, 1.73906943e-03, 4.49014777e-04, 3.66195490e-03, 8.76154421e-04],
    [2.66809900e-05, 4.70388499e-03, -1.33432420e-03, -3.46505064e-03, 1.07454548e-03, -1.69184839e-04],
    [1.73906943e-03, -1.33432420e-03, 9.00885499e-03, 1.80220246e-03, 3.27846190e-03, -1.11786368e-03],
    [4.49014777e-04, -3.46505064e-03, 1.80220246e-03, 5.27060654e-03, 1.01361187e-03, -5.86487142e-04],
    [3.66195490e-03, 1.07454548e-03, 3.27846190e-03, 1.01361187e-03, 7.24994152e-03, -1.36454993e-03],
    [8.76154421e-04, -1.69184839e-04, -1.11786368e-03, -5.86487142e-04, -1.36454993e-03, 1.21162646e-03]
])  # Adjust as needed

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

tag_coordinates = world_corners()

# Initialize particle filter
num_particles = 500  # Adjust as needed
state_dim = 15  # Adjust as needed
particle_filter = ParticleFilter(num_particles, state_dim, process_noise_covariance, observation_noise_covariance)

# Iterate over each time step
for i in range(len(data['data']) - 1):
    # Time step
    dt = data['data'][i + 1]['t'] - data['data'][i]['t']

    # IMU data is present in data['drpy'][i] and data['acc'][i], combine them to get control input
    if args.dataset_number == 0:
        u = np.concatenate((data['data'][i]['drpy'], data['data'][i]['acc']))
    else:
        u = np.concatenate((data['data'][i]['omg'], data['data'][i]['acc']))

    # Predict step
    particle_filter.predict(dt, u)

    if len(data['data'][i]['id']) != 0:
        # Update step
        position, orientation = estimate_pose(data['data'][i], tag_coordinates)
        z = np.concatenate((position, orientation))
        particle_filter.update(z)

        # Resampling step
        particle_filter.resample()

# Estimate state
estimated_state = particle_filter.estimate_state()

# # Plot the estimated trajectory as dots and the ground truth trajectory as a line
# # Plot 3D trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated Trajectory')
# ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth Trajectory')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Estimated vs Ground Truth Trajectory')
# ax.legend()
# plt.show()


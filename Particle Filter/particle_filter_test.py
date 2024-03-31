import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
from extract_pose_final import estimate_pose, world_corners
import sympy as sp
import argparse
import scipy.io

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

class ParticleFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_covariance, observation_noise_covariance, num_particles):
        self.num_particles = num_particles
        self.particles = np.random.multivariate_normal(initial_state, initial_covariance, size=num_particles)
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance

    def predict(self, control_input, delta_t):
        # Predict step using the process model
        for i in range(self.num_particles):
            # Sample noise from process noise covariance
            noise = np.random.multivariate_normal(mean=np.zeros_like(control_input), cov=self.process_noise_covariance)
            # Apply process model
            self.particles[i] = process_model(self.particles[i], control_input + noise, delta_t)

    # def update(self, observation):
    #     # Update step using the new observation model
    #     weights = np.zeros(self.num_particles)
    #     tag_coordinates = world_corners()
    #     for i in range(self.num_particles):
    #         # Use estimate_pose function to calculate observation based on particle state
    #         estimated_position, estimated_orientation = estimate_pose(observation, tag_coordinates)
    #         estimated_pose = np.concatenate((estimated_position, estimated_orientation))
    #         # Calculate likelihood of observation given particle state
    #         likelihood = multivariate_normal.pdf(observation, mean=estimated_pose, cov=self.observation_noise_covariance)
    #         weights[i] = likelihood
    #     # Normalize weights
    #     weights /= np.sum(weights)
    #     # Resample particles based on weights
    #     indices = np.random.choice(range(self.num_particles), size=self.num_particles, replace=True, p=weights)
    #     self.particles = self.particles[indices]

    def update(self, particles, z, data, tag_coordinates):
        # Update step
        for i in range(self.num_particles):
            # Estimate pose using provided data and tag coordinates
            position, orientation = estimate_pose(data, tag_coordinates)
            estimated_pose = np.concatenate((position, orientation))
            # Compute importance weight for each particle using the observation model
            likelihood = observation_model(particles[i], z, estimated_pose)
            # Assign importance weight to particle
            particles[i]['weight'] *= likelihood
        # Normalize weights
        weights_sum = np.sum(particles['weight'])
        particles['weight'] /= weights_sum
        return particles

    def estimate_state(self):
        # Estimate state by taking the mean of particles
        return np.mean(self.particles, axis=0)
    
def observation_model(particle, z, estimated_pose):
    # Observation model
    # Compute the residual between the observation and the estimated pose
    residual = z - estimated_pose
    # Compute the likelihood of the observation given the particle state
    likelihood = multivariate_normal.pdf(residual, mean=np.zeros_like(residual), cov=observation_noise_covariance)
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
initial_state = np.zeros(15)  # Initial state vector
initial_covariance = np.eye(15) * 0.01  # Initial covariance matrix
#process_noise_covariance = np.eye(15) * 1e-6  # Process noise covariance matrix (adjust as needed)
#observation_noise_covariance = R  # Observation noise covariance matrix
num_particles = 1000  # Number of particles

particle_filter = ParticleFilter(initial_state, initial_covariance, process_noise_covariance, observation_noise_covariance, num_particles)

# Main loop
estimated_positions = []
for i in range(len(data['data'])-1):
    if len(data['data'][i]['id']) == 0:
        # Predict step
        dt = data['data'][i+1]['t'] - data['data'][i]['t']  # Time step
        control_input = np.concatenate((data['data'][i]['omg'], data['data'][i]['acc']))
        particle_filter.predict(control_input, dt)

    else:
        # Update step
        observation = data['data'][i]
        z = np.concatenate((observation['p1'], observation['p2'], observation['p3'], observation['p4']))
        particle_filter.update(particle_filter.particles, z, observation, tag_coordinates)

    # Estimate state
    estimated_state = particle_filter.estimate_state()
    estimated_positions.append(estimated_state[:3])  # Extract position from state estimate

# Convert lists to arrays for plotting
estimated_positions = np.array(estimated_positions)

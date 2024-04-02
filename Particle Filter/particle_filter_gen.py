import numpy as np
import scipy.io
import argparse
from scipy.stats import multivariate_normal
from extract_pose_final import world_corners, estimate_pose
from scipy.linalg import block_diag


class ParticleFilter:
    def __init__(self, num_particles, process_noise_covariance, observation_noise_covariance, tag_coordinates):
        self.num_particles = num_particles
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance
        self.tag_coordinates = tag_coordinates
        self.particles = np.zeros((self.num_particles, 6))
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, u):
        # Add process noise to the particles
        noise = np.random.multivariate_normal(np.zeros(6), self.process_noise_covariance, self.num_particles)
        self.particles += noise

    def update(self, z):
        # Calculate the expected observation
        expected_z = np.zeros((self.num_particles, 6))
        for i in range(self.num_particles):
            expected_z[i] = observation_model(self.particles[i], self.tag_coordinates)
        
        # Calculate the likelihood
        likelihood = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            likelihood[i] = multivariate_normal.pdf(z, expected_z[i], self.observation_noise_covariance)
        
        # Update the weights
        self.weights *= likelihood
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Resampling
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, replace=True, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        # Estimate the state
        estimated_state = np.average(self.particles, axis=0, weights=self.weights)
        return estimated_state

def process_model(state, dt, u):
    # Implement the process model
    pass

def observation_model(state, tag_coordinates):
    # Implement the observation model
    pass

def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)


# Create the parser
parser = argparse.ArgumentParser(description="Process the dataset number. The dataset number should be between 0 and 7.")
parser.add_argument('dataset_number', type=int, help='The dataset number to process (0-7)')

# Parse the arguments
args = parser.parse_args()

# Use the dataset number in your filename
filename = f'data/studentdata{args.dataset_number}.mat'
data = load_data(filename)

# Define process noise covariance and observation noise covariance
process_noise_covariance = np.eye(6) * 1e-6  # Adjust as needed
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
process_noise_covariance = np.eye(6) * 1e-6  # Adjust as needed
observation_noise_covariance = np.array([
    [7.09701409e-03, 2.66809900e-05, 1.73906943e-03, 4.49014777e-04, 3.66195490e-03, 8.76154421e-04],
    [2.66809900e-05, 4.70388499e-03, -1.33432420e-03, -3.46505064e-03, 1.07454548e-03, -1.69184839e-04],
    [1.73906943e-03, -1.33432420e-03, 9.00885499e-03, 1.80220246e-03, 3.27846190e-03, -1.11786368e-03],
    [4.49014777e-04, -3.46505064e-03, 1.80220246e-03, 5.27060654e-03, 1.01361187e-03, -5.86487142e-04],
    [3.66195490e-03, 1.07454548e-03, 3.27846190e-03, 1.01361187e-03, 7.24994152e-03, -1.36454993e-03],
    [8.76154421e-04, -1.69184839e-04, -1.11786368e-03, -5.86487142e-04, -1.36454993e-03, 1.21162646e-03]
])  # Adjust as needed

particle_filter = ParticleFilter(num_particles, process_noise_covariance, observation_noise_covariance, tag_coordinates)

ground_truth_positions = []
ground_truth_orientations = []

# Loop through data and store ground truth position and orientation from data['vicon'] and data['time']
for i in range(len(data['vicon'])):
    # Extract ground truth position and orientation from data
    ground_truth_position = data['vicon'][i][:3]
    ground_truth_orientation = data['vicon'][i][3:6]
    
    # Append ground truth data to lists
    ground_truth_positions.append(ground_truth_position)
    ground_truth_orientations.append(ground_truth_orientation)

# Convert lists to numpy arrays
#estimated_positions = np.array(estimated_positions)
ground_truth_positions = np.array(ground_truth_positions)
#estimated_orientations = np.array(estimated_orientations)
ground_truth_orientations = np.array(ground_truth_orientations)

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
    particle_filter.predict(u)

    if len(data['data'][i]['id']) != 0:
        # Update step
        position, orientation = estimate_pose(data['data'][i], tag_coordinates)
        z = np.concatenate((position, orientation))
        particle_filter.update(z)

        # Resampling step
        particle_filter.resample()

# Estimate state
estimated_state = particle_filter.estimate()

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
# plt.show()
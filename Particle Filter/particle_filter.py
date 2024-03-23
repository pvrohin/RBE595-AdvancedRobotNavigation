import numpy as np
import scipy.io
from extract_pose_final import estimate_pose, world_corners
import matplotlib.pyplot as plt

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.zeros((num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, process_noise_covariance):
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), process_noise_covariance, self.num_particles)
        self.particles += noise

    def update(self, measurements, measurement_noise_covariance, measurement_function):
        for i in range(self.num_particles):
            predicted_measurement = measurement_function(self.particles[i])
            measurement_residual = measurements - predicted_measurement
            measurement_likelihood = self.calculate_measurement_likelihood(measurement_residual, measurement_noise_covariance)
            self.weights[i] *= measurement_likelihood

        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def calculate_measurement_likelihood(self, residual, covariance):
        normalization_constant = np.sqrt((2 * np.pi) ** self.state_dim * np.linalg.det(covariance))
        exponent = -0.5 * np.dot(np.dot(residual.T, np.linalg.inv(covariance)), residual)
        likelihood = np.exp(exponent) / normalization_constant
        return likelihood

    def estimate_state(self):
        state_estimate = np.average(self.particles, axis=0, weights=self.weights)
        return state_estimate
    
    def run(self, process_noise_covariance, measurements, measurement_noise_covariance, measurement_function, num_iterations):
        for _ in range(num_iterations):
            self.predict(process_noise_covariance)
            self.update(measurements, measurement_noise_covariance, measurement_function)
            self.resample()
        return self.estimate_state()
    
def measurement_function(state):
    return state
        
def main():
    num_particles = 100
    state_dim = 15
    process_noise_covariance = np.eye(state_dim) * 0.1
    measurement_noise_covariance = np.eye(state_dim) * 0.1
    
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

    R = np.array([
    [7.09701409e-03, 2.66809900e-05, 1.73906943e-03, 4.49014777e-04, 3.66195490e-03, 8.76154421e-04],
    [2.66809900e-05, 4.70388499e-03, -1.33432420e-03, -3.46505064e-03, 1.07454548e-03, -1.69184839e-04],
    [1.73906943e-03, -1.33432420e-03, 9.00885499e-03, 1.80220246e-03, 3.27846190e-03, -1.11786368e-03],
    [4.49014777e-04, -3.46505064e-03, 1.80220246e-03, 5.27060654e-03, 1.01361187e-03, -5.86487142e-04],
    [3.66195490e-03, 1.07454548e-03, 3.27846190e-03, 1.01361187e-03, 7.24994152e-03, -1.36454993e-03],
    [8.76154421e-04, -1.69184839e-04, -1.11786368e-03, -5.86487142e-04, -1.36454993e-03, 1.21162646e-03]
    ])

    # Initialize particle filter
    particle_filter = ParticleFilter(num_particles, state_dim)

    estimated_positions = []
    estimated_orientations = []
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

    # Loop through each data point and estimate the pose
    for i in range(len(data['data'])):
        # If the tag ID has no elements in the array, skip it
        if len(data['data'][i]['id']) == 0:
            continue
        position, orientation = estimate_pose(data['data'][i], tag_coordinates)
        # Store the position and orientation for visualization or further processing
        # Extract ground truth position and orientation from data

        # Append estimated and ground truth data to lists
        estimated_positions.append(position)

        estimated_orientations.append(orientation)
    
        measurements = np.concatenate((position, orientation))
        #particle_filter = ParticleFilter(num_particles, state_dim)
        state_estimate = particle_filter.run(process_noise_covariance, measurements, measurement_noise_covariance, measurement_function, num_iterations=10)
        print("Estimated State:", state_estimate)   
        # Put the estimated state into the array
        estimated_positions.append(state_estimate[:3])
        estimated_orientations.append(state_estimate[3:])

    # Convert lists to numpy arrays
    estimated_positions = np.array(estimated_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    estimated_orientations = np.array(estimated_orientations)
    ground_truth_orientations = np.array(ground_truth_orientations)

    # Plot the estimated and ground truth positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated')
    ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Trajectory')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
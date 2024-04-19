import numpy as np
import scipy.io
import sympy as sp
from sympy import symbols, Matrix
from sympy.utilities.lambdify import lambdify
from extract_pose_final import estimate_pose, world_corners
import argparse
import matplotlib.pyplot as plt

# Function to load data from .mat file
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

class ParticleFilter:
  def __init__(self, num_particles,mat_file):
      self.mat_file = mat_file
      self.num_particles = num_particles
      self.R = np.array([
    [7.09701409e-03, 2.66809900e-05, 1.73906943e-03, 4.49014777e-04, 3.66195490e-03, 8.76154421e-04],
    [2.66809900e-05, 4.70388499e-03, -1.33432420e-03, -3.46505064e-03, 1.07454548e-03, -1.69184839e-04],
    [1.73906943e-03, -1.33432420e-03, 9.00885499e-03, 1.80220246e-03, 3.27846190e-03, -1.11786368e-03],
    [4.49014777e-04, -3.46505064e-03, 1.80220246e-03, 5.27060654e-03, 1.01361187e-03, -5.86487142e-04],
    [3.66195490e-03, 1.07454548e-03, 3.27846190e-03, 1.01361187e-03, 7.24994152e-03, -1.36454993e-03],
    [8.76154421e-04, -1.69184839e-04, -1.11786368e-03, -5.86487142e-04, -1.36454993e-03, 1.21162646e-03]
    ])
      self.Q = np.diag([0.01] * 3 + [0.002] * 3 + [.1] * 3 + [.001] * 6) * 500

      self.filtered_positions = []
      self.filtered_orientations = []
      self.estimated_positions = []
      self.estimated_orientations = []
      self.n = 15

  def create_particles(self):
      N = self.num_particles
      particles = np.zeros((N, 15)) 

      # Position ranges
      x_range = (0, 3)
      y_range = (0, 3)
      z_range = (0, 2)

      # Angle ranges (-pi/2 to pi/2)
      angle_range = (-np.pi/2, np.pi/2)

      # Generate particles for positions
      x_particles = np.random.uniform(low=x_range[0], high=x_range[1], size=N)
      y_particles = np.random.uniform(low=y_range[0], high=y_range[1], size=N)
      z_particles = np.random.uniform(low=z_range[0], high=z_range[1], size=N)

      # Generate particles for angles
      angle_particles = np.random.uniform(low=angle_range[0], high=angle_range[1], size=(N, 3))

      # Combine particles into state vector
      particles[:, :3] = np.column_stack((x_particles, y_particles, z_particles))
      particles[:, 3:6] = angle_particles
      particles[:, 6:9] = np.random.uniform(low=-0.5, high=0.5, size=(self.num_particles, 3))
      particles[:, 9:15] = np.random.uniform(low=-0.5, high=0.5, size=(self.num_particles, 6))

      return particles
  
  def get_G_and_R(self,x):
      """
      Symbolic computation of the Jacobian of the process model.
      """
      phi, theta, psi = x[3,0] ,x[4,0] ,x[5,0]

      # Define the matrix elements
      G_q = np.array([
      [np.cos(theta), 0, -np.cos(phi) * np.sin(theta)],
      [0, 1, np.sin(phi)],
      [np.sin(theta), 0, np.cos(phi) * np.cos(theta)]])

      # Write R_q as a 3x3 matrix just like G_q
      R_q =  np.array([
      [np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(theta) * np.sin(psi), -np.cos(phi) * np.sin(psi),
        np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi)],
      [np.cos(psi) * np.sin(phi) * np.sin(theta) + np.cos(theta) * np.sin(psi), np.cos(phi) * np.cos(psi),
        np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)],
      [-np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)]])

      return np.linalg.inv(G_q) ,R_q

  def predict(self, particles, u, dt):
    uw = u[0:3]
    ua = u[3:6]

    for i in range(self.num_particles):
      x = particles[i,:].reshape((15,1))

      g=np.array([[0],[0],[-9.81]])   # gravity vector

      G_q_inv, R_q = self.get_G_and_R(x)

      x_dot = np.zeros((15,1))
      x_dot[0:3] = x[6:9]
      x_dot[3:6] = G_q_inv @ (uw + x[9:12])
      x_dot[6:9] = g + R_q @ (ua + x[12:15])
      x_dot[9:15] = np.zeros((6,1))
      noise = np.random.multivariate_normal(np.zeros(15), self.Q).reshape(-1, 1)
      particles[i] = (x + (x_dot + noise)*dt).reshape((1,15))

    return particles

  def get_weights(self,particles,z):
    weights = np.zeros((self.num_particles,1)) 
    z = z.reshape(6,1)
    for i in range(self.num_particles):
      x = particles[i,:].reshape((15,1))
      err = z - x[:6]
      constant = 1.0 / ((2 * np.pi) ** (15 / 2) * np.linalg.det(self.R) ** 0.5)
      weights[i] = np.exp(-0.5 * np.dot(np.dot(err.T, np.linalg.inv(self.R)), err)) * constant

    return weights/np.sum(weights)
  
  def highest_weighted_average(self, particles, weights):
     max_weight_index = np.argmax(weights)
     updated_estimate = particles[max_weight_index,:]
     return updated_estimate
  
  def average(self, particles):
    updated_estimate = np.mean(particles, axis=0)
    return updated_estimate
  
  def weighted_average(self, particles, weights):
    weights = weights.reshape((self.num_particles,1,1))
    weights_reshaped = np.tile(weights, (1, 15, 1))
    updated_estimate = np.average(particles, weights=weights_reshaped, axis=0)
    return updated_estimate

  def update(self, particles, weights, method):
    if method == 'highest_weighted':
        # Find the index of the particle with the highest weight
        updated_estimate = self.highest_weighted_average(particles, weights)  

    elif method == 'average':
        # Take the average of all particles
        updated_estimate = self.average(particles)

    elif method == 'weighted_average':
        # Weighted average of all particles
        updated_estimate = self.weighted_average(particles, weights)

    return updated_estimate

  def low_variance_resampling(self, particles, weights):

    resampled_particles = np.zeros_like(particles)

    N = self.num_particles
    # Step size
    step = 1.0 / N

    # Random start index
    r = np.random.rand() * step

    # Initialize the cumulative sum of weights
    c = weights[0]

    # Keep track of which particle is being considered for resampling
    i = 0

    # Resampling loop
    for m in range(N):
      # Move along the weight distribution until we find the particle to resample for the m-th new particle
      U = r + m * step
      while U > c:
        i = i + 1
        c = c + weights[i]
      resampled_particles[m,:] = particles[i,:]

    return resampled_particles

  def run(self, data, dataset_number, method):
    """
    Estimate pose at each time stamp.
    """
    i=0

    particles = self.create_particles()
    prev_t=0

    for datapoint in data['data']:
      dt = prev_t - datapoint['t']
      prev_t = datapoint['t']

      # End condition for the for loop
      print("Iteration : ", i)
      i+=1

      if dataset_number == 0:
        u = np.concatenate((datapoint['drpy'], datapoint['acc']))
      else:
        u = np.concatenate((datapoint['omg'], datapoint['acc']))

      u = u.reshape(-1,1)

      tag_coordinates = world_corners()

      if len(datapoint['id']) == 0:
        continue

      position, orientation = estimate_pose(datapoint, tag_coordinates)

      position = position.reshape(-1,1)
      orientation = orientation.reshape(-1,1)

      self.estimated_positions.append([position[0],position[1],position[2]])
      self.estimated_orientations.append([orientation[0],orientation[1],orientation[2]])  
      
      particles = self.predict(particles,u,dt)

      z = np.concatenate((position, orientation))

      weights = self.get_weights(particles,z)

      estimates = self.update(particles, weights,method)
        
      particles = self.low_variance_resampling(particles, weights)

      self.filtered_positions.append([estimates[0],estimates[1],estimates[2]])

    return self.filtered_positions, self.estimated_positions
  
def main():
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

    # Initialize lists to store ground truth position and orientation
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

    # Create a particle filter object
    pf = ParticleFilter(2000, filename)

    # Run the particle filter
    filtered_positions, estimated_positions = pf.run(data, args.dataset_number, 'highest_weighted')

    filtered_positions = np.array(filtered_positions)
    estimated_positions = np.array(estimated_positions)
    ground_truth_positions = np.array(ground_truth_positions)
    ground_truth_orientations = np.array(ground_truth_orientations)

    # Plot ground truth, estimated, and filtered positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate and plot the rmse errors for position 

    # Plot ground truth position
    ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Observation Model Result')
    ax.plot(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2], label='PF Filtered')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ground Truth vs Estimated vs Filtered Position')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
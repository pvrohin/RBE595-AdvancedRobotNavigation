from __future__ import annotations
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from process_model_fb import *
from haversine import haversine, Unit

def load_data(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    return data

class UKF:
    def __init__(self, data, kappa: float = 1.0, alpha: float = 1.0, beta: float = 0.4):
        self.data = data
        self.len = len(self.data)

        self.gt_lat_lon_alt = self.data[:,1:4]
        self.gt_roll_pitch_yaw = self.data[:,4:7]

        self.gyro_x_y_z = self.data[:,7:10]
        self.accel_x_y_z = self.data[:,10:13]

        self.z_lat_lon_alt = self.data[:,13:16]
        self.z_VN_VE_VD = self.data[:,16:19]

        self.feedback_states = []

        self.dt = 1

        self.n = 15

        # The number of sigma points we do are typically 2*n + 1,
        # so therefore...
        self.number_of_sigma_points = 2 * self.n + 1  # 25 or 31 based on model
        # kappa is a tuning value for the filter
        self.kappa = kappa
        # alpha is used to calculate initial covariance weights
        self.alpha = alpha
        self.beta = beta

        self.measurement_noise = np.identity(self.n) * 1e-3

        # Create our state vector and set it to 0 - for the very first
        # step we need to initialize it to a non zero value
        self.mu = np.zeros((self.n, 1))

        # Calculate our lambda and weights for use throughout the filter
        self.λ = self.alpha**2 * (self.n + self.kappa) - self.n

        # We have three weights - the 0th mean weight, the 0th covariance
        # weight, and weight_i, which is the ith weight for all other
        # mean and covariance calculations (equivalent)

        weights_mean_0 = self.λ / (self.n + self.λ)

        weights_covariance_0 = (
            (self.λ / (self.n + self.λ)) + (1 - (self.alpha**2)) + self.beta
        )

        weight_i = 1 / (2 * (self.n + self.λ))

        self.weights_mean = np.zeros((self.number_of_sigma_points))
        self.weights_covariance = np.zeros((self.number_of_sigma_points))
        self.weights_mean[0] = weights_mean_0
        self.weights_mean[1:] = weight_i
        self.weights_covariance[0] = weights_covariance_0
        self.weights_covariance[1:] = weight_i

    def measurement_function(self, state, gnss) -> np.ndarray:
        noise_adjustment = np.zeros((self.n, 1))
        noise_scale = 2e-3
        noise = np.random.normal(scale=noise_scale, size=(self.n, self.n))

        c = np.zeros((self.n, self.n))
        c[0:6, 0:6] = np.eye(6)
        c[6:9, 6:9] = np.eye(3)

        R = np.diag(noise).reshape(self.n, 1)
        noise_adjustment[0 : self.n] = np.dot(c, state) + R

        return noise_adjustment

    def find_sigma_points(self, mu, sigma):
        # Based on our system, we expect this to be a 15x31 matrix
        sigma_points = np.zeros((self.number_of_sigma_points, self.n, 1))

        # Set the first column of sigma points to be mu, since that is the mean
        # of our distribution (our center point)
        sigma_points[0] = mu

        try:
            S = sqrtm((self.n + self.kappa) * sigma)
        except:
            print(self.n)
            print(self.kappa)
            print(sigma)
            print(sigma.shape)
            raise "sqrtm failure"
        
        # This is an implementation of the Julier Sigma Point Method
        for i in range(self.n):
            sigma_points[i + 1] = mu + S[i].reshape((self.n, 1))
            sigma_points[self.n + i + 1] = mu - S[i].reshape((self.n, 1))

        return sigma_points

    def update(self, gnss, mu, sigma, sigma_points):
        # Apply the measurement function across each new sigma point
        measurement_points = np.zeros_like(sigma_points)
        for i in range(self.number_of_sigma_points):
            measurement_points[i] = self.measurement_function(sigma_points[i], gnss)

        # Calculate the mean of the measurement points by their respective weights.
        zhat = np.zeros((self.n, 1))
        for i in range(0, self.number_of_sigma_points):
            zhat += self.weights_mean[i] * measurement_points[i]

        St = np.zeros((self.n, self.n))
        differences_z = measurement_points - zhat
        for i in range(0, self.number_of_sigma_points):
            St += self.weights_covariance[i] * np.dot(
                differences_z[i], differences_z[i].T
            )

        # Find the cross-covariance
        # Find the differences between the generated sigma points from
        # earlier and mu
        sigmahat_t = np.zeros((self.n, self.n))
        differences_x = sigma_points - mu
        for i in range(0, self.number_of_sigma_points):
            sigmahat_t += self.weights_covariance[i] * np.dot(
                differences_x[i], differences_z[i].T
            )

        kalman_gain = np.dot(sigmahat_t, np.linalg.pinv(St))

        # Update the mean and covariance
        current_position = mu + np.dot(kalman_gain, gnss - zhat)
        covariance = sigma - np.dot(kalman_gain, St).dot(kalman_gain.T)
        covariance = self.fix_covariance(covariance)

        return current_position, covariance

    def fix_covariance(self, covariance: np.ndarray, jitter: float = 1e-3):
        # Is it symmetric?
        symmetric = np.allclose(covariance, covariance.T)
        # Is it positive definite?
        try:
            np.linalg.cholesky(covariance)
            positive_definite = True
        except np.linalg.LinAlgError:
            positive_definite = False

        # If the result is symmetric and positive definite, return it
        if symmetric and positive_definite:
            return covariance

        # Make covariance matrix symmetric
        covariance = (covariance + covariance.T) / 2

        # Set the eigen values to zero
        eig_values, eig_vectors = np.linalg.eig(covariance)
        eig_values[eig_values < 0] = 0
        eig_values += jitter

        # Reconstruct the matrix
        covariance = eig_vectors.dot(np.diag(eig_values)).dot(eig_vectors.T)

        return self.fix_covariance(covariance, jitter=10 * jitter)

    def predict(self, sigma_points: np.ndarray, fb: np.ndarray, wb: np.ndarray, delta_t: float):
        # For each sigma point, run them through our state transition function
        transitioned_points = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            transitioned_points[i, :] = feedback_propagation_model(sigma_points[i], wb, fb, self.dt)

        # Calculate the mean of the transitioned points by their respective weights
        mu = np.zeros((self.n, 1))
        for i in range(0, self.number_of_sigma_points):
            mu += self.weights_mean[i] * transitioned_points[i]

        # Calculate the covariance of the transitioned points by their respective weights. 
        noise_scale = 5e-5
        Q = np.random.normal(scale=noise_scale, size=(self.n, self.n))
        differences = transitioned_points - mu
        sigma = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            sigma += self.weights_covariance[i] * np.dot(differences[i], differences[i].T)
        sigma += Q

        return mu, sigma, transitioned_points

    def run(self):
        # First we need to initialize our initial position to the 0th estimated
        # position. We will use the true value for the init on the 0th only
        state = np.zeros([15,1])
        state[0:3] = self.gt_lat_lon_alt[0,:].reshape(-1,1)
        state[3:6] = self.gt_roll_pitch_yaw[0,:].reshape(-1,1)
        state[6:9] = self.z_VN_VE_VD[0,:].reshape(-1,1)
        state[9:12] = np.zeros([3,1])
        state[12:15] = np.zeros([3,1])
        # print shape of state
        print(state.shape)
        print(state)

        process_covariance_matrix = np.eye(self.n) * 1e-3

        for i in range(self.len):
            if i == 0:
                continue
            print('Iteration: ', i)
            print(self.len)
            # Get the gyro, acc and measurement data
            gyro = self.gyro_x_y_z[i, :].reshape(-1, 1)
            acc = self.accel_x_y_z[i, :].reshape(-1, 1)
            # make z with 0,1,2 as lat, lon, alt and 6,7,8 as VN, VE, VD, set the rest to 0
            z = np.zeros([15,1])
            z[0:3] = self.z_lat_lon_alt[i,:].reshape(-1,1)
            z[6:9] = self.z_VN_VE_VD[i,:].reshape(-1,1)

            # Get sigma points X0 with weights wm, wc
            sigma_points = self.find_sigma_points(state, process_covariance_matrix)

            # Run the prediction step based off of our state transition
            mubar, sigmabar, transitioned_points = self.predict(sigma_points, acc, gyro, self.dt)

            # Run the update step to filter our estimated position and resulting
            # sigma (mu and sigma)
            mu, sigma = self.update(z, mubar, sigmabar, transitioned_points)

            # Our current position is mu, and our new process covariance
            # matrix is sigma
            process_covariance_matrix = sigma
            state = mu

            print(state.shape)

            self.feedback_states.append(state)

        return self.feedback_states
    
    def plot(self, filtered_positions):
        # Plot the ground truth vs filtered state for latitude, longitude, altitude, phi, theta, psi, VN, VE, VD as 6 subplots in one figure
        gt_lat = self.gt_lat_lon_alt[:,0]
        gt_lon = self.gt_lat_lon_alt[:,1]
        gt_alt = self.gt_lat_lon_alt[:,2]
        gt_roll = self.gt_roll_pitch_yaw[:,0]
        gt_pitch = self.gt_roll_pitch_yaw[:,1]
        gt_yaw = self.gt_roll_pitch_yaw[:,2]
        gt_vn = self.z_VN_VE_VD[:,0]
        gt_ve = self.z_VN_VE_VD[:,1]
        gt_vd = self.z_VN_VE_VD[:,2]

        filtered_lat = [x[0] for x in filtered_positions]
        filtered_lon = [x[1] for x in filtered_positions]
        filtered_alt = [x[2] for x in filtered_positions]
        filtered_roll = [x[3] for x in filtered_positions]
        filtered_pitch = [x[4] for x in filtered_positions]
        filtered_yaw = [x[5] for x in filtered_positions]
        filtered_vn = [x[6] for x in filtered_positions]
        filtered_ve = [x[7] for x in filtered_positions]
        filtered_vd = [x[8] for x in filtered_positions]

        plt.figure(figsize=(20, 20))
        plt.subplot(3, 3, 1)
        plt.plot(gt_lat, label='Ground Truth')
        plt.plot(filtered_lat, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('Latitude')
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.plot(gt_lon, label='Ground Truth')
        plt.plot(filtered_lon, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('Longitude')
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(gt_alt, label='Ground Truth')
        plt.plot(filtered_alt, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('Altitude')
        plt.legend()

        plt.subplot(3, 3, 4)
        plt.plot(gt_roll, label='Ground Truth')
        plt.plot(filtered_roll, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('Roll')
        plt.legend()

        plt.subplot(3, 3, 5)
        plt.plot(gt_pitch, label='Ground Truth')
        plt.plot(filtered_pitch, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('Pitch')
        plt.legend()

        plt.subplot(3, 3, 6)
        plt.plot(gt_yaw, label='Ground Truth')
        plt.plot(filtered_yaw, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('Yaw')
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(gt_vn, label='Ground Truth')
        plt.plot(filtered_vn, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('VN')
        plt.legend()

        plt.subplot(3, 3, 8)
        plt.plot(gt_ve, label='Ground Truth')
        plt.plot(filtered_ve, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('VE')
        plt.legend()

        plt.subplot(3, 3, 9)
        plt.plot(gt_vd, label='Ground Truth')
        plt.plot(filtered_vd, label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('VD')
        plt.legend()

        plt.show()
    
def main():
    data = load_data('trajectory_data.csv')
    ukf = UKF(data)
    filtered_positions = ukf.run()
    ukf.plot(filtered_positions)

if __name__ == "__main__":
    main()
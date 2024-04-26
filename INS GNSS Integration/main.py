import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from process_model import *
from ukf import *

def load_data(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    return data

class INS_GNSS:
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

        self.gt_lat_lon_alt = self.data[:,1:4]
        self.gt_roll_pitch_yaw = self.data[:,4:7]

        self.gyro_x_y_z = self.data[:,7:10]
        self.accel_x_y_z = self.data[:,10:13]

        self.z_lat_lon_alt = self.data[:,13:16]
        self.z_VN_VE_VD = self.data[:,16:19]

        self.feedback_states = np.zeros([self.len, 15])

        self.dt = 1
        self.n = 15

        #self.P = np.eye(15) * 0.1
        self.P = np.eye(15) * 0.00001
        self.Q = np.eye(15) * 0.01
        self.R = np.eye(6) * 0.00001

    def unscented_transform(self, X1, wm, wc, noise_cov):
        x = np.sum(wm * X1, axis=1).reshape(-1,1)
        P = wc[0] * (X1[:,0] - x) @ (X1[:,0] - x).T + noise_cov
        for i in range(1, 2*self.n+1):
            P += wc[i] * (X1[:,i] - x) @ (X1[:,i] - x).T
        return x, P
    
    def fix_covariance(self, covariance, jitter: float = 1e-3):
        """
        Fix the covariance matrix to be positive definite with the
        jitter method on its eigen values. Will continually add more
        jitter until the matrix is symmetric positive definite.
        """
        # Check for invalid values
        if np.isnan(covariance).any() or np.isinf(covariance).any():
            raise ValueError("Covariance matrix contains NaN or infinite values")

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

    def run(self):

        # Define the initial state x = [lat, lon, alt, phi, theta, psi, vn, ve, vd, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
        # Consider the ground truth values as the initial state
        x_prior = np.zeros([15,1])
        x_prior[0:3] = self.gt_lat_lon_alt[0,:].reshape(-1,1)
        x_prior[3:6] = self.gt_roll_pitch_yaw[0,:].reshape(-1,1)
        x_prior[6:9] = self.z_VN_VE_VD[0,:].reshape(-1,1)
        x_prior[9:12] = np.zeros([3,1])
        x_prior[12:15] = np.zeros([3,1])

        # Define the bias values x_prior[9:12] and x_prior[12:15] as white gaussian noise
        # x_prior[9:12] = np.random.normal(0, 0.1, [3,1])
        # x_prior[12:15] = np.random.normal(0, 0.1, [3,1])

        P = self.P
        
        for i in range(self.len):
            print('Iteration: ', i)
            print(self.len)
            # Get the gyro, acc and measurement data
            gyro = self.gyro_x_y_z[i, :].reshape(-1, 1)
            acc = self.accel_x_y_z[i, :].reshape(-1, 1)
            z = np.vstack([self.z_lat_lon_alt[i, :].reshape(-1, 1), self.z_VN_VE_VD[i, :].reshape(-1, 1)])

            # Print shape of P
            print('Shape of P: ', np.shape(P))

            # Get sigma points X0 with weights wm, wc
            X0, wm, wc = getSigmaPoints(x_prior, P, self.n)

            # Print shape of X0
            print('Shape of X0: ', np.shape(X0))
            
            # Propagate sigma points through state transition
            X1 = np.zeros([self.n, 2*self.n+1])
            for j in range(2*self.n+1):
                X1[:, j] = feedback_propagation_model(X0[:, j].reshape(-1, 1), gyro, acc, self.dt).squeeze()

            # Compute the predicted state and covariance
            x_prior, P = self.unscented_transform(X1, wm, wc, self.Q)

            # Print shape of P
            print('Shape of P: ', np.shape(P))

            if np.all(np.linalg.eigvals(P) > 0):
                print('P is positive definite')
            else:
                P = self.fix_covariance(P)

            # Get sigma points X0 with weights wm, wc
            X0, wm, wc = getSigmaPoints(x_prior, P, self.n)

            # Propagate sigma points through measurement model
            Z1 = np.zeros([6, 2*self.n+1])
            for j in range(2*self.n+1):
                Z1[:, j] = measurement_model(X0[:, j].reshape(-1, 1)).squeeze()

            # Print the shape of Z1 before computing z_prior
            print('Shape of Z1 before computing z_prior: ', np.shape(Z1))

            # Compute the predicted measurement and covariance
            z_prior, Pz = self.unscented_transform(Z1, wm, wc, self.R)

            # Print the shape of z_prior after computing it within the unscented_transform function
            print('Shape of z_prior after computing within unscented_transform: ', np.shape(z_prior))

            # Compute the cross covariance
            Pxz = np.zeros([self.n, 6])
            for j in range(2*self.n+1):
                Pxz += wc[j] * (X1[:, j].reshape(-1, 1) - x_prior).dot((Z1[:, j].reshape(-1, 1) - z_prior).T)

            # Print the shape of (Z1.flatten() - z_prior)
            print('Shape of (z.flatten() - z_prior): ', np.shape(z.flatten() - z_prior))

            # Reshape z.flatten() to match the shape of z_prior
            z_flattened = z.flatten().reshape(-1, 1)        

            # Compute the Kalman gain
            K = Pxz.dot(np.linalg.inv(Pz))

            # Update the state and covariance
            #x_prior = x_prior + K.dot(z.flatten() - z_prior)
            # Update the state and covariance
            x_prior = x_prior + K.dot(z_flattened - z_prior)
            #x_prior = x_prior + K.dot((z.flatten() - z_prior.reshape(-1, 1)))
            P = P - K.dot(Pz).dot(K.T)

            # Print shape of P
            print('Shape of P: ', np.shape(P))

            # Check if P is positive definite
            if np.all(np.linalg.eigvals(P) > 0):
                print('P is positive definite')
            else:
                P = self.fix_covariance(P)

            # Print shape of P
            print('Shape of P: ', np.shape(P))

            if np.all(np.linalg.eigvals(P) > 0):
                print('P is positive definite')
            else:
                print('P is not positive definite')

            # Print shape of x_prior
            print('x_prior: ', x_prior)

            self.feedback_states[i, :] = x_prior[:, 0].squeeze()

    
    def plot(self, feedback_states):
        # Plot the lalitude
        plt.figure()
        plt.plot(self.gt_lat_lon_alt[:,0], label='Ground Truth')
        plt.plot(feedback_states[:,0], label='Estimated')
        plt.xlabel('Time')
        plt.ylabel('Latitude')
        plt.legend()

        # Plot the longitude
        plt.figure()
        plt.plot(self.gt_lat_lon_alt[:,1], label='Ground Truth')
        plt.plot(feedback_states[:,1], label='Estimated')
        plt.xlabel('Time')
        plt.ylabel('Longitude')
        plt.legend()
        plt.show()

def run():
    # Load data
    data = load_data('trajectory_data.csv')

    # Check for any NaN values in the data
    if np.isnan(data).any():
        print('There are NaN values in the data')

    # Initialize the INS_GNSS class
    ins_gnss = INS_GNSS(data)
    ins_gnss.run()
    #ins_gnss.plot(feedback_states)

if __name__ == '__main__':
    run()
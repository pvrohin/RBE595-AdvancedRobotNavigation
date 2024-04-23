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

        self.P = np.eye(15) * 0.1
        self.Q = np.eye(15) * 0.01
        self.R = np.eye(6) * 0.00001

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
        x_prior[9:12] = np.random.normal(0, 0.1, [3,1])
        x_prior[12:15] = np.random.normal(0, 0.1, [3,1])

        P = self.P
        
        for i in range(self.len):
            # Get the gyro, acc and measurement data
            gyro = self.gyro_x_y_z[i,:].reshape(-1,1)
            acc = self.accel_x_y_z[i,:].reshape(-1,1)
            z = np.vstack([self.z_lat_lon_alt[i,:].reshape(-1,1), self.z_VN_VE_VD[i,:].reshape(-1,1)])

            #x_updated = feedback_propagation_model(x_prior, gyro, acc, self.dt)

            # Get sigma points X0 with wights wm, wc
            X0, wm, wc = getSigmaPoints(x_prior, P, self.n)
            
            # Propogate sigma points through state transition
            X1 = np.zeros(np.shape(X0))
            for j in range(2*self.n+1):
                thisX = X0[:,j]
                X1[:,j] = np.squeeze(feedback_propagation_model(x_prior, gyro, acc, self.dt))
            
            # Recover mean
            x = np.sum(X1 * wm, axis=1)
            # Recover variance
            diff = X1-np.vstack(x)
            P = np.zeros((self.n, self.n))
            diff2 = X1-np.vstack(X1[:,0])
            for j in range(2*self.n+1):
                d = diff2[:, j].reshape(-1,1)
                P += wc[j] * d @ d.T
            P += self.Q

            # Get new sigma points
            X2, wm, wc = getSigmaPoints(x, self.P, self.n)

            # Put sigma points through measurement model
            Z = np.zeros([6,2*self.n+1])
            for j in range(2*self.n+1):
                thisX = X2[:,j]
                Z[:,j] = measurement_model(thisX)

            # Recover mean
            z = np.sum(Z * wm, axis=1)

            # Recover variance
            S = np.zeros((6,6))

            diff2 = Z-np.vstack(Z[:,0])
            for j in range(2 * self.n + 1):
                d = diff2[:, j].reshape(-1,1)
                S += wc[j] * d @ d.T
            S += self.R

            diff = Z-np.vstack(z)
            # Compute cross covariance
            cxz = np.zeros([self.n,6])
            for j in range(2 * self.n + 1):
                cxz += wc[j] * np.outer(X2[:,j] - x, diff[:, j])

            # Compute Kalman Gain
            K = cxz @ np.linalg.inv(S)

            z_meas = z

            # Update estimate
            x = x.reshape(-1,1) + K @ (z_meas - np.vstack(z))
            print(z_meas - np.vstack(z))
            # Update variance
            P = P - K @ S @ np.transpose(K)

            x_updated = x

            x_prior = x_updated
            print(f"i: {i}")

            # Append the updated state to the feedback_states
            self.feedback_states[i,:] = x_updated[:,0]

        return self.feedback_states
    
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
    feedback_states = ins_gnss.run()
    ins_gnss.plot(feedback_states)

if __name__ == '__main__':
    run()
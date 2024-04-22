import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from process_model import *

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
        # x_prior = np.vstack([self.data[0,1:7][:, np.newaxis], np.zeros([9,1])]) write this in a better way
        x_prior = np.zeros([15,1])
        x_prior[0:3] = self.gt_lat_lon_alt[0,:].reshape(-1,1)
        x_prior[3:6] = self.z_VN_VE_VD[0,:].reshape(-1,1)
        x_prior[6:15] = np.zeros([9,1])

        for i in range(self.len):
            # Get the gyro, acc and measurement data
            gyro = self.gyro_x_y_z[i,:].reshape(-1,1)
            acc = self.accel_x_y_z[i,:].reshape(-1,1)
            z = np.vstack([self.z_lat_lon_alt[i,:][:, np.newaxis], self.z_VN_VE_VD[i,:][:, np.newaxis]])

            # Print shape of gyro, acc and x_prior
            print(gyro.shape, acc.shape, x_prior.shape)

            x_updated = feedback_propagation_model(x_prior, gyro, acc, self.dt)

            # Store the feedback states
            #self.feedback_states[i,:] = x_updated.T


        
    def plot(self):
        pass

def run():
    # Load data
    data = load_data('trajectory_data.csv')

    # Check for any NaN values in the data
    if np.isnan(data).any():
        print('There are NaN values in the data')

    # Initialize the INS_GNSS class
    ins_gnss = INS_GNSS(data)
    ins_gnss.run()

    

if __name__ == '__main__':
    run()
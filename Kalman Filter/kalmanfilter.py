import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, type=1, position=True):
        # Read data from file based on user input
        if type==1:
            self.data = np.loadtxt("kalman_filter_data_low_noise.txt", delimiter=",")
        elif type==2:
            self.data = np.loadtxt("kalman_filter_data_high_noise.txt", delimiter=",")
        else:
            self.data = np.loadtxt("kalman_filter_data_velocity.txt", delimiter=",")
        
        self.actual_data = np.loadtxt("kalman_filter_data_mocap.txt", delimiter=",")
        
        self.mass = 0.027  # 27 grams

        # Define initial covariance matrix
        self.initial_covariance = np.eye(6) * 1e-6

        # Define process noise covariance matrix
        self.process_noise_std = 1e-8
        self.process_noise_covariance = np.eye(6) * self.process_noise_std
        
        # Define measurement noise covariance matrix for position and velocity based on user input
        if position:
            self.measurement_noise_std = 0.1
            self.measurement_noise_covariance = np.eye(3) * self.measurement_noise_std**2
        else:
            self.measurement_noise_std = 0.05
            self.measurement_noise_covariance = np.eye(3) * self.measurement_noise_std**2
        
        # Initialize state vector
        self.initial_position = self.data[0, 4:7]
        self.initial_velocity = np.zeros(3)
        self.initial_state = np.hstack((self.initial_position, self.initial_velocity))
        
        # Initialize measurement matrices based on user input (position or velocity)
        if position:
            self.H = np.block([
            [np.eye(3), np.zeros((3, 3))]
            ])
        else:
            self.H = np.block([
            [np.zeros((3, 3)), np.eye(3)]
            ])
        
        self.x_hat = self.initial_state
        self.P = self.initial_covariance
        self.estimated_positions = []

    def Kalmanloop(self):
        for i in range(len(self.data)-1):
            #Extract time step
            delta_t = self.data[i+1,0] - self.data[i,0]

            #Initialize A and B matrices
            A = np.block([
                [np.eye((3)), delta_t*np.eye(3)],
                [np.zeros((3, 3)), np.eye((3))]
            ])
            B = np.block([
                [np.zeros((3, 3))],
                [np.eye(3) * delta_t / self.mass]
            ])

            #Prediction step
            x_hat_minus = A @ self.x_hat + B @ self.data[i, 1:4]
            P_minus = A @ self.P @ A.T + self.process_noise_covariance

            # Extract measurement
            measurement = self.data[i+1, 4:7]

            #Assign H and R
            H = self.H
            R = self.measurement_noise_covariance

            # Calculate Innovation and Kalman Gain
            y = measurement - H @ x_hat_minus
            S = H @ P_minus @ H.T + R
            K = P_minus @ H.T @ np.linalg.inv(S)

            #Update step
            self.x_hat = x_hat_minus + K @ y
            self.P = (np.eye(6) - K @ H) @ P_minus

            #Store estimated position
            self.estimated_positions.append(self.x_hat[:3])
        self.estimated_positions = np.array(self.estimated_positions)
        return self.estimated_positions
        
    def plot(self, estimated_positions_1, estimated_positions_2, estimated_positions_3):
        fig = plt.figure()
        ax1 = fig.add_subplot(131, projection='3d')

        ax1.plot(self.actual_data[:, 4], self.actual_data[:, 5], self.actual_data[:, 6], label='Actual Position', color='red')
        ax1.plot(estimated_positions_1[:, 0], estimated_positions_1[:, 1], estimated_positions_1[:, 2], label='Estimated Position', color='blue')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('Actual vs. Estimated Position (Low Noise)')

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(self.actual_data[:, 4], self.actual_data[:, 5], self.actual_data[:, 6], label='Actual Position', color='red')
        ax2.plot(estimated_positions_2[:, 0], estimated_positions_2[:, 1], estimated_positions_2[:, 2], label='Estimated Position', color='blue')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        ax2.set_title('Actual vs. Estimated Position (High Noise)')


        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(self.actual_data[:, 4], self.actual_data[:, 5], self.actual_data[:, 6], label='Actual Velocity', color='red')
        ax3.plot(estimated_positions_3[:, 0], estimated_positions_3[:, 1], estimated_positions_3[:, 2], label='Estimated Velocity', color='blue')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        ax3.set_title('Actual vs. Estimated Velocity')

        plt.show()

def main():

    type = 1 # Low noise
    position = True # Position tracking
    kalman1 = KalmanFilter(type, position)
    estimated_positions_1 = kalman1.Kalmanloop()

    type = 2 # High noise
    position = True # Position tracking
    kalman2 = KalmanFilter(type, position)
    estimated_positions_2 = kalman2.Kalmanloop()

    type = 3 # Velocity
    position = False # Velocity tracking
    kalman3 = KalmanFilter(type, position)
    estimated_positions_3 = kalman3.Kalmanloop()
    
    kalman3.plot(estimated_positions_1, estimated_positions_2, estimated_positions_3)

if __name__ == "__main__":
    main()
import sympy as sp
import numpy as np

# dt = sp.symbols('dt')

# p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

# # Define symbolic variables
# #phi, theta, psi = sp.symbols('phi theta psi')

# # Define the matrix elements
# G_q = sp.Matrix([
#     [sp.cos(q2), 0, -sp.cos(q1)*sp.sin(q2)],
#     [0, 1, sp.sin(q1)],
#     [sp.sin(q2), 0, sp.cos(q1)*sp.cos(q2)]
# ])

# # Compute the inverse of G_q
# G_q_inv = G_q.inv()

# # Write R_q as a 3x3 matrix just like G_q
# R_q = sp.Matrix([
#                 [ sp.cos(q3)*sp.cos(q2) - sp.sin(q1)*sp.sin(q2)*sp.sin(q3), -sp.cos(q1)*sp.sin(q3), sp.cos(q3)*sp.sin(q2) + sp.cos(q2)*sp.sin(q1)*sp.sin(q3)],
#                 [ sp.cos(q3)*sp.sin(q1)*sp.sin(q2) + sp.cos(q2)*sp.sin(q3), sp.cos(q1)*sp.cos(q3), sp.sin(q3)*sp.sin(q2) - sp.cos(q3)*sp.cos(q2)*sp.sin(q1)],
#                 [ -sp.cos(q1)*sp.sin(q2), sp.sin(q1), sp.cos(q1)*sp.cos(q2)]
#                 ])

# # Define the state vector x = [p, q, p_dot, bg, ba]
# x = sp.Matrix([p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3])

# # Create a new matrix including only p_dot1, p_dot2, p_dot3
# p_dot = sp.Matrix([p_dot1, p_dot2, p_dot3])

# # Define the input vector u = [wx, wy, wz, vx, vy, vz]
# wx, wy, wz, vx, vy, vz = sp.symbols('wx wy wz vx vy vz')
# u = sp.Matrix([wx, wy, wz, vx, vy, vz])

# uw = sp.Matrix([wx, wy, wz])
# ua = sp.Matrix([vx, vy, vz])

# # Define the gravity vector
# g = sp.Matrix([0, 0, -9.81])

# nbg = sp.Matrix([0, 0, 0])
# nba = sp.Matrix([0, 0, 0])

# # Define the x_dot equation x_dot = f(x, u) = [p_dot, G_q_inv * u, g + R_q * u, 0, 0]
# x_dot = sp.Matrix([p_dot, G_q_inv * uw, g + R_q * ua, nbg, nba])

# F = x + x_dot*dt

# # Compute the Jacobian of the process model
# Jacobian_J = F.jacobian(x)

# # Substitute F with sample values
# F = F.subs({p1: 0.0, p2: 0, p3: 0, q1: 0, q2: 0, q3: 0, p_dot1: 0, p_dot2: 0, p_dot3: 0, bg1: 0, bg2: 0, bg3: 0, ba1: 0, ba2: 0, ba3: 0, wx: 0, wy: 0, wz: 0, vx: 0, vy: 0, vz: 0, dt: 0.01})

# print(F)

def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)

def calculate_symbolic():
    dt = sp.symbols('dt')

    p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

    # Define the matrix elements
    G_q = sp.Matrix([
            [sp.cos(q2), 0, -sp.cos(q1)*sp.sin(q2)],
            [0, 1, sp.sin(q1)],
            [sp.sin(q2), 0, sp.cos(q1)*sp.cos(q2)]])

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


def process_model(self, x, delta_t, u):
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

def compute_process_model_jacobian(x, delta_t, u):

    dt = sp.symbols('dt')

    p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

    wx, wy, wz, vx, vy, vz = sp.symbols('wx wy wz vx vy vz')
        
    _, Jacobian_J = calculate_symbolic()

    # Substitute the values of x and u into Jacobian_J
    Jacobian_J = Jacobian_J.subs({p1: x[0], p2: x[1], p3: x[2], q1: x[3], q2: x[4], q3: x[5], p_dot1: x[6], p_dot2: x[7], p_dot3: x[8], bg1: x[9], bg2: x[10], bg3: x[11], ba1: x[12], ba2: x[13], ba3: x[14], wx: u[0], wy: u[1], wz: u[2], vx: u[3], vy: u[4], vz: u[5], dt: delta_t})

    # Convert Jacobian_J to a numpy array
    Jacobian_J_np = np.array(Jacobian_J)

    return Jacobian_J_np

def predict(x, P, dt, u):
    # Predict step
    F = compute_process_model_jacobian(x, dt, u)
    x_pred = process_model(x, dt, u)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


filename = 'data/studentdata0.mat'
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
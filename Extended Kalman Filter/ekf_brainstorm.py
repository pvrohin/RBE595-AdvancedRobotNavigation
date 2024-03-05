import sympy as sp
import numpy as np

dt = sp.symbols('dt')

p1, p2, p3, q1, q2, q3, p_dot1, p_dot2, p_dot3, bg1, bg2, bg3, ba1, ba2, ba3 = sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')

# Define symbolic variables
#phi, theta, psi = sp.symbols('phi theta psi')

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

# Substitute F with sample values
F = F.subs({p1: 0.0, p2: 0, p3: 0, q1: 0, q2: 0, q3: 0, p_dot1: 0, p_dot2: 0, p_dot3: 0, bg1: 0, bg2: 0, bg3: 0, ba1: 0, ba2: 0, ba3: 0, wx: 0, wy: 0, wz: 0, vx: 0, vy: 0, vz: 0, dt: 0.01})

print(F)



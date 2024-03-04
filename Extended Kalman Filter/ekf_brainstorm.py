import sympy as sp
import numpy as np

# Define symbolic variables
phi, theta, psi = sp.symbols('phi theta psi')

# Define the matrix elements
G_q = sp.Matrix([
    [sp.cos(theta), 0, -sp.cos(phi)*sp.sin(theta)],
    [0, 1, sp.sin(phi)],
    [sp.sin(theta), 0, sp.cos(phi)*sp.cos(theta)]
])

# Compute the inverse of G_q
G_q_inv = G_q.inv()

# Write R_q as a 3x3 matrix just like G_q
R_q = sp.Matrix([
                [ sp.cos(psi)*sp.cos(theta) - sp.sin(phi)*sp.sin(psi)*sp.sin(theta), -sp.cos(phi)*sp.sin(psi), sp.cos(psi)*sp.sin(theta) + sp.cos(theta)*sp.sin(phi)*sp.sin(psi)],
                [ sp.cos(theta)*sp.sin(psi) + sp.cos(psi)*sp.sin(phi)*sp.sin(theta), sp.cos(phi)*sp.cos(psi), sp.sin(psi)*sp.sin(theta) - sp.cos(psi)*sp.cos(theta)*sp.sin(phi)],
                [-sp.cos(phi)*sp.sin(theta), sp.sin(phi), sp.cos(phi)*sp.cos(theta)]
                ])

# Define the state vector x = [p, q, p_dot, bg, ba]
x = sp.Matrix([sp.symbols('p1 p2 p3 q1 q2 q3 p_dot1 p_dot2 p_dot3 bg1 bg2 bg3 ba1 ba2 ba3')])

# Define the input vector u = [wx, wy, wz, vx, vy, vz]
u = sp.Matrix([sp.symbols('wx wy wz vx vy vz')])

# Define the gravity vector
g = sp.Matrix([0, 0, 9.81])

# Define the x_dot equation x_dot = f(x, u) = [p_dot, G_q_inv * u, g + R_q * u, 0, 0]
x_dot = sp.Matrix([x[6:9], G_q_inv * u[:3], g + R_q * u[3:], 0, 0])

print(x_dot)
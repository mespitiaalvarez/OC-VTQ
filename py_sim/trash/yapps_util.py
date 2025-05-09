import numpy as np 
from yapss.math import sqrt, sum, cos, sin, zeros, array, hstack, vstack
import yapss as yp
from yapss.math import matmul, add, concatenate

def rotation_matrix(psi_i, theta_i, phi_i):

    # Calculate matrix elements
    r11 = cos(psi_i) * cos(theta_i)
    r12 = cos(psi_i) * sin(theta_i) * sin(phi_i) - sin(psi_i) * cos(phi_i)
    r13 = cos(psi_i) * sin(theta_i) * cos(phi_i) + sin(psi_i) * sin(phi_i)
    
    r21 = sin(psi_i) * cos(theta_i)
    r22 = sin(psi_i) * sin(theta_i) * sin(phi_i) + cos(psi_i) * cos(phi_i)
    r23 = sin(psi_i) * sin(theta_i) * cos(phi_i) - cos(psi_i) * sin(phi_i)
    
    r31 = -sin(theta_i)
    r32 = cos(theta_i) * sin(phi_i)
    r33 = cos(theta_i) * cos(phi_i)
    
    # Build matrix using YAPSS functions
    R = vstack([
        hstack([r11, r12, r13]),
        hstack([r21, r22, r23]),
        hstack([r31, r32, r33])
    ])
    
    return R

def normalize_quaternion(q):
    # To ensure stability during operations involving unit quaternions
    norm_q = sqrt(sum(q**2))
    return q / norm_q if norm_q > 1e-6 else q

def rotation_matrix_q(q):
    # Defining a rotation matrix from a quaternion
    q = normalize_quaternion(q)  
    
    # Calculate matrix elements
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    r11 = 1-2*(c**2 + d**2)
    r12 = 2*(b*c - d*a)
    r13 = 2*(b*d + c*a)
    
    r21 = 2*(b*c + a*d)
    r22 = 1-2*(b**2 + d**2)
    r23 = 2*(c*d - a*b)
    
    r31 = 2*(b*d - c*a)
    r32 = 2*(a*b + c*d)
    r33 = 1 - 2*(b**2 + c**2)
    
    # Build matrix using YAPSS functions
    R = vstack([
        hstack([r11, r12, r13]),
        hstack([r21, r22, r23]),
        hstack([r31, r32, r33])
    ])
    
    return R

def Y(q):
    # Modified Left Operator (A*Upsilon) found in quaternion dynamics equation
    q = normalize_quaternion(q)  
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    return vstack([
        hstack([-b, -c, -d]),
        hstack([a, -d, c]),
        hstack([d, a, -b]),
        hstack([-c, b, a])
    ])

def norm(x):
    """Compute the L2 norm of a vector using YAPSS math functions."""
    return sqrt(sum(x**2))

def cross(a, b):
    """Cross product of two 3D vectors using YAPSS math functions."""
    return hstack([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])
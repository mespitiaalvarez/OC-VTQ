import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def quaternion_product(q1, q2): 
    # Quaternion product
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    q_prod = ca.vertcat(q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0])
    return q_prod

def quaternion_conjugate(q):
    # Quaternion conjugate
    q = normalize_quaternion(q)
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def quaternion_geo_distance(q1, q2):
    # Quaternion geodesic distance
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    q_prod = ca.dot(q1, q2)
    q_prod = ca.fabs(q_prod)
    return 2*ca.acos(q_prod)

def rotation_matrix(psi_i, theta_i, phi_i):
    # Defining the rotation matrix RB_Pi using yaw-pitch-roll (applied right to left)
    R = ca.SX.zeros(3, 3)
    R[0, 0] = ca.cos(psi_i) * ca.cos(theta_i)
    R[0, 1] = ca.cos(psi_i) * ca.sin(theta_i) * ca.sin(phi_i) - ca.sin(psi_i) * ca.cos(phi_i)
    R[0, 2] = ca.cos(psi_i) * ca.sin(theta_i) * ca.cos(phi_i) + ca.sin(psi_i) * ca.sin(phi_i)
    
    R[1, 0] = ca.sin(psi_i) * ca.cos(theta_i)
    R[1, 1] = ca.sin(psi_i) * ca.sin(theta_i) * ca.sin(phi_i) + ca.cos(psi_i) * ca.cos(phi_i)
    R[1, 2] = ca.sin(psi_i) * ca.sin(theta_i) * ca.cos(phi_i) - ca.cos(psi_i) * ca.sin(phi_i)
    
    R[2, 0] = -ca.sin(theta_i)
    R[2, 1] = ca.cos(theta_i) * ca.sin(phi_i)
    R[2, 2] = ca.cos(theta_i) * ca.cos(phi_i)
    
    return R

def normalize_quaternion(q): # To ensure stability during operations involving unit quaternions
    norm_q = ca.sqrt(ca.sumsqr(q))
    return ca.if_else(norm_q > 1e-6, q / norm_q, q)

def rotation_matrix_q(q):
    # Defining a rotation matrix from a quaternion
    q = normalize_quaternion(q)  
    R = ca.SX.zeros(3,3)
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    R[0, 0] = 1-2*(c**2 + d**2)
    R[0, 1] = 2*(b*c - d*a)
    R[0, 2] = 2*(b*d + c*a)
    
    R[1, 0] = 2*(b*c + a*d)
    R[1, 1] = 1-2*(b**2 + d**2)
    R[1, 2] = 2*(c*d - a*b)
    
    R[2, 0] = 2*(b*d - c*a)
    R[2, 1] = 2*(a*b + c*d)
    R[2, 2] = 1 - 2*(b**2 + c**2)
    return R

def Y(q):
    # Modified Left Operator (A*Upsilon) found
    # in quaternion dynamics equation
    q = normalize_quaternion(q)  
    Y = ca.SX.zeros(4,3)
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    Y[0,0] = -b
    Y[0,1] = -c
    Y[0,2] = -d

    Y[1,0] = a
    Y[1,1] = -d
    Y[1,2] = c

    Y[2,0] = d
    Y[2,1] = a
    Y[2,2] = -b

    Y[3,0] = -c
    Y[3,1] = b
    Y[3,2] = a

    return Y

def linear_state_guess(x0, xf, N):
    """Linearly interpolate between x0 and xf for N+1 steps."""
    return np.linspace(x0, xf, N+1).T  # shape: (state_dim, N+1)

def slerp_quaternion(q0, qf, N):
    """Spherical linear interpolation (SLERP) between two quaternions for N+1 steps.
    Assumes quaternions are in [w, x, y, z] format.
    Returns array of shape (N+1, 4) in [w, x, y, z] format.
    """
    # Convert to [x, y, z, w] for scipy
    q0_scipy = np.roll(q0, -1)
    qf_scipy = np.roll(qf, -1)
    key_rots = R.from_quat([q0_scipy, qf_scipy])
    slerp = Slerp([0, 1], key_rots)
    times = np.linspace(0, 1, N+1)
    interp_rots = slerp(times)
    quats = interp_rots.as_quat()  # shape (N+1, 4), [x, y, z, w]
    quats = np.roll(quats, 1, axis=1)  # back to [w, x, y, z]
    return quats

def make_X_guess(x0, xf, N, q_start, q_end):
    """Create a state guess with linear interpolation for non-quaternion states and SLERP for quaternion."""
    X_guess = np.linspace(x0, xf, N+1).T
    q0 = x0[q_start:q_end]
    qf = xf[q_start:q_end]
    quats = slerp_quaternion(q0, qf, N)
    X_guess[q_start:q_end, :] = quats.T
    return X_guess

def constant_control_guess(u_hover, N):
    """
    Create a constant control guess for N steps.
    u_hover: (nu,) array (the hover control for your drone)
    N: number of time steps
    Returns: (nu, N) array with each column = u_hover
    """
    import numpy as np
    return np.tile(u_hover.reshape(-1, 1), (1, N))

def best_time_guess(x0, xf):
    """
    Estimate the best time to reach the target state from the initial state.
    x0: initial state
    xf: target state
    """
    cf = 4.0687e-7
    w_max = 4720
    t_max = cf * w_max**2  * 4
    m = 1.587

    # Guess the best time to reach the target state from the initial state.
    # This is a rough estimate using the position error and the maximum thrust.
    # starting from rest, assuming constant accelleration.
    # d = 1/2 * a * t^2
    # a = F / m
    # F = w_max**2 * 4 * cf
    # d = 1/2 * (F / m) * t^2
    # t = np.sqrt(2*m*d/F)
    F = w_max**2 * 4 * cf
    return np.sqrt(2*m*np.linalg.norm(xf[0:3] - x0[0:3])/F)


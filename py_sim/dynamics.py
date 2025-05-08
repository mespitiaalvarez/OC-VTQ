import numpy as np
import casadi as ca

# Initialize constants
m = 4.34
J = ca.diag([0.082,0.0845,0.1377])
# c_tf = 8.004e-4
cd = 1e-5
cf = 1e-4
g = 9.81
l = 0.315
nx = 21
nu = 12
psi = [np.pi / 4 + (i - 1) * np.pi / 2 for i in range(1, 5)]  
e3 = ca.vertcat(0, 0, 1)  # Unit vector in z-direction

w_max = 1000
theta_dot_max = 10 
phi_dot_max = 10

# State Variables (x in R21)
p_W = ca.SX.sym('p_W', 3)        # Position (x, y, z)
v_W = ca.SX.sym('v_W', 3)        # Velocity (vx, vy, vz)
q_WB = ca.SX.sym('q_WB', 4)      # Quaternion (w, x, y, z)
omega_B = ca.SX.sym('omega_B', 3)  # Angular velocity (ωx, ωy, ωz)
theta = ca.SX.sym('theta', 4)    # Motor tilt angles (4 values)
phi = ca.SX.sym('phi', 4)        # Motor roll angles (4 values)

# Control Variables (u in R20)
u_w = ca.SX.sym('u_w', 4)            # Rotor angular velocities (4 values)
u_theta = ca.SX.sym('u_theta', 4) # Motor pitch rate (4 values)
u_phi = ca.SX.sym('u_phi', 4)    # Motor roll rate (4 values)

# === Scaled Control Efforts ===
w = w_max * u_w                  # Motor speeds (0 to w_max)

def rotation_matrix(psi_i, theta_i, phi_i):
    # Define the rotation matrix RB_Pi using yaw-pitch-roll
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

def normalize_quaternion(q):
    norm_q = ca.sqrt(ca.sumsqr(q))
    # Use CasADi's if_else for symbolic conditional
    return ca.if_else(norm_q > 1e-6, q / norm_q, q)

def rotation_matrix_q(q):
    q = normalize_quaternion(q)  # Ensure the quaternion is a unit quaternion
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
    q = normalize_quaternion(q)  # Ensure the quaternion is a unit quaternion
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

# Initialize thrust and torque
f_B_thrust = ca.SX.zeros(3)
tau_B_thrust = ca.SX.zeros(3)
tau_B_drag = ca.SX.zeros(3)

for i in range(4):
    R_B_Pi = rotation_matrix(psi[i], theta[i], phi[i])  # Rotation for each propeller
    thrust_i = cf * (w_max * u_w[i])**2  * ca.sign(u_w[i]) * e3  # Thrust in local frame (vertical)
    f_B_thrust += R_B_Pi @ thrust_i  # Rotate and add to total thrust
    
    # Drag torque for each rotor (rotational resistance)
    tau_B_drag += R_B_Pi @ ((-1) ** (i + 1) * cd * (w_max * u_w[i])**2 *ca.sign(u_w[i]) * e3)
    
    # Torque from thrust force offset
    rho_Bi = ca.SX([l*ca.cos(ca.pi/4 + i*(ca.pi/2)), l*ca.sin(ca.pi/4 + i*(ca.pi/2)), 0])  # Propeller positions
    tau_B_thrust += ca.cross(rho_Bi, R_B_Pi @ thrust_i)


# Translational Dynamics
a_W = g * -e3 + (1 / m) * ca.mtimes(rotation_matrix_q(q_WB),f_B_thrust)  # Acceleration in world frame

# Rotational Dynamics
omega_dot_B = ca.mtimes(ca.inv(J), (ca.cross(-omega_B, J @ omega_B) + tau_B_thrust + tau_B_drag))

# Quaternion Derivative (Using quaternion multiplication)
q_dot_WB = 0.5 * ca.mtimes(Y(q_WB),omega_B)

theta_dot = ca.SX.sym('theta_dot', 4)
phi_dot = ca.SX.sym('phi_dot', 4)

theta_dot = theta_dot_max * u_theta
phi_dot = phi_dot_max * u_phi 

# State Derivatives
x_dot = ca.vertcat(
    v_W,
    a_W,
    q_dot_WB,
    omega_dot_B,
    theta_dot,
    phi_dot
)


# State and control vectors
x = ca.vertcat(p_W, v_W, q_WB, omega_B, theta, phi)
u = ca.vertcat(u_w, u_theta, u_phi)

# Define the CasADi function for the dynamics
f_dyn = ca.Function('f_dyn', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

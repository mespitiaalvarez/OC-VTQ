import casadi as ca
from numpy import pi
from utils import rotation_matrix, rotation_matrix_q,Y

nx = 21
nu = 12

# Constants
m = 1.587
J = ca.diag([0.0213,0.02217,0.0282])
J_inv = ca.diag([1/0.0213,1/0.02217,1/0.0282])
cd = 8.4367e-9
cf = 4.0687e-7
g = 9.81
l = 0.243
psi = [pi / 4 + (i - 1) * pi / 2 for i in range(1, 5)]  
e3 = ca.vertcat(0, 0, 1) 
u_w_hover = 0.4293844144478733 # m * g / (4 * cf * (w_max ** 2))

# Actuator Limits
w_max = 4720
theta_dot_max = 10
phi_dot_max = 10

# State Variables (x in R21)
p_W = ca.SX.sym('p_W', 3)        # Position (x, y, z)
v_W = ca.SX.sym('v_W', 3)        # Velocity (vx, vy, vz)
q_WB = ca.SX.sym('q_WB', 4)      # Quaternion (w, x, y, z)
omega_B = ca.SX.sym('omega_B', 3)  # Angular velocity (omega_x, omega_y, omega_z)
theta = ca.SX.sym('theta', 4)    # Motor pitch angles (1,2,3,4)
phi = ca.SX.sym('phi', 4)        # Motor roll angles (1,2,3,4)

# Control Variables (u in R12)
u_w = ca.SX.sym('u_w', 4)            # Rotor angular velocities (1,2,3,4)
u_theta = ca.SX.sym('u_theta', 4) # Motor pitch rate (1,2,3,4)
u_phi = ca.SX.sym('u_phi', 4)    # Motor roll rate (1,2,3,4)

# Definition of rotor thrust and moments
f_B_thrust = ca.SX.zeros(3)
tau_B_thrust = ca.SX.zeros(3)
tau_B_drag = ca.SX.zeros(3)

for i in range(4):
    R_B_Pi = rotation_matrix(psi[i], theta[i], phi[i])  # Rotation for each propeller
    thrust_i = (cf * (w_max**2))*u_w[i] * e3  # Thrust in local frame (vertical)
    f_B_thrust += R_B_Pi @ thrust_i  # Rotate and add to total thrust
    
    # Drag torque for each rotor (rotational resistance)
    tau_B_drag += R_B_Pi @ ((-1) ** (i + 1) * (cd * (w_max**2))* u_w[i]  * e3)
    
    # Torque from thrust force offset
    rho_Bi = ca.SX([l*ca.cos(ca.pi/4 + i*(ca.pi/2)), l*ca.sin(ca.pi/4 + i*(ca.pi/2)), 0])  # Propeller positions
    tau_B_thrust += ca.cross(rho_Bi, R_B_Pi @ thrust_i)

# Definition of x_dot
a_W = g * -e3 + (1 / m) * ca.mtimes(rotation_matrix_q(q_WB),f_B_thrust)  # Translational Dynamics
omega_dot_B = ca.mtimes(J_inv, (ca.cross(-omega_B, J @ omega_B) + tau_B_thrust + tau_B_drag)) # Rotational Dynamics
q_dot_WB = 0.5 * ca.mtimes(Y(q_WB),omega_B) # Quaternion Dynamics
theta_dot = theta_dot_max * u_theta
phi_dot = phi_dot_max * u_phi 

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

# Definition of dynamics (As Casadi function)
f_dyn = ca.Function('f_dyn', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

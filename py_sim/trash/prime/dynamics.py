import casadi as ca
from numpy import pi

# Constants
# m = 4.34
# J = ca.diag([0.082,0.0845,0.1377])
# cd = 1e-5
# cf = 1e-4
# g = 9.81
# l = 0.315


m = 1.587
J = ca.diag([0.0213,0.02217,0.0282])
cd = 8.4367e-7
cf = 4.0687-7
g = 9.81
l = 0.243

nx = 20
nu = 12
psi = [pi / 4 + (i - 1) * pi / 2 for i in range(1, 5)]  
e3 = ca.vertcat(0, 0, 1) 

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

# Actuator Limits
w_max = 4720
theta_dot_max = 10 
phi_dot_max = 10

# State Variables (x in R20)
p_W = ca.SX.sym('p_W', 3)        # Position (x, y, z)
v_W = ca.SX.sym('v_W', 3)        # Velocity (vx, vy, vz)
euler_WB = ca.SX.sym('euler_W_B', 3)      # Rotation Matrix Parametrized by Euler phi,theta,psi
R_WB = rotation_matrix(euler_WB[2],euler_WB[1],euler_WB[0])
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
    thrust_i = cf * (w_max**2)*u_w[i]  * e3  # Thrust in local frame (vertical)
    f_B_thrust += R_B_Pi @ thrust_i  # Rotate and add to total thrust
    
    # Drag torque for each rotor (rotational resistance)
    tau_B_drag += R_B_Pi @ ((-1) ** (i + 1) * cd * (w_max**2)* u_w[i] * e3)
    
    # Torque from thrust force offset
    rho_Bi = ca.SX([l*ca.cos(ca.pi/4 + i*(ca.pi/2)), l*ca.sin(ca.pi/4 + i*(ca.pi/2)), 0])  # Propeller positions
    tau_B_thrust += ca.cross(rho_Bi, R_B_Pi @ thrust_i)

# Definition of x_dot
a_W = g * -e3 + (1 / m) * ca.mtimes(R_WB,f_B_thrust)  # Translational Dynamics
omega_dot_B = ca.mtimes(ca.inv(J), (ca.cross(-omega_B, J @ omega_B) + tau_B_thrust + tau_B_drag)) # Rotational Dynamics

# Skew-symmetric matrix for angular velocity (omega_B)
transform_euler = ca.SX.zeros(3, 3)
transform_euler[0, 0] = 1
transform_euler[0, 1] = 0
transform_euler[0, 2] = -ca.sin(euler_WB[1])

transform_euler[1, 0] = 0
transform_euler[1, 1] = ca.cos(euler_WB[0])
transform_euler[1, 2] = ca.sin(euler_WB[0]) * ca.cos(euler_WB[1]) 

transform_euler[2, 0] = 0
transform_euler[2, 1] = -ca.sin(euler_WB[0])
transform_euler[2, 2] = ca.cos(euler_WB[0]) * ca.sin(euler_WB[1]) 
# Rotation Matrix Dynamics
euler_WB_dot = ca.inv(transform_euler)@omega_B
theta_dot = theta_dot_max * u_theta
phi_dot = phi_dot_max * u_phi 

# Concatenate State Derivative
# Note, Casadi is matrix-major
# to have row major, take transpose then reshape
x_dot = ca.vertcat(
    v_W,           # Velocity (p_W dot)
    a_W,           # Acceleration (v_W dot)
    euler_WB_dot,  # Flatten rotation matrix derivative
    omega_dot_B,   # Angular acceleration
    theta_dot,     # Motor pitch rates
    phi_dot        # Motor roll rates
)


# State and control vectors
x = ca.vertcat(p_W, v_W, euler_WB, omega_B, theta, phi)
u = ca.vertcat(u_w, u_theta, u_phi)

# Definition of dynamics (As Casadi function)
f_dyn = ca.Function('f_dyn', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

X0 = ca.MX.sym('x', nx)
U0 = ca.MX.sym('u', nu)
N = 20
M = 4  # RK4 steps per interval
DT = 2 / N / M
XF = X0
for j in range(M):
    k1 = f_dyn(XF, U0)
    k2 = f_dyn(XF + DT / 2 * k1, U0)
    k3 = f_dyn(XF + DT / 2 * k2, U0)
    k4 = f_dyn(XF + DT * k3, U0)
    XF = XF + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

F = ca.Function('F', [X0, U0], [XF], ['x0', 'u0'], ['xf'])
# Initial State Constraint (Starting at rest)
x_init = ca.DM.zeros(nx)
x_int_R = ca.DM.eye(3)
u_init = ca.DM.zeros(nu)
u_init[0] = .8
u_init[1] = .8 
u_init[2] = .8
u_init[3] = .8

xf = F(x_init,u_init)
print(xf)
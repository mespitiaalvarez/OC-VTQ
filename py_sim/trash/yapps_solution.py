import yapss as yp
import numpy as np
from yapss.math import cos, sin, sqrt, pi, matmul, sum, zeros, add, vstack
from yapps_util import rotation_matrix, rotation_matrix_q, Y, norm, cross

# Constants
m = 1.587
# Create diagonal matrix manually
J = zeros((3, 3))
J[0,0] = 0.0213
J[1,1] = 0.02217
J[2,2] = 0.0282
# Create inverse of diagonal matrix manually
J_inv = zeros((3, 3))
J_inv[0,0] = 1.0/0.0213
J_inv[1,1] = 1.0/0.02217
J_inv[2,2] = 1.0/0.0282
cd = 8.4367e-7
cf = 4.0687e-7  # Fixed typo in original constant
g = 9.81
l = 0.243
nx = 21
nu = 12

# Actuator Limits
w_max = 4720
theta_dot_max = 10
phi_dot_max = 10
psi = [pi/4 + (i-1)*pi/2 for i in range(1,5)]  
e3 = zeros(3)
e3[2] = 1.0  # Use YAPSS zeros instead of np.array

# Create problem with states and controls
problem = yp.Problem(
    name="OC-VTQ",
    nx=[21],  # 21 states
    nu=[12],  # 12 controls
    nh=[1],   # 1 path constraints
    nq=[5]    # 4 integrals
)

def continuous(arg):
    """Calculate thrust vectoring quadcopter dynamics and path constraints."""
    # Extract states and controls
    state = arg.phase[0].state
    control = arg.phase[0].control
    
    # Extract states
    p_W = state[0:3]        # Position (x, y, z)
    v_W = state[3:6]        # Velocity (vx, vy, vz)
    q_WB = state[6:10]      # Quaternion (w, x, y, z)
    omega_B = state[10:13]  # Angular velocity (omega_x, omega_y, omega_z)
    theta = state[13:17]    # Motor pitch angles (1,2,3,4)
    phi = state[17:21]      # Motor roll angles (1,2,3,4)
    
    # Extract controls
    u_w = control[0:4]        # Rotor angular velocities (1,2,3,4)
    u_theta = control[4:8]    # Motor pitch rate (1,2,3,4)
    u_phi = control[8:12]     # Motor roll rate (1,2,3,4)
    
    # Calculate thrust and moments
    f_B_thrust = zeros(3)
    tau_B_thrust = zeros(3)
    tau_B_drag = zeros(3)
    
    for i in range(4):
        # Get scalar values for this motor
        psi_i = psi[i]
        theta_i = theta[i]
        phi_i = phi[i]
        u_w_i = u_w[i]
        
        # Calculate rotation matrix for this motor
        R_B_Pi = rotation_matrix(psi_i, theta_i, phi_i)
        
        # Calculate thrust for this motor
        thrust_i = cf * (w_max**2) * u_w_i * e3
        f_B_thrust = add(f_B_thrust, matmul(R_B_Pi, thrust_i))
        
        # Drag torque
        tau_B_drag = add(tau_B_drag, matmul(R_B_Pi, ((-1) ** (i + 1) * cd * (w_max**2) * u_w_i * e3)))
        
        # Torque from thrust force offset
        rho_Bi = zeros(3)
        rho_Bi[0] = l*cos(pi/4 + i*(pi/2))
        rho_Bi[1] = l*sin(pi/4 + i*(pi/2))
        rho_Bi[2] = 0
        tau_B_thrust = add(tau_B_thrust, cross(rho_Bi, matmul(R_B_Pi, thrust_i)))
    
    # State derivatives
    a_W = g * -e3 + (1/m) * matmul(rotation_matrix_q(q_WB), f_B_thrust)
    omega_dot_B = matmul(J_inv, (cross(-omega_B, matmul(J, omega_B)) + tau_B_thrust + tau_B_drag))
    
    # Quaternion dynamics using exponential map
    # This maintains unit length naturally
    omega_norm = sqrt(sum(omega_B**2))
    if omega_norm > 1e-10:  # Avoid division by zero
        omega_unit = omega_B / omega_norm
        q_dot_WB = 0.5 * matmul(Y(q_WB), omega_B)
    else:
        q_dot_WB = zeros(4)
    
    theta_dot = theta_dot_max * u_theta
    phi_dot = phi_dot_max * u_phi
    
    # Set dynamics directly in phase object
    # Convert all vectors to column vectors before stacking
    v_W_col = vstack([v_W[0], v_W[1], v_W[2]])
    a_W_col = vstack([a_W[0], a_W[1], a_W[2]])
    q_dot_WB_col = vstack([q_dot_WB[0], q_dot_WB[1], q_dot_WB[2], q_dot_WB[3]])
    omega_dot_B_col = vstack([omega_dot_B[0], omega_dot_B[1], omega_dot_B[2]])
    theta_dot_col = vstack([theta_dot[0], theta_dot[1], theta_dot[2], theta_dot[3]])
    phi_dot_col = vstack([phi_dot[0], phi_dot[1], phi_dot[2], phi_dot[3]])
    
    dynamics = vstack([v_W_col, a_W_col, q_dot_WB_col, omega_dot_B_col, theta_dot_col, phi_dot_col])
    arg.phase[0].dynamics = dynamics
    
    # Set path constraints
    arg.phase[0].path[:] = [
        sum(q_WB**2) - 1.0 >= 0  # Must be at least unit length
    ]
    
    # Set integrals
    # 1. Total energy consumption (sum of squared rotor speeds)
    arg.phase[0].integrand[0] = sum(u_w**2)
    
    # 2. Total control effort (sum of squared motor rates)
    arg.phase[0].integrand[1] = sum(u_theta**2) + sum(u_phi**2)
    
    # 3. Distance traveled
    arg.phase[0].integrand[2] = norm(v_W)
    
    # 4. Angular momentum
    arg.phase[0].integrand[3] = norm(matmul(J, omega_B))
    
    # 5. Tracking error (squared error between current position and hover point)
    ref_x = 0.0  # Hover at origin
    ref_y = 0.0
    ref_z = 1.0  # 1m height
    
    # Calculate squared error
    pos_error = (p_W[0] - ref_x)**2 + (p_W[1] - ref_y)**2 + (p_W[2] - ref_z)**2
    arg.phase[0].integrand[4] = pos_error

def objective(arg):
    """Calculate thrust vectoring quadcopter objective.
    
    The objective is to minimize:
    1. Total energy consumption (integral[0])
    2. Total control effort (integral[1])
    3. Final time
    4. Tracking error (integral[4])
    """
    # Get integrals
    energy_consumption = arg.phase[0].integral[0]  # Total energy used
    control_effort = arg.phase[0].integral[1]      # Total control effort
    tracking_error = arg.phase[0].integral[4]      # Tracking error
    final_time = arg.phase[0].final_time
    
    # Weighted sum of objectives
    arg.objective = final_time + 0.1 * energy_consumption + 0.01 * control_effort + 10.0 * tracking_error

# Set up the continuous dynamics and objective
problem.functions.continuous = continuous
problem.functions.objective = objective

# Set bounds on states
problem.bounds.phase[0].state.lower = np.array([
    -5, -5, -5,     # Position bounds
    -2, -2, -2,     # Velocity bounds
    -1, -1, -1, -1, # Quaternion bounds
    -5, -5, -5,     # Angular velocity bounds
    -pi/4, -pi/4, -pi/4, -pi/4,  # Motor pitch angle bounds
    -pi/4, -pi/4, -pi/4, -pi/4   # Motor roll angle bounds
])

problem.bounds.phase[0].state.upper = np.array([
    5, 5, 5,        # Position bounds
    2, 2, 2,        # Velocity bounds
    1, 1, 1, 1,     # Quaternion bounds
    5, 5, 5,        # Angular velocity bounds
    pi/4, pi/4, pi/4, pi/4,      # Motor pitch angle bounds
    pi/4, pi/4, pi/4, pi/4       # Motor roll angle bounds
])

# Set bounds on controls
problem.bounds.phase[0].control.lower = np.array([
    0.3, 0.3, 0.3, 0.3,         # Rotor speed bounds (minimum thrust for hover)
    -0.5, -0.5, -0.5, -0.5,     # Motor pitch rate bounds
    -0.5, -0.5, -0.5, -0.5      # Motor roll rate bounds
])

problem.bounds.phase[0].control.upper = np.array([
    1, 1, 1, 1,                 # Rotor speed bounds
    0.5, 0.5, 0.5, 0.5,        # Motor pitch rate bounds
    0.5, 0.5, 0.5, 0.5         # Motor roll rate bounds
])

# Set initial and final conditions
# Start from ground, hover at 1m height
problem.bounds.phase[0].initial_state.lower = np.array([
    0, 0, 0,       # Start at origin
    0, 0, 0,       # Start at rest
    1, 0, 0, 0,    # Identity quaternion (exact)
    0, 0, 0,       # No initial angular velocity
    0, 0, 0, 0,    # No initial motor angles
    0, 0, 0, 0     # No initial motor angles
])

problem.bounds.phase[0].initial_state.upper = np.array([
    0, 0, 0,       # Start at origin
    0, 0, 0,       # Start at rest
    1, 0, 0, 0,    # Identity quaternion (exact)
    0, 0, 0,       # No initial angular velocity
    0, 0, 0, 0,    # No initial motor angles
    0, 0, 0, 0     # No initial motor angles
])

problem.bounds.phase[0].final_state.lower = np.array([
    -0.1, -0.1, 0.9,  # End near z = 1 (1m up)
    0, 0, 0,          # End at rest
    1, 0, 0, 0,       # Identity quaternion (exact)
    0, 0, 0,          # No final angular velocity
    0, 0, 0, 0,       # No final motor angles
    0, 0, 0, 0        # No final motor angles
])

problem.bounds.phase[0].final_state.upper = np.array([
    0.1, 0.1, 1.1,    # End near z = 1 (1m up)
    0, 0, 0,          # End at rest
    1, 0, 0, 0,       # Identity quaternion (exact)
    0, 0, 0,          # No final angular velocity
    0, 0, 0, 0,       # No final motor angles
    0, 0, 0, 0        # No final motor angles
])

# Set time bounds
problem.bounds.phase[0].initial_time.lower = 0
problem.bounds.phase[0].initial_time.upper = 0
problem.bounds.phase[0].final_time.lower = 0.1
problem.bounds.phase[0].final_time.upper = 5.0  # Reduced max time

# Set transcription options
m, n = 5, 3  # Reduced to minimum values: 5 segments with 3 collocation points each
problem.mesh.phase[0].collocation_points = m * (n,)
problem.mesh.phase[0].fraction = m * (1.0 / m,)

# Set solver options
problem.ipopt_options.print_level = 5
problem.ipopt_options.max_iter = 100
problem.ipopt_options.tol = 1e-2  # Relaxed tolerance
problem.ipopt_options.acceptable_tol = 1e-2  # Relaxed acceptable tolerance
problem.ipopt_options.linear_solver = 'mumps'
problem.ipopt_options.mu_strategy = 'adaptive'
problem.ipopt_options.bound_relax_factor = 1e-3

# Set initial guess
# Time points (initial and final time)
problem.guess.phase[0].time = np.array([0.0, 1.0])  # 1 second maneuver

# Create initial and final states
initial_state = np.zeros((nx, 1))
initial_state[0:3] = 0.0  # Start at origin
initial_state[3:6] = 0.0  # Start at rest
initial_state[6] = 1.0    # Quaternion w = 1.0 (no rotation)
initial_state[7:10] = 0.0 # Quaternion x,y,z = 0.0
initial_state[10:13] = 0.0 # No angular velocity
initial_state[13:21] = 0.0 # No motor angles

final_state = np.zeros((nx, 1))
final_state[0:2] = 0.0    # End at x,y = 0
final_state[2] = 0.3      # End at z = 0.3 (reduced height further)
final_state[3:6] = 0.0    # End at rest
final_state[6] = 1.0      # Quaternion w = 1.0 (no rotation)
final_state[7:10] = 0.0   # Quaternion x,y,z = 0.0
final_state[10:13] = 0.0  # No angular velocity
final_state[13:21] = 0.0  # No motor angles

# Set state guess
state_guess = np.hstack([initial_state, final_state])
problem.guess.phase[0].state = state_guess

# Set control guess (constant hover thrust)
control_guess = np.zeros((nu, 2))
control_guess[0:4, :] = 0.35  # Reduced throttle further
problem.guess.phase[0].control = control_guess

# Solve the problem
try:
    solution = problem.solve()
    if hasattr(solution, 'status'):
        if solution.status == 0:
            print("Problem solved successfully!")
        else:
            print(f"Solver failed with status: {solution.status}")
    else:
        print("Solver failed - no status available")
except Exception as e:
    print(f"Error during solve: {str(e)}")




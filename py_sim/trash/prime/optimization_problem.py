import casadi as ca
from dynamics import nx,nu, f_dyn
from integrator import F

def reorthogonalize_gram_schmidt(R):
    """
    Reorthogonalizes a 3x3 rotation matrix using the Gram-Schmidt process.
    """
    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]

    # Step 1: Normalize the first column
    x = x / ca.norm_2(x)
    
    # Step 2: Make y orthogonal to x
    y = y - ca.dot(x, y) * x
    y = y / ca.norm_2(y)
    
    # Step 3: Make z orthogonal to x and y using cross product
    z = ca.cross(x, y)
    z = z / ca.norm_2(z)
    
    # Reassemble the orthogonalized rotation matrix
    R_orth = ca.horzcat(x, y, z)
    return R_orth

def stage_cost(x_current, u_current, x_target):
    """
    A richer stage cost for NMPC.

    Args:
        x_current: Current state vector.
        u_current: Current control vector.
        x_target: Target state vector.
        weights: Dictionary of weight values.

    Returns:
        Cost value.
    """
    # Weights for different objectives
    gamma_pos = 1
    gamma_vel = 1
    gamma_ang_vel = 1
    gamma_orient = 1
    gamma_control = 0.01

    # Position Error
    position_error = ca.sumsqr(x_current[0:3] - x_target[0:3])

    # Velocity Error (if velocity is part of state)
    velocity_error = ca.sumsqr(x_current[3:6] - x_target[3:6])

    # Angular Velocity Error 
    ang_velocity_error = ca.sumsqr(x_current[15:18] - x_target[15:18])

    # Orientation Error (using trace-based geodesic distance)
    R_current = ca.reshape(x_current[6:15], (3, 3)).T
    R_target = ca.reshape(x_target[6:15], (3, 3))
    R_rel = R_target @ R_current
    trace_R_rel = ca.fmin(3, ca.fmax(-1, ca.trace(R_rel)))
    orientation_error = ca.acos((trace_R_rel - 1) / 2)

    # Control Effort (energy usage)
    control_effort = ca.sumsqr(u_current)


    # Total Cost
    cost = (gamma_pos * position_error +
            gamma_vel * velocity_error +
            gamma_orient * orientation_error +
            gamma_control * control_effort)
    
    return cost


def solve_optimal_control(x_init, u_init, x_target):

    # Fixed step Runge-Kutta 4 integrator
    X0 = ca.MX.sym('x', nx)
    U0 = ca.MX.sym('u', nu)
    T0 = ca.MX.sym('t')

    N = 4720
    M = 4  # RK4 steps per interval
    DT = T0 / N / M
    XF = X0
    for j in range(M):
        k1 = f_dyn(XF, U0)
        k2 = f_dyn(XF + DT / 2 * k1, U0)
        k3 = f_dyn(XF + DT / 2 * k2, U0)
        k4 = f_dyn(XF + DT * k3, U0)
        XF = XF + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    F = ca.Function('F', [X0, U0, T0], [XF], ['x0', 'u0','T0'], ['xf'])

    opti = ca.Opti()
    X = opti.variable(nx, N+1)
    U = opti.variable(nu, N)
    T = opti.variable()

    x0 = opti.parameter(nx)
    u0 = opti.parameter(nu)
    opti.set_value(x0, x_init)
    opti.set_value(u0, u_init)

    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(U[:, 0] == u0)


    # Optimized Element-Wise Constraints (Element-Wise)
    for i in range(4):
        opti.subject_to(opti.bounded(0, U[i, :], 1))       # Rotor Speeds (0 to 1)

    for i in range(4, 8):
        opti.subject_to(opti.bounded(-1, U[i, :], 1))      # Tilt Rates (-1 to 1)

    for i in range(8, 12):
        opti.subject_to(opti.bounded(-1, U[i, :], 1))      # Roll Rates (-1 to 1)

    for i in range(18,22):
        opti.subject_to(opti.bounded(ca.pi/4, X[i, :], ca.pi/4))       # Rotor Speeds (0 to 1)

    for i in range(22, 26):
        opti.subject_to(opti.bounded(ca.pi/4, X[i, :], ca.pi/4))      # Roll Rates (-1 to 1)

    opti.subject_to(T > 0)

    # Cost Function Initialization
    J = 0
    for k in range(N):
        x_k = X[:, k]
        u_k = U[:, k]
        x_next = F(x_k, u_k, T)
        R = ca.reshape(X[6:15, k], (3, 3))
        I = ca.DM.eye(3)
        opti.subject_to(R.T @ R == I)

        opti.subject_to(X[:, k+1] == x_next)        
        J += stage_cost(x_k, u_k, x_target)

    # Terminal Cost (Heavily Weighted)
    terminal_error = ca.sumsqr(X[:, -1] - x_target)
    J += 1 * terminal_error + 2*T

    # Define Cost in Optimization
    opti.minimize(J)
    
    # Initial Guess for Control (Slight Hover)
    opti.set_initial(X[:,0], x_init)
    opti.set_initial(U[:,0], u_init)
    opti.set_initial(T, 2)
    
    # Solver Settings
    opts = {
        "ipopt.print_level": 5,             # Detailed printout
        "print_time": 1,                    # Display solve time
        "ipopt.max_iter": 500,             # Increase maximum iterations
        "ipopt.tol": 1e-8,                  # Tight tolerance for high accuracy
        "ipopt.acceptable_tol": 1e-6,       # Slightly relaxed acceptable tolerance
        "ipopt.acceptable_iter": 15,        # Allow more iterations before accepting
        "ipopt.linear_solver": "mumps",     # More robust linear solver
        "ipopt.mu_strategy": "adaptive",    # Adaptive barrier parameter strategy
        "ipopt.mu_target": 1e-10,           # Target barrier parameter
        "ipopt.warm_start_init_point": "yes", # Allow warm start
        "ipopt.warm_start_bound_push": 1e-8, # Aggressive warm start settings
        "ipopt.warm_start_mult_bound_push": 1e-8,
        "ipopt.hessian_approximation": "limited-memory", # Limited-memory Hessian
        "ipopt.derivative_test": "first-order",  # Derivative check (can slow down)
        "ipopt.derivative_test_tol": 1e-6,       # Tight tolerance for derivative check
        "ipopt.max_cpu_time": 1000,             # Increase CPU time limit
    }
    opti.solver('ipopt', opts)

    # Solve Problem
    try:
        sol = opti.solve()
        x_opt = sol.value(X)
        u_opt = sol.value(U)
        t_opt = sol.value(T)
        return x_opt, u_opt, t_opt
    except RuntimeError as e:
        print("❌ Solver Failed:", str(e))
        print("📌 Debugging Information:")
        print("Last Valid State:", opti.debug.value(X))
        print("Last Valid Control:", opti.debug.value(U))
        raise e


# def solve_optimal_control(initial_x, target_x, N):
#     opti = ca.Opti()  # Create Optimization Problem
#     X = opti.variable(nx, N+1)  # State Variables
#     U = opti.variable(nu, N)    # Control Variables
#     T = opti.variable()         # Total Time (Free Variable)
#     opti.subject_to(T > 1e-3)   # Time must be positive

#     # Initial State Constraint
#     opti.subject_to(X[:, 0] == initial_x)

#     # Stage and Terminal Costs
#     J = 0
#     M = 4  # RK4 steps per interval
#     for k in range(N):
#         x_k = X[:, k]
#         u_k = U[:, k]
#         DT = T / N / M

#         # RK4 Integration
#         XF = x_k
#         for j in range(M):
#             k1 = f_dyn(XF, u_k)
#             k2 = f_dyn(XF + DT / 2 * k1, u_k)
#             k3 = f_dyn(XF + DT / 2 * k2, u_k)
#             k4 = f_dyn(XF + DT * k3, u_k)
#             XF = XF + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

#         # Transition Constraint
#         opti.subject_to(X[:, k+1] == XF)

#         # Stage Cost (Distance + Control Effort)
#         J += 1000*ca.sumsqr(X[0:2, k] - target_x[0:2]) 

#     # Terminal Cost
#     J += 1000 * ca.sumsqr(X[0:2, -1] - target_x[0:2])
#     J += 1 * T  # Penalize Time

#     # Define Cost
#     opti.minimize(J)

#     # Control Constraints
#     opti.subject_to(opti.bounded(0, U[0:4], 1))  # Control limited to [-1, 1]
#     opti.subject_to(opti.bounded(-1, U[4:8], 1))
#     opti.subject_to(opti.bounded(-1, U[8:12], 1))

#     # Initial Guess (Smart Guess)
#     opti.set_initial(X[:,0], initial_x)
#     opti.set_initial(U, 0)
#     opti.set_initial(T, 1)  # Start with 5 seconds

#     # Solver Settings
#     opts = {
#         "ipopt.print_level": 5,
#         "print_time": 1,
#         "ipopt.max_iter": 1000,
#         "ipopt.tol": 1e-6,
#         "ipopt.acceptable_tol": 1e-5,
#     }
#     opti.solver('ipopt', opts)

#     # Solve Problem
#     try:
#         sol = opti.solve()
#         x_opt = sol.value(X)
#         u_opt = sol.value(U)
#         T_opt = sol.value(T)
#         return x_opt, u_opt, T_opt
#     except RuntimeError as e:
#         print("❌ Solver Failed:", str(e))
#         print("📌 Debugging Information:")
#         print("Last Valid State:", opti.debug.value(X))
#         print("Last Valid Control:", opti.debug.value(U))
#         print("Last Valid Time:", opti.debug.value(T))
#         raise e

# # def reorthogonalize_gram_schmidt(R):
# #     """
# #     Reorthogonalizes a 3x3 rotation matrix using the Gram-Schmidt process.

# #     Args:
# #         R (casadi.MX or casadi.SX): 3x3 Rotation matrix to be reorthogonalized.

# #     Returns:
# #         ca.MX or ca.SX: Reorthogonalized 3x3 rotation matrix.
# #     """
# #     # Extract columns of the rotation matrix (basis vectors)
# #     x = R[:, 0]
# #     y = R[:, 1]
# #     z = R[:, 2]
    
# #     # Step 1: Normalize the first column
# #     x = x / ca.norm_2(x)
    
# #     # Step 2: Make y orthogonal to x
# #     y = y - ca.dot(x, y) * x
# #     y = y / ca.norm_2(y)
    
# #     # Step 3: Make z orthogonal to x and y using cross product
# #     z = ca.cross(x, y)
# #     z = z / ca.norm_2(z)
    
# #     # Reassemble the orthogonalized rotation matrix
# #     R_orth = ca.horzcat(x, y, z)
# #     return R_orth

# # def stage_cost(x_current, u_current, x_target):
# #     # Weights
# #     gamma_pos = 10
# #     gamma_control = 0.001
# #     # gamma_attitude = 0

# #     # Error calulations
# #     position_error = ca.sumsqr(x_target[0:3] - x_current[0:3])
# #     control_effort_norm = ca.sumsqr(u_current)

# #     # R_current = ca.reshape(x_current[6:15], (3, 3)).T
# #     # R_target = ca.reshape(x_target[6:15], (3, 3))
# #     # R_rel = R_target@R_current
# #     # trace_R_rel = ca.fmin(3, ca.fmax(-1, ca.trace(R_rel)))
# #     # attitude_error = ca.acos((trace_R_rel - 1) / 2) # Geodesic distance

# #     return gamma_pos*position_error + gamma_control*control_effort_norm #+ gamma_attitude*attitude_error

# # def terminal_cost(x_final, x_target, T):
# #     # Terminal Weights
# #     gamma_pos_t = 10000
# #     # gamma_attitude_t = 50
# #     gamma_T = 0.1

# #     position_error_t = ca.sumsqr(x_target[0:3] - x_final[0:3])
# #     # R_final = ca.reshape(x_final[6:15], (3, 3)).T
# #     # R_target = ca.reshape(x_target[6:15], (3, 3))
# #     # R_rel = R_target@R_final
# #     # trace_R_rel = ca.fmin(3, ca.fmax(-1, ca.trace(R_rel)))
# #     # attitude_error_t = ca.acos((trace_R_rel - 1) / 2) # Geodesic distance


# #     return gamma_pos_t*position_error_t + gamma_T * T #+ gamma_attitude_t*attitude_error_t 
    


# # def solve_optimization(initial_x, target_x, N):
# #     # Initialize optimzation problem
# #     opti = ca.Opti()
# #     X = opti.variable(nx, N+1)
# #     U = opti.variable(nu, N)
# #     T = opti.variable()
# #     opti.subject_to(X[:, 0] == initial_x)  # Start at initial state
# #     J = 0

# #     # Fixed step Runge-Kutta 4 integrator
# #     X0 = ca.MX.sym('x', nx)
# #     U0 = ca.MX.sym('u', nu)
# #     T0 = ca.MX.sym('t')
# #     M = 4 # RK4 steps per interval
# #     DT = T0/N/M
# #     XF = X0
# #     for j in range(M):
# #         k1 = f_dyn(XF, U0)
# #         k2 = f_dyn(XF + DT/2 * k1, U0)
# #         k3 = f_dyn(XF + DT/2 * k2, U0)
# #         k4 = f_dyn(XF + DT * k3, U0)
# #         XF=XF+DT/6*(k1 +2*k2 +2*k3 +k4) 
# #     F = ca.Function('F', [X0, U0, T0], [XF],['x0','u0','T0'],['xf'])


# #     # Multishooting constraints + Stage costs
# #     for k in range(N):
# #         x_k = X[:,k]
# #         u_k = U[:,k]
    
# #         J += stage_cost(x_k, u_k, target_x)
# #         x_next = F(x_k, u_k, T)
        
# #         # Reorthogonalize Rotation Matrix
# #         R_k = ca.reshape(x_next[6:15], (3, 3)).T
# #         R_k_orth = reorthogonalize_gram_schmidt(R_k)
# #         x_next[6:15] = ca.reshape(R_k_orth.T, (9, 1))  # Replace with reorthogonalized matrix
# #         opti.subject_to(X[:,k+1]==x_next)

        
# #     #Terminal Cost    
# #     J += terminal_cost(X[:, -1], target_x, T)

# #     # Constraints: Control (Saturation)
# #     opti.subject_to(opti.bounded(0, U[0:4, :], 1))  # Motor Speeds
# #     opti.subject_to(opti.bounded(-1, U[4:8, :], 1))  # Tilt Rates
# #     opti.subject_to(opti.bounded(-1, U[8:12, :], 1))  # Roll Rates


# #     # Constraints: Limits of Motor angles
# #     opti.subject_to(opti.bounded(-ca.pi/2, X[18:22], ca.pi/2)) # Pitch Lmits
# #     opti.subject_to(opti.bounded(-ca.pi/2,X[22:26],ca.pi/2)) # Roll Limits

# #     # Contraint: Ensure T positive
# #     opti.subject_to(T>1e-6) 
    
# #     # Intial Guess
# #     opti.set_initial(X[:,0], initial_x)
# #     opti.set_initial(T, 1.0)  # Start with 5 seconds

# #     # Solver Settings (IPOPT)
# #     opts = {
# #          "ipopt.print_level": 5,  # Detailed printout
# #          "print_time": 1,
# #          "ipopt.max_iter": 1000,
# #          "ipopt.tol": 1e-6,
# #          "ipopt.acceptable_tol": 1e-5,
# #          "ipopt.acceptable_iter": 10,
# #     }
# #     opti.solver('ipopt', opts)
# #     try:
# #         sol = opti.solve()
# #         x_opt = sol.value(X)
# #         u_opt = sol.value(U)
# #         T_opt = sol.value(T)
# #         return x_opt, u_opt, T_opt
# #     except RuntimeError as e:
# #         print("❌ Solver Failed:", str(e))
# #         print("📌 Debugging Information:")
        
# #         # Get Last Valid Values (Even if Solver Fails)
# #         last_valid_X = opti.debug.value(X)
# #         last_valid_U = opti.debug.value(U)
# #         last_valid_T = opti.debug.value(T)
        
# #         print("Last Valid State X:", last_valid_X)
# #         print("Last Valid Control U:", last_valid_U)
# #         print("Last Valid Time T:", last_valid_T)
        
# #         raise e


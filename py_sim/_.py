import casadi as ca
import numpy as np
from dynamics import nx, nu, f_dyn, u_w_hover  # Make sure dynamics.py is properly set up
from utils import normalize_quaternion, make_X_guess, constant_control_guess
from integrator import F_rk4
import os
import pandas as pd


# Discrete Cost Function
def cost_1(x, u, x_target):
    position_error = ca.sumsqr(x[0:3] - x_target[0:3])  # Position error (x, y, z)
    # velocity_error = ca.sumsqr(x[3:6])  # Velocity error (vx, vy, vz)
    # angular_velocity_error = ca.sumsqr(x[10:13])  # Angular velocity error
    # control_effort = ca.sumsqr(u)  # Minimize control effort
    
    # # Quaternion Orientation Error (Track Orientation)
    # qw, qx, qy, qz = x[6], x[7], x[8], x[9]
    # qw_target, qx_target, qy_target, qz_target = x_target[6], x_target[7], x_target[8], x_target[9]
    # dot_product = qw * qw_target + qx * qx_target + qy * qy_target + qz * qz_target
    # dot_product = ca.fmax(-1.0, ca.fmin(1.0, dot_product))
    # quaternion_error = 2 * ca.acos(ca.fabs(dot_product))
    return 5 * position_error  #+ 0.1 * angular_velocity_error + 0.01 * control_effort + 4 * quaternion_error + 0.1 * velocity_error



def solve(cost_fn, x_target, x_init, N):
    """Solve the optimization problem.
    
    Args:
        cost_fn: Cost function to use (stage_cost or terminal_cost)
        x_target: Target state
        x_init: Initial state
        N: Prediction horizon
        dt: Time step
    
    Returns:
        x_opt: Optimal state trajectory
        u_opt: Optimal control sequence
    """
    # Initialize optimization problem
    opti = ca.Opti()
    
    # Decision variables
    X = opti.variable(nx, N+1)  # States
    U = opti.variable(nu, N)    # Controls
    
    u_target = ca.DM.zeros(nu)
    u_target[0:4] = u_w_hover

    test_x = x_init.full().flatten()
    test_u = u_target.full().flatten()
    test_next = F_rk4(test_x, test_u, 0.01)
    print("Test next state:", test_next)    
    print("Initial cost:", cost_1(x_init, u_target, x_target))


    # Initial state constraint
    opti.subject_to(X[:,0] == x_init)
    opti.subject_to(U[:,0] == u_target)
    J = 0

    # Dynamics constraints
    for k in range(N):
        # Get current state and control
        x_k = X[:,k]
        u_k = U[:,k]
        # Add stage cost
        J += cost_fn(x_k, u_k, x_target)
        dt = 0.01
        x_next = F_rk4(x_k, u_k, dt)
        opti.subject_to(X[:,k+1] == x_next)
            
    
    # Control constraints
    opti.subject_to(opti.bounded(0, U[0:4], 1))  # Motor speeds
    opti.subject_to(opti.bounded(-1, U[4:8], 1))  # Tilt rates
    opti.subject_to(opti.bounded(-1, U[8:12], 1))  # Roll rates
    
    # State constraints
    opti.subject_to(opti.bounded(-ca.pi/2, X[13:17], ca.pi/2))  # Pitch limits
    opti.subject_to(opti.bounded(-ca.pi/2, X[17:21], ca.pi/2))  # Roll limits
    
    # Initial guess

    # Convert CasADi DMs to NumPy arrays
    x0 = np.array(x_init.full()).flatten()
    xf = np.array(x_target.full()).flatten()
    u_hover = np.array(u_target.full()).flatten() 
    q_start, q_end = 6, 10

    # Generate improved guesses
    X_guess = make_X_guess(x0, xf, N, q_start, q_end)  # shape: (nx, N+1)
    U_guess = constant_control_guess(u_hover, N)       # shape: (nu, N)
    
    opti.set_initial(X, X_guess)
    opti.set_initial(U, U_guess)
    opti.minimize(J)

    # Solver settings
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.acceptable_iter": 10,
    }
    opti.solver('ipopt', opts)
    
    # Solve problem
    try:
        sol = opti.solve()
        x_opt = sol.value(X)
        u_opt = sol.value(U)
        return x_opt, u_opt
    except RuntimeError as e:
        print("❌ Solver Failed:", str(e))
        print("📌 Debugging Information:")
        print("Last Valid State:", opti.debug.value(X))
        print("Last Valid Control:", opti.debug.value(U))
        raise e

    
    

# Horizon
N = 20

# Target State (Hover at Z = 2 m, 45 deg roll) ===
x_target = ca.DM.zeros(nx)
x_target[2] = 3  # Target Z position (hover)
x_target[6] = 1 # q _a 
x_target[7] = 0  # q_b

u_target = ca.DM.zeros(nu)
u_target[0:4] = u_w_hover

# Initial State Constraint (Starting at rest)
x_init = ca.DM.zeros(nx)
x_init[6] = 1.0  # Unit quaternion (w = 1, no rotation)
x_opt, u_opt = solve(cost_1, x_target, x_init, N)

# Creating a directory and saving NMPC results as CSV
save_dir = "nmpc_results"
os.makedirs(save_dir, exist_ok=True)

# Saving the optimized state trajectory (x_opt) and control inputs (u_opt)
x_opt_df = pd.DataFrame(x_opt.T)  # Transpose to make each column a state
u_opt_df = pd.DataFrame(u_opt.T)  # Transpose to make each column a control

# Save to CSV files
x_opt_df.to_csv(os.path.join(save_dir, "x_opt.csv"), index=False)
u_opt_df.to_csv(os.path.join(save_dir, "u_opt.csv"), index=False)


# import casadi as ca
# import numpy as np
# import matplotlib.pyplot as plt
# from dynamics import f_dyn, nx, nu  # Make sure dynamics.py is properly set up
# from integrator import integrator_cvodes
# import os
# import pandas as pd
# from optimization_problem import solve, cost_1

# # === NMPC Settings ===
# dt = 0.01  # Time step (10 ms)
# N = 20     # Shorter Prediction horizon (0.2 seconds)
# sim_time = 5.0  # Total simulation time (5 seconds)
# n_steps = int(sim_time / dt)

# # === Target State (Hover at Z = 5 m, 45° roll) ===
# x_target = ca.DM.zeros(nx)
# x_target[0] = 0
# x_target[1] = 0
# x_target[2] = 1
# x_target[6] = 0.924  # 45° roll
# x_target[7] = 0.383

# # === NMPC Cost Function ===
# def nmpc_cost(x, u, x_target):
#     position_error = ca.sumsqr(x[0:3] - x_target[0:3])  # Position error
#     velocity_error = ca.sumsqr(x[3:6])  # Velocity error
#     angular_velocity_error = ca.sumsqr(x[10:13])  # Angular velocity error
#     control_effort = ca.sumsqr(u)  # Minimize control effort

#     # Quaternion Orientation Error
#     dot_product = ca.dot(x_target[6:10], x[6:10])
#     dot_product = ca.fmax(-1.0, ca.fmin(1.0, dot_product))
#     quaternion_error = 1 - ca.fabs(dot_product)

#     return 5 * position_error + 0.1 * velocity_error + 0.1 * angular_velocity_error + 0.01 * control_effort + 4 * quaternion_error

# # === Initialize Simulation ===
# x_init = ca.DM.zeros(nx)
# x_init[6] = 1.0  # Unit quaternion (w = 1, no rotation)
# x_current = x_init

# # Storage for results
# x_traj = np.zeros((nx, n_steps))
# u_traj = np.zeros((nu, n_steps - 1))

# # === NMPC Simulation Loop ===
# for t in range(n_steps - 1):
#     # === NMPC Solver Setup ===
#     x_opt, u_opt = solve(cost_1, x_target, x_current, N)
    
    
#     # Apply only the first control action
#     u_applied = u_opt[:, 0]
#     u_traj[:, t] = u_applied

#     # Simulate one step with this control
#     result = integrator_cvodes(x0=x_current, p=u_applied)
#     x_next = result['xf']

#     # Store results
#     x_traj[:, t] = x_current.full().flatten()
#     x_current = x_next
#     print("Finished Step: ",t)

# # Store final state
# x_traj[:, -1] = x_current.full().flatten()

# # === Save NMPC Results ===
# save_dir = "nmpc_results"
# os.makedirs(save_dir, exist_ok=True)
# pd.DataFrame(x_traj.T).to_csv(os.path.join(save_dir, "x_opt.csv"), index=False)
# pd.DataFrame(u_traj.T).to_csv(os.path.join(save_dir, "u_opt.csv"), index=False)

# print("True NMPC (Receding Horizon) complete. Results saved to 'nmpc_results/' directory.")



import casadi as ca
import numpy as np
from dynamics import nx, nu, f_dyn, u_w_hover  # Make sure dynamics.py is properly set up
from utils import normalize_quaternion, make_X_guess, constant_control_guess
from integrator import F_rk4


#Discrete Cost Function
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



def solve(cost_fn, x_target, x_init, N, dt):
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


    # Initial state constraint
    opti.subject_to(X[:,0] == x_init)
    opti.subject_to(U[:,0] == u_target)
    J = 0

    # Dynamics constraints
    for k in range(N):
        # Get current state and control
        x_k = X[:,k]
        u_k = U[:,k]
        opti.subject_to(opti.bounded(0, U[0:4,k], 1))  # Motor speeds
        opti.subject_to(opti.bounded(-1, U[4:8,k], 1))  # Tilt rates
        opti.subject_to(opti.bounded(-1, U[8:12,k], 1))  # Roll rates
        opti.subject_to(opti.bounded(-ca.pi/2, X[13:17, k], ca.pi/2))  # Pitch limits
        opti.subject_to(opti.bounded(-ca.pi/2, X[17:21, k], ca.pi/2))  # Roll limits
    
        # Add stage cost
        J += cost_fn(x_k, u_k, x_target)
        x_next = F_rk4(x_k, u_k, dt)
        opti.subject_to(X[:,k+1] == x_next)
            
    
    
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


import casadi as ca
import numpy as np
from dynamics import nx, nu, f_dyn  
from utils import normalize_quaternion, make_X_guess, constant_control_guess, quaternion_geo_distance
from integrator import F_rk4


#Discrete Cost Function
def cost(x, u, x_target):

    l_pos = 5
    l_vel = 0.1
    l_ang_vel = 0.1
    l_control = 0.01
    l_quat = 0.5

    position_error = ca.sumsqr(x[0:3] - x_target[0:3])  # Position error (x, y, z)
    velocity_error = ca.sumsqr(x[3:6] - x_target[3:6])  # Velocity error (vx, vy, vz)
    angular_velocity_error = ca.sumsqr(x[10:13] - x_target[10:13])  # Angular velocity error
    control_effort = ca.sumsqr(u)  # Minimize control effort

    q_WB = x[6:10]
    q_WB_target = x_target[6:10]
    dot_product = ca.dot(q_WB, q_WB_target)
    quaternion_error = 1 - dot_product**2

    
    return l_pos * position_error + l_vel * velocity_error + l_ang_vel * angular_velocity_error + l_control * control_effort #+ l_quat * quaternion_error



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
    
    # minmimum normalized thrust force from each motor
    # such that the quadcopter can hover
    u_w_hover = 0.4293844144478733 # m * g / (4 * cf * (w_max ** 2))
    u_target = ca.DM.zeros(nu)
    u_target[0:4] = u_w_hover


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


        
        opti.subject_to(opti.bounded(-ca.pi/2, X[13:17, k], ca.pi/2))  # Pitch limits
        opti.subject_to(opti.bounded(-ca.pi/2, X[17:21, k], ca.pi/2))  # Roll limits
    
        # Add stage cost
        J += cost_fn(x_k, u_k, x_target)
        x_next = F_rk4(x_k, u_k, dt)
        opti.subject_to(X[:,k+1] == x_next)
            

    q_WB = X[6:10, -1]
    q_WB_target = x_target[6:10]
    dot_product = ca.dot(q_WB, q_WB_target)
    quaternion_error = 1 - dot_product**2

    J += 10*ca.sumsqr(X[0:3, -1] - x_target[0:3])

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


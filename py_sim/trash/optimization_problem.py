import casadi as ca
from dynamics import nx,nu, f_dyn

def stage_cost(x_current, u_current, x_target):
    # Weights
    gamma_pos = 5
    gamma_vel = 0.1
    gamma_ang_vel = 0.1
    gamma_control = 0.01
    gamma_attitude = 1

    # Error calulations
    position_error = ca.sumsqr(x_current[0:3] - x_target[0:3])
    velocity_error = ca.sumsqr(x_current[3:6] - x_target[3:6])
    angular_velocity_norm = ca.sumsqr(x_current[10:13])
    control_effort_norm = ca.sumsqr(u_current)
    quat_dot = ca.dot(x_current[6:10], x_target[6:10])
    quat_error = 2 * ca.acos(ca.fmin(1.0, ca.fmax(-1.0, quat_dot)))

    return gamma_pos*position_error + gamma_vel*velocity_error + gamma_ang_vel*angular_velocity_norm + gamma_control*control_effort_norm + gamma_attitude*quat_error

def terminal_cost(x_final, x_target, T):
    # Terminal Weights
    gamma_pos_t = 10
    gamma_attitude_t = 5
    gamma_T = 0.1

    # Terminal Errors
    position_error_t =  ca.sumsqr(x_final[0:3] - x_target[0:3])
    quat_dot = ca.dot(x_final[6:10], x_target[6:10])
    quat_error_t = 2 * ca.acos(ca.fmin(1.0, ca.fmax(-1.0, quat_dot)))

    return gamma_pos_t*position_error_t + gamma_T * T  + gamma_attitude_t*quat_error_t

def solve(cost_fn, x_target, x_init, N, dt=0.01):
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
    
    # Initial state constraint
    opti.subject_to(X[:,0] == x_init)
    
    # Dynamics constraints
    for k in range(N):
        # Get current state and control
        x_k = X[:,k]
        u_k = U[:,k]
        
        # Add stage cost
        opti.minimize(cost_fn(x_k, u_k, x_target))
        
        # Add dynamics constraint
        x_next = X[:,k+1]
        k1 = f_dyn(x_k, u_k)
        k2 = f_dyn(x_k + dt/2 * k1, u_k)
        k3 = f_dyn(x_k + dt/2 * k2, u_k)
        k4 = f_dyn(x_k + dt * k3, u_k)
        x_next_pred = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(x_next == x_next_pred)
    
    # Add terminal cost
    opti.minimize(cost_fn(X[:,N], U[:,N-1], x_target))
    
    # Control constraints
    opti.subject_to(opti.bounded(0, U[0:4], 1))  # Motor speeds
    opti.subject_to(opti.bounded(-1, U[4:8], 1))  # Tilt rates
    opti.subject_to(opti.bounded(-1, U[8:12], 1))  # Roll rates
    
    # State constraints
    opti.subject_to(opti.bounded(-ca.pi/2, X[13:17], ca.pi/2))  # Pitch limits
    opti.subject_to(opti.bounded(-ca.pi/2, X[17:21], ca.pi/2))  # Roll limits
    
    # Initial guess
    U_guess = ca.DM.zeros(nu, N)
    U_guess[0:4,:] = 0.5  # Initial motor speeds
    X_guess = ca.repmat(x_init, 1, N+1)
    
    opti.set_initial(X, X_guess)
    opti.set_initial(U, U_guess)
    
    # Solver settings
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-8,
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

def solve_optimization(initial_x, target_x, Ts, H):
    # Initialize optimzation problem
    opti = ca.Opti()
    X = opti.variable(nx, H+1)
    U = opti.variable(nu, H)
    J = 0
    Z = opti.variable(nx, H, 3)   # Collocation states (3 Radau points)
    # Collocation Coefficients (Radau)
    C = ca.DM([[5/12, -1/12, 1/12],
               [3/4,  0,   1/4],
               [11/12, 1/12, 5/12]])

    # Multishooting constraints + Stage costs
    for k in range(H):
        x_k = X[:,k]
        u_k = U[:,k]
        J += stage_cost(x_k, u_k, target_x)

        # Collocation Points Dynamics
        for j in range(3):  # 3 Radau points
            z_kj = Z[:, k, j]
            x_dot_kj = f_dyn(z_kj, u_k)
            collocation_eq = Ts * ca.mtimes(C[j, :], Z[:, k, :]) - (z_kj - x_k)
            opti.subject_to(collocation_eq == Ts * x_dot_kj)
        x_next = Ts * ca.mtimes(C[-1, :], Z[:, k, :]) + x_k
        opti.subject_to(X[:,k+1]==x_next)

    # Constraints: Control (Saturation)
    opti.subject_to(opti.bounded(0, U[0:4], 1))  # Motor Speeds
    opti.subject_to(opti.bounded(-1, U[4:8], 1))  # Tilt Rates
    opti.subject_to(opti.bounded(-1, U[8:12], 1))  # Roll Rates

    # Constraints: Limits of Motor angles
    opti.subject_to(opti.bounded(-ca.pi/2, X[13:17], ca.pi/2)) # Pitch Lmits
    opti.subject_to(opti.bounded(-ca.pi/2,X[17:21],ca.pi/2)) # Roll Limits

    # Intial Guess
    U_guess = ca.DM.zeros(nu,H)
    U_guess[0:4,:] = 0.5
    X_guess = ca.repmat(initial_x, 1, H + 1)
    Z_guess = ca.repmat(initial_x, 1, H, 3)

    opti.set_initial(X, X_guess)
    opti.set_initial(U, U_guess)
    opti.set_initial(Z, Z_guess)

    # Set minimization
    opti.minimize(J)

    # Solver Settings (IPOPT)
    opts = {
         "ipopt.print_level": 0,
         "print_time": 0,
         "ipopt.max_iter": 1000,
         "ipopt.tol": 1e-8,
    }
    opti.solver('ipopt', opts)

    # Solve Problem
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
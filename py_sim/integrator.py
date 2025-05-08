import casadi as ca
from dynamics import f_dyn, nx, nu  # Assuming your f_dyn is in dynamics.py

# Define time step and horizon
dt = 0.001  # Time step
N = 40     # Prediction horizon

# State and control symbols
x = ca.SX.sym('x', nx)
u = ca.SX.sym('u', nu)

# CasADi Integrator (RK4) - Corrected and Optimized
integrator_rk4 = ca.integrator(
    'integrator_rk4',  # Name of the integrator
    'rk',              # Integration method (Runge-Kutta)
    {
        'x': x,       # State variable
        'p': u,       # Control input (parameter)
        'ode': f_dyn(x, u)  # The dynamics function
    },
    {
        'tf': dt,                     # Total integration time (1 step of dt)
        'number_of_finite_elements': 4  # Number of RK4 steps (4 for RK4)
    }
)

# Define initial state and full power control
x_init = ca.DM.zeros(nx)  # Initial state (all zeros for testing)
x_init[6:10] = ca.DM([1, 0, 0, 0])  # Unit quaternion (w, x, y, z)

# Full power control (motors at 100%, no tilt or roll)
u_full_power = ca.DM([1, 1, 1, 1,  # Motor speeds at 100%
                      0, 0, 0, 0,  # No tilt (0% of max)
                      0, 0, 0, 0]) # No roll (0% of max)

# Assert quaternion normalization (Strict check)
assert abs(float(ca.norm_2(x_init[6:10])) - 1) < 1e-6, "Initial quaternion is not normalized"

# Simulate one step using RK4 with full power
result = integrator_rk4(x0=x_init, p=u_full_power)
x_next = result['xf']  # 'xf' is the final state after RK4 step

# Ensure quaternion remains normalized (numerically robust)
q_next = x_next[6:10]
q_next = q_next / ca.norm_2(q_next)
x_next[6:10] = q_next  # Overwrite with normalized value

# Print the results
print("Next state after one RK4 step with full power:", x_next)
print("Normalized quaternion after step:", q_next)

import casadi as ca
from dynamics import f_dyn, nx, nu  

# # CasADi Integrator Setup (RK4) ===
# X0 = ca.SX.sym('x', nx)
# U = ca.SX.sym('u', nu)


# # Fixed step Runge-Kutta 4 integrator
# T = 5
# N = 20
# M = 4 # RK4 steps per interval
# DT = T/N/M

# X = X0
# for j in range(M):
#     k1 = f_dyn(X, U)
#     k2 = f_dyn(X + DT/2 * k1, U)
#     k3 = f_dyn(X + DT/2 * k2, U)
#     k4 = f_dyn(X + DT * k3, U)
#     X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    
# F = ca.Function('F', [X0, U], [X],['x0','u'],['xf'])


# # Initial State (Hovering)
# x_init = ca.DM.zeros(nx)
# x_init[2] = 1.0  # Start at z = 1 meter height
# x_init[6:15] = ca.reshape(ca.DM.eye(3), (9, 1))  # Identity rotation matrix (hover)

# # Control Input (Hovering)
# u_init = ca.DM.zeros(nu)
# u_init[0:4] = 0.6  # 50% motor speed
# u_init[4:8] = 0.0  # Zero pitch rate
# u_init[8:12] = 0.0  # Zero roll rate

# Fk = F(x_init, u_init)
# print(Fk)

# integrator_rk4_dt = lambda dt: ca.integrator(
#     'integrator_rk4',
#     'rk',
#     {
#         'x': x_sym,       # State variable (symbolic)
#         'p': u_sym,       # Control input (symbolic)
#         'ode': f_dyn(x_sym, u_sym)  # The dynamics function
#     },
#     {
#         'tf': dt,                     # Total integration time (1 step of dt)
#         'number_of_finite_elements': 4  # RK4 steps
#     }
# )

# integrator_cvodes_dt = lambda dt: ca.integrator(
#     'integrator_cvodes', 'cvodes', 
#     {'x': x_sym, 'p': u_sym, 'ode': f_dyn(x_sym, u_sym)},
#     {
#         'tf': dt,  # Total simulation time
#         'abstol': 1e-8,  # Absolute tolerance
#         'reltol': 1e-8,  # Relative tolerance
#         'max_num_steps': 1000  # Prevent infinite loops
#     }
# )

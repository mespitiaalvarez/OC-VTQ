import casadi as ca
from dynamics import nx, nu, f_dyn, u_w_hover  # Make sure dynamics.py is properly set up
import os
import pandas as pd
from optimization_problem import cost_1, solve


# Horizon
N = 100
dt = 0.01

# Target State (Hover at Z = 2 m, 45 deg roll) ===
x_target = ca.DM.zeros(nx)
x_target[0] = 3  # Target Z position (hover)
x_target[1] = 3  # Target Z position (hover)
x_target[2] = 3  # Target Z position (hover)
x_target[6] = 1 # q _a 
x_target[7] = 0  # q_b

u_target = ca.DM.zeros(nu)
u_target[0:4] = u_w_hover

# Initial State Constraint (Starting at rest)
x_init = ca.DM.zeros(nx)
x_init[6] = 1.0  # Unit quaternion (w = 1, no rotation)
x_opt, u_opt = solve(cost_1, x_target, x_init, N, dt)

# Creating a directory and saving NMPC results as CSV
save_dir = "nmpc_results"
os.makedirs(save_dir, exist_ok=True)

# Saving the optimized state trajectory (x_opt) and control inputs (u_opt)
x_opt_df = pd.DataFrame(x_opt.T)  # Transpose to make each column a state
u_opt_df = pd.DataFrame(u_opt.T)  # Transpose to make each column a control

# Save to CSV files
x_opt_df.to_csv(os.path.join(save_dir, "x_opt.csv"), index=False)
u_opt_df.to_csv(os.path.join(save_dir, "u_opt.csv"), index=False)



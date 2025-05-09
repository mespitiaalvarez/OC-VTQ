import casadi as ca
from dynamics import nx, nu  
from optimization_problem import solve_optimization 
import os
import pandas as pd


# MPC Settings
Ts = 0.1
H = 20

# Target State (Hover at Z = 2 m, 45 deg roll) 
x_target = ca.DM.zeros(nx)
x_target[2] = 1  # Target Z position (hover)
x_target[6] = 1.0

# Initial State Constraint (Starting at rest)
x_init = ca.DM.zeros(nx)
x_init[6] = 1.0  # Unit quaternion (w = 1, no rotation)

x_opt, u_opt, T_opt = solve_optimization(x_init, x_target, Ts, H)

# Creating a directory and saving NMPC results as CSV
save_dir = "nmpc_results"
os.makedirs(save_dir, exist_ok=True)

# Saving the optimized state trajectory (x_opt) and control inputs (u_opt)
x_opt_df = pd.DataFrame(x_opt.T)  # Transpose to make each column a state
u_opt_df = pd.DataFrame(u_opt.T)  # Transpose to make each column a control

# Save to CSV files
x_opt_df.to_csv(os.path.join(save_dir, "x_opt.csv"), index=False)
u_opt_df.to_csv(os.path.join(save_dir, "u_opt.csv"), index=False)

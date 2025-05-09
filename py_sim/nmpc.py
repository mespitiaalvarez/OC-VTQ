import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from dynamics import f_dyn, nx, nu  # Make sure dynamics.py is properly set up
import os
import pandas as pd
from optimization_problem import solve, cost_1


# === NMPC Settings ===
dt = 0.01  # Time step (10 ms)
N = 20     # Shorter Prediction horizon (0.2 seconds)
sim_time = 2.0  # Total simulation time (5 seconds)
n_steps = int(sim_time / dt)

# Simulation Integrator
x_sym = ca.SX.sym('x', nx)
u_sym = ca.SX.sym('u', nu)
integrator_cvodes = ca.integrator('integrator_cvodes', 'cvodes', {'x': x_sym, 'p': u_sym, 'ode': f_dyn(x_sym, u_sym)}, {'tf': dt, 'abstol': 1e-8, 'reltol': 1e-8, 'max_num_steps': 1000})

# === Target State (Hover at Z = 2)
x_target = ca.DM.zeros(nx)
x_target[2] = 2
x_target[6] = 1


# === Initialize Simulation ===
x_init = ca.DM.zeros(nx)
x_init[6] = 1.0  # Unit quaternion (w = 1, no rotation)
x_current = x_init

# Storage for results
x_traj = np.zeros((nx, n_steps))
u_traj = np.zeros((nu, n_steps - 1))

# === NMPC Simulation Loop ===
for t in range(n_steps - 1):
    # === NMPC Solver Setup ===
    x_opt, u_opt = solve(cost_1, x_target, x_current, N, dt)
    
    # Apply only the first control action
    u_applied = u_opt[:, 0]
    u_traj[:, t] = u_applied

    # Simulate one step with this control
    result = integrator_cvodes(x0=x_current, p=u_applied)
    x_next = result['xf']

    # Store results
    x_traj[:, t] = x_current.full().flatten()
    x_current = x_next
    print("Finished Step: ",t)

# Store final state
x_traj[:, -1] = x_current.full().flatten()

# === Save NMPC Results ===
save_dir = "nmpc_results"
os.makedirs(save_dir, exist_ok=True)
pd.DataFrame(x_traj.T).to_csv(os.path.join(save_dir, "x_opt.csv"), index=False)
pd.DataFrame(u_traj.T).to_csv(os.path.join(save_dir, "u_opt.csv"), index=False)

print("True NMPC (Receding Horizon) complete. Results saved to 'nmpc_results/' directory.")



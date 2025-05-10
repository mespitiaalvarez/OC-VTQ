import casadi as ca
from dynamics import nx, nu  
import os
import pandas as pd
import glob
from optimization_problem import cost, solve
import time

dt = 0.01
def main():
    # Horizon
    N = 200

    # Target State (Hover at Z = 2 m, 45 deg roll) ===
    x_target = ca.DM.zeros(nx)
    x_target[0] = 5   
    x_target[1] = 0   
    x_target[2] = 1        
    x_target[6] = 0.9063
    x_target[7] = .4226

    # Initial State Constraint (Starting at rest)
    x_init = ca.DM.zeros(nx)
    x_init[6] = 1.0  # Unit quaternion (w = 1, no rotation)
    # Benchmark time
    start_time = time.time()
    x_opt, u_opt = solve(cost, x_target, x_init, N, dt)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


    save_dir = "results"
    # Saving the optimized state trajectory (x_opt) and control inputs (u_opt)
    x_ref = ca.repmat(x_target, 1, N + 1)
    x_ref_df = pd.DataFrame(x_ref.T)  
    x_opt_df = pd.DataFrame(x_opt.T)  
    u_opt_df = pd.DataFrame(u_opt.T)  

    os.makedirs(save_dir, exist_ok=True)
    run_id = len(glob.glob(os.path.join(save_dir, "x_opt_*.csv"))) + 1
    x_opt_df.to_csv(os.path.join(save_dir, f"x_opt_{run_id}.csv"), index=False)
    u_opt_df.to_csv(os.path.join(save_dir, f"u_opt_{run_id}.csv"), index=False)
    x_ref_df.to_csv(os.path.join(save_dir, f"x_ref_{run_id}.csv"), index=False)



if __name__ == "__main__":
    main()  
import casadi as ca
from dynamics import nx, nu  
from optimization_problem import solve_optimal_control
import os
import pandas as pd


# Solver Settings

# Target State (Hover at Z = 2 m, 45 deg roll) 
x_target = ca.DM.zeros(nx)
x_target[2] = 10  # Target Z position (hover)
x_target_R = ca.DM.eye(3)
x_target[6:15] = ca.reshape(x_target_R.T,( 9, 1))

# Initial State Constraint (Starting at rest)
x_init = ca.DM.zeros(nx)
x_int_R = ca.DM.eye(3)
x_init[6:15] = ca.reshape(x_int_R.T,( 9, 1))
u_init = ca.DM.zeros(nu)
u_init[0] = .8
u_init[1] = .8 
u_init[2] = .8
u_init[3] = .8

x,u,t = solve_optimal_control(x_init, u_init, x_target)
print(x)


print(u)

print(t)

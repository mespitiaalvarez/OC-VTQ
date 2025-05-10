import casadi as ca
from dynamics import f_dyn, nx, nu  

# Fixed step Runge-Kutta 4 integrator
# Plus quaternion normalization
X0 = ca.MX.sym('x', nx)
U0 = ca.MX.sym('u', nu)
DT = ca.MX.sym('dt')
XF = X0
k1 = f_dyn(XF, U0)
k2 = f_dyn(XF + DT / 2 * k1, U0)
k3 = f_dyn(XF + DT / 2 * k2, U0)
k4 = f_dyn(XF + DT * k3, U0)
XF = XF + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
Q = XF[6:10]
Q_NORM = ca.norm_2(Q)
Q_NORMALIZED = Q / Q_NORM
XF[6:10] = Q_NORMALIZED
F_rk4 = ca.Function('F', [X0, U0, DT], [XF], ['x0', 'u0','dt'], ['xf'])


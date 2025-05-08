import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Quaternions are using scalar first notation, not JPL (scalar last)
# notation, which is what is used in reference notes


def quat_product(q_1,q_2):
  w_1,x_1,y_1,z_1 = np.squeeze(q_1)
  w_2,x_2,y_2,z_2 = np.squeeze(q_2)

  q_c = np.array([[w_1*w_2 - x_1*x_2 - y_1*y_2 - z_1*z_2],
                 [w_1*x_2 + x_1*w_2 + y_1*z_2 - z_1*y_2],
                 [w_1*y_2 - x_1*z_2 + y_1*w_2 + z_1*x_2],
                 [w_1*z_2 + x_1*y_2 - y_1*x_2 + z_1*w_2]])
  return q_c


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def vec(q):
  return q[1:4,:]

def R(q):
  a,b,c,d = np.squeeze(q)
  R = np.array([[a**2 + b**2 - c**2 - d**2, 2*(b*c-a*d), 2*(a*c + b*d)],
                [2*(b*c + a*d), a**2 - b**2 + c**2 -d**2, 2*(c*d - a*b)],
                [2*(b*d - c*a), 2*(a*b + c*d), a**2 -b**2 -c**2 +d**2]])
  return R

x = np.array([[1,0,0]]).T
A = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
x_a = A@x
q_yaw = np.array([[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]]).T
q_yaw_conj = quat_conjugate(q_yaw)
R_q = R(q_yaw)



x_rot = vec(quat_product(quat_product(q_yaw,x_a),q_yaw_conj))

print(x_rot)
print(R_q@x)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original vector in red
ax.quiver(0, 0, 0, x[0,0], x[1,0], x[2,0], color='r', label='Original Vector')

# Rotated vector in blue
ax.quiver(0, 0, 0, x_rot[0,0], x_rot[1,0], x_rot[2,0], color='b', label='Rotated Vector')

# Setup plot
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quaternion Rotation Visualization')
ax.legend()
plt.tight_layout()
plt.show()




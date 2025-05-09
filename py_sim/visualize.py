import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
from mpl_toolkits.mplot3d import Axes3D
# from integrator import dt_cvodes
import numpy as np
use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
dt = 0.01


# === Load NMPC Results ===
x_opt_df = pd.read_csv("nmpc_results/x_opt.csv")
u_opt_df = pd.read_csv("nmpc_results/u_opt.csv")

# Convert to numpy arrays
x_opt = x_opt_df.values.T  # Transpose back to match original format (nx, N+1)
u_opt = u_opt_df.values.T  # Transpose back to match original format (nu, N)

# === Visualization ===

time = np.linspace(0, x_opt.shape[1] * dt, x_opt.shape[1])  # Adjust for your dt

# === 3D Trajectory Visualization ===
x_traj = x_opt[0, :]  # X positions from NMPC
y_traj = x_opt[1, :]  # Y positions from NMPC
z_traj = x_opt[2, :]  # Z positions from NMPC

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_traj, y_traj, z_traj, label='Quadcopter 3D Trajectory', color='blue')
ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='red', label='Final Position')
ax.set_title("Quadcopter NMPC 3D Trajectory")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.legend()

# === Control Effort Visualization ===
plt.figure()
for i in range(4):
    plt.plot(time[:-1], u_opt[i, :], label=f'Motor {i+1} Speed')
plt.title("Quadcopter NMPC Control Effort (Motor Speeds)")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (0 to 1)")
plt.legend()
plt.grid(True)

# Tilt and Roll Rates
plt.figure()
for i in range(4, 8):
    plt.plot(time[:-1], u_opt[i, :], label=f'Tilt Rate {i-3}')
plt.title("Quadcopter NMPC Control Tilt Rates")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (-1 to 1)")
plt.legend()
plt.grid(True)

plt.figure()
for i in range(8, 12):
    plt.plot(time[:-1], u_opt[i, :], label=f'Roll Rate {i-7}')
plt.title("Quadcopter NMPC Control Roll Rates")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (-1 to 1)")
plt.legend()
plt.grid(True)


# Quadcopter Parameters
l = 0.3  # Arm length (adjust as needed)
N = x_opt.shape[1]

# Define Rotation Matrix Function
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Converts a quaternion to a 3x3 rotation matrix."""
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)
    return R

# Motor Yaw Angles (45° + i * 90°)
psi = [np.pi / 4 + i * np.pi / 2 for i in range(4)]

# Initialize 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Animation Loop (Real-Time)
for t in range(N):
    ax.clear()
    ax.set_title("Quadcopter Pose and Motor Positions (45° Rotated)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")

    # Extract Position and Orientation at time t
    x_pos, y_pos, z_pos = x_opt[0, t], x_opt[1, t], x_opt[2, t]
    qw, qx, qy, qz = x_opt[6, t], x_opt[7, t], x_opt[8, t], x_opt[9, t]
    
    # Calculate Rotation Matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    # Motor Positions in Local (Body) Frame (45° rotated)
    motors_local = np.array([
        [l * np.cos(psi[0]), l * np.sin(psi[0]), 0],  # Motor 1
        [l * np.cos(psi[1]), l * np.sin(psi[1]), 0],  # Motor 2
        [l * np.cos(psi[2]), l * np.sin(psi[2]), 0],  # Motor 3
        [l * np.cos(psi[3]), l * np.sin(psi[3]), 0]   # Motor 4
    ])

    # Rotate Motors to World Frame
    motors_world = (R @ motors_local.T).T  # Rotate each motor
    motors_world[:, 0] += x_pos  # Translate to current position
    motors_world[:, 1] += y_pos
    motors_world[:, 2] += z_pos

    # Plot Quadcopter Body (Cross)
    for i in range(4):
        ax.plot(
            [x_pos, motors_world[i, 0]], 
            [y_pos, motors_world[i, 1]], 
            [z_pos, motors_world[i, 2]], 
            color='black', linestyle='--'
        )

    # Plot Motors and Thrust Directions
    for i in range(4):
        ax.scatter(*motors_world[i], color='red', label=f'Motor {i+1}' if t == 0 else "")
        # Thrust Vector (Direction)
        thrust_direction = R @ np.array([0, 0, 0.15])  # Thrust direction in Z-axis
        ax.quiver(
            motors_world[i, 0], motors_world[i, 1], motors_world[i, 2],  # Motor Position
            thrust_direction[0], thrust_direction[1], thrust_direction[2],  # Thrust Vector
            color='blue', length=0.15, normalize=True, arrow_length_ratio=0.2
        )

    # Plot Drone Position (Body)
    ax.scatter(x_pos, y_pos, z_pos, color='green', s=100, label="Quadcopter Body" if t == 0 else "")

    # Set Limits (Zoom)
    ax.set_xlim(x_pos - 1, x_pos + 1)
    ax.set_ylim(y_pos - 1, y_pos + 1)
    ax.set_zlim(z_pos - 0.2, z_pos + 0.5)

    plt.pause(1)  # Small delay for animation
    
plt.show()



# # === Plot Quivers (Body Axes) along the 3D Trajectory ===
# skip = 10  # Skip every 10 points for clarity
# scale_factor = 0.5  # Reduce the scaling factor (try different values)

# for t in range(0, x_opt.shape[1], skip):
#     # Extract Position and Orientation
#     x_pos, y_pos, z_pos = x_opt[0, t], x_opt[1, t], x_opt[2, t]
#     qw, qx, qy, qz = x_opt[6, t], x_opt[7, t], x_opt[8, t], x_opt[9, t]
#     # Calculate Rotation Matrix
#     R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

#     # Body Axes Vectors (X, Y, Z in body frame)
#     body_x = R @ np.array([1, 0, 0]) * scale_factor  # Reduced scaling
#     body_y = R @ np.array([0, 1, 0]) * scale_factor  # Reduced scaling
#     body_z = R @ np.array([0, 0, 1]) * scale_factor  # Reduced scaling

#     # Plot Quivers (Body Axes)
#     ax.quiver(
#         x_pos, y_pos, z_pos,  # Start position of the quiver (quadcopter position)
#         body_x[0], body_x[1], body_x[2],  # Direction of X-axis (Red)
#         color='red', label='X-axis (Body)' if t == 0 else ""
#     )
#     ax.quiver(
#         x_pos, y_pos, z_pos,  # Start position
#         body_y[0], body_y[1], body_y[2],  # Direction of Y-axis (Green)
#         color='green', label='Y-axis (Body)' if t == 0 else ""
#     )
#     ax.quiver(
#         x_pos, y_pos, z_pos,  # Start position
#         body_z[0], body_z[1], body_z[2],  # Direction of Z-axis (Blue)
#         color='blue', label='Z-axis (Body)' if t == 0 else ""
#     )
# # Labels and Limits
# ax.set_title("Quadcopter NMPC 3D Trajectory with Correct Body Axes")
# ax.set_xlim([-2, 2])
# ax.set_ylim([-2, 2])
# ax.set_zlim([-2, 2])

# ax.set_xlabel("X Position (m)")
# ax.set_ylabel("Y Position (m)")
# ax.set_zlabel("Z Position (m)")
# ax.legend()
# plt.show()

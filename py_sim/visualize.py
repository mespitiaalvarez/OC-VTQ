import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from optimal_control import dt


def main():
    # === Load NMPC Results === #
    use('TkAgg')  # Was using Linux Subsystem
    number = "3"

    str_xopt = "results/x_opt_" + number + ".csv"
    str_uopt = "results/u_opt_" + number + ".csv"
    str_xref = "results/x_ref_" + number + ".csv"

    x_opt_df = pd.read_csv(str_xopt)
    u_opt_df = pd.read_csv(str_uopt)
    x_ref_df = pd.read_csv(str_xref)

    # Convert to numpy arrays
    x_opt = x_opt_df.values.T  
    u_opt = u_opt_df.values.T  
    x_ref = x_ref_df.values.T  # Transpose back to match original format (nx, N+1)

    time = np.linspace(0, x_opt.shape[1] * dt, x_opt.shape[1])  

    # === 3D Path Visualization === #
    x_traj = x_opt[0, :]  # X 
    y_traj = x_opt[1, :]  # Y 
    z_traj = x_opt[2, :]  # Z 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_traj, y_traj, z_traj, label='Quadcopter 3D Path', color='blue')
    ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='red', label='Final Position')
    ax.set_title("Quadcopter NMPC 3D Path")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    ax.legend()



    # === Control Effort Visualization === #
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    # === Motor Speeds (Thrusts) ===
    for i in range(4):
        axs[0].plot(time[:-1], u_opt[i, :], label=f'Motor {i+1} Speed')
    axs[0].set_title("Quadcopter NMPC Control Effort (Normalized Thrust)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Control Input (0 to 1)")
    axs[0].legend()
    axs[0].grid(True)
    
    # === Pitch Rates ===
    for i in range(4, 8):
        axs[1].plot(time[:-1], u_opt[i, :], label=f'Pitch Rate {i-3}')
    axs[1].set_title("Quadcopter NMPC Control Pitch Rates")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Control Input (-1 to 1)")
    axs[1].legend()
    axs[1].grid(True)

    # === Roll Rates ===
    for i in range(8, 12):
        axs[2].plot(time[:-1], u_opt[i, :], label=f'Roll Rate {i-7}')
    axs[2].set_title("Quadcopter NMPC Control Roll Rates")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Control Input (-1 to 1)")
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout()


    # === Error Visualization === #
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # === RMSE Error ===
    rmse = np.sqrt(np.mean((x_opt - x_ref) ** 2, axis=0))
    axs[0].plot(time, rmse, label="RMSE Error", color='green')
    axs[0].set_title("Error: RMSE")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("RMSE")
    axs[0].legend()
    axs[0].grid(True)
    
    # === Quaternion Geodesic Error ===
    def quaternion_geodesic_error(quaternion, quaternion_ref):
        dot_product = np.sum(quaternion * quaternion_ref, axis=0)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return 2 * np.arccos(np.abs(dot_product))
    
    # Extracting Quaternions (qw, qx, qy, qz)
    quaternion = x_opt[6:10, :]
    quaternion_ref = x_ref[6:10, :]
    geo_error = quaternion_geodesic_error(quaternion, quaternion_ref)

    axs[1].plot(time, geo_error, label="Quaternion Geodesic Error", color='red')
    axs[1].set_title("Error: Quaternion Geodesic")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Geodesic Error (rad)")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()

    # === Altitude (Z) Plot === #
    plt.figure()
    plt.plot(time, x_opt[2, :], label="Altitude (Z)", color='blue')
    plt.title("Quadcopter Altitude Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.grid(True)

    # === Quadcopter Animation === #
    # Quadcopter Parameters
    l = 0.3  # Arm length (adjust as needed)
    N = x_opt.shape[1]

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

    # Motor Yaw Angles 
    psi = [np.pi / 4 + i * np.pi / 2 for i in range(4)]

    # Initialize 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Animation Loop (Real-Time)
    for t in range(N):
        ax.clear()
        ax.set_title("Quadcopter Pose and Motor Positions")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")

        x_pos, y_pos, z_pos = x_opt[0, t], x_opt[1, t], x_opt[2, t]
        qw, qx, qy, qz = x_opt[6, t], x_opt[7, t], x_opt[8, t], x_opt[9, t]
        
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

        # Motor Positions in Body Frame 
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

        # Plot Quadcopter Body 
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
            thrust_direction = R @ np.array([0, 0, 0.15])  # Thrust direction in Z-axis
            ax.quiver(
                motors_world[i, 0], motors_world[i, 1], motors_world[i, 2],  # Motor Position
                thrust_direction[0], thrust_direction[1], thrust_direction[2],  # Thrust Vector
                color='blue', length=0.15, normalize=True, arrow_length_ratio=0.2
            )

        # Plot Drone Position (Body)
        ax.scatter(x_pos, y_pos, z_pos, color='green', s=100, label="Quadcopter Body" if t == 0 else "")

        # Set Moving   Limits 
        moving_lim = True
        if moving_lim:
            ax.set_xlim(x_pos - 1.5, x_pos + 1.5)
            ax.set_ylim(y_pos - 1.5, y_pos + 1.5)
            ax.set_zlim(z_pos - 0.2, z_pos + 0.5)

        else:
            ax.set_xlim(-5,5)
            ax.set_ylim(-5,5)
            ax.set_zlim(-5,5)

        plt.pause(dt)  # Small delay for animation
        
    plt.show()

if __name__ == "__main__":
    main()

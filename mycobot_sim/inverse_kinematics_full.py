import mujoco
import numpy as np
import math
import imageio

###############################################################################
# 1) DH-based FK, Jacobian, and IK
###############################################################################
def dh_transform(a, alpha, d, theta):
    """Compute the Denavit-Hartenberg transformation matrix.
    Uses a, alpha, d, theta order to match forward_kinematics_debug.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct,       -st*ca,     st*sa,      a*ct],
        [st,        ct*ca,    -ct*sa,      a*st],
        [0,         sa,        ca,         d   ],
        [0,         0,         0,          1   ]
    ], dtype=float)

def convert_joint_angles_sim_to_mycobot(joint_angles):
    """Convert joint angles simulator to mycobot.
    This function is not used in the updated implementation, but kept for reference.
    Args:
        joint_angles ([float]): [joint angles(radian)]
    Returns:
        [float]: [joint angles calculated(radian)]
    """
    conv_mul = [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    conv_add = [0.0, -math.pi / 2, 0.0, -math.pi / 2, math.pi / 2, 0.0]

    joint_angles = [joint_angles[i] * conv_mul[i] for i in range(len(joint_angles))]
    joint_angles = [joint_angles[i] + conv_add[i] for i in range(len(joint_angles))]

    joint_angles_lim = []
    for joint_angle in joint_angles:
        while joint_angle > math.pi:
            joint_angle -= 2 * math.pi

        while joint_angle < -math.pi:
            joint_angle += 2 * math.pi

        joint_angles_lim.append(joint_angle)

    return joint_angles_lim

def forward_kinematics(thetas):
    """
    Compute forward kinematics using MyCobot DH parameters.
    thetas: [theta1, theta2, theta3, theta4, theta5, theta6].
    Returns the position [x, y, z] and orientation [alpha, beta, gamma].
    """
    # Apply joint angle transformations as in forward_kinematics_debug
    t1 = thetas[0] 
    t2 = thetas[1] - np.pi/2
    t3 = thetas[2]
    t4 = thetas[3] - np.pi/2
    t5 = thetas[4] + np.pi/2
    t6 = thetas[5]
    
    # DH parameters (a, alpha, d, theta) as used in forward_kinematics_debug
    dh_params = [
        (0,       np.pi/2, 0.15708, t1),
        (-0.1104, 0,       0,       t2),
        (-0.096,  0,       0,       t3),
        (0,       np.pi/2, 0.06639, t4),
        (0,      -np.pi/2, 0.07318, t5),
        (0,       0,       0.0456,  t6)
    ]
    
    # Calculate transformation matrix
    T = np.eye(4)
    for (a, alpha, d, theta) in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)
    
    # Extract position
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    # Calculate Euler angles
    alpha, beta, gamma = euler_angle_from_matrix(T)
    
    return np.array([x, y, z, alpha, beta, gamma])

def euler_angle_from_matrix(T):
    """Calculate Euler angles (ZYZ) from transformation matrix."""
    alpha = math.atan2(T[1, 2], T[0, 2])
    if not (-math.pi/2 <= alpha <= math.pi/2):
        alpha = math.atan2(T[1, 2], T[0, 2]) + math.pi
    if not (-math.pi/2 <= alpha <= math.pi/2):
        alpha = math.atan2(T[1, 2], T[0, 2]) - math.pi
    
    beta = math.atan2(
        T[0, 2] * math.cos(alpha) + T[1, 2] * math.sin(alpha),
        T[2, 2]
    )
    
    gamma = math.atan2(
        -T[0, 0] * math.sin(alpha) + T[1, 0] * math.cos(alpha),
        -T[0, 1] * math.sin(alpha) + T[1, 1] * math.cos(alpha)
    )
    
    return alpha, beta, gamma

def basic_jacobian(thetas):
    """
    Compute the full (6x6) Jacobian matrix for the manipulator.
    Returns the Jacobian matrix mapping joint velocities to end-effector velocities.
    """
    # Calculate end-effector position
    ee_pos = forward_kinematics(thetas)[:3]
    
    # Apply joint angle transformations as in forward_kinematics_debug
    t1 = thetas[0] 
    t2 = thetas[1] - np.pi/2
    t3 = thetas[2]
    t4 = thetas[3] - np.pi/2
    t5 = thetas[4] + np.pi/2
    t6 = thetas[5]
    
    # DH parameters (a, alpha, d, theta) as used in forward_kinematics_debug
    dh_params = [
        (0,       np.pi/2, 0.15708, t1),
        (-0.1104, 0,       0,       t2),
        (-0.096,  0,       0,       t3),
        (0,       np.pi/2, 0.06639, t4),
        (0,      -np.pi/2, 0.07318, t5),
        (0,       0,       0.0456,  t6)
    ]
    
    # Calculate transformation matrices up to each joint
    T = np.eye(4)
    trans_matrices = [T.copy()]  # T_0 for i=0
    
    for (a, alpha, d, theta) in dh_params:
        T_step = dh_transform(a, alpha, d, theta)
        T = T @ T_step
        trans_matrices.append(T.copy())
    
    # Calculate basic Jacobian
    jacobian = np.zeros((6, 6))
    
    for i in range(6):
        # Position part of Jacobian (linear velocity)
        # Get position and z-axis of previous frame
        T_prev = trans_matrices[i]
        pos_prev = T_prev[:3, 3]
        z_axis_prev = T_prev[:3, 2]
        
        # Linear velocity component
        jacobian[:3, i] = np.cross(z_axis_prev, ee_pos - pos_prev)
        
        # Angular velocity component
        jacobian[3:, i] = z_axis_prev
    
    return jacobian

def inverse_kinematics(thetas_init, target_pose, max_iterations=500, step_size=0.01):
    """
    Perform inverse kinematics to find joint angles for a target pose.
    target_pose: [x, y, z, alpha, beta, gamma]
    Returns: array of joint angles
    """
    thetas = thetas_init.copy()
    
    for iteration in range(max_iterations):
        # Get current pose
        current_pose = forward_kinematics(thetas)
        
        # Calculate pose error
        pose_error = target_pose - current_pose
        
        # Get current Euler angles
        alpha, beta, gamma = current_pose[3:]
        
        K_zyz = np.array([
            [0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
            [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
            [1, 0, math.cos(beta)]
        ])
        
        # Construct the full K_alpha matrix
        K_alpha = np.eye(6)
        K_alpha[3:, 3:] = K_zyz
        
        # Get Jacobian and update joint angles
        J = basic_jacobian(thetas)
        J_pinv = np.linalg.pinv(J)
        
        # Calculate joint velocity
        theta_dot = np.dot(np.dot(J_pinv, K_alpha), pose_error)
        
        # Update joint angles
        thetas += step_size * theta_dot
        
        # Normalize angles to [-pi, pi]
        thetas = np.array([((angle + math.pi) % (2 * math.pi)) - math.pi for angle in thetas])
        
        # Check for convergence
        if np.linalg.norm(pose_error) < 1e-5:
            break
    
    return thetas

###############################################################################
# 2) Helper Functions
###############################################################################
def interpolate_linear(start_pos, end_pos, n_steps=20):
    alphas = np.linspace(0.0, 1.0, n_steps)
    return [(1 - a)*start_pos + a*end_pos for a in alphas]

def interpolate_pose(start_pose, end_pose, n_steps=20):
    """
    Interpolate between start and end poses linearly.
    Both position and orientation (Euler angles) are interpolated.
    """
    alphas = np.linspace(0.0, 1.0, n_steps)
    return [(1 - a)*start_pose + a*end_pose for a in alphas]

def set_joint_angles(data, thetas):
    for i in range(len(thetas)):
        data.qpos[i] = thetas[i]

###############################################################################
# 3) Main Demo
###############################################################################
def main():
    model_path = "config/xml/mycobot_280jn_mujoco.xml"  # <-- Your robot XML
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1088
    render_width, render_height =  1920, 1088
    renderer = mujoco.Renderer(model, render_height, render_width)
    renderer.enable_shadows = True

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")

    # Initial conditions
    thetas_init = np.zeros(6, dtype=float)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)

    # Get initial pose (position and orientation)
    initial_pose = forward_kinematics(thetas_init)
    
    # Define target position
    x_target = np.array([ 0.05826, -0.2752,  0.1566 ])
    
    # Define target orientation - rotate 90 degrees around Z axis
    # This creates a new target orientation different from the initial one
    orientation_target = np.array([initial_pose[3] + np.pi/2, initial_pose[4], initial_pose[5]])
    
    # Combine target position and orientation into a full target pose
    target_pose = np.concatenate([x_target, orientation_target])
    
    # Number of steps for interpolation
    n_steps = 100
    
    # Generate waypoints that interpolate both position and orientation
    waypoints = interpolate_pose(initial_pose, target_pose, n_steps)

    frames = []
    current_thetas = thetas_init.copy()

    # Lists for logs
    joint_angle_log = []
    ee_pos_log = []
    ee_orient_log = []

    print("\nStarting motion along a trajectory with changing position and orientation.")
    for i, wp in enumerate(waypoints):
        current_thetas = inverse_kinematics(current_thetas, wp)
        set_joint_angles(data, current_thetas)
        mujoco.mj_forward(model, data)

        mujoco_ee_pos = data.xpos[ee_body_id]
        our_ee_pose = forward_kinematics(current_thetas)
        our_ee_pos = our_ee_pose[:3]
        our_ee_orient = our_ee_pose[3:]
        
        pos_diff = np.linalg.norm(mujoco_ee_pos - our_ee_pos)

        # Record logs
        joint_angle_log.append(current_thetas.copy())
        ee_pos_log.append(our_ee_pos.copy())
        ee_orient_log.append(our_ee_orient.copy())

        renderer.update_scene(data)
        frames.append(renderer.render())

        # Calculate errors to target
        pos_err_to_final = np.linalg.norm(target_pose[:3] - our_ee_pos)
        orient_err_to_final = np.linalg.norm(target_pose[3:] - our_ee_orient)
        
        print(f"Step {i+1}/{n_steps} | "
              f"Pos err: {pos_err_to_final:.5f}, "
              f"Orient err: {orient_err_to_final:.5f}, "
              f"MuJoCo-Our diff: {pos_diff:.5f}")

    print("Saving video to 'robot_pose_trajectory.mp4'...")
    imageio.mimsave("robot_pose_trajectory.mp4", frames, fps=60, quality=10, bitrate="32M")
    print("Video saved.")

    # Save logs to NPZ
    np.savez("robot_pose_logs.npz",
             joint_angles=np.array(joint_angle_log),
             ee_positions=np.array(ee_pos_log),
             ee_orientations=np.array(ee_orient_log))
    print("Logs saved to 'robot_pose_logs.npz'.")

    renderer.close()

if __name__ == "__main__":
    main()
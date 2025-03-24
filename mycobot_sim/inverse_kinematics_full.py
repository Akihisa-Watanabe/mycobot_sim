import mujoco
import numpy as np
import math
import imageio
import time

###############################################################################
# 1) Angle Conversions and DH Transformations
###############################################################################
def dh_transform(a, alpha, d, theta):
    """Compute the Denavit-Hartenberg transformation matrix.
    
    Args:
        a: Link length
        alpha: Link twist
        d: Link offset
        theta: Joint angle
        
    Returns:
        4x4 homogeneous transformation matrix
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

def euler_to_quaternion(euler):
    """Convert ZYZ Euler angles to quaternion [w, x, y, z].
    
    Args:
        euler: Euler angles [alpha, beta, gamma] (ZYZ convention)
        
    Returns:
        Quaternion [w, x, y, z]
    """
    alpha, beta, gamma = euler
    
    # Half angles
    ha = alpha / 2.0
    hb = beta / 2.0
    hg = gamma / 2.0
    
    # Pre-compute sines and cosines
    ca = np.cos(ha)
    sa = np.sin(ha)
    cb = np.cos(hb)
    sb = np.sin(hb)
    cg = np.cos(hg)
    sg = np.sin(hg)
    
    # Compute quaternion components
    w = ca*cb*cg - sa*cb*sg
    x = ca*sb*cg + sa*sb*sg
    y = sa*sb*cg - ca*sb*sg
    z = ca*cb*sg + sa*cb*cg
    
    return normalize_quaternion(np.array([w, x, y, z]))

def quaternion_to_euler(quat):
    """Convert quaternion to ZYZ Euler angles.
    
    Args:
        quat: Quaternion [w, x, y, z]
        
    Returns:
        Euler angles [alpha, beta, gamma] (ZYZ convention)
    """
    quat = normalize_quaternion(quat)
    w, x, y, z = quat
    
    # Compute Euler angles
    # atan2(2(wx+yz), 1-2(x²+y²)) -> first angle alpha
    # acos(1-2(x²+z²)) -> second angle beta
    # atan2(2(wz+xy), 1-2(y²+z²)) -> third angle gamma
    
    alpha = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    # Handle numerical issues for beta
    beta_val = 2*(w*y - x*z)
    beta_val = np.clip(beta_val, -1.0, 1.0)
    beta = np.arcsin(beta_val)
    
    gamma = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    
    # Keep angles in [-pi, pi] range
    alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi
    beta = ((beta + np.pi) % (2 * np.pi)) - np.pi
    gamma = ((gamma + np.pi) % (2 * np.pi)) - np.pi
    
    return np.array([alpha, beta, gamma])

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [w, x, y, z].
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    return normalize_quaternion(np.array([w, x, y, z]))

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    quat = normalize_quaternion(quat)
    w, x, y, z = quat
    
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Return the conjugate of a quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def normalize_quaternion(q):
    """Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default to identity quaternion
    return q / norm

###############################################################################
# 2) Forward Kinematics
###############################################################################
def forward_kinematics(thetas, return_quaternion=False):
    """
    Compute forward kinematics using MyCobot DH parameters.
    
    Args:
        thetas: Joint angles [theta1, theta2, theta3, theta4, theta5, theta6]
        return_quaternion: If True, returns quaternion instead of Euler angles
        
    Returns:
        If return_quaternion is True:
            End-effector pose [x, y, z, qw, qx, qy, qz]
        Otherwise:
            End-effector pose [x, y, z, alpha, beta, gamma]
    """
    # Apply joint angle transformations
    t1 = thetas[0] 
    t2 = thetas[1] - np.pi/2
    t3 = thetas[2]
    t4 = thetas[3] - np.pi/2
    t5 = thetas[4] + np.pi/2
    t6 = thetas[5]
    
    # DH parameters (a, alpha, d, theta)
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
    position = T[:3, 3]
    
    # Extract orientation
    rotation_matrix = T[:3, :3]
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    
    if return_quaternion:
        # Return position and quaternion
        return np.concatenate([position, quaternion])
    else:
        # Convert quaternion to Euler angles
        euler = quaternion_to_euler(quaternion)
        # Return position and Euler angles
        return np.concatenate([position, euler])

###############################################################################
# 3) Jacobian and Inverse Kinematics
###############################################################################
def basic_jacobian(thetas):
    """
    Compute the geometric Jacobian matrix for the manipulator.
    
    Args:
        thetas: Joint angles [theta1, theta2, theta3, theta4, theta5, theta6]
        
    Returns:
        6x6 Jacobian matrix [J_v; J_w]
    """
    # Calculate end-effector pose with quaternion
    ee_pose = forward_kinematics(thetas, return_quaternion=True)
    ee_pos = ee_pose[:3]
    
    # Apply joint angle transformations
    t1 = thetas[0] 
    t2 = thetas[1] - np.pi/2
    t3 = thetas[2]
    t4 = thetas[3] - np.pi/2
    t5 = thetas[4] + np.pi/2
    t6 = thetas[5]
    
    # DH parameters (a, alpha, d, theta)
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
    
    # Calculate Jacobian
    jacobian = np.zeros((6, 6))
    
    for i in range(6):
        # Get position and z-axis of previous frame
        T_prev = trans_matrices[i]
        pos_prev = T_prev[:3, 3]
        z_axis_prev = T_prev[:3, 2]  # z-axis of the joint frame
        
        # Linear velocity component (J_v)
        jacobian[:3, i] = np.cross(z_axis_prev, ee_pos - pos_prev)
        
        # Angular velocity component (J_w)
        jacobian[3:, i] = z_axis_prev
    
    return jacobian

def quaternion_error(q1, q2):
    """Calculate quaternion error for control (q2 - q1).
    
    Args:
        q1: Current quaternion [w, x, y, z]
        q2: Target quaternion [w, x, y, z]
        
    Returns:
        3D angular velocity error vector
    """
    # Normalize quaternions
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    
    # Calculate quaternion error qe = q2 * q1*
    q1_conj = quaternion_conjugate(q1)
    qe = quaternion_multiply(q2, q1_conj)
    
    # Check if the rotation is the long way around
    if qe[0] < 0:
        qe = -qe  # Take the shorter path
    
    # Extract the error vector (imaginary part)
    # For small rotations, the vector part of the quaternion
    # is approximately half the angular velocity vector
    error_vector = 2.0 * qe[1:4]
    
    return error_vector

def inverse_kinematics_step(thetas, target_pose, max_angvel=1.0, target_is_quaternion=False):
    """
    Perform a single step of inverse kinematics calculation.
    
    Args:
        thetas: Current joint angles
        target_pose: Target pose [x, y, z, ...] with orientation as either:
                     - Euler angles [alpha, beta, gamma] (if target_is_quaternion=False)
                     - Quaternion [qw, qx, qy, qz] (if target_is_quaternion=True)
        max_angvel: Maximum allowed joint angular velocity
        target_is_quaternion: Whether target_pose contains quaternion (True) or Euler angles (False)
        
    Returns:
        Joint velocities (dq)
    """
    # Get current pose with quaternion
    current_pose = forward_kinematics(thetas, return_quaternion=True)
    
    # Split position and orientation
    current_pos = current_pose[:3]
    current_quat = current_pose[3:7]
    
    target_pos = target_pose[:3]
    
    # Convert target orientation to quaternion if needed
    if target_is_quaternion:
        target_quat = target_pose[3:7]
    else:
        # Target pose contains Euler angles
        target_euler = target_pose[3:6]
        target_quat = euler_to_quaternion(target_euler)
    
    # Calculate position error
    pos_error = target_pos - current_pos
    
    # Calculate orientation error (as angular velocity)
    orient_error = quaternion_error(current_quat, target_quat)
    
    # Combine position and orientation errors
    error = np.concatenate([pos_error, orient_error])
    
    # Get Jacobian
    J = basic_jacobian(thetas)
    
    # Calculate joint velocities using pseudoinverse (without damping)
    # Simple pseudoinverse: J⁺ = J^T(JJ^T)^(-1)
    J_pinv = J.T @ np.linalg.inv(J @ J.T) 
    # check if singular value are too small
    if np.linalg.cond(J) > 1/np.finfo(J.dtype).eps:
        print("Singular values are too small")
    dq = J_pinv @ error
    
    # Limit maximum joint velocity if specified
    if max_angvel > 0:
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max
    
    return dq

def set_joint_angles(data, thetas):
    """Set joint angles in the MuJoCo simulation.
    
    Args:
        data: MuJoCo simulation data
        thetas: Joint angles to set
    """
    for i in range(len(thetas)):
        data.qpos[i] = thetas[i]

###############################################################################
# 4) Main Demo
###############################################################################
def main():
    """Main function to demonstrate robot control with FK and IK using Euler angles."""
    try:
        # Load the MuJoCo model
        model_path = "config/xml/mycobot_280jn_mujoco.xml"
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # Configure rendering
        model.vis.global_.offwidth = 1920
        model.vis.global_.offheight = 1088
        render_width, render_height = 1920, 1088
        renderer = mujoco.Renderer(model, render_height, render_width)
        renderer.enable_shadows = True
        
        # Get the end-effector body ID
        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
        
        # Set initial joint angles + random perturbation
        thetas_init = np.zeros(6, dtype=float) + np.random.uniform(-0.1, 0.1, size=6)
        set_joint_angles(data, thetas_init)
        mujoco.mj_forward(model, data)
        
        # Get initial pose with Euler angles
        initial_pose = forward_kinematics(thetas_init, return_quaternion=False)
        
        # Define target position
        x_target = np.array([0.04926, -0.2852, 0.0566])
        # Define target orientation using Euler angles
        # Rotate 90 degrees around Z axis from initial orientation
        target_euler = initial_pose[3:6].copy()
        target_euler[0] += np.pi/2  # Add 90 degrees to alpha (first angle, Z rotation)
        
        # Combine target position and orientation into a full target pose with Euler angles
        target_pose = np.concatenate([x_target, target_euler])
        
        # Simulation settings
        n_steps = 500          # Total steps
        dt = 0.02       # Simulation timestep (s)
        integration_dt = 0.02   # Integration timestep (s)
        max_angvel = 1.0       # Maximum joint angular velocity (rad/s)
        
        # Logs for recording simulation data
        frames = []
        joint_angle_log = []
        ee_pos_log = []
        ee_euler_log = []
        
        print("\nStarting motion to target pose using simple IK control with Euler angles.")
        print(f"Initial pose: Position {initial_pose[:3]}, Euler angles {initial_pose[3:6]} (rad)")
        print(f"Target pose: Position {target_pose[:3]}, Euler angles {target_pose[3:6]} (rad)")
        
        current_thetas = thetas_init.copy()
        
        for i in range(n_steps):
            step_start = time.time()
            
            # Calculate one step of inverse kinematics (using Euler angles as target)
            dq = inverse_kinematics_step(current_thetas, target_pose, max_angvel, target_is_quaternion=False)
            
            # Integrate joint velocities to update joint angles
            q = current_thetas.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            
            # Update current joint angles
            current_thetas = q
            
            # Apply joint angles to the simulation
            set_joint_angles(data, current_thetas)
            mujoco.mj_forward(model, data)
            
            # Get current end-effector position from MuJoCo
            mujoco_ee_pos = data.xpos[ee_body_id]
            
            # Get current end-effector pose from our FK calculation (with Euler angles)
            our_ee_pose = forward_kinematics(current_thetas, return_quaternion=False)
            our_ee_pos = our_ee_pose[:3]
            our_ee_euler = our_ee_pose[3:6]
            
            # Calculate error between MuJoCo and our FK
            pos_diff = np.linalg.norm(mujoco_ee_pos - our_ee_pos)
            
            # Calculate error to target pose
            pos_err = np.linalg.norm(target_pose[:3] - our_ee_pos)
            
            # Calculate orientation error using quaternions internally
            our_ee_quat = euler_to_quaternion(our_ee_euler)
            target_quat = euler_to_quaternion(target_pose[3:6])
            orient_err = np.linalg.norm(quaternion_error(our_ee_quat, target_quat))
            
            # Record logs
            joint_angle_log.append(current_thetas.copy())
            ee_pos_log.append(our_ee_pos.copy())
            ee_euler_log.append(our_ee_euler.copy())
            
            # Render the scene
            renderer.update_scene(data)
            frames.append(renderer.render())
            
            # Display progress
            print(f"Step {i+1}/{n_steps} | "
                  f"Pos err: {pos_err:.5f}, "
                  f"Orient err: {orient_err:.5f}, "
                  f"MuJoCo-Our diff: {pos_diff:.5f}")
            
            # Adjust timing to maintain real-time simulation
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        print("Saving video to 'robot_euler_ik.mp4'...")
        imageio.mimsave("robot_euler_ik.mp4", frames, fps=60, quality=10, bitrate="32M")
        print("Video saved.")
        
        # Save logs
        np.savez("robot_euler_ik_logs.npz",
                 joint_angles=np.array(joint_angle_log),
                 ee_positions=np.array(ee_pos_log),
                 ee_euler_angles=np.array(ee_euler_log))
        print("Logs saved to 'robot_euler_ik_logs.npz'.")
        
        renderer.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
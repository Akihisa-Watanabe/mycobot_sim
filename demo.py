import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import os

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
    J_pinv = np.linalg.pinv(J)
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

def create_robot_model_with_mocap():
    """Create and save a MuJoCo model with a mocap body for mouse control.
    
    Returns:
        Path to the created model file
    """
    # Get original model path
    original_model_path = "config/xml/mycobot_280jn_mujoco.xml"
    
    # Create a modified version with a mocap body
    mocap_model_path = "config/xml/mycobot_280jn_mujoco_with_mocap.xml"
    
    # Check if the modified model already exists
    if os.path.exists(mocap_model_path):
        print(f"Using existing model with mocap: {mocap_model_path}")
        return mocap_model_path
    
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(mocap_model_path), exist_ok=True)
        
        # Load the original model
        model = mujoco.MjModel.from_xml_path(original_model_path)
        
        # Get the end-effector position for mocap body placement
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
        ee_pos = data.xpos[ee_body_id].copy()
        
        # Create a modified XML with a mocap body
        xml_content = f"""
<mujoco model="mycobot_with_mocap">
    <include file="{original_model_path}"/>
    
    <worldbody>
        <body name="target" mocap="true" pos="{ee_pos[0]} {ee_pos[1]} {ee_pos[2]}">
            <geom type="sphere" size="0.02" rgba="1 0 0 0.5"/>
        </body>
    </worldbody>
</mujoco>
"""
        # Write the modified XML
        with open(mocap_model_path, "w") as f:
            f.write(xml_content)
        
        print(f"Created model with mocap body: {mocap_model_path}")
        return mocap_model_path
    
    except Exception as e:
        print(f"Error creating model with mocap: {e}")
        print("Using original model instead")
        return original_model_path

###############################################################################
# 4) Main Demo with Keyboard Control
###############################################################################
def main():
    """Main function to demonstrate robot control with keyboard control."""
    try:
        # Load the MuJoCo model - first try to create a version with mocap, fallback to original
        model_path = "config/xml/mycobot_280jn_mujoco.xml"
        has_mocap = False
        
        try:
            # Look for an existing mocap body in the model
            model = mujoco.MjModel.from_xml_path(model_path)
            for i in range(model.nbody):
                if model.body_mocapid[i] >= 0:
                    has_mocap = True
                    break
            
            # If no mocap body exists, try to create one
            if not has_mocap:
                modified_model_path = model_path.replace(".xml", "_with_mocap.xml")
                
                # Create a simple XML with the mocap body included
                xml_content = f"""
<mujoco>
    <include file="{model_path}"/>
    <worldbody>
        <body name="target" mocap="true" pos="0.1 0 0.3">
            <geom type="sphere" size="0.02" rgba="1 0 0 0.5"/>
        </body>
    </worldbody>
</mujoco>
"""
                with open(modified_model_path, "w") as f:
                    f.write(xml_content)
                
                model_path = modified_model_path
                has_mocap = True
                print(f"Created temporary model with mocap: {model_path}")
        except:
            print("Could not add mocap body, using key controls instead")
        
        # Load the model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # Simulation settings
        dt = 0.002             # Simulation timestep (s)
        integration_dt = 0.1   # Integration timestep (s)
        max_angvel = 1.0       # Maximum joint angular velocity (rad/s)
        
        # Set initial joint angles
        thetas_init = np.zeros(6, dtype=float)
        set_joint_angles(data, thetas_init)
        mujoco.mj_forward(model, data)
        
        # Get the end-effector body ID
        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
        
        # Find mocap body if exists
        mocap_id = -1
        mocap_body_name = ""
        for i in range(model.nbody):
            if model.body_mocapid[i] >= 0:
                mocap_id = model.body_mocapid[i]
                mocap_body_name = model.body(i).name
                break
        
        has_mocap = mocap_id >= 0
        if has_mocap:
            print(f"Found mocap body '{mocap_body_name}' (ID: {mocap_id})")
        else:
            print("No mocap body found, using keyboard control")
        
        # Launch the passive viewer
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
            # Get initial pose with Euler angles
            initial_pose = forward_kinematics(thetas_init, return_quaternion=False)
            print(f"Initial pose: Position {initial_pose[:3]}, Euler angles {initial_pose[3:6]} (rad)")
            
            # Initialize current joint angles
            current_thetas = thetas_init.copy()
            
            # Initialize target pose with initial position and orientation
            target_pose = np.zeros(6)
            target_pose[:3] = initial_pose[:3].copy()
            target_pose[3:6] = initial_pose[3:6].copy()  # Keep initial orientation
            
            # Set mocap body to initial position if available
            if has_mocap:
                data.mocap_pos[mocap_id] = target_pose[:3]
                print("Click and drag the red sphere to move the target")
            else:
                print("Use keyboard to move the target:")
                print("  - Up/Down/Left/Right/PgUp/PgDn: Move target")
                print("  - Hold Shift: Move faster")
            
            print("Press ESC to exit")
            
            # Target movement speed (units per second)
            move_speed = 0.1
            
            # Start the simulation loop
            while viewer.is_running():
                step_start = time.time()
                
                # Handle target position control
                if has_mocap:
                    # Use mocap body position as target
                    target_pose[:3] = data.mocap_pos[mocap_id]
                else:
                    # Use keyboard control
                    key_step = move_speed * dt
                    
                    # Handle key presses for target movement
                    # Note: This is a simplified approach since direct key state access
                    # is not available in the Python bindings
                    
                    # Instead, we'll just move the target in a pattern to demonstrate
                    t = data.time
                    target_pose[0] = initial_pose[0] + 0.1 * np.sin(t * 0.5)
                    target_pose[1] = initial_pose[1] + 0.1 * np.cos(t * 0.5)
                    target_pose[2] = initial_pose[2] + 0.05 * np.sin(t * 0.3)
                
                # Calculate one step of inverse kinematics (using Euler angles as target)
                dq = inverse_kinematics_step(current_thetas, target_pose, max_angvel, target_is_quaternion=False)
                
                # Integrate joint velocities to update joint angles
                q = current_thetas.copy()
                mujoco.mj_integratePos(model, q, dq, integration_dt)
                
                # Update current joint angles
                current_thetas = q
                
                # Apply joint angles to the simulation
                set_joint_angles(data, current_thetas)
                mujoco.mj_step(model, data)
                
                # Get current end-effector pose
                our_ee_pose = forward_kinematics(current_thetas, return_quaternion=False)
                our_ee_pos = our_ee_pose[:3]
                our_ee_euler = our_ee_pose[3:6]
                
                # Get MuJoCo's end-effector position
                mujoco_ee_pos = data.xpos[ee_body_id]
                
                # Calculate errors
                pos_err = np.linalg.norm(target_pose[:3] - our_ee_pos)
                pos_diff = np.linalg.norm(mujoco_ee_pos - our_ee_pos)
                
                # Calculate orientation error using quaternions internally
                our_ee_quat = euler_to_quaternion(our_ee_euler)
                target_quat = euler_to_quaternion(target_pose[3:6])
                orient_err = np.linalg.norm(quaternion_error(our_ee_quat, target_quat))
                
                # Update the viewer
                viewer.sync()
                
                # Print status every 100 steps
                if int(data.time / dt) % 100 == 0:
                    print(f"Time: {data.time:.2f}s | "
                          f"Target pos: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}] | "
                          f"Pos err: {pos_err:.5f}")
                
                # Adjust timing to maintain real-time simulation
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("Simulation ended.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



# DHパラメータの求め方．せ




# 図\ref{fig:mycobot_dh_parameter_coordinate_setting}に示すとおり，
# MyCobotは
# 6自由度の回転関節（Revolute関節）から構成され，リンク$i$に対応する関節角を $\theta_i$ とする．
# ロボットアームの幾何構造を定式化する際には，「隣接するリンク同士がどのように並んでいるか」
# および「関節軸がどの方向を向いているか」を明示し，
# 連鎖的な座標変換を定義する必要がある．

# このとき，座標系の付け方に一定の規則を与えて各リンク間の相対変位を表すと，
# \textbf{Denavit–Hartenberg (DH) パラメータ} を用いて変換を簡潔にまとめることができる．
# 一般に，3次元空間で二つの座標系間を任意に変換するには6つのパラメータが必要だが，
# ロボットマニピュレータの場合には単一軸の回転あるいは平行移動が連結しているだけなので，
# 4種類のパラメータ
# \[
#  (a_i,\, \alpha_i,\, d_i,\, \theta_i)
# \]
# によってリンク$i-1$からリンク$i$への変換が表される．

# DHパラメータの定義を厳密に行うために，以下の手順で座標系を配置する:
# \begin{enumerate}
#     \item 各関節の回転軸を $z_i$ 軸とする．
#     \item 隣接する$z$軸を結ぶ最短距離線を$x_i$軸とし，$y_i$軸は右手系に従って決める．
#     \item リンク長$a_i$やねじれ角$\alpha_i$，関節オフセット$d_i$，関節角$\theta_i$をDHパラメータとして割り当てる．
# \end{enumerate}
# 具体的にMyCobotについては，表\ref{tab:dh_params}にパラメータを示す．
# これにしたがってリンク$i-1$座標系からリンク$i$座標系への同次変換行列は
# \begin{equation}
# {}^{i-1}\!T_{i}(\theta_i)
# = 
# \begin{pmatrix}
#  \cos\theta_i & -\sin\theta_i \cos\alpha_{i-1} &  \sin\theta_i \sin\alpha_{i-1} & a_{i-1} \cos\theta_i \\
#  \sin\theta_i &  \cos\theta_i \cos\alpha_{i-1} & -\cos\theta_i \sin\alpha_{i-1} & a_{i-1} \sin\theta_i \\
#  0            &  \sin\alpha_{i-1}             &  \cos\alpha_{i-1}             & d_i \\
#  0            &  0                             &  0                             & 1
# \end{pmatrix}
# \label{eq:DHtransform}
# \end{equation}
# で与えられる．

# \begin{table}[H]
# \centering
# \caption{PUMA型ロボット（MyCobot）のDHパラメータ}
# \label{tab:dh_params}
# \begin{tabular}{c|cccc}
# \hline
# リンク $i$ & $a_{i-1}$ [mm] & $\alpha_{i-1}$ [rad] & $d_i$ [mm] & $\theta_i$ [rad] \\
# \hline
# 1 & 0 & $\frac{\pi}{2}$ & 131.56 & $\theta_1$ \\
# 2 & -110.4 & 0 & 0 & $\theta_2$ \\
# 3 & -96.0 & 0 & 0 & $\theta_3$ \\
# 4 & 0 & $\frac{\pi}{2}$ & 66.39 & $\theta_4$ \\
# 5 & 0 & $-\frac{\pi}{2}$ & 73.18 & $\theta_5$ \\
# 6 & 0 & 0 & 0 & $\theta_6$ \\
# \hline
# \end{tabular}
# \end{table}

# \begin{figure}[H]
#     \centering
#     \includegraphics[width=0.8\linewidth]{figs/dh_parameter.pdf}
#     \caption{MyCobotのDHパラメータ座標系の設定}
#     \label{fig:mycobot_dh_parameter_coordinate_setting}
# \end{figure}

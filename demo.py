#!/usr/bin/env python3
"""
Robot arm demo visualizing target position and orientation
Moves the robot arm to user-specified position and orientation, visualizing the process.
Maintains the original FK/IK implementation with improved Euler angle handling.
"""

import mujoco
import numpy as np
import math
import imageio
import argparse
import os
import sys
from pathlib import Path
import re

###############################################################################
# 1) DH-based FK, Jacobian, and IK - Maintaining original implementation
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

def unwrap_angles(angles, reference=None):
    """
    Unwrap angles to ensure continuous motion without jumps.
    When reference is provided, unwraps angles to be nearest to reference values.
    
    Args:
        angles: Array of angles to unwrap
        reference: Optional reference angles for consistent unwrapping
    
    Returns:
        Unwrapped angles with minimized discontinuities
    """
    unwrapped = np.array(angles)
    
    if reference is not None:
        # Unwrap relative to reference angles
        for i in range(len(unwrapped)):
            # Calculate difference and ensure it's in [-pi, pi]
            diff = unwrapped[i] - reference[i]
            diff = ((diff + math.pi) % (2 * math.pi)) - math.pi
            
            # Apply the adjusted difference
            unwrapped[i] = reference[i] + diff
    else:
        # Ensure all angles are in [-pi, pi] range
        for i in range(len(unwrapped)):
            unwrapped[i] = ((unwrapped[i] + math.pi) % (2 * math.pi)) - math.pi
            
    return unwrapped

def euler_angle_from_matrix(T):
    """
    Calculate Euler angles (ZYZ) from transformation matrix with improved handling
    to prevent jumps and singularities.
    """
    # Extract the rotation matrix part
    R = T[:3, :3]
    
    # Calculate beta with proper handling (range [0, pi])
    beta = math.atan2(
        math.sqrt(R[0, 2]**2 + R[1, 2]**2),
        R[2, 2]
    )
    
    # Handle special cases to avoid gimbal lock issues
    if math.isclose(beta, 0, abs_tol=1e-10) or math.isclose(beta, math.pi, abs_tol=1e-10):
        # Gimbal lock case - alpha and gamma are coupled
        # Set alpha to 0 and calculate appropriate gamma
        alpha = 0
        if math.isclose(beta, 0, abs_tol=1e-10):
            gamma = math.atan2(R[1, 0], R[0, 0])
        else:  # beta ≈ pi
            gamma = -math.atan2(R[1, 0], R[0, 0])
    else:
        # Normal case - calculate alpha and gamma based on beta
        alpha = math.atan2(R[1, 2]/math.sin(beta), R[0, 2]/math.sin(beta))
        gamma = math.atan2(R[2, 1]/math.sin(beta), -R[2, 0]/math.sin(beta))
    
    # Ensure all angles are in [-pi, pi] range
    alpha = ((alpha + math.pi) % (2 * math.pi)) - math.pi
    beta = ((beta + math.pi) % (2 * math.pi)) - math.pi
    gamma = ((gamma + math.pi) % (2 * math.pi)) - math.pi
    
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
    Enhanced to prevent Euler angle jumps during convergence.
    
    Args:
        thetas_init: Initial joint angles
        target_pose: [x, y, z, alpha, beta, gamma]
        max_iterations: Maximum iterations for convergence
        step_size: Step size for joint angle updates
    
    Returns: 
        Array of joint angles
    """
    thetas = thetas_init.copy()
    
    # Ensure target Euler angles are normalized
    target_pose_norm = target_pose.copy()
    target_pose_norm[3:] = unwrap_angles(target_pose[3:])
    
    # Previous pose for tracking orientation changes
    prev_pose = None
    
    for iteration in range(max_iterations):
        # Get current pose
        current_pose = forward_kinematics(thetas)
        
        # On first iteration, use current pose as reference
        if prev_pose is None:
            prev_pose = current_pose.copy()
        
        # Unwrap current Euler angles relative to previous pose to ensure consistency
        current_pose_unwrapped = current_pose.copy()
        current_pose_unwrapped[3:] = unwrap_angles(current_pose[3:], prev_pose[3:])
        prev_pose = current_pose_unwrapped.copy()
        
        # Calculate pose error
        pose_error = target_pose_norm - current_pose_unwrapped
        
        # For orientation error, ensure shortest path
        pose_error[3:] = unwrap_angles(pose_error[3:])
        
        # Get Euler angles from the unwrapped current pose
        alpha, beta, gamma = current_pose_unwrapped[3:]
        
        # Compute K_zyz matrix which maps angular velocities to Euler angle rates
        # Avoid singularities by adding small epsilon to sin(beta)
        sin_beta = math.sin(beta)
        if abs(sin_beta) < 1e-6:
            sin_beta = 1e-6 if sin_beta >= 0 else -1e-6
            
        K_zyz = np.array([
            [0, -math.sin(alpha), math.cos(alpha) * sin_beta],
            [0, math.cos(alpha), math.sin(alpha) * sin_beta],
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
        
        # Adaptive step size to improve convergence
        adaptive_step = step_size / (1 + 0.3 * np.linalg.norm(pose_error))
        
        # Update joint angles
        thetas += adaptive_step * theta_dot
        
        # Normalize angles to [-pi, pi]
        thetas = unwrap_angles(thetas)
        
        # Check for convergence
        if np.linalg.norm(pose_error) < 1e-5:
            break
    
    return thetas

###############################################################################
# 2) Helper Functions - Modified for better Euler angle handling
###############################################################################
def interpolate_linear(start_pos, end_pos, n_steps=20):
    alphas = np.linspace(0.0, 1.0, n_steps)
    return [(1 - a)*start_pos + a*end_pos for a in alphas]

def interpolate_pose(start_pose, end_pose, n_steps=20):
    """
    Interpolate between start and end poses with improved Euler angle handling.
    Uses angle unwrapping to prevent discontinuities during interpolation.
    
    Args:
        start_pose: Starting pose [x, y, z, alpha, beta, gamma]
        end_pose: Target pose [x, y, z, alpha, beta, gamma]
        n_steps: Number of interpolation steps
        
    Returns:
        List of interpolated poses
    """
    # Extract positions and orientations
    start_pos, start_orient = start_pose[:3], start_pose[3:]
    end_pos, end_orient = end_pose[:3], end_pose[3:]
    
    # Linear interpolation for positions
    alphas = np.linspace(0.0, 1.0, n_steps)
    interp_pos = [(1 - a) * start_pos + a * end_pos for a in alphas]
    
    # Unwrap end orientation to be consistent with start orientation
    # This ensures the shortest path when interpolating between angles
    unwrapped_end = unwrap_angles(end_orient, start_orient)
    
    # Interpolate angles with unwrapped values to prevent jumps
    interp_orient = []
    for a in alphas:
        # Linear interpolation between start and unwrapped end orientation
        orient = (1 - a) * start_orient + a * unwrapped_end
        interp_orient.append(orient)
    
    # Combine interpolated positions and orientations
    return [np.concatenate([pos, orient]) for pos, orient in zip(interp_pos, interp_orient)]

def set_joint_angles(data, thetas):
    for i in range(len(thetas)):
        data.qpos[i] = thetas[i]

def calculate_workspace_limits():
    """
    Calculate approximate workspace limits of the robot based on link lengths.
    Returns min_reach and max_reach.
    """
    # Extract link lengths from DH parameters
    # Values from the DH parameters in the code
    link_lengths = [0.15708, 0.1104, 0.096, 0.06639, 0.07318, 0.0456]
    
    # Calculate maximum reach by summing all link lengths
    # This is a theoretical maximum - the actual reach will be less due to joint limits
    max_reach = sum(link_lengths) * 0.9  # 90% of theoretical maximum
    
    # Minimum reach - approximation based on robot geometry
    min_reach = 0.12  # Simplified minimum reach (robot base radius + safety margin)
    
    return min_reach, max_reach

def is_point_reachable(target_pos, target_orientation, thetas_init=None):
    """
    Check if a target position and orientation is reachable.
    
    Args:
        target_pos: Target position [x, y, z]
        target_orientation: Target orientation [alpha, beta, gamma]
        thetas_init: Initial joint angles to start the IK from
        
    Returns:
        (reachable, thetas): A tuple containing a boolean indicating if the target is reachable
                             and the joint angles if reachable
    """
    if thetas_init is None:
        thetas_init = np.zeros(6)
    
    target_pose = np.concatenate([target_pos, target_orientation])
    
    # Get rough workspace limits based on link lengths
    min_reach, max_reach = calculate_workspace_limits()
    
    # Rough check: Is the target within a reasonable distance?
    distance = np.linalg.norm(target_pos)
    if distance < min_reach * 0.8 or distance > max_reach * 1.1:
        return False, None
    
    # Try to find an IK solution
    try:
        thetas = inverse_kinematics(thetas_init, target_pose, max_iterations=200, step_size=0.01)
        
        # Verify the solution by computing forward kinematics
        achieved_pose = forward_kinematics(thetas)
        
        # Calculate errors
        pos_error = np.linalg.norm(achieved_pose[:3] - target_pos)
        orient_error = np.linalg.norm(unwrap_angles(achieved_pose[3:], target_orientation) - target_orientation)
        
        # Check if the solution is accurate enough
        if pos_error < 0.01 and orient_error < 0.1:
            return True, thetas
        else:
            return False, None
    except Exception:
        return False, None

def find_closest_reachable_target(target_pos, target_orientation, num_samples=10):
    """
    Find the closest reachable target to the specified target position and orientation.
    Uses a sampling-based approach to find a valid target.
    
    Args:
        target_pos: Target position [x, y, z]
        target_orientation: Target orientation [alpha, beta, gamma]
        num_samples: Number of samples to try when searching
        
    Returns:
        (closest_pos, closest_orientation): The closest reachable position and orientation
    """
    # Get the direction vector from origin to target
    distance = np.linalg.norm(target_pos)
    if distance < 1e-6:
        # If target is at origin, choose a default direction
        direction = np.array([0, 0, 1])
    else:
        direction = target_pos / distance
    
    # Get rough workspace limits
    min_reach, max_reach = calculate_workspace_limits()
    
    # If target is too far, scale it back
    if distance > max_reach:
        # Try points at different distances along the same direction
        for scale in np.linspace(0.9, 0.5, num_samples):
            test_pos = direction * (max_reach * scale)
            reachable, _ = is_point_reachable(test_pos, target_orientation)
            if reachable:
                return test_pos, target_orientation
        
        # If still not found, try with different orientations
        for scale in np.linspace(0.9, 0.5, 5):
            test_pos = direction * (max_reach * scale)
            # Try different orientations (simplistic approach - just rotate around z)
            for angle in np.linspace(0, np.pi, 5):
                test_orientation = np.array([angle, 0, 0])
                reachable, _ = is_point_reachable(test_pos, test_orientation)
                if reachable:
                    return test_pos, test_orientation
    
    # If target is too close, move it outward
    elif distance < min_reach:
        for scale in np.linspace(1.2, 2.0, num_samples):
            test_pos = direction * (min_reach * scale)
            reachable, _ = is_point_reachable(test_pos, target_orientation)
            if reachable:
                return test_pos, target_orientation
        
        # If still not found, try with different orientations
        for scale in np.linspace(1.2, 2.0, 5):
            test_pos = direction * (min_reach * scale)
            for angle in np.linspace(0, np.pi, 5):
                test_orientation = np.array([angle, 0, 0])
                reachable, _ = is_point_reachable(test_pos, test_orientation)
                if reachable:
                    return test_pos, test_orientation
    
    # If distance is within bounds but still unreachable, try scaling back slightly
    else:
        # Try points at slightly different distances along the same direction
        for scale in np.linspace(0.95, 0.5, num_samples):
            test_pos = direction * (distance * scale)
            reachable, _ = is_point_reachable(test_pos, target_orientation)
            if reachable:
                return test_pos, target_orientation
        
        # If still not found, try with different orientations
        for scale in np.linspace(0.95, 0.5, 5):
            test_pos = direction * (distance * scale)
            for angle in np.linspace(0, np.pi, 5):
                test_orientation = np.array([angle, 0, 0])
                reachable, _ = is_point_reachable(test_pos, test_orientation)
                if reachable:
                    return test_pos, test_orientation
    
    # Fallback - return a safe position and orientation that's likely reachable
    fallback_pos = direction * min_reach * 1.5  # 50% beyond minimum reach
    fallback_orientation = np.zeros(3)  # Default orientation
    
    return fallback_pos, fallback_orientation

def validate_and_adjust_target(target_pos, target_orientation):
    """
    Validate if the target is reachable, and if not, adjust it to the closest reachable position.
    
    Args:
        target_pos: Target position [x, y, z]
        target_orientation: Target orientation [alpha, beta, gamma]
        
    Returns:
        (adjusted_pos, adjusted_orientation, reachable): 
            Adjusted position, orientation and a flag indicating if the original target was reachable
    """
    # Check if the target is reachable
    reachable, _ = is_point_reachable(target_pos, target_orientation)
    
    if reachable:
        return target_pos, target_orientation, True
    else:
        # Find the closest reachable target
        closest_pos, closest_orientation = find_closest_reachable_target(
            target_pos, target_orientation)
        
        return closest_pos, closest_orientation, False
###############################################################################
# 3) Target Visualization Functions
###############################################################################
def euler_to_rotation_matrix(alpha, beta, gamma):
    """
    Calculate rotation matrix from ZYZ Euler angles
    """
    # Calculate sine and cosine for each angle
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    # Calculate ZYZ rotation matrix
    R = np.array([
        [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
        [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
        [-sb*cg, sb*sg, cb]
    ])
    
    return R

def axis_to_quat(vec1, vec2):
    """
    Calculate quaternion representing rotation between two unit vectors
    Shows rotation from vec1 to vec2
    """
    # Normalize vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Calculate dot product and cross product
    dot_product = np.dot(vec1, vec2)
    cross_product = np.cross(vec1, vec2)
    
    # Special case: vectors are parallel (same or opposite direction)
    if np.isclose(dot_product, 1.0):
        return "1 0 0 0"  # Identity quaternion (no rotation)
    elif np.isclose(dot_product, -1.0):
        # 180-degree rotation - choose any perpendicular axis
        perp = np.array([1, 0, 0]) if not np.allclose(vec1, [1, 0, 0]) else np.array([0, 1, 0])
        perp = perp - vec1 * np.dot(perp, vec1)
        perp = perp / np.linalg.norm(perp)
        return f"0 {perp[0]} {perp[1]} {perp[2]}"
        
    # Calculate rotation angle
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Calculate rotation axis (normalized cross product)
    axis = cross_product / np.linalg.norm(cross_product)
    
    # Calculate quaternion components
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    
    return f"{w} {xyz[0]} {xyz[1]} {xyz[2]}"

def remove_existing_target(model_xml):
    """
    Remove existing target site definitions from the model XML
    """
    # Regex pattern to find target sites
    site_pattern = r'<site\s+name="target[^"]*"[^>]*>'
    
    # Remove existing target sites
    model_xml = re.sub(site_pattern, '<!-- Removed target site -->', model_xml)
    return model_xml

def add_target_visualization(model_xml, target_pos, target_orientation):
    """
    Add elements to visualize target position and orientation to the model XML
    """
    # Remove any existing target sites
    model_xml = remove_existing_target(model_xml)
    
    # Find the </worldbody> tag
    worldbody_end = model_xml.find("</worldbody>")
    if worldbody_end == -1:
        print("Warning: Could not find </worldbody> tag in the XML.")
        return model_xml
    
    # Calculate rotation matrix from Euler angles
    R = euler_to_rotation_matrix(*target_orientation)
    
    # Add target position sphere and XYZ axis markers
    target_sites = f"""
    <!-- Target position marker (red sphere) -->
    <site name="target_pos" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" size="0.01 0.01 0.01" type="sphere" rgba="1 0 0 0.7"/>
    
    <!-- Target orientation markers (RGB = XYZ axes) -->
    <site name="target_x_axis" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" 
          size="0.001 0.03 0.001" type="cylinder" rgba="1 0 0 1" 
          quat="{axis_to_quat(np.array([0, 0, 1]), R[:, 0])}"/>
    
    <site name="target_y_axis" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" 
          size="0.001 0.03 0.001" type="cylinder" rgba="0 1 0 1" 
          quat="{axis_to_quat(np.array([0, 0, 1]), R[:, 1])}"/>
    
    <site name="target_z_axis" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" 
          size="0.001 0.03 0.001" type="cylinder" rgba="0 0 1 1" 
          quat="{axis_to_quat(np.array([0, 0, 1]), R[:, 2])}"/>
    
    <!-- Line connecting target position and current position -->
    <site name="target_line" pos="0 0 0" size="0.001 0.001 0.001" type="cylinder" rgba="1 1 0 0.5"/>
    """
    
    # Insert before the closing worldbody tag
    xml_with_target = model_xml[:worldbody_end] + target_sites + model_xml[worldbody_end:]
    return xml_with_target

def update_target_line(model, data, ee_pos, target_pos):
    """
    Update the line connecting end effector to target
    """
    # Get site ID
    line_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_line")
    if line_id == -1:
        return  # Site not found, do nothing
    
    # Calculate midpoint and direction
    midpoint = (ee_pos + target_pos) / 2
    direction = target_pos - ee_pos
    length = np.linalg.norm(direction)
    
    if length < 1e-6:
        return  # Distance too small, don't update
    
    # Set line position
    model.site_pos[line_id] = midpoint
    
    # Set line size - MuJoCo API needs three values
    model.site_size[line_id] = np.array([0.001, length/2, 0.001])
    
    # Set line direction (convert to unit vector)
    direction = direction / length
    
    # Calculate rotation from Z-axis to target direction
    quat_str = axis_to_quat(np.array([0, 0, 1]), direction)
    quat_vals = np.array([float(x) for x in quat_str.split()])
    model.site_quat[line_id] = quat_vals

###############################################################################
# 4) Main Demo Function
###############################################################################
def run_robot_demo(target_pos, target_orientation, output_file="robot_pose_trajectory.mp4"):
    """
    Run robot arm demo visualizing target position and orientation
    """
    # Remove temporary files
    if os.path.exists("temp_model_with_target.xml"):
        try:
            os.remove("temp_model_with_target.xml")
            print("Removed previous temporary file")
        except Exception as e:
            print(f"Error removing temporary file: {e}")
    
    # Find XML file path
    try:
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "config" / "xml" / "mycobot_280jn_mujoco.xml"
        if not model_path.exists():
            print(f"XML file not found: {model_path}")
            
            # Also search in current directory
            model_path = Path("config/xml/mycobot_280jn_mujoco_demo.xml")
            if not model_path.exists():
                print(f"XML file not found: {model_path}")
                raise FileNotFoundError("Model XML file not found")
        
        model_path = str(model_path)
    except Exception as e:
        print(f"Error searching for model path: {e}")
        return False
    
    # Load robot model
    try:
        # Load XML as string
        with open(model_path, 'r') as f:
            model_xml = f.read()
        
        # Fix mesh directory path
        script_dir = Path(__file__).resolve().parent
        mesh_dir = script_dir / "config" / "meshes_mujoco"
        
        if not mesh_dir.exists():
            print(f"Mesh directory not found: {mesh_dir}")
            # Search in current directory
            mesh_dir = Path("config/meshes_mujoco")
            if not mesh_dir.exists():
                print(f"Mesh directory not found: {mesh_dir}")
                raise FileNotFoundError("Mesh directory not found")
        
        # Convert to absolute path
        mesh_dir = str(mesh_dir.absolute())
        
        # Replace meshdir attribute with correct path
        if 'meshdir=' in model_xml:
            start_idx = model_xml.find('meshdir=') + 9  # Length of 'meshdir="' is 9
            end_idx = model_xml.find('"', start_idx)
            old_meshdir = model_xml[start_idx:end_idx]
            
            # Replace with new path
            model_xml = model_xml.replace(f'meshdir="{old_meshdir}"', f'meshdir="{mesh_dir}"')
        
        # Add target visualization elements
        model_xml = add_target_visualization(model_xml, target_pos, target_orientation)
        
        # Save to temporary file
        temp_model_path = "temp_model_with_target.xml"
        with open(temp_model_path, 'w') as f:
            f.write(model_xml)
        
        # Load modified model
        model = mujoco.MjModel.from_xml_path(temp_model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error configuring model: {e}")
        return False
    
    # Configure renderer
    try:
        render_width, render_height = 1920, 1088
        model.vis.global_.offwidth = render_width
        model.vis.global_.offheight = render_height
        
        renderer = mujoco.Renderer(model, render_height, render_width)
        renderer.enable_shadows = True
    except Exception as e:
        print(f"Error configuring renderer: {e}")
        return False
    
    # Set initial conditions
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    
    thetas_init = np.zeros(6, dtype=float)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)
    
    # Get initial pose and set target pose
    initial_pose = forward_kinematics(thetas_init)
    target_pose = np.concatenate([target_pos, target_orientation])
    
    # Generate waypoints for smooth movement
    n_steps = 100
    waypoints = interpolate_pose(initial_pose, target_pose, n_steps)
    
    # Initialize frames and logs
    frames = []
    current_thetas = thetas_init.copy()
    
    # Initialize log data
    joint_angle_log = []
    ee_pos_log = []
    ee_orient_log = []
    
    print("\nMoving robot to target position and orientation")
    print(f"Target position: {target_pos}")
    print(f"Target orientation (ZYZ Euler angles): {target_orientation}")
    
    for i, wp in enumerate(waypoints):
        # Calculate joint angles using inverse kinematics
        current_thetas = inverse_kinematics(current_thetas, wp)
        
        # Set joint angles in simulation
        set_joint_angles(data, current_thetas)
        mujoco.mj_forward(model, data)
        
        # Get current end effector position and orientation
        mujoco_ee_pos = data.xpos[ee_body_id]
        our_ee_pose = forward_kinematics(current_thetas)
        our_ee_pos = our_ee_pose[:3]
        our_ee_orient = our_ee_pose[3:]
        
        # Update target line
        update_target_line(model, data, mujoco_ee_pos, target_pos)
        
        # Calculate errors
        pos_diff = np.linalg.norm(mujoco_ee_pos - our_ee_pos)
        pos_err_to_target = np.linalg.norm(target_pos - our_ee_pos)
        orient_err_to_target = np.linalg.norm(unwrap_angles(target_orientation, our_ee_orient) - our_ee_orient)
        
        # Record logs
        joint_angle_log.append(current_thetas.copy())
        ee_pos_log.append(our_ee_pos.copy())
        ee_orient_log.append(our_ee_orient.copy())
        
        # Display progress
        print(f"Step {i+1}/{n_steps} | "
              f"Position error: {pos_err_to_target:.5f}, "
              f"Orientation error: {orient_err_to_target:.5f}, "
              f"MuJoCo-FK difference: {pos_diff:.5f}")
        
        # Render and save frame
        renderer.update_scene(data)
        frames.append(renderer.render())
    
    # Save video
    try:
        print(f"Saving video: {output_file}")
        imageio.mimsave(output_file, frames, fps=30, quality=8, bitrate="16M")
        print(f"Video saved: {output_file}")
    except Exception as e:
        print(f"Error saving video: {e}")
        return False
    
    # Save log data
    log_file = os.path.splitext(output_file)[0] + "_log.npz"
    try:
        np.savez(log_file, 
                 target_position=target_pos,
                 target_orientation=target_orientation,
                 joint_angles=np.array(joint_angle_log),
                 ee_positions=np.array(ee_pos_log),
                 ee_orientations=np.array(ee_orient_log))
        print(f"Log saved: {log_file}")
    except Exception as e:
        print(f"Error saving log: {e}")
    
    # Cleanup
    renderer.close()
    if os.path.exists("temp_model_with_target.xml"):
        os.remove("temp_model_with_target.xml")
    
    return True

###############################################################################
# 5) Interactive Mode
###############################################################################
def run_interactive_mode(initial_target_pos, initial_target_orientation):
    """
    Run in interactive mode using MuJoCo viewer
    """
    print("Starting interactive mode")
    
    try:
        from mujoco import viewer
    except ImportError:
        print("Interactive mode requires mujoco.viewer")
        print("Make sure the latest MuJoCo package is installed")
        return False
    
    # Remove temporary files
    if os.path.exists("temp_model_with_target.xml"):
        try:
            os.remove("temp_model_with_target.xml")
        except Exception:
            pass
    
    # Find XML file path
    try:
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "config" / "xml" / "mycobot_280jn_mujoco.xml"
        if not model_path.exists():
            model_path = Path("config/xml/mycobot_280jn_mujoco_demo.xml")
        model_path = str(model_path)
    except Exception as e:
        print(f"Error searching for model path: {e}")
        return False
    
    # Load robot model
    try:
        with open(model_path, 'r') as f:
            model_xml = f.read()
        
        # Fix mesh directory path
        mesh_dir = Path("config/meshes_mujoco").absolute()
        
        if 'meshdir=' in model_xml:
            start_idx = model_xml.find('meshdir=') + 9
            end_idx = model_xml.find('"', start_idx)
            old_meshdir = model_xml[start_idx:end_idx]
            model_xml = model_xml.replace(f'meshdir="{old_meshdir}"', f'meshdir="{mesh_dir}"')
        
        # Add target visualization elements
        model_xml = add_target_visualization(model_xml, initial_target_pos, initial_target_orientation)
        
        # Save to temporary file
        temp_model_path = "temp_model_with_target.xml"
        with open(temp_model_path, 'w') as f:
            f.write(model_xml)
        
        # Load modified model
        model = mujoco.MjModel.from_xml_path(temp_model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error configuring model: {e}")
        return False
    
    # Set initial conditions
    thetas_init = np.zeros(6, dtype=float)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)
    
    # Move to initial target
    target_pose = np.concatenate([initial_target_pos, initial_target_orientation])
    current_thetas = inverse_kinematics(thetas_init, target_pose)
    set_joint_angles(data, current_thetas)
    mujoco.mj_forward(model, data)
    
    # Start viewer
    print("\nLaunching interactive viewer")
    print("Controls:")
    print("  Left mouse drag: Rotate view")
    print("  Right mouse drag: Move view")
    print("  Mouse wheel: Zoom")
    print("  Esc: Exit")
    
    viewer.launch(model, data)
    
    return True

###############################################################################
# 6) Main Function
###############################################################################
def main():
    """
    Main function to parse command line arguments and run the demo
    """
    parser = argparse.ArgumentParser(description='Robot arm demo - Visualizing target position and orientation')
    
    # Target position arguments
    parser.add_argument('--x', type=float, default=0.05826, help='Target position X (default: 0.06026)')
    parser.add_argument('--y', type=float, default=-0.2752, help='Target position Y (default: -0.2752)')
    parser.add_argument('--z', type=float, default=0.1566 , help='Target position Z (default: 0.0566)')
    
    # Target orientation arguments (in radians)
    parser.add_argument('--alpha', type=float, default=0.0, help='Target orientation Alpha - rotation around Z axis (default: 0.0)')
    parser.add_argument('--beta', type=float, default=0.0, help='Target orientation Beta - rotation around Y axis (default: 0.0)')
    parser.add_argument('--gamma', type=float, default=0.0, help='Target orientation Gamma - rotation around Z axis (default: 0.0)')
    
    # Output file
    parser.add_argument('--output', type=str, default="robot_pose_trajectory.mp4", help='Output video file (default: robot_pose_trajectory.mp4)')
    
    # Interactive mode flag
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode (default: False)')
    
    # Workspace checking flag
    parser.add_argument('--skip-check', action='store_true', help='Skip workspace checking (default: False)')
    
    args = parser.parse_args()
    
    # Create target position and orientation arrays
    original_target_pos = np.array([args.x, args.y, args.z])
    original_target_orientation = np.array([args.alpha, args.beta, args.gamma])
    
    # Validate and adjust the target if needed
    if not args.skip_check:
        print("\nChecking if target is within reachable workspace...")
        target_pos, target_orientation, was_reachable = validate_and_adjust_target(
            original_target_pos, original_target_orientation)
        
        # Print information about the target
        if not was_reachable:
            print("\nWarning: Original target is not reachable!")
            print(f"Original target position: {original_target_pos}")
            print(f"Original target orientation: {original_target_orientation}")
            print(f"Adjusted to closest reachable target position: {target_pos}")
            print(f"Adjusted target orientation: {target_orientation}")
        else:
            print("\nTarget is reachable.")
            print(f"Target position: {target_pos}")
            print(f"Target orientation: {target_orientation}")
    else:
        print("\nSkipping workspace checking as requested.")
        target_pos = original_target_pos
        target_orientation = original_target_orientation
    
    # Run in appropriate mode
    if args.interactive:
        success = run_interactive_mode(target_pos, target_orientation)
    else:
        success = run_robot_demo(target_pos, target_orientation, args.output)
    
    if success:
        print("Demo completed successfully")
    else:
        print("Error occurred during demo execution")

if __name__ == "__main__":
    main()


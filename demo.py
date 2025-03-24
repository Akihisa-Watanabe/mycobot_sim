#!/usr/bin/env python3
"""
Robot arm demo visualizing target position and orientation using diff_ik algorithm
Moves the robot arm to user-specified position and orientation, visualizing the process.
Uses quaternion as internal representation with Euler angle input from user.
"""

import mujoco
import numpy as np
import math
import imageio
import argparse
import time
import os
import re
from pathlib import Path

# Import functionality from diff_ik.py
from mycobot_sim.inverse_kinematics_full import (
    dh_transform, 
    euler_to_quaternion, 
    quaternion_to_euler, 
    rotation_matrix_to_quaternion, 
    quaternion_to_rotation_matrix, 
    quaternion_multiply, 
    quaternion_conjugate, 
    normalize_quaternion, 
    forward_kinematics, 
    basic_jacobian, 
    inverse_kinematics_step, 
    set_joint_angles,
    quaternion_error
)

###############################################################################
# Target Visualization Functions
###############################################################################
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
    
    Args:
        model_xml: Model XML string
        target_pos: Target position [x, y, z]
        target_orientation: Target orientation as Euler angles [alpha, beta, gamma]
        
    Returns:
        Modified model XML string
    """
    # Remove any existing target sites
    model_xml = remove_existing_target(model_xml)
    
    # Find the </worldbody> tag
    worldbody_end = model_xml.find("</worldbody>")
    if worldbody_end == -1:
        print("Warning: Could not find </worldbody> tag in the XML.")
        return model_xml
    
    # Convert Euler angles to rotation matrix
    target_quat = euler_to_quaternion(target_orientation)
    R = quaternion_to_rotation_matrix(target_quat)
    
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

def axis_to_quat(vec1, vec2):
    """
    Calculate quaternion representing rotation between two unit vectors
    Shows rotation from vec1 to vec2
    
    Args:
        vec1: First unit vector
        vec2: Second unit vector
        
    Returns:
        Quaternion as string "w x y z"
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

def update_target_line(model, data, ee_pos, target_pos):
    """
    Update the line connecting end effector to target
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        ee_pos: End effector position [x, y, z]
        target_pos: Target position [x, y, z]
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
# Main Demo Function
###############################################################################
def run_robot_demo(target_pos, target_orientation, output_file="robot_euler_ik.mp4"):
    """
    Run robot arm demo visualizing target position and orientation
    using the diff_ik algorithm with quaternion internal representation
    
    Args:
        target_pos: Target position [x, y, z]
        target_orientation: Target orientation as Euler angles [alpha, beta, gamma]
        output_file: Output video file path
        
    Returns:
        True if successful, False otherwise
    """
    # Remove temporary files
    temp_model_path = "temp_model_with_target.xml"
    if os.path.exists(temp_model_path):
        try:
            os.remove(temp_model_path)
            print("Removed previous temporary file")
        except Exception as e:
            print(f"Error removing temporary file: {e}")
    
    # Find XML file path
    try:
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "config" / "xml" / "mycobot_280jn_mujoco.xml"
        if not model_path.exists():
            # Also search in current directory
            model_path = Path("config/xml/mycobot_280jn_mujoco.xml")
            if not model_path.exists():
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
        
        # Fix mesh directory path if needed
        script_dir = Path(__file__).resolve().parent
        mesh_dir = script_dir / "config" / "meshes_mujoco"
        
        if not mesh_dir.exists():
            # Search in current directory
            mesh_dir = Path("config/meshes_mujoco")
            if not mesh_dir.exists():
                print(f"Warning: Mesh directory not found: {mesh_dir}")
        else:
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
        with open(temp_model_path, 'w') as f:
            f.write(model_xml)
        
        # Load modified model
        model = mujoco.MjModel.from_xml_path(temp_model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error configuring model: {e}")
        return False
    
    # Configure rendering
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
    
    # Set initial joint angles with small random perturbation like in diff_ik.py
    thetas_init = np.zeros(6, dtype=float) + np.random.uniform(-0.1, 0.1, size=6)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)
    
    # Get initial pose
    initial_pose = forward_kinematics(thetas_init, return_quaternion=False)
    
    # Define target position and orientation using Euler angles
    # This section follows diff_ik.py exactly
    # Define target orientation
    target_euler = initial_pose[3:6].copy()
    target_euler[0] += np.pi/2  # Add 90 degrees to alpha (first rotation around Z)
    
    # Use command line args if provided (otherwise use the default rotation)
    if not np.allclose(target_orientation, np.zeros(3)):
        target_euler = target_orientation
        
    # Combine target position and orientation into a full target pose
    target_pose = np.concatenate([target_pos, target_euler])
    
    # Simulation settings
    n_steps = 500          # Total steps
    
    # Simulation settings
    dt = 0.02                # Simulation timestep (s)
    integration_dt = 0.02    # Integration timestep (s)
    max_angvel = 1.0         # Maximum joint angular velocity (rad/s)
    
    # Initialize frames and logs
    frames = []
    joint_angle_log = []
    ee_pos_log = []
    ee_euler_log = []
    
    print("\nStarting motion to target pose using diff_ik algorithm.")
    print(f"Initial pose: Position {initial_pose[:3]}, Euler angles {initial_pose[3:6]} (rad)")
    print(f"Target pose: Position {target_pose[:3]}, Euler angles {target_pose[3:6]} (rad)")
    
    current_thetas = thetas_init.copy()
    
    for i in range(n_steps):
        step_start = time.time()
        
        # Calculate one step of inverse kinematics (using Euler angles as target)
        # Directly using target_pose - exactly like in diff_ik.py
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
        
        # Get current end-effector pose from our FK calculation
        our_ee_pose = forward_kinematics(current_thetas, return_quaternion=False)
        our_ee_pos = our_ee_pose[:3]
        our_ee_euler = our_ee_pose[3:6]
        
        # Update the target line visualization
        update_target_line(model, data, mujoco_ee_pos, target_pos)
        
        # Calculate errors
        pos_diff = np.linalg.norm(mujoco_ee_pos - our_ee_pos)
        pos_err = np.linalg.norm(target_pos - our_ee_pos)
        
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
    
    # Save video
    try:
        print(f"Saving video to '{output_file}'...")
        imageio.mimsave(output_file, frames, fps=60, quality=10, bitrate="32M")
        print("Video saved.")
    except Exception as e:
        print(f"Error saving video: {e}")
        return False
    
    # Save logs
    log_file = os.path.splitext(output_file)[0] + "_logs.npz"
    try:
        np.savez(log_file,
                 target_position=target_pos,
                 target_orientation=target_orientation,
                 joint_angles=np.array(joint_angle_log),
                 ee_positions=np.array(ee_pos_log),
                 ee_euler_angles=np.array(ee_euler_log))
        print(f"Logs saved to '{log_file}'.")
    except Exception as e:
        print(f"Error saving logs: {e}")
    
    # Cleanup
    renderer.close()
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    return True

###############################################################################
# Main Function
###############################################################################
def main():
    """
    Main function to parse command line arguments and run the demo
    """
    parser = argparse.ArgumentParser(description='Robot arm demo using diff_ik algorithm - Visualizing target position and orientation')
    
    # Target position arguments
    parser.add_argument('--x', type=float, default=0.05826, help='Target position X (default: 0.06026)')
    parser.add_argument('--y', type=float, default=-0.2752, help='Target position Y (default: -0.2752)')
    parser.add_argument('--z', type=float, default=0.1566 , help='Target position Z (default: 0.0566)')
    
    # Target orientation arguments (in radians)
    parser.add_argument('--alpha', type=float, default=np.pi/4, 
                        help='Target orientation Alpha - first Euler angle (default: pi/2)')
    parser.add_argument('--beta', type=float, default=0.0, 
                        help='Target orientation Beta - second Euler angle (default: 0.0)')
    parser.add_argument('--gamma', type=float, default=0.0, 
                        help='Target orientation Gamma - third Euler angle (default: 0.0)')
    
    # Output file
    parser.add_argument('--output', type=str, default="robot_quat_ik.mp4", 
                        help='Output video file (default: robot_quat_ik.mp4)')
    
    args = parser.parse_args()
    
    # Create target position and orientation arrays
    target_pos = np.array([args.x, args.y, args.z])
    target_orientation = np.array([args.alpha, args.beta, args.gamma])
    
    # Run the demo
    success = run_robot_demo(target_pos, target_orientation, args.output)
    
    if success:
        print("Demo completed successfully")
    else:
        print("Error occurred during demo execution")

if __name__ == "__main__":
    main()
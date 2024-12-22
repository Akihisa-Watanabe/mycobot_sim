"""
inverse_kinematics_debug.py

Usage:
    python inverse_kinematics_debug.py
"""

import sys
import os
import time

import numpy as np
import mediapy as media

# Enable faulthandler for more detailed crash info (optional)
import faulthandler
faulthandler.enable()

# MuJoCo imports
try:
    import mujoco
    from mujoco.glfw import glfw
    print("[DEBUG] Successfully imported mujoco and glfw.")
except Exception as e:
    print("[ERROR] Failed to import mujoco or glfw.", e)
    sys.exit(1)

##############################################################################
# 1) Simple Forward Kinematics (Using DH from your snippet)
##############################################################################
def dh_transform(a, alpha, d, theta):
    """Compute the Denavit-Hartenberg transformation matrix for given parameters."""
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

def forward_kinematics_debug(thetas):
    """
    Compute forward kinematics using DH parameters. 
    thetas: [theta1, theta2, theta3, theta4, theta5, theta6]
    Returns the 4x4 transformation matrix from base to end-effector.
    """
    t1 = thetas[0] 
    t2 = thetas[1] - np.pi/2
    t3 = thetas[2]
    t4 = thetas[3] - np.pi/2
    t5 = thetas[4] + np.pi/2
    t6 = thetas[5] 

    dh_params = [
        (0,       np.pi/2, 0.15708, t1),
        (-0.1104, 0,       0,       t2),
        (-0.096,  0,       0,       t3),
        (0,       np.pi/2, 0.06639, t4),
        (0,      -np.pi/2, 0.07318, t5),
        (0,      0,        0.0456,  t6)
    ]

    T = np.eye(4)
    for (a, alpha, d, theta) in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)

    return T

##############################################################################
# 2) Numerical Jacobian (Finite Difference)
##############################################################################
def numerical_jacobian(fk_func, thetas, eps=1e-6):
    """
    Approximates the 3xN Jacobian w.r.t. the joint angles (thetas).
    We'll treat only the first 3 rows of the transform as x, y, z.
    """
    x0 = fk_func(thetas)[:3, 3]
    J = np.zeros((3, len(thetas)))
    for i in range(len(thetas)):
        perturbed = np.copy(thetas)
        perturbed[i] += eps
        x_perturbed = fk_func(perturbed)[:3, 3]
        J[:, i] = (x_perturbed - x0) / eps
    return J

##############################################################################
# 3) Simple One-Step IK Update (Jacobian Pseudoinverse)
##############################################################################
def inverse_kinematics_step(thetas, x_desired, fk_func, alpha=0.5):
    """
    Perform a single small step of Jacobian-based IK.
      thetas: current joint angles (N,)
      x_desired: (3,) desired end-effector position
      fk_func: forward kinematics function
      alpha: step size (smaller = safer but slower)

    Returns updated_thetas, error_norm
    """
    x_current = fk_func(thetas)[:3, 3]
    error = x_desired - x_current
    error_norm = np.linalg.norm(error)

    if error_norm < 1e-9:
        # Already basically at the target
        return thetas, error_norm

    J = numerical_jacobian(fk_func, thetas)  # 3xN
    dtheta = alpha * np.linalg.pinv(J) @ error  # (N,)

    # Update angles
    new_thetas = thetas + dtheta
    return new_thetas, error_norm

##############################################################################
# 4) Set Joint Angles in MuJoCo
##############################################################################
def set_joint_angles(data, thetas):
    """
    Set each of the first len(thetas) qpos in the MuJoCo data to the given angles.
    Adjust as needed for your model's joint indexing.
    """
    for i in range(len(thetas)):
        data.qpos[i] = thetas[i]

##############################################################################
# 5) Straight-Line Interpolation in Cartesian Space
##############################################################################
def interpolate_linear(start_pos, end_pos, n_steps=20):
    """
    Creates a list of 3D waypoints from start_pos to end_pos (inclusive).
    """
    alphas = np.linspace(0.0, 1.0, n_steps)
    waypoints = []
    for alpha in alphas:
        wp = (1 - alpha) * start_pos + alpha * end_pos
        waypoints.append(wp)
    return waypoints

##############################################################################
# 6) Capture Frame (Offscreen)
##############################################################################
def capture_frame(model, data, scene, context, width=600, height=600):
    """
    Renders the current simulation state and returns an RGB image (H,W,3).
    """
    viewport = mujoco.MjrRect(0, 0, width, height)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()

    # Example camera settings:
    cam.distance = 1.0
    cam.azimuth = -145
    cam.elevation = -30
    cam.lookat[:] = [0, 0, 0]

    # Update scene
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, scene)

    # Render the scene
    mujoco.mjr_render(viewport, scene, context)

    # Read the pixels
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    mujoco.mjr_readPixels(rgb, depth, viewport, context)

    # Flip to match typical image coordinates
    rgb = np.flipud(rgb)
    return rgb

##############################################################################
# 7) Main Demo
##############################################################################
def main():
    print("[DEBUG] Entering main()...")

    # --------------------- Load MuJoCo Model ---------------------
    model_path = "config/xml/mycobot_280jn_mujoco.xml"  # Replace with your XML

    # Try loading model
    print(f"[DEBUG] Attempting to load MuJoCo model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file does not exist: {model_path}")
        sys.exit(1)

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        print("[DEBUG] MuJoCo model loaded successfully.")
    except Exception as e:
        print("[ERROR] Failed to load MuJoCo model:", e)
        sys.exit(1)

    # Create data
    try:
        data = mujoco.MjData(model)
        print("[DEBUG] MjData created successfully.")
    except Exception as e:
        print("[ERROR] Failed to create MjData:", e)
        sys.exit(1)

    # Create scene & context for rendering offscreen
    print("[DEBUG] Creating offscreen scene and context...")
    try:
        scene = mujoco.MjvScene(model, maxgeom=100)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        print("[DEBUG] Scene and context created successfully.")
    except Exception as e:
        print("[ERROR] Failed to create MjvScene or MjrContext:", e)
        sys.exit(1)

    # --------------------- Initial Setup -------------------------
    # Example initial angles
    thetas_init = np.zeros(6, dtype=float)
    print("[DEBUG] Setting initial thetas to:", thetas_init)

    try:
        set_joint_angles(data, thetas_init)
        mujoco.mj_forward(model, data)  # update FK
        print("[DEBUG] Joint angles set and mj_forward called.")
    except Exception as e:
        print("[ERROR] Failed to set joint angles or mj_forward:", e)
        sys.exit(1)

    # Get the initial EE position from forward kinematics
    x_init = forward_kinematics_debug(thetas_init)[:3, 3]
    print("[DEBUG] Computed x_init from forward_kinematics_debug:", x_init)

    # Define final target position
    x_target = np.array([1.0, 0.0, 0.0])  # Move 1m in x-direction, for example
    print("[DEBUG] Final target position is:", x_target)

    # Interpolate a path of, e.g., 30 points
    waypoints = interpolate_linear(x_init, x_target, n_steps=30)
    print(f"[DEBUG] Created {len(waypoints)} waypoints from x_init to x_target.")

    # --------------------- Simulation / IK Loop ------------------
    frames = []
    thetas = thetas_init.copy()
    stop_threshold = 0.01  # stop once close to final target

    print("[DEBUG] Starting IK loop over waypoints...")
    for i, wp in enumerate(waypoints):
        try:
            # Single-step IK for the current waypoint
            new_thetas, err = inverse_kinematics_step(thetas, wp, forward_kinematics_debug, alpha=0.3)
            thetas = new_thetas

            # Update simulation state
            set_joint_angles(data, thetas)
            mujoco.mj_forward(model, data)

            # Capture frame
            frame = capture_frame(model, data, scene, context, width=600, height=600)
            frames.append(frame)

            # Check error w.r.t. final target
            final_error = np.linalg.norm(x_target - forward_kinematics_debug(thetas)[:3, 3])
            print(f"[DEBUG] Step {i+1}/{len(waypoints)}: "
                  f"Current WP Error={err:.5f}, Final Target Error={final_error:.5f}")

            if final_error < stop_threshold:
                print("[DEBUG] Reached target region, stopping early.")
                break

        except Exception as e:
            print(f"[ERROR] IK or simulation step failed at waypoint {i+1}: {e}")
            break

    # --------------------- Show Video & Cleanup ------------------
    print("[DEBUG] Showing video of the motion (total frames = {})...".format(len(frames)))
    try:
        media.show_video(frames, fps=10)
    except Exception as e:
        print("[ERROR] Failed to display video:", e)

    print("[DEBUG] Terminating GLFW...")
    glfw.terminate()
    print("[DEBUG] Done.")

if __name__ == "__main__":
    main()

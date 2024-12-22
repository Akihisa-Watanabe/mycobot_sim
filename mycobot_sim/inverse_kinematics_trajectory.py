import numpy as np
import mujoco
from mujoco.glfw import glfw
from PIL import Image
import argparse

###############################################################################
#                           Utility: Add Ephemeral Geoms
###############################################################################

import numpy as np
import mujoco

def add_trajectory_markers(scene, positions, color=(0,1,0,1), radius=0.005):
    """
    Add small spheres to the scene for each 3D point in `positions`.
    These are ephemeral geoms that only live for the current render call.
    """
    # Pre-allocate the common geometry data for each sphere
    size_array = np.array([radius, radius, radius], dtype=np.float32)
    rgba_array = np.array(color, dtype=np.float32)
    mat_array  = np.eye(3, dtype=np.float32).ravel()  # Flattened 3x3 = 9 elems

    for point in positions:
        # Create a blank MjvGeom
        g = mujoco.MjvGeom()

        # Position must be float32
        pos_array = np.array(point, dtype=np.float32)

        # Initialize it as a sphere
        mujoco.mjv_initGeom(
            geom=g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size_array,
            pos=pos_array,
            mat=mat_array,
            rgba=rgba_array
        )

        # Use mjv_copyGeom to copy 'g' into the next free slot of scene.geoms
        geom_index = scene.ngeom
        if geom_index >= scene.maxgeom:
            print("Warning: scene has reached maxgeom; no more ephemeral geoms can be added.")
            return

        mujoco.mjv_copyGeom(scene.geoms[geom_index], g)
        scene.ngeom += 1


###############################################################################
#                               FK Utilities
###############################################################################

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
    # Adjust angles similarly to your original code
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
        (0,       0,       0.0456,  t6)
    ]

    T = np.eye(4)
    for (a, alpha, d, theta) in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)
    return T

###############################################################################
#                           Analytic Jacobian
###############################################################################

def compute_jacobian(thetas):
    """
    Computes the 3x6 position Jacobian using DH parameters analytically.
    thetas: [theta1, theta2, theta3, theta4, theta5, theta6]
    returns: J (3x6), the Jacobian for end-effector position.
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
        (0,       0,       0.0456,  t6)
    ]
    
    # Forward transforms up to each joint frame
    T = np.eye(4)
    origins = [T[:3, 3]]   # p_0 in base frame
    z_axes  = [T[:3, 2]]   # z_0 in base frame

    for (a, alpha, d, theta) in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    p_ee = origins[-1]  
    J = np.zeros((3, 6))
    for i in range(6):
        z_i = z_axes[i]
        p_i = origins[i]
        J[:, i] = np.cross(z_i, (p_ee - p_i))
    return J

###############################################################################
#                             Inverse Kinematics
###############################################################################

def inverse_kinematics_step(thetas, x_des, alpha=1.0):
    """One IK update for position-only."""
    T_0_6 = forward_kinematics_debug(thetas)
    x_current = T_0_6[:3, 3]
    error = x_des - x_current
    J = compute_jacobian(thetas)
    dtheta = alpha * np.linalg.pinv(J) @ error
    return thetas + dtheta

def inverse_kinematics(thetas_init, x_des, max_iters=100, tol=1e-4, alpha=1.0):
    """Iterative IK for position-only (3D)."""
    thetas = np.copy(thetas_init)
    for _ in range(max_iters):
        T_0_6 = forward_kinematics_debug(thetas)
        x_current = T_0_6[:3, 3]
        error = x_des - x_current
        if np.linalg.norm(error) < tol:
            break
        thetas = inverse_kinematics_step(thetas, x_des, alpha=alpha)
    return thetas

###############################################################################
#                     MuJoCo Visualization (with trajectory)
###############################################################################

def set_joint_angles(data, thetas):
    """Set the joint angles in the MuJoCo data structure."""
    for i in range(6):
        data.qpos[i] = thetas[i]


def visualize(model, data, 
              trajectory_positions,
              cam_distance,
              cam_azimuth,
              cam_elevation,
              cam_lookat,
              screenshot_name="robot_screenshot.png"):
    """
    Create a GLFW window, render the robot, and show ephemeral spheres for
    the entire trajectory. The user can interact until ESC is pressed.
    """
    if not glfw.init():
        raise Exception("Failed to initialize GLFW")

    width, height = 1200, 1200
    window = glfw.create_window(width, height, "Robot Visualization", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window")
    glfw.make_context_current(window)

    # Camera settings
    cam = mujoco.MjvCamera()
    cam.distance = cam_distance
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation
    cam.lookat[:] = cam_lookat

    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    viewport = mujoco.MjrRect(0, 0, width, height)

    # Render once and save screenshot
    mujoco.mjv_updateScene(model, data, opt, None, cam, 
                           mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    # Add ephemeral geoms for the entire trajectory
    add_trajectory_markers(scene, trajectory_positions, color=(0,1,0,1), radius=0.005)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
    mujoco.mjr_render(viewport, scene, context)

    # Take screenshot
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    mujoco.mjr_readPixels(rgb, depth, viewport, context)
    rgb = np.flipud(rgb)
    img = Image.fromarray(rgb)
    img.save(screenshot_name)
    print(f"Screenshot saved as {screenshot_name}")

    print("\nViewer started. Press Esc to close.")

    while not glfw.window_should_close(window):
        glfw.poll_events()
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # Update scene
        scene.ngeom = 0  # reset ephemeral geoms each frame
        mujoco.mjv_updateScene(model, data, opt, None, cam, 
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)

        # Add ephemeral geoms again
        add_trajectory_markers(scene, trajectory_positions, color=(0,1,0,1), radius=0.005)

        # Render
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)

        # Press ESC to close
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()

###############################################################################
#                                 MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Robot Visualization + Trajectory IK Comparison")
    
    parser.add_argument("--joint_angles", type=float, nargs=6, default=[0.0, 0, 0, 0, 0, 0],
                        help="Initial joint angles in radians for the 6 joints")
    parser.add_argument("--cam_distance", type=float, default=1.0, help="Camera distance from the scene center")
    parser.add_argument("--cam_azimuth", type=float, default=-145, help="Camera azimuth angle in degrees")
    parser.add_argument("--cam_elevation", type=float, default=-30, help="Camera elevation angle in degrees")
    parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, 0.1],
                        help="Camera look-at point as [x, y, z]")
    parser.add_argument("--model_path", type=str, default="config/xml/mycobot_280jn_mujoco.xml",
                        help="Path to the MuJoCo model XML file")
    parser.add_argument("--screenshot_name", type=str, default="mycobot_forward_kinematics_DH.png",
                        help="File name for the screenshot")

    # Trajectory definition
    parser.add_argument("--ik_target_start", type=float, nargs=3, default=None,
                        help="Start point [x, y, z] of the line trajectory for IK")
    parser.add_argument("--ik_target_end", type=float, nargs=3, default=None,
                        help="End point [x, y, z] of the line trajectory for IK")
    parser.add_argument("--ntraj_points", type=int, default=10,
                        help="Number of points to sample along the line from start to end")
    
    # IK iteration parameters
    parser.add_argument("--max_iters", type=int, default=100, help="Max iterations for IK per point")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for IK per point")
    parser.add_argument("--alpha", type=float, default=1.0, help="Step size scale for IK")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Setup MuJoCo model & data
    # -------------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data  = mujoco.MjData(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")

    # Initial angles
    thetas = np.array(args.joint_angles, dtype=float)

    # We'll keep track of the final angles as we move along the line
    solution_trajectory = []
    dh_positions_trajectory = []  # DH-based positions
    mj_positions_trajectory = []  # MuJoCo positions

    # -------------------------------------------------------------------------
    # 2) Straight-Line Trajectory and IK
    # -------------------------------------------------------------------------
    if args.ik_target_start is not None and args.ik_target_end is not None:
        start_pos = np.array(args.ik_target_start, dtype=float)
        end_pos   = np.array(args.ik_target_end,   dtype=float)
        
        npoints = args.ntraj_points
        for i in range(npoints):
            alpha_line = i / float(npoints - 1)
            x_des = (1.0 - alpha_line)*start_pos + alpha_line*end_pos

            # Solve IK from the current thetas
            thetas = inverse_kinematics(
                thetas_init=thetas,
                x_des=x_des,
                max_iters=args.max_iters,
                tol=args.tol,
                alpha=args.alpha
            )
            # Compute DH-based position
            T_0_6_dh = forward_kinematics_debug(thetas)
            p_ee_dh = T_0_6_dh[:3, 3]

            # Now set MuJoCo angles to see what MuJoCo computes
            set_joint_angles(data, thetas)
            mujoco.mj_forward(model, data)
            p_ee_mj = data.xpos[body_id]

            solution_trajectory.append(np.copy(thetas))
            dh_positions_trajectory.append(np.copy(p_ee_dh))
            mj_positions_trajectory.append(np.copy(p_ee_mj))

        print(f"\nTrajectory of {npoints} points completed.\n")

        # Display comparisons
        for i, (sol, p_dh, p_mj) in enumerate(zip(solution_trajectory,
                                                 dh_positions_trajectory,
                                                 mj_positions_trajectory)):
            diff = p_mj - p_dh
            print(f"Point {i+1}/{npoints}:")
            print(f"  IK Joint Angles: {sol}")
            print(f"  DH-based Pos:    {p_dh}")
            print(f"  MuJoCo Pos:      {p_mj}")
            print(f"  Difference:      {diff},  ||diff||= {np.linalg.norm(diff)}\n")

    else:
        # Single pose
        print("\nNo trajectory specified. Using only the provided initial pose.")
        T_0_6_dh = forward_kinematics_debug(thetas)
        p_ee_dh = T_0_6_dh[:3, 3]
        set_joint_angles(data, thetas)
        mujoco.mj_forward(model, data)
        p_ee_mj = data.xpos[body_id]

        solution_trajectory.append(thetas)
        dh_positions_trajectory.append(p_ee_dh)
        mj_positions_trajectory.append(p_ee_mj)

        diff = p_ee_mj - p_ee_dh
        print("Single Pose Comparison:")
        print(f"  Joint Angles: {thetas}")
        print(f"  DH-based Pos:  {p_ee_dh}")
        print(f"  MuJoCo Pos:    {p_ee_mj}")
        print(f"  Difference:    {diff},  ||diff||= {np.linalg.norm(diff)}\n")

    # -------------------------------------------------------------------------
    # 3) Final Pose Visualization
    # -------------------------------------------------------------------------
    thetas_final = solution_trajectory[-1]
    set_joint_angles(data, thetas_final)
    mujoco.mj_forward(model, data)
    print("Final angles:", thetas_final)

    # We can choose to visualize the MuJoCo positions or the DH positions.
    # Here, let's collect the MuJoCo positions as the trajectory to display.
    # (Alternatively, you could display dh_positions_trajectory or both.)
    all_positions = np.array(mj_positions_trajectory)

    visualize(
        model, 
        data, 
        trajectory_positions=all_positions,
        cam_distance=args.cam_distance,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
        cam_lookat=args.cam_lookat,
        screenshot_name=args.screenshot_name
    )


if __name__ == "__main__":
    main()

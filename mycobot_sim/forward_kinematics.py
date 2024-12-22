import numpy as np
import mujoco
from mujoco.glfw import glfw
from PIL import Image
import argparse

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

def set_joint_angles(data, thetas):
    """Set the joint angles in the MuJoCo data structure."""
    for i in range(6):
        data.qpos[i] = thetas[i]

def visualize(model, data, cam_distance, cam_azimuth, cam_elevation, cam_lookat, screenshot_name="robot_screenshot.png"):
    """
    Create a GLFW window, render the robot, and allow interaction until ESC is pressed.
    A screenshot is taken before the main rendering loop.
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

    # Create scene and context
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    viewport = mujoco.MjrRect(0, 0, width, height)

    # Update scene once and save a screenshot
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
    mujoco.mjr_render(viewport, scene, context)

    # Read pixels for screenshot
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    mujoco.mjr_readPixels(rgb, depth, viewport, context)
    rgb = np.flipud(rgb)

    # Save screenshot
    img = Image.fromarray(rgb)
    img.save(screenshot_name)
    print(f"Screenshot saved as {screenshot_name}")

    print("\nViewer started. Press Esc to close.")

    # Main rendering loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)

        # Press ESC to close
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Robot Visualization") #[0.0, np.pi/4, -np.pi/6, np.pi/3, -np.pi/2, np.pi/6]
    parser.add_argument("--joint_angles", type=float, nargs=6, default=[0.0, 0, 0, 0, 0, 0],
                        help="Joint angles in radians for the 6 joints, e.g. --joint_angles 0.0 0.78 -0.52 1.04 -1.57 0.52")
    parser.add_argument("--cam_distance", type=float, default=1.0, help="Camera distance from the scene center")
    parser.add_argument("--cam_azimuth", type=float, default=-145, help="Camera azimuth angle in degrees")
    parser.add_argument("--cam_elevation", type=float, default=-30, help="Camera elevation angle in degrees")
    parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, 0.1],
                        help="Camera look-at point as [x, y, z]")
    parser.add_argument("--model_path", type=str, default="config/xml/mycobot_280jn_mujoco.xml",
                        help="Path to the MuJoCo model XML file")
    parser.add_argument("--screenshot_name", type=str, default="mycobot_forward_kinematics_DH.png",
                        help="File name for the screenshot")
    args = parser.parse_args()

    # Compute forward kinematics using DH parameters
    T_0_6 = forward_kinematics_debug(args.joint_angles)
    fk_pos = T_0_6[:3, 3]
    print("DH-based FK End-Effector Position:", fk_pos)

    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)

    # Set the joint angles in MuJoCo and update forward kinematics
    set_joint_angles(data, args.joint_angles)
    mujoco.mj_forward(model, data)

    # Retrieve the end-effector position from MuJoCo
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    mj_pos = data.xpos[body_id]
    print("MuJoCo End-Effector Position:", mj_pos)

    # Get the ID of the target site
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")

    # Move the target site to the computed FK position
    model.site_pos[site_id] = fk_pos

    # Update the simulation
    mujoco.mj_forward(model, data)

    # Visualize the scene and take a screenshot
    visualize(model, data, 
              cam_distance=args.cam_distance,
              cam_azimuth=args.cam_azimuth,
              cam_elevation=args.cam_elevation,
              cam_lookat=args.cam_lookat,
              screenshot_name=args.screenshot_name)

if __name__ == "__main__":
    main()

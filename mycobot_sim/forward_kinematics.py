"""
MuJoCo robot visualiser that evaluates forward kinematics with the exact
quaternion recursion described in the “順運動学” section.

Requirements:
    • mujoco >= 3.1  
    • glfw   (via mujoco.glfw)  
    • pillow (PIL) for screenshots

Place this file and `quaternion_utils.py` in the same directory.
"""
import argparse
import numpy as np
import mujoco
from mujoco.glfw import glfw
from PIL import Image

# ── local quaternion helpers ─────────────────────────────────────────────
from quaternion_utils import quat_mul, quat_conj, rotvec, rotmat_to_quat


# ── classic DH transform (needed only to extract the relative translation) ──
def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Compute a standard Denavit–Hartenberg 4-×-4 homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.,       sa,       ca,      d],
        [0.,        0,        0,      1]
    ], dtype=float)


def dh_to_quat_and_d(a: float, alpha: float, d: float, theta: float):
    """
    Convert one DH line to a relative transform represented by
       – quaternion q_rel(θ)
       – translation  d_rel(θ) (3-vector in Σ_{k-1})
    """
    T = dh_transform(a, alpha, d, theta)
    R = T[:3, :3]
    p = T[:3, 3]
    q = rotmat_to_quat(R)
    return q, p


# ── recursive quaternion forward kinematics ──────────────────────────────
def forward_kinematics_quat(thetas):
    """
    Quaternion-based forward kinematics.

    Returns:
        T_0_n : 4 × 4 homogeneous transform base → EE
        p_n   : 3-vector position
        q_n   : 4-vector unit quaternion (scalar first)
    """
    # Joint offsets identical to the original script
    t1 = thetas[0]
    t2 = thetas[1] - np.pi / 2
    t3 = thetas[2]
    t4 = thetas[3] - np.pi / 2
    t5 = thetas[4] + np.pi / 2
    t6 = thetas[5]

    dh_params = [
        (0.0,        np.pi/2, 0.15708, t1),
        (-0.1104,    0.0,     0.0,     t2),
        (-0.096,     0.0,     0.0,     t3),
        (0.0,        np.pi/2, 0.06639, t4),
        (0.0,       -np.pi/2, 0.07318, t5),
        (0.0,        0.0,     0.0456,  t6),
    ]

    # Base frame initial state
    p = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])     # identity quaternion

    # Recursive propagation
    for (a, alpha, d, theta) in dh_params:
        q_rel, d_rel = dh_to_quat_and_d(a, alpha, d, theta)
        p += rotvec(q, d_rel)              # translate in current base frame
        q = quat_mul(q, q_rel)             # compose rotations

    # Assemble final 4 × 4 homogeneous transform
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p
    return T, p, q


# ── Mujoco helpers – unchanged from the original script ──────────────────
def set_joint_angles(data, thetas):
    """Write joint angles into the MuJoCo data buffer."""
    for i, angle in enumerate(thetas):
        data.qpos[i] = angle


def visualize(model, data,
              cam_distance, cam_azimuth, cam_elevation, cam_lookat,
              screenshot_name="robot_screenshot.png"):
    """Open a GLFW window, render one screenshot, then run an interactive viewer."""
    if not glfw.init():
        raise RuntimeError("Failed to initialise GLFW")

    width, height = 1200, 1200
    window = glfw.create_window(width, height, "Robot Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)

    # Camera setup
    cam = mujoco.MjvCamera()
    cam.distance = cam_distance
    cam.azimuth  = cam_azimuth
    cam.elevation = cam_elevation
    cam.lookat[:] = cam_lookat

    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)

    # Save screenshot
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    mujoco.mjr_readPixels(rgb, depth, viewport, context)
    Image.fromarray(np.flipud(rgb)).save(screenshot_name)
    print(f"Screenshot saved as {screenshot_name}")

    print("Viewer active – press Esc to close.")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()


# ── entry point ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo visualiser with quaternion forward kinematics")
    parser.add_argument("--joint_angles", type=float, nargs=6,
                        default=[0.0, -np.pi/4, -np.pi/6, -np.pi/3, -np.pi/2, -np.pi/6],
                        help="Six joint angles in radians")
    parser.add_argument("--cam_distance", type=float, default=1.0)
    parser.add_argument("--cam_azimuth",  type=float, default=-145)
    parser.add_argument("--cam_elevation", type=float, default=-30)
    parser.add_argument("--cam_lookat", type=float, nargs=3,
                        default=[0.0, 0.0, 0.1], help="Camera look-at [x y z].")
    parser.add_argument("--model_path", type=str,
                        default="config/xml/mycobot_280jn_mujoco.xml")
    parser.add_argument("--screenshot_name", type=str,
                        default="mycobot_fk_quat.png")
    args = parser.parse_args()

    # Evaluate forward kinematics
    T_0_6, fk_pos, fk_quat = forward_kinematics_quat(args.joint_angles)
    print("FK position (quat method):", fk_pos)
    print("FK quaternion            :", fk_quat)

    # Load MuJoCo model and push joint angles
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data  = mujoco.MjData(model)
    set_joint_angles(data, args.joint_angles)
    mujoco.mj_forward(model, data)

    # Compare with MuJoCo’s own kinematics for sanity
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    print("MuJoCo EE position       :", data.xpos[ee_body_id])

    # Visualise
    visualize(model, data,
              cam_distance=args.cam_distance,
              cam_azimuth=args.cam_azimuth,
              cam_elevation=args.cam_elevation,
              cam_lookat=args.cam_lookat,
              screenshot_name=args.screenshot_name)


if __name__ == "__main__":
    main()

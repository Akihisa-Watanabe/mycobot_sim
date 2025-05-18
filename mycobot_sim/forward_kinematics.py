"""
MuJoCo robot visualiser that performs forward kinematics using
**only quaternion & vector operations** (no per-link matrices).

Usage example
-------------
python forward_kinematics_demo.py \
       --joint_angles 0.0 -0.785 -0.524 -1.047 -1.571 -0.524
"""

import argparse
import numpy as np
import mujoco
from mujoco.glfw import glfw
from PIL import Image

# Quaternion helpers
from quaternion_utils import quat_mul, quat_conj, rotvec


# ---------------------------------------------------------------------- #
# DH line  →  (relative quaternion, relative translation)
# ---------------------------------------------------------------------- #
def dh_to_quat_and_d(a: float, alpha: float, d: float, theta: float):
    """
    Convert standard DH parameters directly to

        q_rel : unit quaternion  (rotation Σ_{k-1} → Σ_k)
        d_rel : 3-vector         (translation of Σ_k origin,
                                  expressed in frame Σ_{k-1})

    No intermediate 4×4 matrix is created.
    """
    # --- rotation part --------------------------------------------------
    half_t = theta / 2.0
    half_a = alpha / 2.0
    q_z = np.array([np.cos(half_t), 0.0, 0.0, np.sin(half_t)])   # Rz(theta)
    q_x = np.array([np.cos(half_a), np.sin(half_a), 0.0, 0.0])   # Rx(alpha)
    q_rel = quat_mul(q_z, q_x)   # Rz ⊗ Rx  ⇒  Rz(theta) · Rx(alpha)

    # --- translation part ----------------------------------------------
    d_rel = np.array([a * np.cos(theta),
                      a * np.sin(theta),
                      d], dtype=float)
    return q_rel, d_rel


# ---------------------------------------------------------------------- #
# Quaternion recursion
# ---------------------------------------------------------------------- #
def forward_kinematics_quat(thetas):
    """
    Quaternion-based forward kinematics.

    Parameters
    ----------
    thetas : iterable of 6 joint angles [rad]

    Returns
    -------
    T_0_6 : 4×4 homogeneous matrix base → EE (for legacy use)
    p_ee  : 3-vector EE position
    q_ee  : 4-vector unit quaternion EE orientation (scalar first)
    """
    # Joint-specific offsets (unchanged from the original script)
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

    # Base frame
    p = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])   # identity quaternion

    # Recursive propagation Σ_{k-1} → Σ_k
    for (a, alpha, d, theta) in dh_params:
        q_rel, d_rel = dh_to_quat_and_d(a, alpha, d, theta)
        p += rotvec(q, d_rel)            # translate in current base frame
        q  = quat_mul(q, q_rel)          # compose rotations

    # Convert final quaternion to a rotation matrix (explicit formula)
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


# ---------------------------------------------------------------------- #
# MuJoCo helpers
# ---------------------------------------------------------------------- #
def set_joint_angles(data, thetas):
    """Copy the six joint angles into MuJoCo’s qpos array."""
    for i, ang in enumerate(thetas):
        data.qpos[i] = ang


def visualize(model, data,
              cam_distance, cam_azimuth, cam_elevation, cam_lookat,
              screenshot_name="robot_screenshot.png"):
    """Render one screenshot, then open an interactive viewer."""
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    w_win, h_win = 1200, 1200
    window = glfw.create_window(w_win, h_win, "Robot Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")
    glfw.make_context_current(window)

    cam = mujoco.MjvCamera()
    cam.distance  = cam_distance
    cam.azimuth   = cam_azimuth
    cam.elevation = cam_elevation
    cam.lookat[:] = cam_lookat

    scene   = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    opt     = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    vp = mujoco.MjrRect(0, 0, w_win, h_win)
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(vp, scene, context)

    # Screenshot
    rgb   = np.zeros((h_win, w_win, 3), dtype=np.uint8)
    depth = np.zeros((h_win, w_win), dtype=np.float32)
    mujoco.mjr_readPixels(rgb, depth, vp, context)
    Image.fromarray(np.flipud(rgb)).save(screenshot_name)
    print(f"Screenshot saved as {screenshot_name}")

    print("Viewer active – press Esc to exit.")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        vp = mujoco.MjrRect(0, 0, fb_w, fb_h)
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(vp, scene, context)
        glfw.swap_buffers(window)

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
    glfw.terminate()


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo visualiser with quaternion-only forward kinematics")
    parser.add_argument("--joint_angles", type=float, nargs=6,
                        default=[0.0, -np.pi/4, -np.pi/6, -np.pi/3, -np.pi/2, -np.pi/6],
                        help="Six joint angles in radians")
    parser.add_argument("--cam_distance",  type=float, default=1.0)
    parser.add_argument("--cam_azimuth",   type=float, default=-145)
    parser.add_argument("--cam_elevation", type=float, default=-30)
    parser.add_argument("--cam_lookat",    type=float, nargs=3,
                        default=[0.0, 0.0, 0.1], help="[x y z] camera look-at point")
    parser.add_argument("--model_path",    type=str,
                        default="config/xml/mycobot_280jn_mujoco.xml")
    parser.add_argument("--screenshot_name", type=str,
                        default="mycobot_fk_quat.png")
    args = parser.parse_args()

    # Forward kinematics (quaternion recursion)
    T, p_ee, q_ee = forward_kinematics_quat(args.joint_angles)
    print("FK position :", p_ee)
    print("FK quaternion:", q_ee)

    # MuJoCo simulation for verification
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data  = mujoco.MjData(model)
    set_joint_angles(data, args.joint_angles)
    mujoco.mj_forward(model, data)

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    print("MuJoCo EE position:", data.xpos[ee_body_id])

    # Visualise
    visualize(model, data,
              cam_distance=args.cam_distance,
              cam_azimuth=args.cam_azimuth,
              cam_elevation=args.cam_elevation,
              cam_lookat=args.cam_lookat,
              screenshot_name=args.screenshot_name)


if __name__ == "__main__":
    main()

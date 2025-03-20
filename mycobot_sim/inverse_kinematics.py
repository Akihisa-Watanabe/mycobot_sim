import mujoco
import numpy as np
import mediapy as media
import time

###############################################################################
# 1) DH-based FK, Jacobian, and IK
###############################################################################
def dh_transform(a, alpha, d, theta):
    """Compute the Denavit-Hartenberg transformation matrix."""
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
    Compute forward kinematics using your MyCobot-like DH parameters.
    thetas: [theta1, theta2, theta3, theta4, theta5, theta6].
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
        (0,       0,       0.0456,  t6)
    ]

    T = np.eye(4)
    for (a, alpha, d, theta) in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)

    return T

def analytic_jacobian(thetas):
    """
    Build the 3x6 position Jacobian for a 6-DOF manipulator
    by accumulating transforms to get each intermediate frame.
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

    # Accumulate transforms up to each joint
    T_i = np.eye(4)
    all_T = [T_i.copy()]  # T_0 for i=0

    for (a, alpha, d, theta) in dh_params:
        T_step = dh_transform(a, alpha, d, theta)
        T_i = T_i @ T_step
        all_T.append(T_i.copy())

    p = []
    z = []
    for i in range(7):
        p_i = all_T[i][:3, 3]
        z_i = all_T[i][:3, 2]
        p.append(p_i)
        z.append(z_i)

    p_ee = p[6]
    J_pos = np.zeros((3, 6))
    for i in range(6):
        J_pos[:, i] = np.cross(z[i], p_ee - p[i])
    return J_pos

def inverse_kinematics_simple(thetas_init, x_desired, 
                              fk_func, 
                              ):
    """
    Simple iterative IK using a first-order (Jacobian-based) approach (position only).
    """
    thetas = np.array(thetas_init, dtype=float)

    T = fk_func(thetas)
    x_current = T[:3, 3]
    error = x_desired - x_current
    J = analytic_jacobian(thetas)
    J_inv = np.linalg.pinv(J)
    dtheta = J_inv @ error
    thetas += dtheta

    return thetas

###############################################################################
# 2) Helper Functions
###############################################################################
def interpolate_linear(start_pos, end_pos, n_steps=20):
    alphas = np.linspace(0.0, 1.0, n_steps)
    return [(1 - a)*start_pos + a*end_pos for a in alphas]

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
    # render_width, render_height = 640, 480
    renderer = mujoco.Renderer(model, render_height, render_width)
    renderer.enable_shadows = True

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")

    # Initial conditions
    thetas_init = np.zeros(6, dtype=float)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)

    # Our DH-based initial EE position
    x_init = forward_kinematics_debug(thetas_init)[:3, 3]
    x_target = np.array([ 0.05826, -0.1752,  0.3566 ])
    n_steps = 100
    waypoints = interpolate_linear(x_init, x_target, n_steps)

    frames = []
    current_thetas = thetas_init.copy()

    # NEW: Lists for logs
    joint_angle_log = []
    ee_pos_log = []

    print("\nStarting motion along a line in Cartesian space.")
    for i, wp in enumerate(waypoints):
        current_thetas = inverse_kinematics_simple(current_thetas, wp, forward_kinematics_debug)
        set_joint_angles(data, current_thetas)
        mujoco.mj_step(model, data)

        mujoco_ee_pos = data.xpos[ee_body_id]
        our_ee_pos = forward_kinematics_debug(current_thetas)[:3, 3] 
        diff = np.linalg.norm(mujoco_ee_pos - our_ee_pos)

        # Record logs (minimal change: just append to your lists)
        joint_angle_log.append(current_thetas.copy())
        ee_pos_log.append(our_ee_pos.copy())

        renderer.update_scene(data)
        frames.append(renderer.render())

        err_to_final = np.linalg.norm(x_target - our_ee_pos)
        print(f"Step {i+1}/{n_steps} | "
              f"MuJoCo EE = {mujoco_ee_pos.round(5)} "
              f" Our EE = {our_ee_pos.round(5)} "
              f"(diff={diff:.5f}), "
              f"Error->final={err_to_final:.5f}")


    import imageio
    print("Saving video to 'my_robot_trajectory.mp4'...")
    imageio.mimsave("my_robot_trajectory.mp4", frames, fps=60, quality=10, bitrate="32M")
    print("Video saved.")

    # Save logs to NPZ
    np.savez("robot_logs.npz",
             joint_angles=np.array(joint_angle_log),
             ee_positions=np.array(ee_pos_log))
    print("Logs saved to 'robot_logs.npz'.")

    renderer.close()

if __name__ == "__main__":
    main()

import mujoco
import numpy as np
import mediapy as media
import time

###############################################################################
# 1) User-Provided FK, Jacobian, and IK
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


def numerical_jacobian(fk_func, thetas, eps=1e-6):
    """
    Compute a numerical approximation of the 3xN position Jacobian
    via finite differences around current joint angles 'thetas'.
    """
    x_current = fk_func(thetas)[:3, 3]
    J = np.zeros((3, len(thetas)))

    for i in range(len(thetas)):
        thetas_perturbed = thetas.copy()
        thetas_perturbed[i] += eps
        x_perturbed = fk_func(thetas_perturbed)[:3, 3]
        J[:, i] = (x_perturbed - x_current)

    return J


def analytic_jacobian(thetas):
    """
    Build the 3x6 position Jacobian for a 6-DOF manipulator
    by accumulating transforms to get each intermediate frame.
    
    Returns a np.array of shape (3,6).
    """
    # Same DH manipulation as forward_kinematics_debug, 
    # but we'll store partial transforms along the way
    t1 = thetas[0] 
    t2 = thetas[1] - np.pi/2
    t3 = thetas[2]
    t4 = thetas[3] - np.pi/2
    t5 = thetas[4] + np.pi/2
    t6 = thetas[5] 
    
    # DH parameters
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

    # Now, all_T[i] = ^0T_i for i in {0..6}
    # positions p_i = ^0p_i and z_i = ^0z_i (the 3rd column of the rotation)
    p = []
    z = []
    for i in range(7):  # i=0..6
        p_i = all_T[i][:3, 3]
        # Extract the Z axis from the rotation portion
        z_i = all_T[i][:3, 2]
        p.append(p_i)
        z.append(z_i)

    # End-effector position p_6
    p_ee = p[6]

    # Build 3x6 Jacobian for a revolute manipulator
    J_pos = np.zeros((3, 6))
    for i in range(6):
        # Revolute joint => cross( z_i, (p_ee - p_i) )
        J_pos[:, i] = np.cross(z[i], p_ee - p[i])

    return J_pos

def inverse_kinematics_simple(thetas_init, x_desired, 
                              fk_func, 
                              max_iterations=100, 
                              tolerance=1e-4):
    """
    Simple iterative IK using a first-order (Jacobian-based) approach.
    """
    thetas = np.array(thetas_init, dtype=float)

    for _ in range(max_iterations):
        T = fk_func(thetas)
        x_current = T[:3, 3]
        error = x_desired - x_current
        if np.linalg.norm(error) < tolerance:
            break

        J = analytic_jacobian(thetas)
        dtheta = np.linalg.pinv(J) @ error
        thetas += dtheta

    return thetas

###############################################################################
# 2) Helper Functions
###############################################################################
def interpolate_linear(start_pos, end_pos, n_steps=20):
    """
    Return a list of 3D points linearly interpolated between 'start_pos' and 'end_pos'.
    """
    alphas = np.linspace(0.0, 1.0, n_steps)
    return [(1 - a)*start_pos + a*end_pos for a in alphas]

def set_joint_angles(data, thetas):
    """
    Set the joint angles in MuJoCo data.qpos to 'thetas'.
    Adjust as needed for your robot's joint indexing.
    """
    for i in range(len(thetas)):
        data.qpos[i] = thetas[i]

###############################################################################
# 3) Main Demo
###############################################################################
def main():
    # -------------------------------------------------------------------------
    # A) Load MuJoCo Model & Create Renderer
    # -------------------------------------------------------------------------
    model_path = "config/xml/mycobot_280jn_mujoco.xml"  # Update to your XML
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    render_width, render_height = 640, 480 # 1200, 1200
    renderer = mujoco.Renderer(model, render_height, render_width)
    renderer.enable_shadows = True

    # -------------------------------------------------------------------------
    # B) Get Initial Pose & End-Effector Position
    # -------------------------------------------------------------------------
    # Let's say the robot starts with all joints = 0
    # Or you can read them from data.qpos if your model is initialized differently.
    thetas_init = np.zeros(6, dtype=float)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)  # update the model state

    # The initial end-effector position according to your FK
    x_init = forward_kinematics_debug(thetas_init)[:3, 3]

    # Define a target end-effector position
    x_target =  np.array([0.22253875, -0.11022109,  0.20772435])  # example 3D target

    # -------------------------------------------------------------------------
    # C) Generate a Straight-Line Trajectory in Cartesian Space
    # -------------------------------------------------------------------------
    n_steps = 1000
    waypoints = interpolate_linear(x_init, x_target, n_steps=n_steps)

    # -------------------------------------------------------------------------
    # D) For Each Waypoint, Solve IK & Render
    # -------------------------------------------------------------------------
    frames = []
    current_thetas = thetas_init.copy()

    for i, wp in enumerate(waypoints):
        # 1) IK for this waypoint in Cartesian space
        current_thetas = inverse_kinematics_simple(current_thetas, wp, 
                                                   forward_kinematics_debug)
        # 2) Set the resulting angles in MuJoCo
        set_joint_angles(data, current_thetas)
        mujoco.mj_forward(model, data)

        # 3) Capture the frame
        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame)

        # Print progress
        err_to_target = np.linalg.norm(x_target - forward_kinematics_debug(current_thetas)[:3, 3])
        print(f"Waypoint {i+1}/{n_steps}, Error to final target = {err_to_target:.4f}")

    # E) Save video
    import imageio
    imageio.mimsave("my_robot_trajectory.mp4", frames, fps=100)
    
    # -------------------------------------------------------------------------
    # F) Cleanup
    # -------------------------------------------------------------------------
    renderer.close()

if __name__ == "__main__":
    main()

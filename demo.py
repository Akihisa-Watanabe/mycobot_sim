#!/usr/bin/env python3
"""
ターゲット位置と姿勢を可視化するロボットアームデモ
ユーザーが指定した位置と姿勢にロボットアームを動かし、その過程を可視化します。
元のFK/IK実装を維持しています。
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
# 1) DH-based FK, Jacobian, and IK - 元の実装を保持
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

def euler_angle_from_matrix(T):
    """Calculate Euler angles (ZYZ) from transformation matrix."""
    alpha = math.atan2(T[1, 2], T[0, 2])
    if not (-math.pi/2 <= alpha <= math.pi/2):
        alpha = math.atan2(T[1, 2], T[0, 2]) + math.pi
    if not (-math.pi/2 <= alpha <= math.pi/2):
        alpha = math.atan2(T[1, 2], T[0, 2]) - math.pi
    
    beta = math.atan2(
        T[0, 2] * math.cos(alpha) + T[1, 2] * math.sin(alpha),
        T[2, 2]
    )
    
    gamma = math.atan2(
        -T[0, 0] * math.sin(alpha) + T[1, 0] * math.cos(alpha),
        -T[0, 1] * math.sin(alpha) + T[1, 1] * math.cos(alpha)
    )
    
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
    target_pose: [x, y, z, alpha, beta, gamma]
    Returns: array of joint angles
    """
    thetas = thetas_init.copy()
    
    for iteration in range(max_iterations):
        # Get current pose
        current_pose = forward_kinematics(thetas)
        
        # Calculate pose error
        pose_error = target_pose - current_pose
        
        # Get current Euler angles
        alpha, beta, gamma = current_pose[3:]
        
        K_zyz = np.array([
            [0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
            [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
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
        
        # Update joint angles
        thetas += step_size * theta_dot
        
        # Normalize angles to [-pi, pi]
        thetas = np.array([((angle + math.pi) % (2 * math.pi)) - math.pi for angle in thetas])
        
        # Check for convergence
        if np.linalg.norm(pose_error) < 1e-5:
            break
    
    return thetas

###############################################################################
# 2) Helper Functions - 元の実装を保持
###############################################################################
def interpolate_linear(start_pos, end_pos, n_steps=20):
    alphas = np.linspace(0.0, 1.0, n_steps)
    return [(1 - a)*start_pos + a*end_pos for a in alphas]

def interpolate_pose(start_pose, end_pose, n_steps=20):
    """
    Interpolate between start and end poses linearly.
    Both position and orientation (Euler angles) are interpolated.
    """
    alphas = np.linspace(0.0, 1.0, n_steps)
    return [(1 - a)*start_pose + a*end_pose for a in alphas]

def set_joint_angles(data, thetas):
    for i in range(len(thetas)):
        data.qpos[i] = thetas[i]

###############################################################################
# 3) ターゲット可視化のための新しい関数
###############################################################################
def euler_to_rotation_matrix(alpha, beta, gamma):
    """
    ZYZオイラー角から回転行列を計算する
    """
    # 各角度の正弦と余弦を計算
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    # ZYZ回転行列を計算
    R = np.array([
        [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
        [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
        [-sb*cg, sb*sg, cb]
    ])
    
    return R

def axis_to_quat(vec1, vec2):
    """
    2つの単位ベクトル間の回転を表す四元数を計算
    vec1から見てvec2がどの回転で到達するか
    """
    # 正規化
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # 内積とクロス積を計算
    dot_product = np.dot(vec1, vec2)
    cross_product = np.cross(vec1, vec2)
    
    # 特殊ケース: ベクトルが並行（同じ方向または逆方向）
    if np.isclose(dot_product, 1.0):
        return "1 0 0 0"  # 単位四元数（回転なし）
    elif np.isclose(dot_product, -1.0):
        # 180度回転 - 任意の垂直軸を選択
        perp = np.array([1, 0, 0]) if not np.allclose(vec1, [1, 0, 0]) else np.array([0, 1, 0])
        perp = perp - vec1 * np.dot(perp, vec1)
        perp = perp / np.linalg.norm(perp)
        return f"0 {perp[0]} {perp[1]} {perp[2]}"
        
    # 回転角度を計算
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # 回転軸を計算（正規化されたクロス積）
    axis = cross_product / np.linalg.norm(cross_product)
    
    # 四元数を計算
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    
    return f"{w} {xyz[0]} {xyz[1]} {xyz[2]}"

def remove_existing_target(model_xml):
    """
    既存のターゲットサイトの定義を削除
    """
    # ターゲットサイトを見つける正規表現パターン
    site_pattern = r'<site\s+name="target[^"]*"[^>]*>'
    
    # 既存のターゲットサイトを削除
    model_xml = re.sub(site_pattern, '<!-- Removed target site -->', model_xml)
    return model_xml

def add_target_visualization(model_xml, target_pos, target_orientation):
    """
    ターゲット位置と姿勢を可視化するための要素をモデルXMLに追加
    """
    # 既存のターゲットサイトを削除
    model_xml = remove_existing_target(model_xml)
    
    # </worldbody>タグを見つける
    worldbody_end = model_xml.find("</worldbody>")
    if worldbody_end == -1:
        print("Warning: Could not find </worldbody> tag in the XML.")
        return model_xml
    
    # オイラー角から回転行列を計算
    R = euler_to_rotation_matrix(*target_orientation)
    
    # ターゲット位置の球体とXYZ軸マーカーを追加
    target_sites = f"""
    <!-- ターゲット位置マーカー（赤色の球） -->
    <site name="target_pos" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" size="0.01 0.01 0.01" type="sphere" rgba="1 0 0 0.7"/>
    
    <!-- ターゲット姿勢マーカー（RGB = XYZ軸） -->
    <site name="target_x_axis" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" 
          size="0.001 0.03 0.001" type="cylinder" rgba="1 0 0 1" 
          quat="{axis_to_quat(np.array([0, 0, 1]), R[:, 0])}"/>
    
    <site name="target_y_axis" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" 
          size="0.001 0.03 0.001" type="cylinder" rgba="0 1 0 1" 
          quat="{axis_to_quat(np.array([0, 0, 1]), R[:, 1])}"/>
    
    <site name="target_z_axis" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" 
          size="0.001 0.03 0.001" type="cylinder" rgba="0 0 1 1" 
          quat="{axis_to_quat(np.array([0, 0, 1]), R[:, 2])}"/>
    
    <!-- ターゲット位置と現在位置の間の線 -->
    <site name="target_line" pos="0 0 0" size="0.001 0.001 0.001" type="cylinder" rgba="1 1 0 0.5"/>
    """
    
    # worldbodyの閉じタグの前に挿入
    xml_with_target = model_xml[:worldbody_end] + target_sites + model_xml[worldbody_end:]
    return xml_with_target

def update_target_line(model, data, ee_pos, target_pos):
    """
    エンドエフェクタからターゲットまでの線を更新
    """
    # サイトIDを取得
    line_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_line")
    if line_id == -1:
        return  # サイトが見つからない場合は何もしない
    
    # 線の中間点と方向を計算
    midpoint = (ee_pos + target_pos) / 2
    direction = target_pos - ee_pos
    length = np.linalg.norm(direction)
    
    if length < 1e-6:
        return  # 距離が非常に小さい場合は更新しない
    
    # 線の位置を設定
    model.site_pos[line_id] = midpoint
    
    # 線のサイズを設定 - MuJoCo APIでは3つの値が必要
    model.site_size[line_id] = np.array([0.001, length/2, 0.001])
    
    # 線の方向を設定（単位ベクトルに変換）
    direction = direction / length
    
    # Z軸方向から目標方向への回転を計算
    quat_str = axis_to_quat(np.array([0, 0, 1]), direction)
    quat_vals = np.array([float(x) for x in quat_str.split()])
    model.site_quat[line_id] = quat_vals

###############################################################################
# 4) メインのデモ関数
###############################################################################
def run_robot_demo(target_pos, target_orientation, output_file="robot_pose_trajectory.mp4"):
    """
    ターゲット位置と姿勢を可視化したロボットアームのデモを実行
    """
    # 一時ファイルの削除
    if os.path.exists("temp_model_with_target.xml"):
        try:
            os.remove("temp_model_with_target.xml")
            print("前回の一時ファイルを削除しました")
        except Exception as e:
            print(f"一時ファイルの削除中にエラーが発生: {e}")
    
    # XMLファイルのパスを見つける
    try:
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "config" / "xml" / "mycobot_280jn_mujoco.xml"
        if not model_path.exists():
            print(f"XMLファイルが見つかりません: {model_path}")
            
            # カレントディレクトリからも探す
            model_path = Path("config/xml/mycobot_280jn_mujoco_demo.xml")
            if not model_path.exists():
                print(f"XMLファイルが見つかりません: {model_path}")
                raise FileNotFoundError("モデルXMLファイルが見つかりません")
        
        model_path = str(model_path)
    except Exception as e:
        print(f"モデルパスの検索中にエラーが発生: {e}")
        return False
    
    # ロボットモデルの読み込み
    try:
        # XMLを文字列として読み込む
        with open(model_path, 'r') as f:
            model_xml = f.read()
        
        # メッシュディレクトリのパスを修正
        script_dir = Path(__file__).resolve().parent
        mesh_dir = script_dir / "config" / "meshes_mujoco"
        
        if not mesh_dir.exists():
            print(f"メッシュディレクトリが見つかりません: {mesh_dir}")
            # カレントディレクトリからも探す
            mesh_dir = Path("config/meshes_mujoco")
            if not mesh_dir.exists():
                print(f"メッシュディレクトリが見つかりません: {mesh_dir}")
                raise FileNotFoundError("メッシュディレクトリが見つかりません")
        
        # 絶対パスに変換
        mesh_dir = str(mesh_dir.absolute())
        
        # meshdir属性を正しいパスに置換
        if 'meshdir=' in model_xml:
            start_idx = model_xml.find('meshdir=') + 9  # 'meshdir="'の長さは9
            end_idx = model_xml.find('"', start_idx)
            old_meshdir = model_xml[start_idx:end_idx]
            
            # 新しいパスに置換
            model_xml = model_xml.replace(f'meshdir="{old_meshdir}"', f'meshdir="{mesh_dir}"')
        
        # ターゲット可視化要素を追加
        model_xml = add_target_visualization(model_xml, target_pos, target_orientation)
        
        # 一時ファイルに保存
        temp_model_path = "temp_model_with_target.xml"
        with open(temp_model_path, 'w') as f:
            f.write(model_xml)
        
        # 修正したモデルを読み込む
        model = mujoco.MjModel.from_xml_path(temp_model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"モデルの設定中にエラーが発生: {e}")
        return False
    
    # レンダラーの設定
    try:
        render_width, render_height = 1920, 1088
        model.vis.global_.offwidth = render_width
        model.vis.global_.offheight = render_height
        
        renderer = mujoco.Renderer(model, render_height, render_width)
        renderer.enable_shadows = True
    except Exception as e:
        print(f"レンダラーの設定中にエラーが発生: {e}")
        return False
    
    # 初期条件の設定
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    
    thetas_init = np.zeros(6, dtype=float)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)
    
    # 初期姿勢を取得し、ターゲット姿勢を設定
    initial_pose = forward_kinematics(thetas_init)
    target_pose = np.concatenate([target_pos, target_orientation])
    
    # スムーズな動きのためにウェイポイントを生成
    n_steps = 100
    waypoints = interpolate_pose(initial_pose, target_pose, n_steps)
    
    # フレームとログの初期化
    frames = []
    current_thetas = thetas_init.copy()
    
    # ログデータの初期化
    joint_angle_log = []
    ee_pos_log = []
    ee_orient_log = []
    
    print("\n目標位置と姿勢へロボットを移動しています (Moving robot to target position and orientation)")
    print(f"目標位置 (Target position): {target_pos}")
    print(f"目標姿勢 (Target orientation in ZYZ Euler angles): {target_orientation}")
    
    for i, wp in enumerate(waypoints):
        # 逆運動学で関節角度を計算
        current_thetas = inverse_kinematics(current_thetas, wp)
        
        # シミュレーションに関節角度を設定
        set_joint_angles(data, current_thetas)
        mujoco.mj_forward(model, data)
        
        # 現在のエンドエフェクタの位置と姿勢を取得
        mujoco_ee_pos = data.xpos[ee_body_id]
        our_ee_pose = forward_kinematics(current_thetas)
        our_ee_pos = our_ee_pose[:3]
        our_ee_orient = our_ee_pose[3:]
        
        # ターゲット線を更新
        update_target_line(model, data, mujoco_ee_pos, target_pos)
        
        # 誤差を計算
        pos_diff = np.linalg.norm(mujoco_ee_pos - our_ee_pos)
        pos_err_to_target = np.linalg.norm(target_pos - our_ee_pos)
        orient_err_to_target = np.linalg.norm(target_orientation - our_ee_orient)
        
        # ログを記録
        joint_angle_log.append(current_thetas.copy())
        ee_pos_log.append(our_ee_pos.copy())
        ee_orient_log.append(our_ee_orient.copy())
        
        # 進捗を表示
        print(f"ステップ {i+1}/{n_steps} | "
              f"位置誤差: {pos_err_to_target:.5f}, "
              f"姿勢誤差: {orient_err_to_target:.5f}, "
              f"MuJoCo-FK差分: {pos_diff:.5f}")
        
        # レンダリングとフレーム保存
        renderer.update_scene(data)
        frames.append(renderer.render())
    
    # 動画の保存
    try:
        print(f"ビデオを保存しています: {output_file}")
        imageio.mimsave(output_file, frames, fps=30, quality=8, bitrate="16M")
        print(f"ビデオが保存されました: {output_file}")
    except Exception as e:
        print(f"ビデオの保存中にエラーが発生: {e}")
        return False
    
    # ログデータの保存
    log_file = os.path.splitext(output_file)[0] + "_log.npz"
    try:
        np.savez(log_file, 
                 target_position=target_pos,
                 target_orientation=target_orientation,
                 joint_angles=np.array(joint_angle_log),
                 ee_positions=np.array(ee_pos_log),
                 ee_orientations=np.array(ee_orient_log))
        print(f"ログが保存されました: {log_file}")
    except Exception as e:
        print(f"ログの保存中にエラーが発生: {e}")
    
    # クリーンアップ
    renderer.close()
    if os.path.exists("temp_model_with_target.xml"):
        os.remove("temp_model_with_target.xml")
    
    return True

###############################################################################
# 5) インタラクティブモード
###############################################################################
def run_interactive_mode(initial_target_pos, initial_target_orientation):
    """
    MuJoCoのビューアを使ってインタラクティブモードで実行
    """
    print("インタラクティブモードを開始します")
    
    try:
        from mujoco import viewer
    except ImportError:
        print("インタラクティブモードにはmujoco.viewerが必要です")
        print("最新のMuJoCoパッケージがインストールされていることを確認してください")
        return False
    
    # 一時ファイルを削除
    if os.path.exists("temp_model_with_target.xml"):
        try:
            os.remove("temp_model_with_target.xml")
        except Exception:
            pass
    
    # XMLファイルのパスを見つける
    try:
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "config" / "xml" / "mycobot_280jn_mujoco.xml"
        if not model_path.exists():
            model_path = Path("config/xml/mycobot_280jn_mujoco_demo.xml")
        model_path = str(model_path)
    except Exception as e:
        print(f"モデルパスの検索中にエラーが発生: {e}")
        return False
    
    # ロボットモデルの読み込み
    try:
        with open(model_path, 'r') as f:
            model_xml = f.read()
        
        # メッシュディレクトリのパスを修正
        mesh_dir = Path("config/meshes_mujoco").absolute()
        
        if 'meshdir=' in model_xml:
            start_idx = model_xml.find('meshdir=') + 9
            end_idx = model_xml.find('"', start_idx)
            old_meshdir = model_xml[start_idx:end_idx]
            model_xml = model_xml.replace(f'meshdir="{old_meshdir}"', f'meshdir="{mesh_dir}"')
        
        # ターゲット可視化要素を追加
        model_xml = add_target_visualization(model_xml, initial_target_pos, initial_target_orientation)
        
        # 一時ファイルに保存
        temp_model_path = "temp_model_with_target.xml"
        with open(temp_model_path, 'w') as f:
            f.write(model_xml)
        
        # 修正したモデルを読み込む
        model = mujoco.MjModel.from_xml_path(temp_model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"モデルの設定中にエラーが発生: {e}")
        return False
    
    # 初期条件の設定
    thetas_init = np.zeros(6, dtype=float)
    set_joint_angles(data, thetas_init)
    mujoco.mj_forward(model, data)
    
    # 初期ターゲットに移動
    target_pose = np.concatenate([initial_target_pos, initial_target_orientation])
    current_thetas = inverse_kinematics(thetas_init, target_pose)
    set_joint_angles(data, current_thetas)
    mujoco.mj_forward(model, data)
    
    # ビューアを開始
    print("\nインタラクティブビューアを起動します")
    print("操作方法:")
    print("  マウス左ドラッグ: 視点回転")
    print("  マウス右ドラッグ: 視点移動")
    print("  マウスホイール: ズーム")
    print("  Esc: 終了")
    
    viewer.launch(model, data)
    
    return True

###############################################################################
# 6) メイン関数
###############################################################################
def main():
    """
    コマンドライン引数を解析してデモを実行するメイン関数
    """
    parser = argparse.ArgumentParser(description='ロボットアームデモ - ターゲット位置と姿勢の可視化')
    
    # ターゲット位置の引数
    parser.add_argument('--x', type=float, default=0.1, help='目標位置 X (デフォルト: 0.1)')
    parser.add_argument('--y', type=float, default=-0.2, help='目標位置 Y (デフォルト: -0.2)')
    parser.add_argument('--z', type=float, default=0.15, help='目標位置 Z (デフォルト: 0.2)')
    
    # ターゲット姿勢の引数 (ラジアン単位)
    parser.add_argument('--alpha', type=float, default=0.0, help='目標姿勢 Alpha - X軸周りの回転 (デフォルト: 0.0)')
    parser.add_argument('--beta', type=float, default=-np.pi/4, help='目標姿勢 Beta - Y軸周りの回転 (デフォルト: 0.0)')
    parser.add_argument('--gamma', type=float, default=0.0, help='目標姿勢 Gamma - Z軸周りの回転 (デフォルト: 0.0)')
    
    # 出力ファイル
    parser.add_argument('--output', type=str, default="robot_pose_trajectory.mp4", help='出力ビデオファイル (デフォルト: robot_pose_trajectory.mp4)')
    
    # インタラクティブモードフラグ
    parser.add_argument('--interactive', action='store_true', help='インタラクティブモードで実行 (デフォルト: False)')
    
    args = parser.parse_args()
    
    # ターゲット位置と姿勢の配列を作成
    target_pos = np.array([args.x, args.y, args.z])
    target_orientation = np.array([args.alpha, args.beta, args.gamma])
    
    # 適切なモードで実行
    if args.interactive:
        success = run_interactive_mode(target_pos, target_orientation)
    else:
        success = run_robot_demo(target_pos, target_orientation, args.output)
    
    if success:
        print("デモが正常に完了しました")
    else:
        print("デモの実行中にエラーが発生しました")

if __name__ == "__main__":
    main()
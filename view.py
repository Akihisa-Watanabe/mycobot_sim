import mujoco
import numpy as np
from mujoco.glfw import glfw
import time

def get_jacobian(model, data):
    """ヤコビ行列の計算"""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
    
    return jacp

def get_end_effector_pos(model, data):
    """エンドエフェクタの現在位置を取得"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    return data.xpos[body_id]

def solve_ik_to_target(model, data, target_pos, max_iterations=1000, threshold=0.001):
    """目標位置に最も近い到達可能な姿勢を計算"""
    damping = 0.5
    step_size = 0.01
    
    for i in range(max_iterations):
        current_pos = get_end_effector_pos(model, data)
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)
        
        if error_norm < threshold:
            print(f"Target reached within threshold after {i} iterations")
            return True
            
        J = get_jacobian(model, data)
        JT = J.T
        lambda_squared = damping ** 2
        dq = step_size * (JT @ np.linalg.solve(J @ JT + lambda_squared * np.eye(3), error))
        
        # 関節角度の更新
        data.qpos[:model.nv] += dq
        mujoco.mj_forward(model, data)
        
        # 5回ごとに進捗を表示
        if i % 5 == 0:
            print(f"\rIteration {i}: Error = {error_norm:.4f}", end="")
    
    print("\nMaximum iterations reached")
    return False

def main():
    # 目標位置の設定
    target_position = np.array([0.3, -0.3, 0.1])  # 目標位置を設定
    
    # Initialize GLFW
    glfw.init()

    # Load the model from the XML file
    model = mujoco.MjModel.from_xml_path("config/xml/mycobot_280jn_mujoco.xml")
    data = mujoco.MjData(model)

    # IKを解いて目標位置に最も近い姿勢を計算
    print("Calculating nearest reachable position...")
    solve_ik_to_target(model, data, target_position)
    
    # 到達した位置を表示
    final_pos = get_end_effector_pos(model, data)
    print(f"\nTarget position: {target_position}")
    print(f"Reached position: {final_pos}")
    print(f"Final error: {np.linalg.norm(target_position - final_pos):.4f}")

    # Create a window
    window = glfw.create_window(1200, 900, "MyCobot Target Position", None, None)
    glfw.make_context_current(window)

    # Create camera
    cam = mujoco.MjvCamera()
    cam.distance = 0.8
    cam.azimuth = 90
    cam.elevation = -15
    cam.lookat[0] = 0
    cam.lookat[1] = 0
    cam.lookat[2] = 0.3

    # Create scene and context
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # ライティングとビジュアライゼーション設定
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    # 目標位置のマーカーを追加するための設定
    scene.ngeom += 1  # 新しいジオメトリ用のスペースを確保
    g = scene.geoms[scene.ngeom-1]
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = [0.02, 0, 0]  # マーカーのサイズ
    g.pos[:] = target_position  # マーカーの位置
    g.rgba[:] = [1, 0, 0, 1]   # 赤色

    # Create viewport
    viewport = mujoco.MjrRect(0, 0, 1200, 900)

    print("\nViewer started. Press Esc to close.")

    # Visualization loop
    while not glfw.window_should_close(window):
        # Clear buffers and render
        glfw.poll_events()
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        viewport.width = max(1, viewport.width)
        viewport.height = max(1, viewport.height)

        # Update scene and render
        mujoco.mjv_updateScene(
            model, data, opt,
            None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene
        )
        
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
        mujoco.mjr_render(viewport, scene, context)

        # Swap buffers
        glfw.swap_buffers(window)

        # Check for escape key
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()
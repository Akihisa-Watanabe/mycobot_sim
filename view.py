import mujoco
import numpy as np
from mujoco.glfw import glfw
import time

def main():
    # Initialize GLFW
    glfw.init()

    # Load the model from the XML file
    model = mujoco.MjModel.from_xml_path("config/xml/mycobot_280jn_mujoco.xml")
    data = mujoco.MjData(model)

    # Create a window
    window = glfw.create_window(1200, 900, "MyCobot Visualization", None, None)
    glfw.make_context_current(window)

    # Create camera
    cam = mujoco.MjvCamera()
    cam.distance = 0.8  # カメラ距離を近づけました
    cam.azimuth = 90   # 水平回転
    cam.elevation = -15  # 垂直回転
    cam.lookat[0] = 0  # 注視点のx座標
    cam.lookat[1] = 0  # 注視点のy座標
    cam.lookat[2] = 0.3  # 注視点のz座標（少し上向きに）

    # Create scene and context
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # ライティングの設定
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1  # 影を有効化
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1  # 静的なジオメトリを表示

    # Create viewport
    viewport = mujoco.MjrRect(0, 0, 1200, 900)

    print("Viewer started. Press Esc to close.")

    # Animation loop
    while not glfw.window_should_close(window):
        time_prev = data.time

        # Advance simulation by one step
        mujoco.mj_step(model, data)

        # Clear buffers and render
        glfw.poll_events()
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        viewport.width = max(1, viewport.width)
        viewport.height = max(1, viewport.height)

        # Update scene and render
        mujoco.mj_forward(model, data)
        
        # シーンの更新時にライティングオプションを適用
        mujoco.mjv_updateScene(
            model, data, opt,
            None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene
        )
        
        # レンダリング設定
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
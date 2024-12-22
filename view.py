import mujoco
import numpy as np
from mujoco.glfw import glfw
from PIL import Image

def main():
    # Initialize GLFW
    if not glfw.init():
        raise Exception("Failed to initialize GLFW")

    # Load the model from the XML file
    model = mujoco.MjModel.from_xml_path("config/xml/mycobot_280jn_mujoco.xml")
    data = mujoco.MjData(model)
    # if model.nq >= 6:
    #     # 90 degrees in radians
    #     data.qpos[1] = np.pi / 2
    #     data.qpos[3] = np.pi / 2
    #     data.qpos[4] = -np.pi / 2
    mujoco.mj_forward(model, data)

    # Create a window
    width, height = 1200, 1200
    window = glfw.create_window(width, height, "MyCobot Visualization", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window")

    glfw.make_context_current(window)

    # Create camera and scene
    cam = mujoco.MjvCamera()

    cam.distance = 0.65
    cam.azimuth = -20
    cam.elevation = -15.5
    cam.lookat[0] = 0
    cam.lookat[1] = 0
    cam.lookat[2] = 0.25

    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    viewport = mujoco.MjrRect(0, 0, width, height)

    # 一度描画してからスクリーンショットを撮る
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
    mujoco.mjr_render(viewport, scene, context)

    # スクリーンショット用のバッファ作成
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)

    # ピクセル読み込み（現在のフレーム）
    mujoco.mjr_readPixels(rgb, depth, viewport, context)

    # OpenGLは上下反転なので反転
    rgb = np.flipud(rgb)

    # 画像として保存
    img = Image.fromarray(rgb)
    img.save("robot_closeup.png")
    print("Screenshot saved as robot_closeup.png")

    print("\nViewer started. Press Esc to close.")
    # 表示ループ
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # 再描画
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        viewport.width = max(1, viewport.width)
        viewport.height = max(1, viewport.height)
        
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)

        # ESCキーで終了
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()

if __name__ == "__main__":
    main()

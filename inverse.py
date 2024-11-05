import mujoco
import numpy as np
import time
import mediapy as media

class RobotController:
    def __init__(self, model_path):
        # MuJoCoモデルの読み込み
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初期化
        self.damping = 0.5  # ダンピングを小さくして動作を速く
        self.max_vel = 10.0  # 最大速度を上げる
        self.nv = self.model.nv
        
        # レンダリング用の設定
        self.render_resolution = (480, 360)
        self.frames = []
        self.renderer = mujoco.Renderer(self.model, height=self.render_resolution[1], 
                                      width=self.render_resolution[0])
        
        # カメラの設定
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 1.5
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.lookat = np.array([0, 0, 0.5])
        
        # シーンとオプションの初期化
        self.scene = mujoco.MjvScene(self.model, maxgeom=100)
        self.opt = mujoco.MjvOption()
        
        # 軌道の生成
        self.trajectory = self.generate_trajectory()
        self.trajectory_index = 0
        self.target_pos = self.trajectory[0]
        
        # 軌道追従の制御パラメータ
        self.position_threshold = 0.05  # 位置誤差の閾値を大きく
        self.time_per_point = 0.05     # 各点での滞在時間を短く
        self.time_at_current_point = 0
        
        # シミュレーション完了フラグ
        self.is_running = True
    
    def generate_trajectory(self):
        """単純化された軌道生成（少ない点数の四角形）"""
        points = [
            [0.3, 0.2, 0.5],   # 右前
            [0.3, -0.2, 0.5],  # 右後
            [0.1, -0.2, 0.5],  # 左後
            [0.1, 0.2, 0.5],   # 左前
        ]
        
        # 各辺を5点で補間（合計20点）
        trajectory = []
        for i in range(len(points)):
            start = np.array(points[i])
            end = np.array(points[(i+1) % len(points)])
            for t in np.linspace(0, 1, 5):
                trajectory.append(start * (1-t) + end * t)
                
        return trajectory

    
    def get_jacobian(self):
        """ヤコビ行列の計算"""
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[body_id], body_id)
        
        return jacp
    
    def get_end_effector_pos(self):
        """エンドエフェクタの現在位置を取得"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
        return self.data.xpos[body_id]
    
    def solve_ik(self):
        """インバース・キネマティクスの計算"""
        current_pos = self.get_end_effector_pos()
        error = self.target_pos - current_pos
        
        J = self.get_jacobian()
        JT = J.T
        lambda_squared = self.damping ** 2
        joint_vel = JT @ np.linalg.solve(J @ JT + lambda_squared * np.eye(3), error)
        
        norm = np.linalg.norm(joint_vel)
        if norm > self.max_vel:
            joint_vel = joint_vel * self.max_vel / norm
            
        return joint_vel, np.linalg.norm(error)
    
    def update_target(self, dt):
        """目標位置の更新"""
        current_error = np.linalg.norm(self.target_pos - self.get_end_effector_pos())
        self.time_at_current_point += dt
        
        if current_error < self.position_threshold and self.time_at_current_point >= self.time_per_point:
            self.trajectory_index = (self.trajectory_index + 1) % len(self.trajectory)
            self.target_pos = self.trajectory[self.trajectory_index]
            self.time_at_current_point = 0
            return True
        return False

    def run(self):
        """メインループ"""
        print("Starting trajectory following...")
        
        while self.is_running:
            time_start = time.time()
            
            try:
                # IKを解いて関節速度を取得
                joint_vel, error = self.solve_ik()
                self.data.qvel[:self.nv] = joint_vel
                
                # シミュレーションステップ
                mujoco.mj_step(self.model, self.data)
                
                # 目標位置の更新
                point_updated = self.update_target(1/60.0)
                
                try:
                    # シーンの更新とレンダリング
                    mujoco.mjv_updateScene(
                        self.model,
                        self.data,
                        self.opt,
                        None,
                        self.cam,
                        mujoco.mjtCatBit.mjCAT_ALL.value,
                        self.scene
                    )
                    self.renderer.update_scene(self.data, self.cam)
                    pixels = self.renderer.render()
                    self.frames.append(pixels)
                except Exception as render_error:
                    print(f"\nRendering error: {render_error}")
                    raise render_error
                
                # 現在の状態を表示
                current_pos = self.get_end_effector_pos()
                print(f"\rPoint: {self.trajectory_index:3d}/{len(self.trajectory):3d}, "
                      f"Error: {error:.3f}m, "
                      f"Target: {self.target_pos.round(3)}, "
                      f"Current: {current_pos.round(3)}", end="")
                
                # 1周完了で終了
                if self.trajectory_index == len(self.trajectory) - 1 and error < self.position_threshold:
                    self.is_running = False
                
            except Exception as e:
                print(f"\nError during simulation: {e}")
                break
            
            # フレームレート制御
            time_to_wait = 1/60 - (time.time() - time_start)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
        
        # 動画の保存
        if self.frames:
            print("\nSaving video...")
            try:
                import imageio
                imageio.mimsave('robot_trajectory.mp4', self.frames, fps=60)
                print("Video saved as robot_trajectory.mp4")
            except Exception as e:
                print(f"Error saving video: {e}")

def main():
    model_path = "config/xml/mycobot_280jn_mujoco.xml"
    try:
        controller = RobotController(model_path)
        controller.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
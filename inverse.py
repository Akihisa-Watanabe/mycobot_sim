import mujoco
import numpy as np
import time
import mediapy as media

class RobotController:
    def __init__(self, model_path, target_position=None):
        try:
            # MuJoCo model loading with error handling
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            
            # Set model parameters for better visualization
            self.model.vis.global_.offwidth = 1920
            self.model.vis.global_.offheight = 1080
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")
            
        # Initialize parameters
        self.damping = 0.01
        self.max_vel = 500.0
        self.nv = self.model.nv
        
        # Enhanced rendering settings
        self.duration = float(7)
        self.framerate = int(60)
        self.frames = []
        self.render_width = 1920
        self.render_height = 1080
        
        # Target position setup with validation
        if target_position is not None:
            self.target_pos = np.array(target_position, dtype=np.float64)
        else:
            self.target_pos = np.array([0.3, 0.0, 0.5], dtype=np.float64)
            
        # Control parameters
        self.position_threshold = 0.1
        self.time_at_target = 0.0
        self.time_threshold = 0.5
        
        # Simulation state
        self.is_running = True
        
        # バリデーションを実行
        self.validate_initial_state()
    
    def validate_initial_state(self):
        """初期状態の検証"""
        if self.target_pos.shape != (3,):
            raise ValueError("Target position must be a 3D vector")
        if not isinstance(self.duration, (int, float)) or self.duration <= 0:
            raise ValueError("Duration must be a positive number")
        if not isinstance(self.framerate, int) or self.framerate <= 0:
            raise ValueError("Framerate must be a positive integer")
    
    def get_jacobian(self):
        """ヤコビ行列の計算"""
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
            mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[body_id], body_id)
        except Exception as e:
            raise RuntimeError(f"Failed to compute Jacobian: {e}")
        
        return jacp
    
    def get_end_effector_pos(self):
        """エンドエフェクタの現在位置を取得"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
            return np.array(self.data.xpos[body_id])
        except Exception as e:
            raise RuntimeError(f"Failed to get end effector position: {e}")
    
    def solve_ik(self):
        """インバース・キネマティクスを解く"""
        try:
            current_pos = self.get_end_effector_pos()
            error = self.target_pos - current_pos
            
            J = self.get_jacobian()
            JT = J.T
            lambda_squared = self.damping ** 2
            
            JJT = J @ JT
            damped_inversion = JJT + lambda_squared * np.eye(3)
            joint_vel = JT @ np.linalg.solve(damped_inversion, error)
            
            norm = np.linalg.norm(joint_vel)
            if norm > self.max_vel:
                joint_vel = joint_vel * self.max_vel / norm
                
            return joint_vel, float(np.linalg.norm(error))
            
        except Exception as e:
            raise RuntimeError(f"IK solving failed: {e}")
    
    def check_target_reached(self, dt, error):
        """目標位置到達判定"""
        try:
            dt = float(dt)
            error = float(error)
            
            if error < self.position_threshold:
                self.time_at_target += dt
                return self.time_at_target >= self.time_threshold
            else:
                self.time_at_target = 0
            return False
        except Exception as e:
            raise RuntimeError(f"Target check failed: {e}")

    def run(self):
        """メインシミュレーションループ"""
        print("Moving to target position...")
        print(f"Target position: {self.target_pos}")
        
        try:
            # データリセット
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            
            # レンダラーの作成（高解像度設定）
            renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
            
            try:
                # 追加のレンダリング設定
                renderer.enable_shadows = True
                
                last_render_time = 0.0
                
                while self.is_running and self.data.time < self.duration:
                    time_start = time.time()
                    
                    # IKを解いて関節速度を更新
                    joint_vel, error = self.solve_ik()
                    self.data.qvel[:self.nv] = joint_vel
                    
                    # シミュレーションステップ
                    mujoco.mj_step(self.model, self.data)
                    
                    # フレーム取得
                    current_frame_time = int(self.data.time * self.framerate)
                    if current_frame_time > last_render_time:
                        renderer.update_scene(self.data)
                        pixels = renderer.render()
                        self.frames.append(pixels)
                        last_render_time = current_frame_time
                    
                    # 状態更新
                    current_pos = self.get_end_effector_pos()
                    print(f"\rTime: {self.data.time:.2f}s, Error: {error:.3f}m, Current: {current_pos.round(3)}", end="")
                    
                    # 目標位置到達判定
                    if self.check_target_reached(1.0/self.framerate, error):
                        print("\nTarget position reached!")
                        break
                    
                    # フレームレート制御
                    time_to_wait = 1.0/self.framerate - (time.time() - time_start)
                    if time_to_wait > 0:
                        time.sleep(time_to_wait)
            
            finally:
                renderer.close()
            
            # ビデオ処理（高品質設定）
            if self.frames:
                print("\nDisplaying video...")
                media.show_video(self.frames, fps=self.framerate)
                
                print("\nSaving video...")
                try:
                    import imageio
                    imageio.mimsave('robot_movement.mp4', self.frames, fps=self.framerate, 
                                  quality=10, bitrate="32M")
                    print("Video saved as robot_movement.mp4")
                except Exception as e:
                    print(f"Error saving video: {e}")
                    
        except Exception as e:
            print(f"\nSimulation error: {e}")
            raise

def main():
    model_path = "config/xml/mycobot_280jn_mujoco.xml"
    target_position = [0.19425692, -0.18526572, 0.13998125]
    
    try:
        controller = RobotController(model_path, target_position)
        controller.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
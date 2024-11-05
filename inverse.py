# MuJoCo Robot Controller with Inverse Kinematics
# This code implements a robot controller using MuJoCo physics engine with damped least squares IK.
# The controller simulates robot movement, logs trajectory data, and creates visualization.

import mujoco
import numpy as np
import time
import mediapy as media

class RobotController:
    """
    Robot controller class implementing inverse kinematics control for a MuJoCo robot model.
    Handles simulation, visualization, and trajectory logging.
    """
    def __init__(self, model_path, target_position=None):
        """
        Initialize robot controller with model and parameters.
        
        Args:
            model_path (str): Path to MuJoCo XML model file
            target_position (list/array, optional): 3D target position [x, y, z]
        """
        try:
            # Load MuJoCo model and create data instance
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            
            # Set visualization resolution
            self.model.vis.global_.offwidth = 1920
            self.model.vis.global_.offheight = 1080
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")
            
        # IK control parameters
        self.damping = 0.1          # Damping coefficient for numerical stability
        self.step_size = 0.01       # Step size for IK convergence
        self.max_iterations = 100    # Maximum number of IK iterations
        self.nv = self.model.nv     # Number of degrees of freedom
        
        # Rendering parameters
        self.duration = float(7)     # Simulation duration in seconds
        self.framerate = int(60)     # Rendering framerate
        self.frames = []             # Store rendered frames
        self.render_width = 1920     # Render width in pixels
        self.render_height = 1080    # Render height in pixels
        
        # Set and validate target position
        if target_position is not None:
            self.target_pos = np.array(target_position, dtype=np.float64)
        else:
            self.target_pos = np.array([0.3, 0.0, 3.5], dtype=np.float64)
            
        # Control thresholds
        self.position_threshold = 0.01  # Position error threshold (meters)
        self.time_at_target = 0.0       # Time spent at target position
        self.time_threshold = 0.5       # Required time at target for success
        
        # Simulation state
        self.is_running = True
        
        # Validate initialization parameters
        self.validate_initial_state()

        # Initialize trajectory logging arrays
        self.time_log = []              # Time stamps
        self.joint_angles_log = []       # Joint angles over time
        self.end_effector_pos_log = []   # End-effector positions
        self.error_log = []             # Position errors
        self.step_size_log = []         # Step sizes used
        
        # Output file for trajectory data
        self.log_filename = "robot_trajectory.npz"
    
    def log_state(self, time, joint_angles, error):
        """
        Log current robot state for analysis.
        Records time, joint angles, end-effector position, error, and step size.
        """
        self.time_log.append(time)
        self.joint_angles_log.append(joint_angles.copy())
        self.end_effector_pos_log.append(self.get_end_effector_pos().copy())
        self.error_log.append(error)
        self.step_size_log.append(self.step_size)

    def save_logs(self):
        """Save logged trajectory data to NPZ file for later analysis."""
        try:
            np.savez(self.log_filename,
                    time=np.array(self.time_log),
                    joint_angles=np.array(self.joint_angles_log),
                    end_effector_pos=np.array(self.end_effector_pos_log),
                    error=np.array(self.error_log),
                    step_size=np.array(self.step_size_log),
                    target_position=self.target_pos)
            
            print(f"\nTrajectory data saved to {self.log_filename}")
        except Exception as e:
            print(f"Error saving trajectory data: {e}")

    def validate_initial_state(self):
        """Validate initialization parameters and their types."""
        if self.target_pos.shape != (3,):
            raise ValueError("Target position must be a 3D vector")
        if not isinstance(self.duration, (int, float)) or self.duration <= 0:
            raise ValueError("Duration must be a positive number")
        if not isinstance(self.framerate, int) or self.framerate <= 0:
            raise ValueError("Framerate must be a positive integer")
    
    def get_jacobian(self):
        """
        Calculate the Jacobian matrix for the end-effector.
        Returns the positional Jacobian (3 x nv matrix).
        """
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
            mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[body_id], body_id)
        except Exception as e:
            raise RuntimeError(f"Failed to compute Jacobian: {e}")
        
        return jacp
    
    def get_end_effector_pos(self):
        """Get current end-effector position in world coordinates."""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
            return np.array(self.data.xpos[body_id])
        except Exception as e:
            raise RuntimeError(f"Failed to get end effector position: {e}")
    
    def solve_ik(self):
        """
        Solve inverse kinematics using damped least squares method.
        Returns new joint angles and position error norm.
        """
        try:
            current_pos = self.get_end_effector_pos()
            error = self.target_pos - current_pos
            error_norm = float(np.linalg.norm(error))
            
            # Get current joint configuration
            current_q = np.copy(self.data.qpos[:self.nv])
            
            # Compute IK solution using damped least squares
            J = self.get_jacobian()
            JT = J.T
            lambda_squared = self.damping ** 2
            
            # Calculate pseudo-inverse with damping
            JJT = J @ JT
            damped_inversion = JJT + lambda_squared * np.eye(3)
            delta_q = self.step_size * (JT @ np.linalg.solve(damped_inversion, error))
            
            # Update joint angles with limits
            new_q = current_q + delta_q
            
            # Apply joint limits
            for i in range(self.nv):
                new_q[i] = np.clip(new_q[i], 
                                 self.model.jnt_range[i][0], 
                                 self.model.jnt_range[i][1])
            
            return new_q, error_norm
            
        except Exception as e:
            raise RuntimeError(f"IK solving failed: {e}")
    
    def check_target_reached(self, dt, error):
        """
        Check if target position has been reached and maintained.
        Returns True if position error is below threshold for sufficient time.
        """
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
        """
        Main simulation loop.
        Executes IK control, renders visualization, and logs trajectory data.
        """
        print("Moving to target position...")
        print(f"Target position: {self.target_pos}")
        previous_error = None
        try:
            # Initialize simulation state
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
            
            try:
                # Setup renderer
                renderer.enable_shadows = True
                last_render_time = 0.0
                
                # Main control loop
                while self.is_running and self.data.time < self.duration:
                    time_start = time.time()
                    
                    # Compute and apply IK solution
                    new_q, error = self.solve_ik()
                    self.log_state(self.data.time, new_q, error)
                    
                    previous_error = error
                    self.data.qpos[:self.nv] = new_q
                    
                    # Update forward kinematics
                    mujoco.mj_forward(self.model, self.data)
                    
                    # Render and save frame if needed
                    current_frame_time = int(self.data.time * self.framerate)
                    if current_frame_time > last_render_time:
                        renderer.update_scene(self.data)
                        pixels = renderer.render()
                        self.frames.append(pixels)
                        last_render_time = current_frame_time
                    
                    # Print current state
                    current_pos = self.get_end_effector_pos()
                    print(f"\rTime: {self.data.time:.2f}s, Error: {error:.3f}m, Current: {current_pos.round(3)}", end="")
                    
                    # Check termination condition
                    if self.check_target_reached(1.0/self.framerate, error):
                        print("\nTarget position reached!")
                        break
                    
                    # Update simulation time
                    self.data.time += 1.0/self.framerate
                    
                    # Maintain framerate
                    time_to_wait = 1.0/self.framerate - (time.time() - time_start)
                    if time_to_wait > 0:
                        time.sleep(time_to_wait)
            
            finally:
                renderer.close()
            
            # Save trajectory data
            self.save_logs()
            
            # Process and save video
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
    """
    Main entry point of the program.
    Sets up and runs the robot controller with specified target position.
    """
    model_path = "config/xml/mycobot_280jn_mujoco.xml"
    target_position = [0.19425692, -0.18526572, 0.2]
    
    try:
        controller = RobotController(model_path, target_position)
        controller.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
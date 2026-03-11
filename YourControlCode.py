import mujoco
import numpy as np
from BoxControlHandler import BoxControlHandle
import os
import csv

class YourCtrl:

    # Constructor: Initialize model, data, box controller, logging, and starting state
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d
        self.init_qpos = d.qpos.copy()  # Save initial joint positions

        # Instantiate box controller
        self.boxCtrlhdl = BoxControlHandle(self.m, self.d)
        self.boxCtrlhdl.set_difficulty(0.2) # set task difficulty level
        print("Initial Position(qpos):", self.d.qpos[:6])

        # Initialize internal state
        self.start_time = None
        self.q_target = None

        # Initialize logging for sensor data
        self.last_print_time = 0.0
        self.sensor_log = []
        self.log_file_path = "sensor_log.csv"

        # Remove any old log file to avoid appending to previous results
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

    # Newton-Raphson IK solver: Iteratively solve joint angles to reach target pose
    def newton_raphson(self, target_pos, target_quat, num_steps, lr):
        original_qpos = self.d.qpos[:6].copy()
        q_hist = []       # Store joint history
        error_hist = []   # Store pose error history

        for steps in range(num_steps):
            # Get current end-effector position and orientation
            ee_pos = self.boxCtrlhdl._get_ee_position()
            ee_quat = self.boxCtrlhdl._get_ee_orientation()

            # Compute position and orientation error
            pos_error = target_pos - ee_pos
            quat_err = self.boxCtrlhdl.quat_multiply(target_quat, self.boxCtrlhdl.quat_inv(ee_quat))
            rot_err = self.boxCtrlhdl.quat2so3(quat_err)
            error = np.hstack([pos_error, rot_err])  # 6D error vector

            error_hist.append(error)

            # Compute geometric Jacobian at current pose
            jacp = np.zeros((3, self.m.nv))
            jacr = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jacp, jacr, ee_pos, self.boxCtrlhdl.ee_id)
            J = np.vstack([jacp[:, :6], jacr[:, :6]])

            # Solve for joint update
            try:
                dq = np.linalg.solve(J, error)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(J) @ error  # Use pseudoinverse if Jacobian is singular

            q_hist.append(self.d.qpos[:6].copy())

            # Apply joint update
            self.d.qpos[:6] += lr * dq
            mujoco.mj_forward(self.m, self.d)  # Update kinematics

        # Restore original joint state
        self.d.qpos[:6] = original_qpos
        mujoco.mj_forward(self.m, self.d)

        return q_hist, error_hist

    # Main control loop called at each simulation step
    def update(self):
        # Set initial simulation time
        if self.start_time is None:
            self.start_time = self.d.time

        # Read current box sensor positions
        box_sensor_names = ["mould_pos_sensor1", "mould_pos_sensor2", "mould_pos_sensor3", "mould_pos_sensor4"]
        box_sensors = []
        for name in box_sensor_names:
            idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            pos = self.d.sensordata[idx*3 : idx*3+3]
            box_sensors.append(pos)

        # Compute target box orientation and rotate 90 degrees around Y axis
        target_quat, _ = self.boxCtrlhdl.box_orientation(*box_sensors)
        target_quat = self.boxCtrlhdl.rotate_quat_90_y(target_quat)

        # Prior Option: Compute position error using custom insertion logic
        pos_err = self.compute_custom_pos_err()

        ### Or Use given function get_EE_pos_err, but performs not that well
        # pos_err = self.boxCtrlhdl.get_EE_pos_err()

        target_pos = self.boxCtrlhdl._get_ee_position() + pos_err

        # Solve IK to get target joint configuration
        q_hist, err_hist = self.newton_raphson(target_pos, target_quat, num_steps=10, lr=0.8)
        if q_hist:
            q_target = q_hist[-1]
        else:
            print("[update] IK failed — reverting to home")
            q_target = self.q_home.copy()

        # Log sensor data every 0.1 seconds
        if self.d.time - self.last_print_time >= 0.1:
            row = [f"{self.d.time:.3f}"]
            sensor_names = box_sensor_names
            for name in sensor_names:
                idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
                pos = self.d.sensordata[idx*3 : idx*3+3]
                row.extend([f"{x:.6f}" for x in pos])
            self.sensor_log.append(row)
            self.last_print_time = self.d.time

            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["time", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z",
                                     "s3_x", "s3_y", "s3_z", "s4_x", "s4_y", "s4_z"])
                writer.writerow(row)

        # PD control: apply torque to drive joints to target configuration
        for i in range(6):
            self.d.ctrl[i] = 150.0 * (q_target[i] - self.d.qpos[i]) - 5.2 * self.d.qvel[i]

    # Custom function to compute desired EE position error toward an insertion point
    def compute_custom_pos_err(self):
        try:
            # Read sensor positions
            s1 = self.d.sensordata[0:3]
            s2 = self.d.sensordata[3:6]
            s3 = self.d.sensordata[6:9]
            s4 = self.d.sensordata[9:12]

            # Validate sensor data
            if (np.linalg.norm(s1) < 1e-5 or 
                np.linalg.norm(s2 - s1) < 1e-6 or 
                np.linalg.norm(s3 - s1) < 1e-6):
                print("[WARN] sensor data not ready")
                return np.zeros(3)

            # Compute box center
            box_center = (s1 + s2 + s3 + s4) / 4.0

            # Build local frame using sensor points
            v1 = s2 - s1
            v2 = s3 - s1
            z = np.cross(v1, v2)
            z = z / np.linalg.norm(z) if np.linalg.norm(z) > 1e-6 else np.array([0, 0, 1])
            x = s4 - s1
            x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1, 0, 0])
            y = np.cross(z, x)
            R = np.column_stack([x, y, z])  # 3×3 rotation matrix

            # Offset insertion target by 2cm in X direction (in local frame)
            insertion_target = box_center + R @ np.array([0.02, 0.0, 0.0])

            # Compute EE position error
            ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_box")
            ee_pos = self.d.xpos[ee_id].copy()

            return insertion_target - ee_pos
        except Exception as e:
            print("[ERROR] compute_custom_pos_err failed:", e)
            return np.zeros(3)

"""Drop-in replacement for FrankaRobot that uses avantbot's CRISP controllers instead of Polymetis.

Provides the same zerorpc interface so the DROID laptop container works unchanged.
"""

import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from avantbot.robot.robot import Robot
from avantbot.robot.gripper.gripper import Gripper

from droid.misc.transformations import add_poses, pose_diff
from droid.robot_ik.robot_ik_solver_lite import RobotIKSolver


class CrispFrankaRobot:
    """FrankaRobot-compatible interface backed by avantbot CRISP controllers.

    Implements the same methods as droid.franka.robot.FrankaRobot so it can be
    served via zerorpc as a drop-in replacement for the Polymetis-based version.
    """

    def __init__(self):
        self._robot = None
        self._gripper = None
        self._ik_solver = None
        self._controller_not_loaded = False
        self._joint_velocities = [0.0] * 7
        self._joint_efforts = [0.0] * 7
        self._active_controller = None

    def launch_controller(self):
        """No-op: CRISP controllers are started separately via Docker."""
        pass

    def launch_robot(self):
        """Connect to the CRISP controllers via avantbot's Robot and Gripper."""
        if not rclpy.ok():
            rclpy.init()

        self._robot = Robot.from_yaml("fr3", name="droid_crisp_client")
        self._robot.wait_until_ready(timeout=30.0)

        self._robot.node.create_subscription(
            JointState,
            "joint_states",
            self._joint_state_callback,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )

        try:
            self._gripper = Gripper.from_yaml(
                "gripper_franka", node=self._robot.node, spin_node=False
            )
            self._gripper.wait_until_ready(timeout=10.0)
        except Exception:
            self._gripper = None

        self._ik_solver = RobotIKSolver()

        self._ensure_controller("joint_impedance_controller")
        self._controller_not_loaded = False

    def kill_controller(self):
        """Shut down the ROS2 connection."""
        if self._robot is not None:
            self._robot.shutdown()

    def _joint_state_callback(self, msg: JointState):
        """Capture joint velocities and efforts from the JointState topic."""
        joint_names = self._robot.config.joint_names
        vel = [0.0] * len(joint_names)
        eff = [0.0] * len(joint_names)
        prefix = self._robot._prefix
        for i, name in enumerate(msg.name):
            clean = name.removeprefix(prefix)
            if clean in joint_names:
                idx = joint_names.index(clean)
                if msg.velocity:
                    vel[idx] = msg.velocity[i]
                if msg.effort:
                    eff[idx] = msg.effort[i]
        self._joint_velocities = vel
        self._joint_efforts = eff

    def _ensure_controller(self, controller_name: str):
        """Switch to the given controller if not already active."""
        if self._active_controller == controller_name:
            return
        try:
            self._robot.controller_switcher_client.switch_controller(controller_name)
            self._active_controller = controller_name
        except Exception:
            pass

    def get_joint_positions(self):
        return self._robot.joint_values.tolist()

    def get_joint_velocities(self):
        return list(self._joint_velocities)

    def get_gripper_position(self):
        """Return gripper position in DROID convention (0=open, 1=closed)."""
        if self._gripper is None:
            return 0.0
        return 1.0 - self._gripper.value

    def get_gripper_state(self):
        return self.get_gripper_position()

    def get_ee_pose(self):
        """Return EE pose as [x, y, z, rx, ry, rz] with euler angles."""
        pose = self._robot.end_effector_pose
        pos = pose.position.tolist()
        euler = pose.orientation.as_euler("xyz").tolist()
        return pos + euler

    def get_robot_state(self):
        """Return (state_dict, timestamp_dict) matching FrankaRobot's format."""
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()
        gripper_pos = self.get_gripper_position()
        ee_pose = self.get_ee_pose()
        efforts = list(self._joint_efforts)

        now = self._robot.node.get_clock().now()
        sec, nsec = now.seconds_nanoseconds()

        state_dict = {
            "cartesian_position": ee_pose,
            "gripper_position": gripper_pos,
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "joint_torques_computed": efforts,
            "prev_joint_torques_computed": efforts,
            "prev_joint_torques_computed_safened": efforts,
            "motor_torques_measured": efforts,
            "prev_controller_latency_ms": 0.0,
            "prev_command_successful": True,
        }

        timestamp_dict = {
            "robot_timestamp_seconds": sec,
            "robot_timestamp_nanos": nsec,
        }

        return state_dict, timestamp_dict

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        action_dict = self.create_action_dict(
            command, action_space=action_space, gripper_action_space=gripper_action_space
        )
        self.update_joints(action_dict["joint_position"], velocity=False, blocking=blocking)
        self.update_gripper(action_dict["gripper_position"], velocity=False, blocking=blocking)
        return action_dict

    def update_pose(self, command, velocity=False, blocking=False):
        if blocking:
            if velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(command)
                command = add_poses(cartesian_delta, curr_pose)
            robot_state = self.get_robot_state()[0]
            joint_velocity = self._ik_solver.cartesian_velocity_to_joint_velocity(
                self._ik_solver.cartesian_delta_to_velocity(
                    pose_diff(command, self.get_ee_pose())
                ),
                robot_state=robot_state,
            )
            joint_delta = self._ik_solver.joint_velocity_to_delta(joint_velocity)
            desired_joints = (joint_delta + np.array(robot_state["joint_positions"])).tolist()
            self.update_joints(desired_joints, velocity=False, blocking=True)
        else:
            if not velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = pose_diff(command, curr_pose)
                command = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)

            robot_state = self.get_robot_state()[0]
            joint_velocity = self._ik_solver.cartesian_velocity_to_joint_velocity(
                command, robot_state=robot_state
            )
            self.update_joints(joint_velocity, velocity=True, blocking=False)

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None):
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)

        command = np.array(command, dtype=np.float64)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + np.array(self.get_joint_positions())

        if blocking:
            self._ensure_controller("joint_trajectory_controller")
            time_to_go = self.adaptive_time_to_go(command)
            self._robot.joint_trajectory_controller_client.send_joint_config(
                self._robot.config.joint_names,
                command.tolist(),
                time_to_goal=time_to_go,
                blocking=True,
            )
            self._ensure_controller("joint_impedance_controller")
            self._robot.wait_until_ready(timeout=5.0)
        else:
            self._ensure_controller("joint_impedance_controller")
            self._robot.set_target_joint(command)

    def update_gripper(self, command, velocity=True, blocking=False):
        if self._gripper is None:
            return

        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        # DROID: 0=open, 1=closed; avantbot: 0=closed, 1=open
        self._gripper.set_target(1.0 - command)

        if blocking:
            time.sleep(1.0)

    def add_noise_to_joints(self, original_joints, cartesian_noise):
        """Add cartesian noise to joint positions via forward+inverse kinematics."""
        original_joints = np.array(original_joints)
        ee_pose = self.get_ee_pose()
        new_pose = add_poses(cartesian_noise, ee_pose)

        robot_state = self.get_robot_state()[0]
        cart_delta = pose_diff(new_pose, ee_pose)
        cart_vel = self._ik_solver.cartesian_delta_to_velocity(cart_delta)
        joint_vel = self._ik_solver.cartesian_velocity_to_joint_velocity(
            cart_vel, robot_state=robot_state
        )
        joint_delta = self._ik_solver.joint_velocity_to_delta(joint_vel)
        noisy_joints = original_joints + joint_delta

        return noisy_joints.tolist()

    def adaptive_time_to_go(self, desired_joint_position, t_min=0.5, t_max=4.0):
        curr = np.array(self.get_joint_positions())
        desired = np.array(desired_joint_position)
        max_displacement = np.max(np.abs(desired - curr))
        time_to_go = max_displacement / 0.5
        return float(np.clip(time_to_go, t_min, t_max))

    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        assert action_space in [
            "cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"
        ]
        if robot_state is None:
            robot_state = self.get_robot_state()[0]
        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"
        assert gripper_action_space in ["velocity", "position"]

        if gripper_action_space == "velocity":
            action_dict["gripper_velocity"] = action[-1]
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(action[-1])
            gripper_position = robot_state["gripper_position"] + gripper_delta
            action_dict["gripper_position"] = float(np.clip(gripper_position, 0, 1))
        else:
            action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
            gripper_delta = action_dict["gripper_position"] - robot_state["gripper_position"]
            gripper_velocity = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
            action_dict["gripper_delta"] = gripper_velocity

        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1]
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                action_dict["cartesian_position"] = add_poses(
                    cartesian_delta, robot_state["cartesian_position"]
                ).tolist()
            else:
                action_dict["cartesian_position"] = action[:-1]
                cartesian_delta = pose_diff(action[:-1], robot_state["cartesian_position"])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                action_dict["cartesian_velocity"], robot_state=robot_state
            ).tolist()
            joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
            action_dict["joint_position"] = (
                joint_delta + np.array(robot_state["joint_positions"])
            ).tolist()

        if "joint" in action_space:
            if velocity:
                action_dict["joint_velocity"] = action[:-1]
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                action_dict["joint_position"] = (
                    joint_delta + np.array(robot_state["joint_positions"])
                ).tolist()
            else:
                action_dict["joint_position"] = action[:-1]
                joint_delta = np.array(action[:-1]) - np.array(robot_state["joint_positions"])
                joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                action_dict["joint_velocity"] = joint_velocity.tolist()

        return action_dict

"""Lightweight IK solver using raw mujoco instead of dm_control/dm_robotics.

Drop-in replacement for RobotIKSolver that avoids the dm_control version
compatibility issues with conda-installed mujoco.
"""

import os

import mujoco
import numpy as np

from droid.misc.parameters import robot_type


class RobotIKSolver:
    def __init__(self):
        self.relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.max_joint_delta = self.relative_max_joint_delta.max()
        self.max_gripper_delta = 0.25
        self.max_lin_delta = 0.075
        self.max_rot_delta = 0.15
        self.control_hz = 15

        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "franka")
        model_file = os.path.join(dir_path, f"{robot_type}.xml")
        self._model = mujoco.MjModel.from_xml_path(model_file)
        self._data = mujoco.MjData(self._model)

        self._nv = self._model.nv
        self._wrist_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "wrist_site"
        )
        self._regularization = 1e-2

    def cartesian_velocity_to_joint_velocity(self, cartesian_velocity, robot_state):
        cartesian_delta = self.cartesian_velocity_to_delta(cartesian_velocity)
        qpos = np.array(robot_state["joint_positions"])
        qvel = np.array(robot_state["joint_velocities"])

        self._data.qpos[:7] = qpos
        self._data.qvel[:7] = qvel
        mujoco.mj_forward(self._model, self._data)

        jacp = np.zeros((3, self._nv))
        jacr = np.zeros((3, self._nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._wrist_body_id)

        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        JtJ = J.T @ J + self._regularization * np.eye(7)
        joint_delta = np.linalg.solve(JtJ, J.T @ cartesian_delta)

        joint_delta = np.clip(
            joint_delta,
            -self.relative_max_joint_delta,
            self.relative_max_joint_delta,
        )

        joint_velocity = self.joint_delta_to_velocity(joint_delta)
        return joint_velocity

    def gripper_velocity_to_delta(self, gripper_velocity):
        gripper_vel_norm = np.linalg.norm(gripper_velocity)
        if gripper_vel_norm > 1:
            gripper_velocity = gripper_velocity / gripper_vel_norm
        gripper_delta = gripper_velocity * self.max_gripper_delta
        return gripper_delta

    def cartesian_velocity_to_delta(self, cartesian_velocity):
        if isinstance(cartesian_velocity, list):
            cartesian_velocity = np.array(cartesian_velocity)
        lin_vel, rot_vel = cartesian_velocity[:3], cartesian_velocity[3:6]
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm
        lin_delta = lin_vel * self.max_lin_delta
        rot_delta = rot_vel * self.max_rot_delta
        return np.concatenate([lin_delta, rot_delta])

    def joint_velocity_to_delta(self, joint_velocity):
        if isinstance(joint_velocity, list):
            joint_velocity = np.array(joint_velocity)
        relative_max_joint_vel = self.joint_delta_to_velocity(self.relative_max_joint_delta)
        max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()
        if max_joint_vel_norm > 1:
            joint_velocity = joint_velocity / max_joint_vel_norm
        joint_delta = joint_velocity * self.max_joint_delta
        return joint_delta

    def gripper_delta_to_velocity(self, gripper_delta):
        return gripper_delta / self.max_gripper_delta

    def cartesian_delta_to_velocity(self, cartesian_delta):
        if isinstance(cartesian_delta, list):
            cartesian_delta = np.array(cartesian_delta)
        cartesian_velocity = np.zeros_like(cartesian_delta)
        cartesian_velocity[:3] = cartesian_delta[:3] / self.max_lin_delta
        cartesian_velocity[3:6] = cartesian_delta[3:6] / self.max_rot_delta
        return cartesian_velocity

    def joint_delta_to_velocity(self, joint_delta):
        if isinstance(joint_delta, list):
            joint_delta = np.array(joint_delta)
        return joint_delta / self.max_joint_delta

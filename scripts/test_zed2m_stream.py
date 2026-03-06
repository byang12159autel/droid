"""Stream the ZED 2 Mini point cloud overlayed on the FR3 URDF in Viser.

Requires the CRISP stack and ZED cameras to be running.

Usage::

    python scripts/test_zed2m_stream.py [--port 8085] [--camera-serial 17605999]
"""

import dataclasses
import os
import time

import numpy as np
import pyzed.sl as sl
import tyro
from avantbot.utils.viser_camera_viewer import ViserCameraViewer, decode_zed_xyzrgba
from droid.robot_env import RobotEnv


@dataclasses.dataclass
class Args:
    port: int = 8085
    camera_serial: str = "17605999"
    urdf_path: str = os.path.join(
        os.path.expanduser("~"),
        "avantbot",
        "robots",
        "franka_re3",
        "crisp_controllers_demos",
        "crisp_controllers_robot_demos",
        "config",
        "fr3",
        "fr3_robotiq.urdf",
    )


def main(args: Args) -> None:
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("RobotEnv created")

    viewer = ViserCameraViewer(port=args.port)

    avantbot_root = os.path.join(os.path.expanduser("~"), "avantbot")
    package_paths = {
        "franka_description": os.path.join(
            os.path.expanduser("~"), "franka_ros2", "franka_description"
        ),
        "robotiq_description": os.path.join(
            avantbot_root,
            "robots",
            "robotiq_2f85",
            "robotiq_ws",
            "src",
            "ros2_robotiq_gripper",
            "robotiq_description",
        ),
        "crisp_controllers_robot_demos": os.path.join(
            avantbot_root,
            "robots",
            "franka_re3",
            "crisp_controllers_demos",
            "crisp_controllers_robot_demos",
        ),
    }
    viewer.load_urdf(args.urdf_path, package_paths)
    viewer.update_urdf(env.reset_joints, 0.0)
    print(f"URDF loaded: {args.urdf_path}")

    zed_cam = env.camera_reader.camera_dict.get(args.camera_serial)
    if zed_cam is None:
        raise RuntimeError(
            f"Camera {args.camera_serial} not found. "
            f"Available: {list(env.camera_reader.camera_dict.keys())}"
        )
    pc_mat = sl.Mat()
    print(f"Streaming ZED {args.camera_serial} point cloud — http://localhost:{args.port}")

    try:
        while True:
            obs = env.get_observation()
            robot_state = obs["robot_state"]
            joint_pos = np.array(robot_state["joint_positions"])
            gripper_pos = np.array([robot_state["gripper_position"]])

            viewer.update_urdf(joint_pos, gripper_pos)

            wrist_img = obs.get("image", {}).get(args.camera_serial + "_left")
            if wrist_img is not None:
                viewer.update(wrist=wrist_img[..., :3][..., ::-1])

            zed_cam._cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
            xyzrgba = pc_mat.get_data().copy()
            points, colors = decode_zed_xyzrgba(xyzrgba, rotate_to_z_up=False)
            points = points / 1000.0
            points[:, [0, 1]] = points[:, [1, 0]]
            points[:, 1] *= -1
            if len(points) > 0:
                viewer.update_ee_point_cloud(points, colors)

            time.sleep(1 / 15)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(tyro.cli(Args))

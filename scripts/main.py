# ruff: noqa

import collections
import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from moviepy import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from avantbot.perception.droid_camera_reader import DroidCameraReader
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "38082408"  # e.g., "24259877"
    right_camera_id: str = "32923065"  # e.g., "24514023"
    wrist_camera_id: str = "17605999"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    open_loop_horizon: int = 50
    # Fraction of open_loop_horizon at which to trigger async inference for the next chunk
    action_queue_threshold: float = 0.5

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # Viser camera viewer (set to a port like 8085 to enable, None to disable)
    viser_port: int | None = None

    # ZED camera serial to stream point clouds for (requires --viser_port, None to disable)
    pointcloud_camera_id: str | None = "17605999"

    # URDF visualization (requires --viser_port; set to "" to disable)
    urdf_path: str = os.path.join(
        os.path.expanduser("~"),
        "avantbot", "robots", "franka_re3", "crisp_controllers_demos",
        "crisp_controllers_robot_demos", "config", "fr3", "fr3_robotiq.urdf",
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _infer_actions(policy_client, request_data):
    return policy_client.infer(request_data)["actions"]


def _build_request_data(args: Args, curr_obs: dict, instruction: str) -> dict:
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            curr_obs[f"{args.external_camera}_image"], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
        "observation/joint_position": curr_obs["joint_position"],
        "observation/gripper_position": curr_obs["gripper_position"],
        "prompt": instruction,
    }


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    env.camera_reader.disable_cameras()
    env.camera_reader = DroidCameraReader(
        camera_kwargs={"resolution": "HD720", "fps": 15, "enable_depth": True},
    )
    print("Created the droid env!")

    required_cameras = {"wrist": args.wrist_camera_id}
    if args.external_camera == "left":
        required_cameras["left (external)"] = args.left_camera_id
    elif args.external_camera == "right":
        required_cameras["right (external)"] = args.right_camera_id

    available = set(env.camera_reader.camera_dict.keys())
    missing = {name: sid for name, sid in required_cameras.items() if sid not in available}
    if missing:
        missing_str = ", ".join(f"{name} serial={sid}" for name, sid in missing.items())
        raise RuntimeError(
            f"Required camera(s) not found: {missing_str}. "
            f"Available cameras: {sorted(available)}. "
            f"Check connections or update --left_camera_id / --right_camera_id / --wrist_camera_id."
        )

    # Optional Viser camera viewer
    cam_panel = None
    urdf_overlay = None
    pc_mgr = None
    pc_zed_cam = None
    pc_mat = None
    if args.viser_port is not None:
        from avantbot.viz import ViserViewer
        from avantbot.viz.components import CameraPanel, PointCloudManager, UrdfOverlay

        viewer = ViserViewer(port=args.viser_port)

        cam_panel = CameraPanel(jpeg_quality=80)
        viewer.add_component("cameras", cam_panel)

        if args.urdf_path:
            avantbot_root = os.path.join(os.path.expanduser("~"), "avantbot")
            package_paths = {
                "franka_description": os.path.join(
                    os.path.expanduser("~"), "franka_ros2", "franka_description"
                ),
                "robotiq_description": os.path.join(
                    avantbot_root, "robots", "robotiq_2f85", "robotiq_ws",
                    "src", "ros2_robotiq_gripper", "robotiq_description",
                ),
                "crisp_controllers_robot_demos": os.path.join(
                    avantbot_root, "robots", "franka_re3", "crisp_controllers_demos",
                    "crisp_controllers_robot_demos",
                ),
            }
            urdf_overlay = UrdfOverlay(
                args.urdf_path,
                [f"fr3_joint{i}" for i in range(1, 8)],
                package_paths=package_paths,
                gripper_joint="robotiq_85_left_knuckle_joint",
                gripper_range=(0.0, 0.8),
            )
            viewer.add_component("urdf", urdf_overlay)
            urdf_overlay.update(env.reset_joints, 0.0)
            print(f"URDF visualisation loaded: {args.urdf_path}")

        if args.pointcloud_camera_id is not None:
            import pyzed.sl as sl
            from avantbot.perception.zed_utils import decode_zed_xyzrgba

            pc_mgr = PointCloudManager(default_subsample=4)
            viewer.add_component("pointclouds", pc_mgr)

            pc_zed_cam = env.camera_reader.camera_dict.get(args.pointcloud_camera_id)
            if pc_zed_cam is not None:
                pc_mat = sl.Mat()
                print(f"Point cloud enabled for ZED {args.pointcloud_camera_id}")
            else:
                print(
                    f"WARNING: pointcloud_camera_id={args.pointcloud_camera_id} "
                    f"not found in cameras: {list(env.camera_reader.camera_dict.keys())}"
                )

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # Async action queue for real-time chunking
        action_queue: collections.deque = collections.deque()
        pending_future: Future | None = None
        queue_threshold = int(args.open_loop_horizon * args.action_queue_threshold)
        executor = ThreadPoolExecutor(max_workers=1)

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                if cam_panel is not None:
                    cam_panel.update(
                        left=curr_obs.get("left_image"),
                        right=curr_obs.get("right_image"),
                        wrist=curr_obs.get("wrist_image"),
                    )
                    if pc_zed_cam is not None and pc_mgr is not None and urdf_overlay is not None:
                        pc_zed_cam.zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
                        xyzrgba = pc_mat.get_data().copy()
                        pc = decode_zed_xyzrgba(xyzrgba, stride=1, rotate_to_z_up=False)
                        stride = pc_mgr.subsample
                        points = pc.points[::stride]
                        colors = pc.colors[::stride]
                        points[:, [0, 1]] = points[:, [1, 0]]
                        points[:, 1] *= -1
                        if len(points) > 0:
                            urdf_overlay.update_ee_point_cloud(points, colors, point_size=pc_mgr.point_size)
                    if urdf_overlay is not None:
                        urdf_overlay.update(
                            curr_obs["joint_position"],
                            curr_obs["gripper_position"],
                        )

                # Check if pending async inference has completed
                if pending_future is not None and pending_future.done():
                    new_chunk = pending_future.result()
                    assert new_chunk.ndim == 2 and new_chunk.shape[1] == 8, (
                        f"Unexpected action chunk shape: {new_chunk.shape}, expected (N, 8)"
                    )
                    action_queue.clear()
                    action_queue.extend(new_chunk[: args.open_loop_horizon])
                    pending_future = None

                # Request new chunk if queue is at/below threshold and no request in flight
                if len(action_queue) <= queue_threshold and pending_future is None:
                    request_data = _build_request_data(args, curr_obs, instruction)
                    pending_future = executor.submit(_infer_actions, policy_client, request_data)

                # If queue is empty, block until inference completes (fallback to synchronous)
                if len(action_queue) == 0:
                    assert pending_future is not None
                    with prevent_keyboard_interrupt():
                        new_chunk = pending_future.result()
                    assert new_chunk.ndim == 2 and new_chunk.shape[1] == 8, (
                        f"Unexpected action chunk shape: {new_chunk.shape}, expected (N, 8)"
                    )
                    action_queue.extend(new_chunk[: args.open_loop_horizon])
                    pending_future = None

                action = action_queue.popleft()

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        executor.shutdown(wait=False)

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    env.camera_reader.stop()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict.get("image", {})
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    if save_to_disk:
        available = [img for img in [left_image, wrist_image, right_image] if img is not None]
        if available:
            combined_image = np.concatenate(available, axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

    ext_key = f"{args.external_camera}_image"
    ext_img = {"left": left_image, "right": right_image}.get(args.external_camera)
    if ext_img is None:
        cam_id = args.left_camera_id if args.external_camera == "left" else args.right_camera_id
        available = list(image_observations.keys())
        raise RuntimeError(
            f"External camera '{args.external_camera}' (serial {cam_id}) returned no image. "
            f"Available image keys: {available}. "
            f"Check that the camera is connected, or use a different --external_camera / camera ID."
        )
    if wrist_image is None:
        raise RuntimeError(
            f"Wrist camera (serial {args.wrist_camera_id}) returned no image. "
            f"Available image keys: {list(image_observations.keys())}. "
            f"Check that the wrist camera is connected."
        )

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

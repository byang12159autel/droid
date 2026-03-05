# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
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
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # Viser camera viewer (set to a port like 8085 to enable, None to disable)
    viser_port: int | None = None

    # ZED camera serial to stream point clouds for (requires --viser_port, None to disable)
    pointcloud_camera_id: str | None = "32923065"

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


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Optional Viser camera viewer
    viewer = None
    pc_zed_cam = None
    pc_mat = None
    if args.viser_port is not None:
        from avantbot.utils.viser_camera_viewer import ViserCameraViewer
        viewer = ViserCameraViewer(port=args.viser_port)

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
            }
            viewer.load_urdf(args.urdf_path, package_paths)
            viewer.update_urdf(env.reset_joints, 0.0)
            print(f"URDF visualisation loaded: {args.urdf_path}")

        if args.pointcloud_camera_id is not None:
            import pyzed.sl as sl
            from avantbot.utils.viser_camera_viewer import decode_zed_xyzrgba
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

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

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
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                if viewer is not None:
                    viewer.update(
                        left=curr_obs.get("left_image"),
                        right=curr_obs.get("right_image"),
                        wrist=curr_obs.get("wrist_image"),
                    )
                    if pc_zed_cam is not None:
                        pc_zed_cam._cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
                        xyzrgba = pc_mat.get_data().copy()
                        points, colors = decode_zed_xyzrgba(xyzrgba)
                        if len(points) > 0:
                            viewer.update_point_cloud("zed2i", points, colors)
                    viewer.update_urdf(
                        curr_obs["joint_position"],
                        curr_obs["gripper_position"],
                    )

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.ndim == 2 and pred_action_chunk.shape[1] == 8, (
                        f"Unexpected action chunk shape: {pred_action_chunk.shape}, expected (N, 8)"
                    )

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

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

    # Drop the alpha dimension and convert to RGB (BGRA -> RGB)
    if left_image is not None:
        left_image = left_image[..., :3][..., ::-1]
    if right_image is not None:
        right_image = right_image[..., :3][..., ::-1]
    if wrist_image is not None:
        wrist_image = wrist_image[..., :3][..., ::-1]

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

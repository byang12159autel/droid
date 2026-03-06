"""Standalone test for EE-mounted point cloud in Viser scene graph.

Loads the FR3+Robotiq URDF, generates a synthetic point cloud in ZED camera
convention, and parents it under the ``zed2m_mount`` link.  Joint sliders let
you verify the point cloud tracks the end-effector correctly.

Usage::

    python scripts/test_ee_pointcloud.py [--port 8085]

Open http://localhost:<port> in a browser to interact.
"""

import dataclasses
import os
import time
from functools import partial

import numpy as np
import tyro
import viser
import viser.extras
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras._urdf import _viser_name_from_frame

_FR3_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
_GRIPPER_JOINT_NAME = "robotiq_85_left_knuckle_joint"
_GRIPPER_JOINT_MAX = 0.8


@dataclasses.dataclass
class Args:
    port: int = 8085
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
    mount_link: str = "zed2m_mount"


def _package_filename_handler(
    fname: str, dir: str, package_paths: dict[str, str]
) -> str:
    if fname.startswith("package://"):
        remainder = fname[len("package://") :]
        pkg_name, _, pkg_rel_path = remainder.partition("/")
        if pkg_name in package_paths:
            return os.path.join(package_paths[pkg_name], pkg_rel_path)
    return yourdfpy.filename_handler_relative(fname, dir)


def _make_synthetic_pointcloud(n_x: int = 50, n_y: int = 10, n_z: int = 50):
    """Colored grid matching real ZED SDK XYZRGBA output convention.

    The ZED SDK returns points with X/Y swapped compared to the textbook
    camera convention.  This synthetic cloud mimics that so the optical-frame
    rotation calibrated here transfers directly to real data.

    Creates a 0.3 x 0.2 x 0.3 m volume placed 0.3--0.6 m in front of the
    camera.  Color encodes position: R = Y (wide axis), G = Z-depth,
    B = constant.
    """
    wide = np.linspace(-0.15, 0.15, n_x)
    narrow = np.linspace(-0.1, 0.1, n_y)
    depth = np.linspace(0.3, 0.6, n_z)

    ww, nn, dd = np.meshgrid(wide, narrow, depth, indexing="ij")
    points = np.stack([nn.ravel(), ww.ravel(), dd.ravel()], axis=1).astype(np.float32)

    r = ((points[:, 1] - points[:, 1].min()) / (np.ptp(points[:, 1]) + 1e-8) * 255).astype(np.uint8)
    g = ((points[:, 2] - points[:, 2].min()) / (np.ptp(points[:, 2]) + 1e-8) * 255).astype(np.uint8)
    b = np.full_like(r, 100)
    colors = np.stack([r, g, b], axis=1)
    return points, colors


def main(args: Args) -> None:
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (1.5, -1.5, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.3)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    # --- Load URDF --------------------------------------------------------
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

    urdf_path = os.path.realpath(args.urdf_path)
    handler = partial(
        _package_filename_handler,
        dir=os.path.dirname(urdf_path),
        package_paths=package_paths,
    )
    urdf = yourdfpy.URDF.load(
        urdf_path,
        filename_handler=handler,
        build_scene_graph=True,
        load_meshes=True,
    )

    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")

    # --- Resolve mount-link scene path ------------------------------------
    prefixed_root = "/base/visual"
    mount_scene_path = _viser_name_from_frame(
        urdf.scene, args.mount_link, prefixed_root
    )
    print(f"Mount link scene path: {mount_scene_path}")

    # --- Add optical frame + synthetic point cloud ------------------------
    optical_path = f"{mount_scene_path}/zed_optical"
    optical_frame = server.scene.add_frame(
        optical_path,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.003,
    )

    points, colors = _make_synthetic_pointcloud()
    pc_handle = server.scene.add_point_cloud(
        f"{optical_path}/pointcloud",
        points=points,
        colors=colors,
        point_size=0.005,
    )
    print(f"Synthetic point cloud: {len(points)} points")

    # --- GUI: joint sliders -----------------------------------------------
    with server.gui.add_folder("Joints"):
        joint_limits = urdf_vis.get_actuated_joint_limits()
        joint_sliders: dict[str, viser.GuiInputHandle] = {}
        for name in _FR3_JOINT_NAMES:
            lo, hi = joint_limits.get(name, (-np.pi, np.pi))
            lo = lo if lo is not None else -np.pi
            hi = hi if hi is not None else np.pi
            joint_sliders[name] = server.gui.add_slider(
                name,
                min=lo,
                max=hi,
                step=0.01,
                initial_value=float(np.clip(0.0, lo, hi)),
            )
        gripper_slider = server.gui.add_slider(
            "gripper", min=0.0, max=1.0, step=0.01, initial_value=0.0
        )

    # --- GUI: optical-frame rotation (euler XYZ, degrees) -----------------
    with server.gui.add_folder("Camera Frame Rotation (euler XYZ, deg)"):
        rot_x = server.gui.add_slider("rot_x", min=-180, max=180, step=1, initial_value=0)
        rot_y = server.gui.add_slider("rot_y", min=-180, max=180, step=1, initial_value=20)
        rot_z = server.gui.add_slider("rot_z", min=-180, max=180, step=1, initial_value=0)

    # --- GUI: optical-frame position offset (metres) ----------------------
    with server.gui.add_folder("Camera Frame Offset (m)"):
        off_x = server.gui.add_slider("off_x", min=-0.15, max=0.15, step=0.001, initial_value=-0.072)
        off_y = server.gui.add_slider("off_y", min=-0.15, max=0.15, step=0.001, initial_value=0.031)
        off_z = server.gui.add_slider("off_z", min=-0.15, max=0.15, step=0.001, initial_value=0.032)

    print(f"\nViser running at http://localhost:{args.port}")
    print("Move joint sliders to verify the point cloud follows the EE.")
    print("Adjust camera-frame rotation/offset to match your physical ZED mount.")

    try:
        while True:
            # Update URDF joint configuration
            cfg: dict[str, float] = {}
            for name, slider in joint_sliders.items():
                cfg[name] = float(slider.value)
            cfg[_GRIPPER_JOINT_NAME] = float(gripper_slider.value) * _GRIPPER_JOINT_MAX
            urdf_vis.update_cfg(cfg)

            # Update optical frame transform from sliders
            rot = Rotation.from_euler(
                "xyz", [rot_x.value, rot_y.value, rot_z.value], degrees=True
            )
            xyzw = rot.as_quat()
            optical_frame.wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
            optical_frame.position = (off_x.value, off_y.value, off_z.value)

            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(tyro.cli(Args))

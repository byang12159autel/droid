# The DROID Robot Platform

This repository contains the code for setting up your DROID robot platform and using it to collect teleoperated demonstration data. This platform was used to collect the [DROID dataset](https://droid-dataset.github.io), a large, in-the-wild dataset of robot manipulations.

If you are interested in using the DROID dataset for training robot policies, please check out our [policy learning repo](https://github.com/droid-dataset/droid_policy_learning).
For more information about DROID, please see the following links: 

[**[Homepage]**](https://droid-dataset.github.io) &ensp; [**[Documentation]**](https://droid-dataset.github.io/droid) &ensp; [**[Paper]**](https://arxiv.org/abs/2403.12945) &ensp; [**[Dataset Visualizer]**](https://droid-dataset.github.io/dataset.html).

![](https://droid-dataset.github.io/droid/assets/index/droid_teaser.jpg)

---------
## Droid Setup Guide

We assembled a step-by-step guide for setting up the DROID robot platform in our [developer documentation](https://droid-dataset.github.io/droid).
This guide has been used to set up 18 DROID robot platforms over the course of the DROID dataset collection. Please refer to the steps in this guide for setting up your own robot. Specifically, you can follow these key steps:

1. [Hardware Assembly and Setup](https://droid-dataset.github.io/droid/docs/hardware-setup)
2. [Software Installation and Setup](https://droid-dataset.github.io/droid/docs/software-setup)
3. [Example Workflows to collect data or calibrate cameras](https://droid-dataset.github.io/droid/docs/example-workflows)

If you encounter issues during setup, please raise them as issues in this github repo.

---------
## Avantbot Droid Setup (Single Workstation)

This fork replaces the Polymetis control stack with [avantbot](https://github.com/autel-robotics/avantbot) CRISP controllers, allowing everything to run on a single workstation without a separate NUC or RT kernel. Tested on Ubuntu 22.04 with cpu core isolation.

### Prerequisites

- Avantbot repo cloned at `/home/ben/avantbot` with pixi installed
- CRISP Docker stack working (`launch_single_franka_panes.sh`)
- Franka robot accessible at `192.168.1.13`

### One-Time Setup

```bash
# Enter the pixi shell with ROS2 + avantbot
cd /home/ben/avantbot
pixi shell -e humble-full

# Install pip in the pixi env (if not already available)
python -c "import ensurepip; ensurepip.bootstrap()"

# Install droid package (without heavy deps) and zerorpc
python -m pip install --no-deps -e /home/ben/droid
python -m pip install zerorpc

# Install IK dependencies (dm-control compatible with mujoco 3.2.6)
python -m pip install --no-deps dm-control==1.0.14 dm-robotics-moma dm-robotics-transformations dm-robotics-geometry dm-robotics-agentflow
```

### Launch (3 Terminals)

**Terminal 1 -- CRISP Controllers (Docker)**
```bash
cd /home/ben/avantbot/robots/franka_re3/crisp_controllers_demos
./launch_single_franka_panes.sh
# With Franka gripper: LOAD_GRIPPER=true ./launch_single_franka_panes.sh
```

**Terminal 2 -- CRISP zerorpc Adapter (Host)**
```bash
cd /home/ben/avantbot
pixi shell -e humble-full
python /home/ben/droid/scripts/server/run_server_crisp.py
# Should print: CRISP zerorpc server listening on tcp://0.0.0.0:4242
```

**Terminal 3 -- DROID Laptop Container (Docker)**
```bash
cd /home/ben/droid/.docker/laptop
export ROOT_DIR=/home/ben/droid
export ROBOT_TYPE=fr3
export LIBFRANKA_VERSION=0.10.0
export NUC_IP=127.0.0.1
export ROBOT_IP=192.168.1.13
export LAPTOP_IP=127.0.0.1
export DISPLAY=$DISPLAY
export DOCKER_XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $DOCKER_XAUTH nmerge -
docker compose -f docker-compose-laptop.yaml up
```

### Quick Test

From a separate pixi shell terminal, verify the connection:

```bash
python -c "
import zerorpc
c = zerorpc.Client(heartbeat=20)
c.connect('tcp://127.0.0.1:4242')
print('EE pose:', c.get_ee_pose())
print('Joints:', c.get_joint_positions())
print('Gripper:', c.get_gripper_position())
"
```

---------
## Running Pi 0.5 Policy (OpenPI)

Run a pretrained pi0.5 DROID policy using the [OpenPI](https://github.com/Physical-Intelligence/openpi) inference server. This requires 3 terminals.

### Prerequisites

- Avantbot CRISP Docker stack working (see Avantbot Setup above)
- OpenPI repo cloned at `~/openpi` with `uv` installed
- DROID repo (`~/droid`) installed in the avantbot pixi environment (`humble-openpi-droid`)
- ZED cameras connected (external + wrist)

### Launch (3 Terminals)

**Terminal 1 -- CRISP Controllers + Robotiq Gripper (Docker)**
```bash
cd ~/avantbot/robots/franka_re3/crisp_controllers_demos
./launch_single_franka_panes.sh "" "" robotiq
```

**Terminal 2 -- OpenPI Policy Server**
```bash
cd ~/openpi
uv run scripts/serve_policy.py --env=DROID
```
Wait for the server to print that it is listening on port 8000 before proceeding.

**Terminal 3 -- DROID Policy Rollout (Host)**
```bash
cd ~/droid
pixi shell -e humble-openpi-droid
python3 scripts/main.py \
  --remote_host=127.0.0.1 \
  --remote_port=8000 \
  --external_camera=right
```
You will be prompted to enter a language instruction. The robot will then execute the policy and save a video of the rollout.

### Notes

- `--external_camera` selects which external ZED camera to feed to the policy (`left` or `right`).
- Camera serial IDs are configured in `scripts/main.py` (`left_camera_id`, `right_camera_id`, `wrist_camera_id`). Update these if your hardware differs.
- The policy runs at 15 Hz (DROID control frequency). Each inference returns an action chunk; by default 8 actions are executed open-loop before re-querying (`--open_loop_horizon`).
- Press Ctrl+C during a rollout to stop it early. After each rollout you can rate success and optionally run another.
- Results are saved to `results/` as CSV files with per-rollout success scores and video filenames.

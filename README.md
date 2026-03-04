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
## Avantbot Setup (Single Workstation)

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

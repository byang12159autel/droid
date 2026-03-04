source ~/anaconda3/etc/profile.d/conda.sh
conda activate polymetis-local
pkill -9 run_server
pkill -9 franka_panda_cl
taskset -c 2 chrt -f 90 launch_robot.py robot_client=franka_hardware

"""zerorpc server backed by avantbot CRISP controllers instead of Polymetis.

Run this on the host inside a pixi shell (humble-full) with the CRISP Docker
already running, then start the DROID laptop container as usual.

Usage:
    cd /home/ben/droid
    pixi shell -e humble-full
    pip install zerorpc
    python scripts/server/run_server_crisp.py
"""

import zerorpc

from droid.franka.crisp_robot import CrispFrankaRobot

if __name__ == "__main__":
    robot_client = CrispFrankaRobot()
    s = zerorpc.Server(robot_client, heartbeat=60)
    s.bind("tcp://0.0.0.0:4242")
    print("CRISP zerorpc server listening on tcp://0.0.0.0:4242")
    s.run()

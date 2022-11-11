import time

from ext_robot.client import ExtInstance

inst = ExtInstance()
for i in range(5):
    for j in [0, 1, 3]:#range(4):
        q = [0]*4
        q[j] = 120
        inst.send_q_esp(q)
        time.sleep(3)

inst.send_q_esp([0]*4)
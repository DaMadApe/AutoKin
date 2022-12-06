import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from autokin.robot import ExternRobot, SofaRobot
from autokin.muestreo import FKset
from autokin.utils import restringir

DS_PATH = 'out_dataset.pt'
N_PER_STEP = 100

robot = SofaRobot(config='LSL') #ExternRobot(n=3)

test_trayec = torch.concat([torch.zeros(N_PER_STEP, robot.n),
                            torch.tensor([1,0,0]).repeat(N_PER_STEP,1),
                            torch.zeros(N_PER_STEP, robot.n),
                            torch.tensor([0,1,0]).repeat(N_PER_STEP,1),
                            torch.zeros(N_PER_STEP, robot.n),
                            torch.tensor([0,0,1]).repeat(N_PER_STEP,1),
                            torch.zeros(N_PER_STEP, robot.n),])

test_trayec = restringir(test_trayec)

d_set = FKset(robot, test_trayec)
if isinstance(robot, SofaRobot):
    robot.stop_instance()
# d_set.robot = None
# torch.save(d_set, DS_PATH)

# d_set = torch.load(DS_PATH)

q_set = d_set[:][0].numpy()
p_set = d_set[:][1].numpy()

q_diff = np.linalg.norm(np.diff(q_set, axis=0), axis=-1)
p_diff = np.linalg.norm(np.diff(p_set, axis=0), axis=-1)

t = np.arange(len(q_diff))

fig = plt.figure()
ax = fig.add_subplot()

# ax.scatter(t, q_diff, color='orangered')
ax.plot(t, q_diff, color='orange', linewidth=1)

# ax.scatter(t, p_diff, color='royalblue')
ax.plot(t, 0.1*p_diff, color='lightblue', linewidth=1)

plt.tight_layout()
plt.show()
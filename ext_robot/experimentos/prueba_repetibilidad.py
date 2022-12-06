import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from autokin.robot import ExternRobot, SofaRobot
from autokin.muestreo import FKset
from autokin.utils import restringir

DS_PATH = 'out_dataset.pt'
N_PER_STEP = 500

robot = ExternRobot(n=3)
robot.q_max = 800 * torch.ones(robot.n)

test_trayec1 = torch.concat([torch.zeros(N_PER_STEP, robot.n),
                            torch.tensor([1,0,0]).repeat(N_PER_STEP,1),
                            torch.zeros(N_PER_STEP, robot.n),
                            torch.tensor([0,1,0]).repeat(N_PER_STEP,1),
                            torch.zeros(N_PER_STEP, robot.n),
                            torch.tensor([0,0,1]).repeat(N_PER_STEP,1),
                            torch.zeros(N_PER_STEP, robot.n),])

test_trayec2 = torch.concat

test_trayec = restringir(test_trayec2)

d_set = FKset(robot, test_trayec)
if isinstance(robot, SofaRobot):
    robot.stop_instance()
# d_set.robot = None
# torch.save(d_set, DS_PATH)

# d_set = torch.load(DS_PATH)

p_set = d_set[:][1].numpy().transpose()

fig, ax = plt.subplots()
ax.scatter(*p_set, color='royalblue')
ax.plot(*p_set, color='lightblue', linewidth=1)
plt.tight_layout()
plt.show()
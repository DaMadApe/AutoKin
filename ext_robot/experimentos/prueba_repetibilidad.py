import os
import logging

import torch
import numpy as np
from matplotlib import pyplot as plt

from autokin.robot import ExternRobot, SofaRobot
from autokin.muestreo import FKset
from autokin.utils import restringir, suavizar

DS_PATH = 'rep_dataset.pt'
N_PER_STEP = 5
STATIC_MULT = 10


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


robot = ExternRobot(n=3)
robot.q_max = 800 * torch.ones(robot.n)
# robot = SofaRobot(config='LLL')

test_trayec = torch.concat([torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([1,0,0]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,1,0]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,0,1]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([1,0,0]).repeat(STATIC_MULT*N_PER_STEP,1), #
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,1,0]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,0,1]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([1,0,0]).repeat(STATIC_MULT*N_PER_STEP,1), #
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,1,0]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,0,1]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([1,0,0]).repeat(STATIC_MULT*N_PER_STEP,1), #
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,1,0]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),
                             torch.tensor([0,0,1]).repeat(STATIC_MULT*N_PER_STEP,1),
                             torch.zeros(N_PER_STEP, robot.n),])


# print(test_trayec)

# test_trayec = suavizar(test_trayec)

# d_set = FKset(robot, test_trayec)
# if isinstance(robot, SofaRobot):
#     robot.stop_instance()
# d_set.robot = None
# torch.save(d_set, DS_PATH)

d_set = torch.load(DS_PATH)

p_set = d_set[:][1].numpy().transpose()

for i, axis in enumerate(p_set):
    plt.subplot(3,1,i+1)
    plt.plot(axis, color='royalblue')

plt.tight_layout()
plt.show()

# Rangos 1: [18:73], 
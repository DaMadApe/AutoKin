import os
import logging

import torch
import numpy as np
from matplotlib import pyplot as plt

from autokin.robot import ExternRobot, SofaRobot
from autokin.muestreo import FKset
from autokin.utils import restringir

DS_PATH = 'act_dataset.pt'
N_PER_STEP = 15

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


robot = ExternRobot(n=3)
robot.max_dq = 5

# 1/16 de paso
# max_steps = 1000
# robot.q_max = max_steps * torch.ones(robot.n)
# test_trayec = torch.concat([torch.tensor([0,0,200/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,400/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,600/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,800/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,1000/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,0]).unsqueeze(0)])

# # 1/8 de paso
# max_steps = 500
# robot.q_max = max_steps * torch.ones(robot.n)
# test_trayec = torch.concat([torch.tensor([0,0,100/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,200/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,300/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,400/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,500/max_steps]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,0]).unsqueeze(0)])

# # 1/2 de paso
# max_steps = 250
# robot.q_max = max_steps * torch.ones(robot.n)
# test_trayec = torch.concat([torch.tensor([50/max_steps,0,0]).repeat(N_PER_STEP,1),
#                             torch.tensor([100/max_steps,0,0]).repeat(N_PER_STEP,1),
#                             torch.tensor([150/max_steps,0,0]).repeat(N_PER_STEP,1),
#                             torch.tensor([200/max_steps,0,0]).repeat(N_PER_STEP,1),
#                             torch.tensor([250/max_steps,0,0]).repeat(N_PER_STEP,1),
#                             torch.tensor([0,0,0]).unsqueeze(0)])

# Paso completo
max_steps = 120
robot.q_max = max_steps * torch.ones(robot.n)
test_trayec = torch.concat([torch.tensor([0,0,24/max_steps]).repeat(N_PER_STEP,1),
                            torch.tensor([0,0,48/max_steps]).repeat(N_PER_STEP,1),
                            torch.tensor([0,0,72/max_steps]).repeat(N_PER_STEP,1),
                            torch.tensor([0,0,96/max_steps]).repeat(N_PER_STEP,1),
                            torch.tensor([0,0,120/max_steps]).repeat(N_PER_STEP,1),
                            torch.tensor([0,0,0]).unsqueeze(0)])


# test_trayec = restringir(test_trayec)

d_set = FKset(robot, test_trayec)
if isinstance(robot, SofaRobot):
    robot.stop_instance()
d_set.robot = None
# torch.save(d_set, DS_PATH)

# d_set = torch.load(DS_PATH)

# p_set = d_set[:][1].numpy().transpose()

# for i, axis in enumerate(p_set):
#     plt.subplot(3,1,i+1)
#     plt.plot(axis, color='royalblue')

# plt.tight_layout()
# plt.show()
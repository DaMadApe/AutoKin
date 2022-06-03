import torch
from torch.utils.data import TensorDataset
import numpy as np

from robot import SofaRobot
from utils import coprime_sines
from muestreo import FKset


def static_point(robot, q: torch.Tensor):
    pad = torch.zeros((50, q.size()[-1]))
    rep_q = q.repeat(50,1)
    exp_q = torch.concat([pad, rep_q])
    _, p = robot.fkine(exp_q)
    return p[-2]

# dataset = TensorDataset(torch.tensor(np.load('sofa/q_in.npy')),
#                         torch.tensor(np.load('sofa/p_out.npy')))
robot = SofaRobot(config='LL')

c_sines = coprime_sines(robot.n, 300, densidad=0)
dataset = FKset(robot, c_sines)
q_sim, p_sim = dataset[200]

# t = torch.linspace(0, torch.pi, 300)

static_p = static_point(robot, q_sim)
_, dyn_p = robot.fkine(q_sim.repeat(100,1))

print(p_sim)
print(static_p)
print(dyn_p[-2])
import torch
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from autokin.robot import ModelRobot, SofaRobot
from autokin.trayectorias import coprime_sines
from autokin.utils import restringir
from autokin.experimentos.experim import setup_logging, ejecutar_experimento

# def static_point(robot, q: torch.Tensor):
#     pad = torch.zeros((50, q.size()[-1]))
#     interp = q.size()[-1]
#     rep_q = q.repeat(50,1)
#     exp_q = torch.concat([pad, rep_q])
#     _, p = robot.fkine(exp_q)
#     return p[-2]

robot = SofaRobot(config='LL')
model = ModelRobot.load('models/Trunk4C_exp1.pt') 

# dataset = TensorDataset(torch.tensor(np.load('sofa/q_in.npy'), dtype=torch.float),
#                         torch.tensor(np.load('sofa/p_out.npy'), dtype=torch.float))

q = coprime_sines(robot.n, 300, densidad=0)
q = restringir(q)
q, p = robot.fkine(q)

q_test, p_sim = q[::80], p[::80]
_, p_mod = model.fkine(q_test)

# n_points = 10
# indices = torch.randperm(len(dataset))[:n_points]
# q_sim, p_sim = dataset[indices]

# _, p_mod = model.fkine(q_sim)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(p_sim[:,0], p_sim[:,1], p_sim[:,2])
ax.plot(p_sim[:,0], p_sim[:,1], p_sim[:,2])
ax.scatter(p_mod[:,0], p_mod[:,1], p_mod[:,2])
ax.plot(p_mod[:,0], p_mod[:,1], p_mod[:,2])

ax.legend(['Puntos reales', 'Puntos predecidos'])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
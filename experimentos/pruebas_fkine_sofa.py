import torch
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from robot import ModelRobot, SofaRobot
from experim import setup_logging, ejecutar_experimento

robot = SofaRobot()
model = ModelRobot.load('models/Trunk4C_exp1.pt') 


dataset = TensorDataset(torch.tensor(np.load('sofa/q_in.npy'), dtype=float),
                        torch.tensor(np.load('sofa/p_out.npy'), dtype=float))

n_points = 10
indices = torch.randperm(len(dataset))[:n_points]
q_sim, p_sim = dataset[indices]

_, p_mod = model.fkine(q_sim.float())
p_mod = p_mod.detach()

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
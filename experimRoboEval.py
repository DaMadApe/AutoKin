"""
Cargar el modelo entrenado por experimRobo.py para
comparar con el robot original y evaluar
"""

import roboticstoolbox as rtb
import numpy as np
import torch

from experim0 import Regressor
from experimRobo import denorm_q

"""
args
"""
depth = 10
mid_layer_size = 10
activation = torch.relu

robot = rtb.models.DH.Cobra600()

input_dim = robot.n
output_dim = 3
""""""

path = 'models/experimRobo/v1.pt'
model = Regressor(input_dim, output_dim,
                    depth, mid_layer_size,
                    activation)
model.load_state_dict(torch.load(path))
model.eval()

q_test = np.array([0.5, 0.6, 0.2, 0.9])
real = robot.fkine(denorm_q(robot, q_test)).t
pred = model(torch.tensor(q_test).float()).detach()


print(robot.qlim)
print(real)
print(pred)
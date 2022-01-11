"""
Cargar el modelo entrenado por experimR.py para
comparar con el robot original y evaluar
"""

import roboticstoolbox as rtb
import torch

from experim0 import MLP, load
from experimR import denorm_q

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

path = 'models/experimR'
name = 'v1'
model = MLP(input_dim, output_dim,
                    depth, mid_layer_size,
                    activation)
load(model, path, name)
model.eval()

q_test = [0.9, 0.1, 0.9, 0.1]
real = robot.fkine(denorm_q(robot, q_test)).t
pred = model(torch.tensor(q_test).float()).detach()


# print(robot.qlim)
# print(real)
# print(pred)
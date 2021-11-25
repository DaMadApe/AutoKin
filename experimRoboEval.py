import roboticstoolbox as rtb
import numpy as np
import torch

from experim0 import Regressor
from experimRobo import denorm_q

"""
args
"""
lr = 1e-3
depth = 10
mid_layer_size = 10
activation = torch.relu
batch_size = 512
epochs = 500

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
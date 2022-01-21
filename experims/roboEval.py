"""
Cargar el modelo entrenado por experimR.py para
comparar con el robot original y evaluar
"""

import roboticstoolbox as rtb
import torch

from experimR import denorm_q


robot = rtb.models.DH.Cobra600()

model = torch.load('models/experim14/v1.pt')
model.eval()

q_test = [0.9, 0.1, 0.9, 0.1]
real = robot.fkine(denorm_q(robot, q_test)).t
pred = model(torch.tensor(q_test).float()).detach()

print(real)
print(pred)
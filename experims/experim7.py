"""
Probar algoritmos de experim6.py para CI con
jacobiano de la red de regresión (consultar
experim5.py para obtención de jacobiano)
"""
import roboticstoolbox as rtb
import torch
from torch.autograd.functional import jacobian

from experim0 import MLP
from experimR import denorm_q
from experim6 import ikine_pi_jacob

"""
Cargar modelo con args originales de entrenamiento
"""
robot = rtb.models.DH.Cobra600()

model = torch.load('models/experimR_v1.pt')
model.eval()

# q_test = [0.1, 0.9, 0.1, 0.2]
# real = robot.fkine(denorm_q(robot, q_test)).t
# pred = model(torch.tensor(q_test).float()).detach()


q0 = denorm_q(robot, [0.1, 0.1, 0.1, 0.1])

q_target = denorm_q(robot, [0.1, 0.4, 0.2, 0.3])
x_target = robot.fkine(q_target).t

q_inv = ikine_pi_jacob(q0, x_target,
                    lambda q: model(torch.tensor(q).float()).detach().numpy(), 
                    lambda q: jacobian(model, torch.tensor(q).float()).squeeze().numpy(),
                    min_error=0)

print(f"""
    Initial q: {q0}
    Initial x: {robot.fkine(q0).t}
    Requested q: {q_target}
    Requested x: {x_target}
    Found q: {q_inv}
    Reached x: {robot.fkine(q_inv).t}""")
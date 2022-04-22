import roboticstoolbox as rtb
import torch
from torch.autograd.functional import jacobian

from ikine import ikine_pi_jacob
from utils import denorm_q, norm_q

robot = rtb.models.DH.Cobra600() #Puma560()

model = torch.load('models/cobra600_v1.pt')
model.eval()

def model_fkine(q):
    return model(norm_q(robot, q)).detach()

def model_jacobian(q):
    return jacobian(model, norm_q(robot, q)).squeeze()

def robot_fkine(q):
    return torch.tensor(robot.fkine(q.numpy()).t, dtype=torch.float)

def robot_jacobian(q):
    return torch.tensor(robot.jacob0(q.numpy())[:3], dtype=torch.float)


q_start = denorm_q(robot, torch.rand(robot.n))
p_start = robot_fkine(q_start)

q_target = denorm_q(robot, torch.rand(robot.n))
p_target = robot_fkine(q_target)


q_inv = ikine_pi_jacob(q_start, p_target, eta=0.1,
                       fkine=model_fkine, 
                       jacob=model_jacobian)

p_reached = robot_fkine(q_inv)

# TODO : escribir experimento, score = error_posición

print(f"""
q inicial: {q_start}
q objetivo: {q_target}
q encontrada: {q_inv}
Pos inicial: {p_start}
Pos objetivo: {p_target}
Pos resultante: {p_reached}
""")
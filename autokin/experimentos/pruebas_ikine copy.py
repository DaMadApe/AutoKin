import roboticstoolbox as rtb
import torch
from torch.autograd.functional import jacobian

from autokin.robot import ModelRobot, RTBrobot
from autokin.experimentos.experim import setup_logging, ejecutar_experimento

robot = RTBrobot.from_name('Cobra600')
model = ModelRobot.load('models/Cobra600_.pt')

def ikine_experiment():
    q_start = torch.rand(robot.n)
    q_target = torch.rand(robot.n)

    _, p_start = robot.fkine(q_start)
    _, p_target = robot.fkine(q_target)


    q_inv = model.ikine_pi_jacob(q_start, p_target, eta=0.1)

    _, p_reached = robot.fkine(q_inv)

    # TODO : escribir experimento, score = error_posición

    error = torch.linalg.norm(p_target - p_reached)

    print(f"""
    q inicial: {q_start}
    q objetivo: {q_target}
    q encontrada: {q_inv}
    Pos inicial: {p_start}
    Pos objetivo: {p_target}
    Pos resultante: {p_reached}
    """)

    return error, (q_start, q_target)

ejecutar_experimento(5, ikine_experiment)


p = robot.fkine(torch.rand((10, robot.n)))
import roboticstoolbox as rtb
import torch
from torch.autograd.functional import jacobian

from ikine import ikine_pi_jacob
from robot import ModelRobot, RTBrobot
from experim import setup_logging, ejecutar_experimento

robot = RTBrobot.from_name('Puma560') # 'Cobra600')
model = ModelRobot.load('models/Puma560.pt') # cobra600_refactor.pt')

def ikine_experiment():
    q_start = torch.rand(robot.n)
    q_target = torch.rand(robot.n)

    _, p_start = robot.fkine(q_start)
    _, p_target = robot.fkine(q_target)


    q_inv = ikine_pi_jacob(q_start, p_target, eta=0.1,
                        fkine=model.fkine, 
                        jacob=model.jacobian)

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
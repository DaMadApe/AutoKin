import torch
from torch.utils.data import TensorDataset
import numpy as np

from robot import ModelRobot, SofaRobot
from experim import setup_logging, ejecutar_experimento

robot = SofaRobot()
model = ModelRobot.load('models/Trunk4C_exp1.pt') 

def ikine_experiment():
    q_start = torch.rand(robot.n)
    q_target = torch.rand(robot.n)

    # TODO: Rutina para conectar puntos "suavemente"
    q1 = torch.stack([torch.zeros(robot.n), q_start, q_start, q_target, q_target])

    _, r1 = robot.fkine(q1)

    p_start = r1[1]
    p_target = r1[-1]

    # _, p_start = robot.fkine(q_start)
    # _, p_target = robot.fkine(q_target)


    q_inv = robot.ikine_pi_jacob(q_start, p_target, eta=0.1)

    q2 = torch.stack([q_inv, q_inv])
    _, r2 = robot.fkine(q2)
    p_reached = r2[1]

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

ejecutar_experimento(1, ikine_experiment)
# Hacer q arbitraria que incluya firmas de detección
# Sacar p
# Agregar padding aleatorio a p y alterar sample rate
# Hacer alineación
import torch

from robot import RTBrobot
from utils import FKset, coprime_sines
from experimentos.experim import ejecutar_experimento

def sign(q):
    n = q.size()[-1]
    step_len = 2
    signature = torch.cat([torch.ones(step_len, n),
                           torch.zeros(step_len, n),
                           torch.ones(step_len, n),
                           torch.zeros(step_len, n)])
    on_ramp = torch.linspace()
    signed_q = torch.cat([signature, q, signature])
    return signed_q

robot_name = 'Cobra600' #'Puma560'
robot = RTBrobot.from_name(robot_name)
n_samples = 30
q = coprime_sines(robot.n, n_samples, wiggle=3)

q = sign(q)

p = robot.fkine(q)

print(q)
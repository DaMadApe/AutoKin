"""
Automatizar el entrenamiento de múltiples
robots para comparar el efecto de distintas
arquitecturas de la red neuronal.
"""
import numpy as np
import roboticstoolbox as rtb
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from experimR import RoboKinSet
from experim3 import MLP_PL

np.random.seed(42)

# rtb.DHLink([d, alpha, theta, a, joint_type])
# rev=0, prism=1
p_P = 0.5 # Probabilidad de junta prismática
min_DH = np.array([0, 0, 0, 0] )
max_DH = np.array([2*np.pi, 2, 2*np.pi, 2])

links = []

for n_joint in range(np.random.randint(2, 10)):

    DH_vals = (np.random.rand(4) - min_DH) / (max_DH - min_DH)
    d, alpha, theta, a = DH_vals
    is_prism = np.random.rand() < p_P

    if is_prism:
        links.append(rtb.DHLink(alpha=alpha,theta=theta, a=a, sigma=1))
    else:
        links.append(rtb.DHLink(d=d, alpha=alpha, a=a, sigma=0))


robot = rtb.DHRobot(links)
q = np.random.rand(100, robot.n)
"""
fkine_all devuelve la transformación para cada junta, por lo que
podría hacer todos los robots de 9 juntas, y aprovechar la función
para sacar también datos de los subconjuntos de la cadena cinemática
"""
#robot.fkine_all(q).t
robot.plot(q)
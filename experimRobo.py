"""
Regresión de cinemática de un robot
[q1 q2...qn] -> [x,y,z,*R]
"""
import roboticstoolbox as rtb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from experim0 import Regressor

class RoboKinSet(Dataset):
    # Es permisible tener todo el dataset en un tensor porque no es
    # particularmente grande
    def __init__(self, robot, q_samples:list,
                 norm=True):
        """
        q_samples = [(q_i_min, q_i_max, n_i_samples), (...), (...)]
        
        self.qs = [[0, 0, 0], [0, 0, 0]]
        """
        self.robot = robot
        self.n = self.robot.n # Número de ejes

        are_enough = len(q_samples) == self.n
        assert are_enough, f'Expected 3 int tuple list of len{self.n}'

        self.qs = []
        for q_min, q_max, n_samples in q_samples:
            self.qs.append(np.linspace(q_min, q_max, n_samples))
        
        self.q_vecs = np.meshgrid(self.qs)


robot = rtb.models.DH.Panda()
print(robot.fkine(np.zeros(robot.n)).t)


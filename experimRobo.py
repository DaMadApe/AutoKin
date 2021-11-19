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
        assert are_enough, f'Expected 3 int tuple list of len {self.n}'

        # Lista de puntos para cada parámetro
        self.qs = []
        for q_min, q_max, n_samples in q_samples:
            self.qs.append(np.linspace(q_min, q_max, n_samples))
        
        # Magia negra para producir todas las combinaciones de puntos
        self.q_vecs = np.meshgrid(*self.qs)
        self.q_vecs = np.stack(self.q_vecs, -1).reshape(-1, self.n)

        # Producir las etiquetas correspondientes a cada vec de paráms
        self.poses = [self.robot.fkine(q_vec).t for q_vec in self.q_vecs]
        self.poses = np.array(self.poses)


if __name__ == '__main__':
    robot = rtb.models.DH.Panda()
    #print(robot.fkine(np.zeros(robot.n)).t)

    dataset = RoboKinSet(robot, [(0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3)])
    print(dataset.q_vecs.shape)
    print(dataset.poses.shape)
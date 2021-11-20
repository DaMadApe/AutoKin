"""
Regresión de cinemática de un robot
[q1 q2...qn] -> [x,y,z,*R]
"""
import numpy as np
import roboticstoolbox as rtb
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset

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
        self.poses = torch.tensor(self.poses, dtype=torch.float)
        self.q_vecs = torch.tensor(self.q_vecs, dtype=torch.float)


if __name__ == '__main__':

    # args
    lr = 3e-3
    depth = 3
    mid_layer_size = 10
    activation = torch.relu
    n_samples = 512
    batch_size = 512
    epochs = 500

    robot = rtb.models.DH.Panda()

    input_dim = robot.n
    output_dim = 3

    dataset = RoboKinSet(robot, [(0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3),
                                 (0, 1, 3)])
    print(dataset.q_vecs.shape)
    print(dataset.poses.shape)

    train_set = TensorDataset(dataset.q_vecs, dataset.poses)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    #from tqdm import tqdm


    model = Regressor(input_dim, output_dim,
                      depth, mid_layer_size,
                      activation)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for t in range(epochs):
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if t%(epochs//10) == 0:
            print(f'Epoch {t}: Loss={loss.item()}')


#    criterion(dataset.poses)
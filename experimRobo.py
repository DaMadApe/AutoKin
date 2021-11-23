"""
Regresión de cinemática de un robot
[q1 q2...qn] -> [x,y,z,*R]
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class RoboKinSet(Dataset):
    # Es permisible tener todo el dataset en un tensor porque no es
    # particularmente grande
    def __init__(self, robot, q_samples, norm=True):
        """
        q_samples = [(q_i_min, q_i_max, n_i_samples), (...), (...)]
        """
        self.robot = robot
        self.n = self.robot.n # Número de ejes
        self.q_vecs = q_samples

        # Producir las etiquetas correspondientes a cada vec de paráms
        self.poses = [self.robot.fkine(q_vec).t for q_vec in self.q_vecs]

        # Acomodar en tensores con tipo float
        self.poses = torch.tensor(self.poses, dtype=torch.float)
        self.q_vecs = torch.tensor(self.q_vecs, dtype=torch.float)

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        q_vec = self.q_vecs[idx]
        pos = self.poses[idx]
        return q_vec, pos


def q_grid(*q_samples): #q_samples: tuple[int, int, int])
    # Lista de puntos para cada parámetro
    qs = []
    for q_min, q_max, n_samples in q_samples:
        qs.append(np.linspace(q_min, q_max, n_samples))
    
    # Magia negra para producir todas las combinaciones de puntos
    q_vecs = np.meshgrid(*qs) # Probar np->torch
    q_vecs = np.stack(q_vecs, -1).reshape(-1, len(q_samples))
    return q_vecs


if __name__ == '__main__':

    import roboticstoolbox as rtb
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from experim0 import Regressor

    # args
    lr = 3e-3
    depth = 3
    mid_layer_size = 10
    activation = torch.relu
    batch_size = 512
    epochs = 500

    robot = rtb.models.DH.Puma560()

    input_dim = robot.n
    output_dim = 3

    # Muestras
    q_min = 0
    q_max = 2*np.pi
    train_set = RoboKinSet(robot, q_grid((q_min, q_max, 3),
                                         (q_min, q_max, 3),
                                         (q_min, q_max, 3),
                                         (q_min, q_max, 3),
                                         (q_min, q_max, 3),
                                         (q_min, q_max, 3)))

    val_set = RoboKinSet(robot, np.random.uniform(q_min, q_max, (1000, robot.n)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


    model = Regressor(input_dim, output_dim,
                      depth, mid_layer_size,
                      activation)

    # Entrenamiento
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    progress = tqdm(range(epochs), desc='Training')
    for _ in progress:
        # Train step
        for X, Y in train_loader:
            model.train()
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Val step
        with torch.no_grad():
            for X, Y in val_loader:
                model.eval()
                pred = model(X)
                val_loss = criterion(pred, Y)

        progress.set_postfix(Loss=loss.item(), Val=val_loss.item())
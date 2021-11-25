"""
Regresión de cinemática de un robot
[q1 q2...qn] -> [x,y,z,*R]
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class RoboKinSet(Dataset):
    def __init__(self, robot, n_samples: int, normed_q=True,
                 output_transform=None):
        self.robot = robot
        self.n = self.robot.n # Número de ejes
        self.normed_q_vecs = np.random.rand(n_samples, robot.n)

        self.output_transform = output_transform
        self.normed_q = normed_q
        self.generate_labels()

    @classmethod
    def grid_sampling(cls, robot, n_samples: list, normed_q=True):
        dataset = cls(robot, 0, normed_q)

        # Magia negra para producir todas las combinaciones de puntos
        q_vecs = np.meshgrid(*[np.linspace(0,1, int(n)) for n in n_samples])
        q_vecs = np.stack(q_vecs, -1).reshape(-1, robot.n)

        dataset.normed_q_vecs = q_vecs
        dataset.generate_labels()
        return dataset

    def generate_labels(self):
        q_min, q_max = self.robot.qlim # Límites de las juntas
        self.q_vecs = self.normed_q_vecs * (q_max-q_min) + q_min
        # Hacer cinemática directa
        self.poses = [self.robot.fkine(q_vec).t for q_vec in self.q_vecs]

        # Acomodar en tensores con tipo float
        self.poses = torch.tensor(self.poses, dtype=torch.float)
        self.normed_q_vecs = torch.tensor(self.normed_q_vecs, dtype=torch.float)
        self.q_vecs = torch.tensor(self.q_vecs, dtype=torch.float)

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        if self.normed_q:
            q_vec = self.normed_q_vecs[idx]
        else:
            q_vec = self.q_vecs[idx]
        pos = self.poses[idx]
        if self.output_transform is not None:
            q_vec = self.output_transform(pos)
        return q_vec, pos


def norm_q(robot, q_vec):
    q_min, q_max = robot.qlim # Límites de las juntas
    return (q_vec - q_min) / (q_max-q_min)

def denorm_q(robot, q_vec):
    q_min, q_max = robot.qlim # Límites de las juntas
    return q_vec * (q_max-q_min) + q_min


if __name__ == '__main__':

    import roboticstoolbox as rtb
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from experim0 import Regressor

    """
    args
    """
    lr = 1e-3
    depth = 10
    mid_layer_size = 10
    activation = torch.relu
    batch_size = 512
    epochs = 500

    robot = rtb.models.DH.Cobra600()

    input_dim = robot.n
    output_dim = 3

    """
    Muestras
    """
    n_per_q = 10
    n_samples = n_per_q ** robot.n

    ns_samples = [n_per_q] * robot.n
    train_set = RoboKinSet.grid_sampling(robot, ns_samples)

    val_set = RoboKinSet(robot, n_samples//5)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    """
    Entrenamiento
    """
    model = Regressor(input_dim, output_dim,
                      depth, mid_layer_size,
                      activation)

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

    """
    Guardar modelo
    """
    path = 'models/experimRobo/v1.pt'
    torch.save(model.state_dict(), path)

    """
    Cargar modelo
    """"""
    model = Regressor(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    """
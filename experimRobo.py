"""
Regresión de cinemática de un robot
[q1 q2...qn] -> [x,y,z,*R]
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class RoboKinSet(Dataset):
    # TODO: Normalización en __getitem__
    def __init__(self, robot, q_samples, input_transform=None,
                 output_transform=None):
        self.robot = robot
        self.n = self.robot.n # Número de ejes
        self.q_vecs = q_samples
        self.input_transform = input_transform
        self.output_transform = output_transform

        # Producir las etiquetas correspondientes a cada vec de paráms
        self.poses = [self.robot.fkine(q_vec).t for q_vec in self.q_vecs]

        # Acomodar en tensores con tipo float
        self.poses = torch.tensor(self.poses, dtype=torch.float)
        self.q_vecs = torch.tensor(self.q_vecs, dtype=torch.float)

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        q_vec = self.q_vecs[idx]
        if self.input_transform is not None:
            q_vec = self.input_transform(q_vec)
        pos = self.poses[idx]
        if self.output_transform is not None:
            q_vec = self.output_transform(pos)
        return q_vec, pos


def q_grid(*q_samples): 
    #q_samples = [(q_i_min, q_i_max, n_i_samples), (...), (...)]
    qs = [] # Lista de puntos para cada parámetro
    for q_min, q_max, n_samples in q_samples:
        qs.append(np.linspace(q_min, q_max, int(n_samples)))
    
    # Magia negra para producir todas las combinaciones de puntos
    q_vecs = np.meshgrid(*qs) # Probar np->torch
    q_vecs = np.stack(q_vecs, -1).reshape(-1, len(q_samples))
    return q_vecs

def robot_q_grid(robot, samples_per_q):
    n_samples = samples_per_q*np.ones((robot.n,1))
    q_samples = np.concatenate((robot.qlim.T, n_samples), axis=1)
    return q_grid(*q_samples)

def robot_rand_sampling(robot, n_samples):
    samples = np.random.rand(n_samples, robot.n)
    q_min, q_max = robot.qlim
    return samples * (q_max-q_min) + q_min


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

    train_qs = robot_q_grid(robot, n_per_q)
    train_set = RoboKinSet(robot, train_qs)

    val_qs = robot_rand_sampling(robot, n_samples//5)
    val_set = RoboKinSet(robot, val_qs)

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
    model = Regressor(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    """
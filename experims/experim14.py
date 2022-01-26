"""
Métodos de entrenamiento múltiple con pytorch puro.
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def train(model, train_loader, val_loader=None, epochs=10,
          lr=1e-3, criterion=nn.MSELoss(), optim=torch.optim.Adam,
          lr_scheduler=False):
    optimizer = optim(model.parameters(), lr=lr)
    progress = tqdm(range(epochs), desc='Training')
    scheduler = ReduceLROnPlateau(optimizer)#, patience=5)
    for _ in progress:
        # Train step
        # Invertir muestras para fines de exp
        for X, Y in train_loader:
            model.train()
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Val step
        if val_loader is not None:
            with torch.no_grad():
                for X, Y in val_loader:
                    model.eval()
                    pred = model(X)
                    val_loss = criterion(pred, Y)

                    if lr_scheduler:
                        scheduler.step(val_loss)
            progress.set_postfix(Loss=loss.item(), Val=val_loss.item())
        else:
            progress.set_postfix(Loss=loss.item())


if __name__ == "__main__":

    import roboticstoolbox as rtb
    from torch.utils.data import DataLoader

    from experimR import RoboKinSet
    from experim0 import MLP

    robot = rtb.models.DH.Cobra600()

    """
    Conjunto de datos
    """
    n_per_q = 6
    n_samples = n_per_q ** robot.n

    ns_samples = [n_per_q] * robot.n
    train_set = RoboKinSet.grid_sampling(robot, ns_samples)

    val_set = RoboKinSet(robot, n_samples//5)

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1024)
    # Tiempo de entrenamiento aumenta si declaro num_workers

    """
    Entrenamiento
    """
    model = MLP(input_dim=robot.n, output_dim=3,
                depth=4, mid_layer_size=10,
                activation=torch.tanh)

    train(model, train_loader, val_loader, epochs=1000,
          lr=1e-3, lr_scheduler=False)

    path = 'models/experim14_v1.pt'
    torch.save(model, path)
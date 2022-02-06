"""
Métodos de entrenamiento múltiple con pytorch puro.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def train(model, train_set, val_set=None,
          epochs=10, lr=1e-3, batch_size=32,
          criterion=nn.MSELoss(), optim=torch.optim.Adam,
          lr_scheduler=False, silent=False, log_dir=None):

    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer)#, patience=5)
    model.train()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size)
    
    if silent:
        epoch_iter = range(epochs)
    else:
        epoch_iter = tqdm(range(epochs), desc='Training')
    for epoch in epoch_iter:
        # Train step
        # Invertir muestras para fines de exp
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if log_dir is not None:
                writer.add_scalar('Loss/train', loss.item(), epoch)
                #writer.flush()
        # Val step
        if val_set is not None:
            with torch.no_grad():
                for X, Y in val_loader:
                    model.eval()
                    pred = model(X)
                    val_loss = criterion(pred, Y)
                    if lr_scheduler:
                        scheduler.step(val_loss)
                    if log_dir is not None:
                        writer.add_scalar('Loss/val', val_loss.item(), epoch)
                        #writer.flush()
            if not silent:
                epoch_iter.set_postfix(Loss=loss.item(), Val=val_loss.item())
        else:
            if not silent:
                epoch_iter.set_postfix(Loss=loss.item())
        
    if log_dir is not None:
        writer.close()


if __name__ == "__main__":

    import roboticstoolbox as rtb

    from experim1 import RoboKinSet
    from experim0 import MLP
    from experim13 import ResNet

    """
    Conjunto de datos
    """
    robot = rtb.models.DH.Puma560()

    n_per_q = 4
    n_samples = n_per_q ** robot.n
    ns_samples = [n_per_q] * robot.n

    train_set = RoboKinSet.grid_sampling(robot, ns_samples)
    val_set = RoboKinSet(robot, n_samples//5)

    """
    Entrenamiento
    """
    base_params = {'input_dim': robot.n, 
                   'output_dim': 3}

    mlp_p0 = {**base_params,
              'depth': 3,
              'mid_layer_size': 10,
              'activation': torch.tanh}
    mlp_p1 = {**base_params,
              'depth': 6,
              'mid_layer_size': 10,
              'activation': torch.tanh}

    models = [MLP(**mlp_p0), MLP(**mlp_p1), ResNet(**base_params)]


    for i, model in enumerate(models):
        train(model, train_set, val_set,
              epochs=10,
              lr=1e-3,
              lr_scheduler=False,
              log_dir='tb_logs/exp14')

        torch.save(model, f'models/experim14_v1_m{i}.pt')
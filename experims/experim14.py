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

    if lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer)#, patience=5)

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
        model.train()
        for X, Y in train_loader:
            pred = model(X)
            train_loss = criterion(pred, Y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if log_dir is not None:
                writer.add_scalar('Loss/train', train_loss.item(), epoch)
                #writer.flush()

        progress_info = {'Loss': train_loss.item()}

        # Val step
        if val_set is not None:
            with torch.no_grad():
                model.eval()
                for X, Y in val_loader:
                    pred = model(X)
                    val_loss = criterion(pred, Y)

                    if lr_scheduler:
                        scheduler.step(val_loss)

                    if log_dir is not None:
                        writer.add_scalar('Loss/val', val_loss.item(), epoch)
                        #writer.flush()
            progress_info.update({'Val': val_loss.item()})

        if not silent:
            epoch_iter.set_postfix(progress_info)

            
        
    if log_dir is not None:
        if val_set is not None:
            metrics = {'Last val loss': val_loss.item()}
        else:
            metrics = {'Last train loss': train_loss.item()}

        writer.add_hparams({**model.hparams, 'lr':lr, 'batch_size':batch_size},
                           metrics)
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

    mlp_params = [{'depth': 3,
                   'mid_layer_size': 10,
                   'activation': torch.tanh},
                   {'depth': 3,
                   'mid_layer_size': 10,
                   'activation': torch.relu},
                  {'depth': 6,
                   'mid_layer_size': 10,
                   'activation': torch.tanh}]

    resnet_params = [{'depth': 3,
                      'block_depth': 3,
                      'block_width': 6,
                      'activation': torch.tanh},
                     {'depth': 6,
                      'block_depth': 3,
                      'block_width': 6,
                      'activation': torch.tanh}]

    models = []
    for params in mlp_params:
        models.append(MLP(**base_params, **params))
    for params in resnet_params:
        models.append(ResNet(**base_params, **params))

    for i, model in enumerate(models):
        train(model, train_set, val_set,
              epochs=10,
              lr=1e-3,
              lr_scheduler=False,
              log_dir='tb_logs/exp14')

        torch.save(model, f'models/experim14/v1_m{i}.pt')
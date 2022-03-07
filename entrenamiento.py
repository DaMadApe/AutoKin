import sched
from tabnanny import check
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def train(model, train_set, val_set=None,
          epochs=10, lr=1e-3, batch_size=32,
          criterion=nn.MSELoss(), optim=torch.optim.Adam,
          lr_scheduler=False, silent=False, log_dir=None,
          checkpoint=None):
    """
    Rutina de entrenamiento para módulo de torch
    
    args:

    train_set (Dataset) : Conjunto de datos para entrenamiento
    val_set (Dataset) : Conjunto de datos para validación (opcional)
    epochs (int) : Número de recorridos al dataset
    lr (float) : Learning rate para el optimizador
    batch_size (int) : Número de muestras propagadas a la vez
    criterion (callable) : Función para evaluar la pérdida
    optim () : Clase de optimizador
    lr_scheduler (bool) : Reducir lr al frenar disminución de val_loss
    silent (bool) : Mostrar barra de progreso del entrenamiento
    log_dir (str) : Dirección para almacenar registros de Tensorboard
    checkpoint () : Cargar el estado resultante de un entrenaminto previo

    returns:
    
    checkpoint : Estado del optimizador y lr_scheduler, para reanudar entrenamiento
    """

    # TODO: Transferir datos y modelo a GPU si está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)

    optimizer = optim(model.parameters(), lr=lr)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer)#, patience=5)
        if checkpoint is not None:
            scheduler.load_state_dict(checkpoint['sheduler_state_dict'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=len(val_set))

    if silent:
        epoch_iter = range(epochs)
    else:
        epoch_iter = tqdm(range(epochs), desc='Training')

    for epoch in epoch_iter:
        # Train step
        model.train()
        for X, Y in train_loader:
            pred = model(X)
            train_loss = criterion(pred, Y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if log_dir is not None:
                writer.add_scalar('Loss/train', train_loss.item(), epoch)

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

            progress_info.update({'Val': val_loss.item()})

        if not silent:
            epoch_iter.set_postfix(progress_info)

    if log_dir is not None:
        if val_set is not None:
            metrics = {'Last val loss': val_loss.item()}
        else:
            metrics = {'Last train loss': train_loss.item()}

        writer.add_hparams({**model.hparams, 'lr':lr, 'batch_size':batch_size},
                           metric_dict=metrics, run_name='.')
        writer.close()

    checkpoint = {'optimizer_state_dict': optimizer.state_dict()}
    if lr_scheduler:
        checkpoint.update({'sheduler_state_dict': scheduler.state_dict()})

    return checkpoint


def test(model, test_set, criterion=nn.MSELoss()):
    test_loader = DataLoader(test_set, batch_size=len(test_set))
    with torch.no_grad():
        model.eval()
        for X, Y in test_loader:
            pred = model(X)
            test_loss = criterion(pred, Y)

    return test_loss


if __name__ == "__main__":

    import torch
    from torch.utils.data import random_split
    import roboticstoolbox as rtb

    from modelos import MLP
    from utils import RoboKinSet

    """
    Conjuntos de datos
    """
    robot = rtb.models.DH.Cobra600() #Puma560()
    n_samples = 10000

    full_set = RoboKinSet.random_sampling(robot, n_samples)

    # Repartir muestras entre conjuntos
    split_proportions = [0.6, 0.2, 0.2]
    # Convertir proporciones al número de muestras correspondiente
    split = [round(prop*len(full_set)) for prop in split_proportions]

    train_set, val_set, test_set = random_split(full_set, split)

    """
    Definición de modelo
    """
    model = MLP(input_dim=robot.n,
                output_dim=3,
                depth=3,
                mid_layer_size=12,
                activation=torch.tanh)

    """
    Entrenamiento
    """
    train(model, train_set, val_set=val_set,
          epochs=100,
          lr=1e-3,
          batch_size=256,
          lr_scheduler=True,
          log_dir='tb_logs/entrenamiento/cobra600')

    torch.save(model, 'models/cobra600_v1.pt')
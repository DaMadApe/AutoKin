"""
Aproximar la cinemática directa de posición de un robot con
el modelo de experim0, usando Lightning para empaquetar y
entrenar el modelo.
"""

import roboticstoolbox as rtb
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import pytorch_lightning as pl

from experimRobo import RoboKinSet
from experim0 import Regressor


class RegressorPL(pl.LightningModule):

    def __init__(self, input_dim=1, output_dim=1,
                 depth=1, mid_layer_size=10, activation=torch.tanh, lr=1e-3,):
        super().__init__()
        self.save_hyperparameters()
        self.model = Regressor(input_dim, output_dim, depth,
                               mid_layer_size, activation)

        # Se necesita definir para save_graph
        self.example_input_array = torch.zeros(1, self.hparams.input_dim)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        point, target = batch
        pred = self(point)
        loss = F.mse_loss(pred, target)
        self.log('train_loss', loss)
        self.log('hp_metric', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        point, target = batch
        pred = self(point)
        val_loss = F.mse_loss(pred, target)
        self.log('val_loss', val_loss)
        return val_loss


if __name__ == '__main__':


    from pytorch_lightning.loggers import TensorBoardLogger

    pl.seed_everything(36)

    # Hyperparams
    cpu_cores = 8 # Núcleos usados por el programa
    depth = 10
    mid_layer_size = 10
    activation = torch.relu
    lr = 1e-3
    batch_size = 512
    epochs = 500

    robot = rtb.models.DH.Cobra600()

    input_dim = robot.n
    output_dim = 3

    """
    Conjunto de datos
    """
    n_per_q = 10
    n_samples = n_per_q ** robot.n

    ns_samples = [n_per_q] * robot.n
    train_set = RoboKinSet.grid_sampling(robot, ns_samples)

    val_set = RoboKinSet(robot, n_samples//5)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    # Tiempo de entrenamiento aumenta si declaro num_workers

    """
    Entrenamiento
    """
    logger = TensorBoardLogger('lightning_logs', 'Exp3',
                               log_graph=True)

    model = RegressorPL(input_dim, output_dim,
                        depth, mid_layer_size,
                        activation, lr)

    # trainer = pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(max_epochs=epochs,
                         logger=logger)
    trainer.fit(model, train_loader, val_loader)
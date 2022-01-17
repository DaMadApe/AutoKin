"""
Aproximar la cinemática directa de posición de un robot con
el modelo de experim0, usando Lightning para empaquetar y
entrenar el modelo.
"""

from pytorch_lightning import callbacks
import roboticstoolbox as rtb
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from experimR import RoboKinSet
from experim0 import MLP


class MLP_Regressor(pl.LightningModule):

    def __init__(self, train_set, val_set, input_dim=1, output_dim=1,
                 depth=1, mid_layer_size=10, activation=torch.tanh,
                 optim=torch.optim.Adam, batch_size=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['train_set', 'val_set'])
        self.model = MLP(input_dim, output_dim, depth,
                               mid_layer_size, activation)
        self.train_set = train_set
        self.val_set = val_set
        # Se necesita definir para save_graph
        self.example_input_array = torch.zeros(1, self.hparams.input_dim)

    def configure_optimizers(self):
        optimizer = self.hparams.optim(self.parameters(), lr=self.hparams.lr)
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

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size)


if __name__ == '__main__':

    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from experim9 import FKRegressionTask

    # pl.seed_everything(36)

    # Hyperparams
    depth = 3
    mid_layer_size = 10
    activation = torch.tanh
    lr = 1e-3
    batch_size = 1024
    epochs = 10

    robot = rtb.models.DH.Orion5()

    input_dim = robot.n
    output_dim = 3

    """
    Datos
    """
    n_per_q = 12
    ns_samples = [n_per_q] * robot.n
    n_samples = n_per_q ** robot.n // 5
    train_set = RoboKinSet.grid_sampling(robot, ns_samples)
    val_set = RoboKinSet(robot, n_samples)

    """
    Entrenamiento
    """
    logger = TensorBoardLogger('lightning_logs', 'exp3',
                               log_graph=True)

    regressor = MLP(input_dim=input_dim, output_dim=output_dim,
                    depth=depth, mid_layer_size=mid_layer_size,
                    activation=activation)
    model = FKRegressionTask(regressor, train_set, val_set, batch_size, lr)

    # model = MLP_Regressor(train_set, val_set,
    #                       input_dim=input_dim, output_dim=output_dim,
    #                       depth=depth, mid_layer_size=mid_layer_size,
    #                       activation=activation, batch_size=batch_size, lr=lr)

    # Callbacks de entrenamiento
    # train_checkpoint = ModelCheckpoint()
    # last_checkpoint = ModelCheckpoint(monitor='val_loss',
    #                                   dirpath='models/experim3/')
    #                                   #save_top_k=

    early_stop_cb = EarlyStopping(monitor="val_loss")

    # trainer = pl.Trainer(fast_dev_run=True, logger=logger)
    trainer = pl.Trainer(max_epochs=epochs,
                         logger=logger,
                         #auto_lr_find=True,
                         #auto_scale_batch_size=True, # Resultado: 10000 - 20736
                         callbacks=[#checkpoint_cb,
                                    early_stop_cb])
    # trainer.tune(model)
    trainer.fit(model)
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

    def __init__(self, robot, n_per_q, input_dim=1, output_dim=1,
                 depth=1, mid_layer_size=10, activation=torch.tanh,
                 optim=torch.optim.Adam, batch_size=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()#ignore='robot'
        self.model = MLP(input_dim, output_dim, depth,
                               mid_layer_size, activation)

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
        ns_samples = [self.hparams.n_per_q] * self.hparams.robot.n
        train_set = RoboKinSet.grid_sampling(self.hparams.robot, ns_samples)
        loader = DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        n_samples = self.hparams.n_per_q ** self.hparams.robot.n // 5
        val_set = RoboKinSet(self.hparams.robot, n_samples)
        loader = DataLoader(val_set, batch_size=self.hparams.batch_size)
        return loader


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
    n_per_q = 12

    input_dim = robot.n
    output_dim = 3

    """
    Entrenamiento
    """
    logger = TensorBoardLogger('lightning_logs', 'exp3',
                               log_graph=True)

    # regressor = MLP(input_dim=input_dim, output_dim=output_dim,
    #                 depth=depth, mid_layer_size=mid_layer_size,
    #                 activation=activation)
    # model = FKRegressionTask(regressor, robot, n_per_q, batch_size, lr)

    model = MLP_Regressor(robot, n_per_q,
                          input_dim=input_dim, output_dim=output_dim,
                          depth=depth, mid_layer_size=mid_layer_size,
                          activation=activation, batch_size=batch_size, lr=lr)

    # Callbacks de entrenamiento
    # train_checkpoint = ModelCheckpoint()
        
    # last_checkpoint = ModelCheckpoint(monitor='val_loss',
    #                                   dirpath='models/experim3/')
    #                                   #save_top_k=

    early_stop_cb = EarlyStopping(monitor="val_loss")

    # trainer = pl.Trainer(fast_dev_run=True, logger=logger)
    trainer = pl.Trainer(max_epochs=epochs,
                         logger=logger,
    #                      auto_lr_find=True,
    #                      auto_scale_batch_size=True,
                         callbacks=[#checkpoint_cb,
                                    early_stop_cb])
    #trainer.tune(model)
    trainer.fit(model)
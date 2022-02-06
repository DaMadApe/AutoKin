"""
Automatizar el entrenamiento de múltiples
robots para comparar el efecto de distintas
arquitecturas de la red neuronal.
"""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from experim1 import RoboKinSet


class FKRegressionTask(pl.LightningModule):

    def __init__(self, model, train_set, val_set,
                 batch_size=64, lr=1e-3, optim=torch.optim.Adam):
        super().__init__()
        self.save_hyperparameters(ignore=['train_set', 'val_set'])
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        # Para save_graph
        self.example_input_array = torch.zeros(1, self.model.input_dim)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.hparams.optim(self.parameters(), lr=self.hparams.lr)
        return optimizer
        
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


if __name__ == "__main__":

    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from experim0 import MLP
    from experim13 import ResNet
    from experim1 import random_robot

    min_DH = np.array([0, 0, 0, 0] )
    max_DH = np.array([2*np.pi, 2, 2*np.pi, 2])
    prob_prism = 0.5
    min_n, max_n = (2, 4)

    robot = random_robot(min_DH, max_DH, prob_prism, min_n, max_n)
    # TODO: Registrar params DH del robot

    """
    fkine_all devuelve la transformación para cada junta, por lo que
    podría hacer todos los robots de 9 juntas, y aprovechar la función
    para sacar también datos de los subconjuntos de la cadena cinemática
    """
    # q = np.random.rand(100, robot.n)
    # robot.fkine_all(q).t
    # robot.plot(q)


    # Data
    n_per_q = 12
    ns_samples = [n_per_q] * robot.n
    n_samples = n_per_q ** robot.n // 5
    train_set = RoboKinSet.grid_sampling(robot, ns_samples)
    val_set = RoboKinSet(robot, n_samples)


    # Learnign params
    max_epochs = 50

    input_dim = robot.n
    output_dim = 3
    
    base_params = {'input_dim': input_dim, 
                   'output_dim': output_dim}

    mlp_p0 = {**base_params,
              'depth': 3,
              'mid_layer_size': 10,
              'activation': torch.tanh}
    mlp_p1 = {**base_params,
              'depth': 6,
              'mid_layer_size': 10,
              'activation': torch.tanh}

    logger = TensorBoardLogger('lightning_logs', 'exp9',
                               log_graph=True)

    early_stop_cb = EarlyStopping(monitor="val_loss")

    # Training
    for model in [MLP(**mlp_p0), MLP(**mlp_p1), ResNet(**base_params)]:
        task = FKRegressionTask(model, train_set, val_set)
        trainer = pl.Trainer(max_epochs=max_epochs,
                             logger=logger,
                             #auto_lr_find=True,
                             #auto_scale_batch_size=True,
                             callbacks=[early_stop_cb])
        # trainer.tune(task)
        trainer.fit(task)
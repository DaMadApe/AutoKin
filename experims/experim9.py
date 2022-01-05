"""
Automatizar el entrenamiento de múltiples
robots para comparar el efecto de distintas
arquitecturas de la red neuronal.
"""

import roboticstoolbox as rtb
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from experimR import RoboKinSet
from experim3 import RegressorPL


pl.seed_everything(36)

# Hyperparams
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
logger = TensorBoardLogger('lightning_logs', 'Exp9',
                            log_graph=True)

model = RegressorPL(input_dim, output_dim,
                    depth, mid_layer_size,
                    activation, lr)

# trainer = pl.Trainer(fast_dev_run=True)
trainer = pl.Trainer(max_epochs=epochs,
                        logger=logger)
trainer.fit(model, train_loader, val_loader)
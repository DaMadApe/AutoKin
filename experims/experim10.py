"""
Comparar métodos de optimización,
L-BFGS, Adam, SGD, etc.
"""
import roboticstoolbox as rtb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from experim1 import RoboKinSet
from experim0 import MLP
from experim9 import FKRegressionTask


pl.seed_everything(42)

# Hyperparams
depth = 10
mid_layer_size = 10
activation = torch.relu
optim = torch.optim.LBFGS
lr = 1e-3
batch_size = 512
epochs = 500

"""
Conjunto de datos
"""
robot = rtb.models.DH.Cobra600()

n_per_q = 10
n_samples = n_per_q ** robot.n
ns_samples = [n_per_q] * robot.n

train_set = RoboKinSet.grid_sampling(robot, ns_samples)
val_set = RoboKinSet(robot, n_samples//5)

# Tiempo de entrenamiento aumenta si declaro num_workers

"""
Entrenamiento
"""
logger = TensorBoardLogger('lightning_logs', 'exp10',
                            log_graph=True)

model = MLP(input_dim=robot.n, output_dim=3,
            depth=depth, mid_layer_size=mid_layer_size,
            activation=activation)

task = FKRegressionTask(model, train_set, val_set,
                         batch_size=batch_size,
                         lr=lr)


# trainer = pl.Trainer(fast_dev_run=True, logger=logger)
trainer = pl.Trainer(max_epochs=epochs,
                    logger=logger)
trainer.fit(task)
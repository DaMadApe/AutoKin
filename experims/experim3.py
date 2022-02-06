"""
Aproximar la cinemática directa de posición de un robot con
el modelo de experim0, usando Lightning para empaquetar y
entrenar el modelo.
"""
import roboticstoolbox as rtb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from experimR import RoboKinSet
from experim0 import MLP
from experim9 import FKRegressionTask

pl.seed_everything(42) # Resultados repetibles

"""
Datos
"""
robot = rtb.models.DH.Orion5()

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
# Callbacks de entrenamiento
# train_checkpoint = ModelCheckpoint()
# last_checkpoint = ModelCheckpoint(monitor='val_loss',
#                                   dirpath='models/experim3/')
#                                   #save_top_k=
early_stop_cb = EarlyStopping(monitor="val_loss")

model = MLP(input_dim=robot.n, output_dim=3,
                depth=3,
                mid_layer_size=10,
                activation=torch.tanh)

task = FKRegressionTask(model, train_set, val_set,
                         batch_size=1024,
                         lr=1e-3)


# trainer = pl.Trainer(fast_dev_run=True, logger=logger)
trainer = pl.Trainer(max_epochs=10,
                    logger=logger,
                    #auto_lr_find=True,
                    #auto_scale_batch_size=True, # Resultado: 10000 - 20736
                    callbacks=[#checkpoint_cb,
                               early_stop_cb])
# trainer.tune(task) # Buscar lr, bath_size, etc, óptimos
trainer.fit(task)
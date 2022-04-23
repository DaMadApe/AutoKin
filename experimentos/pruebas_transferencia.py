from functools import partial

import torch
import roboticstoolbox as rtb

from modelos import MLP
from utils import RoboKinSet, rand_data_split, random_robot
from experim import setup_logging, ejecutar_experimento

"""
Conjuntos de datos
"""
trans_robot = random_robot(n=4) #rtb.models.DH.UR10() # Puma560()
n_trans_samples = 500
trans_set = RoboKinSet.random_sampling(trans_robot, n_trans_samples)
trans_train_set, trans_val_set = rand_data_split(trans_set, [0.8, 0.2,])

robot = random_robot(n=4) #rtb.models.DH.Stanford()
n_samples = 500
full_set = RoboKinSet.random_sampling(robot, n_samples)
train_set, val_set, test_set = rand_data_split(full_set, [0.7, 0.2, 0.1])

assert trans_robot.n == robot.n

"""
Definición de modelo y entrenamiento
"""
n_reps = 2

model_params = {'input_dim': robot.n, 
                'output_dim': 3,
                'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.tanh}

train_params = {
                #'epochs': 1000,
                'lr': 1e-3,
                'batch_size': 256,
                'optim': partial(torch.optim.Adam, weight_decay=1e-5),
                'lr_scheduler': True}


def experim_no_trans():
    model = MLP(**model_params)
    model.fit(train_set, val_set, epochs=2000, **train_params)
    score = model.test(test_set)

    return score, model # Considerar None

def experim_trans():
    model = MLP(**model_params)
    model.fit(trans_train_set, trans_val_set, epochs=1000, **train_params)
    model.fit(train_set, val_set, epochs=1000, **train_params)
    score = model.test(test_set)

    return score, model


logger = setup_logging()

logger.info(trans_robot)
logger.info(robot)

logger.info('Sin transferencia')
ejecutar_experimento(n_reps, experim_no_trans,
                     log_all_products=False, anotar=False)

logger.info('Con transferencia')
ejecutar_experimento(n_reps, experim_trans,
                     log_all_products=False)
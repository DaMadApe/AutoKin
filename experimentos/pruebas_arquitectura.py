from functools import partial

import torch

from modelos import MLP, ResNet
from utils import FKset
from robot import RTBrobot
from experim import setup_logging, ejecutar_experimento

"""
Conjunto de datos
"""
# DH = [d, alpha, theta, a]
# robot = RTBrobot.random_robot(n=3)
robot = RTBrobot.from_name('Cobra600') #Puma560()
n_samples = 2000

dataset = FKset.random_sampling(robot, n_samples)

"""
Parámetros del experimento
"""
n_prueba = 1 # Cambiar para tener registros separados

def experim_hparams_MLP(model_params, train_params):
    train_set, val_set, test_set = dataset.rand_split([0.7, 0.15, 0.15])
    model = MLP(robot.n, 3, **model_params)
    model.fit(train_set, val_set, **train_params)
    score = model.test(test_set)
    return score, model


# TODO: Organizar archivos de conjuntos de hiperparámetros
#       Cada uno con objetivo específico (e.g. tanh vs relu)
#       para no estar atascando aquí
model_params = [{'depth': 3,
                'mid_layer_size': 8,
                'activation': torch.relu},
                {'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.relu},
                {'depth': 2,
                'mid_layer_size': 10,
                'activation': torch.relu},
                {'depth': 2,
                'mid_layer_size': 12,
                'activation': torch.relu},
                {'depth': 1,
                'mid_layer_size': 10,
                'activation': torch.relu},
                {'depth': 1,
                'mid_layer_size': 12,
                'activation': torch.relu},
                {'depth': 1,
                'mid_layer_size': 16,
                'activation': torch.relu}]

train_params = {'epochs': 500,
                'batch_size': 512,
                'lr': 1e-3,
                'lr_scheduler': True,
                'optim': partial(torch.optim.Adam, weight_decay=5e-5)}

Model_class = MLP

"""
Ejecución
"""
logger = setup_logging()

logger.info(f'Params de entrenamiento: {train_params}')

for i, params in enumerate(model_params):

    logger.info(f'Probando parámetros: {params}')

    ejecutar_experimento(3, experim_hparams_MLP,
                         params, train_params,
                         model_save_dir=f'models/arq_test{n_prueba}_m{i}.pt',
                         log_all_products=False,
                         anotar=False)
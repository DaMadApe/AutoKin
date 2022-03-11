from functools import partial
from random import random

import torch
import roboticstoolbox as rtb

from modelos import MLP
from entrenamiento import train, test
from utils import RoboKinSet, rand_data_split, random_robot

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

print(trans_robot)
print(robot)
"""
Definición de modelo y entrenamiento
"""
n_tests = 5

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


print('No trans')
for _ in range(n_tests):
    model = MLP(**model_params)
    train(model, train_set, val_set, epochs=2000, **train_params)
    print(f'Test score: {test(model, test_set)}')

print('With trans')
for _ in range(n_tests):
    model = MLP(**model_params)
    train(model, trans_train_set, trans_val_set, epochs=1000, **train_params)
    train(model, train_set, val_set, epochs=1000, **train_params)
    print(f'Test score: {test(model, test_set)}')
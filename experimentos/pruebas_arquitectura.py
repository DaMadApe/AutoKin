from functools import partial

import torch
import roboticstoolbox as rtb

from modelos import MLP, ResNet
from utils import RoboKinSet, rand_data_split, random_robot
from experim import repetir_experimento

"""
Conjunto de datos
"""
# DH = [d, alpha, theta, a]
# robot = random_robot(n=3)
robot = rtb.models.DH.Cobra600() #Puma560()
n_samples = 2000

dataset = RoboKinSet.random_sampling(robot, n_samples)

"""
Parámetros del experimento
"""
n_prueba = 1 # Cambiar para tener registros separados

def experim_hparams_MLP(model_params, train_params):
    train_set, val_set, test_set = rand_data_split(dataset, [0.7, 0.15, 0.15])
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
best_score = torch.inf

for i, params in enumerate(model_params):

    print(f'Probando parámetros: {params}')

    score, model = repetir_experimento(3, experim_hparams_MLP,
                                       params, train_params)
    
    print(f'Mejor rendimiento del conjunto: {score}')
    torch.save(model, f'models/arq_test{n_prueba}_m{i}.pt')

    if score < best_score:
        best_score = score
        best_model = model
        best_model_params = params

print(f'''Mejor conjunto de params: {best_model_params}
          {best_model}
          Con rendimiento: {best_score}''')
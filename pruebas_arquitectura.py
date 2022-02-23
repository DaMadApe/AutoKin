import torch
from torch.utils.data import random_split
import roboticstoolbox as rtb

from modelos import MLP, ResNet
from entrenamiento import train, test
from utils import RoboKinSet, random_robot

"""
Conjunto de datos
"""
# DH = [d, alpha, theta, a]
# robot = random_robot(min_DH = [0, 0, 0, 0],
#                      max_DH = [1, 2*torch.pi, 2*torch.pi, 1],
#                      n=3)
robot = rtb.models.DH.Cobra600() #Puma560()
n_samples = 3000

full_set = RoboKinSet(robot, n_samples)

# Repartir muestras entre conjuntos
split_proportions = [0.6, 0.2, 0.2]
# Convertir proporciones al número de muestras correspondiente
split = [round(prop*len(full_set)) for prop in split_proportions]

train_set, val_set, test_set = random_split(full_set, split)

"""
Definición de modelos
"""
n_prueba = 1 # Cambiar para tener registros separados

base_params = {'input_dim': robot.n, 
                'output_dim': 3}

# TODO: Organizar archivos de conjuntos de hiperparámetros
#       Cada uno con objetivo específico (e.g. tanh vs relu)
#       para no estar atascando aquí
mlp_params = [{'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.tanh},
                {'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.relu},
                {'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.tanh},
                {'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.relu},
                {'depth': 6,
                'mid_layer_size': 5,
                'activation': torch.tanh},
                {'depth': 6,
                'mid_layer_size': 5,
                'activation': torch.relu},
                {'depth': 6,
                'mid_layer_size': 10,
                'activation': torch.tanh},
                {'depth': 6,
                'mid_layer_size': 10,
                'activation': torch.relu}]

# resnet_params = [{'depth': 3,
#                     'block_depth': 3,
#                     'block_width': 6,
#                     'activation': torch.tanh},
#                     {'depth': 6,
#                     'block_depth': 3,
#                     'block_width': 6,
#                     'activation': torch.tanh}]

# training_params = [{'epochs': 100,
#                     'lr': 1e-3,
#                     'lr_scheduler': True}]

models = []
for params in mlp_params:
    models.append(MLP(**base_params, **params))
# for params in resnet_params:
#     models.append(ResNet(**base_params, **params))

"""
Entrenamiento
"""
best_score = torch.inf

for i, model in enumerate(models):

    train(model, train_set, val_set,
          epochs=100,
          lr=1e-3,
          # lr_scheduler=True,
          log_dir=f'tb_logs/arq_test{n_prueba}/modelo{i}')

    score = test(model, test_set)

    if score < best_score:
        best_score = score
        best_model = i

    # Guardar sólo el mejor modelo?
    torch.save(model, f'models/arq_test{n_prueba}_m{i}.pt')

print(f'Best model idx: {best_model}')
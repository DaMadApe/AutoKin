import torch

from modelos import MLP, ResNet
from entrenamiento import train
from utils import RoboKinSet, random_robot

"""
Conjunto de datos
"""
# DH = [d, alpha, theta, a]
robot = random_robot(min_DH = [0, 0, 0, 0],
                     max_DH = [5, 2*torch.pi, 2*torch.pi, 5])

n_per_q = 4
n_samples = n_per_q ** robot.n
ns_samples = [n_per_q] * robot.n

train_set = RoboKinSet.grid_sampling(robot, ns_samples)
val_set = RoboKinSet(robot, n_samples//5)

"""
Definición de modelos
"""
n_prueba = 1 # Cambiar para tener registros separados

base_params = {'input_dim': robot.n, 
                'output_dim': 3}

mlp_params = [{'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.tanh},
                {'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.relu},
                {'depth': 6,
                'mid_layer_size': 10,
                'activation': torch.tanh}]

resnet_params = [{'depth': 3,
                    'block_depth': 3,
                    'block_width': 6,
                    'activation': torch.tanh},
                    {'depth': 6,
                    'block_depth': 3,
                    'block_width': 6,
                    'activation': torch.tanh}]


models = []
for params in mlp_params:
    models.append(MLP(**base_params, **params))
for params in resnet_params:
    models.append(ResNet(**base_params, **params))

"""
Entrenamiento
"""
for i, model in enumerate(models):
    train(model, train_set, val_set,
            epochs=10,
            lr=1e-3,
            lr_scheduler=False,
            log_dir=f'tb_logs/arq_test{n_prueba}')

    torch.save(model, f'models/arq_test{n_prueba}_m{i}.pt')
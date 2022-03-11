from functools import partial

import torch
import roboticstoolbox as rtb

from modelos import MLP, ResNet
from utils import RoboKinSet, rand_data_split, random_robot

"""
Conjunto de datos
"""
# DH = [d, alpha, theta, a]
# robot = random_robot(n=3)
robot = rtb.models.DH.Cobra600() #Puma560()
n_samples = 2000

full_set = RoboKinSet.random_sampling(robot, n_samples)
train_set, val_set, test_set = rand_data_split(full_set, [0.6, 0.2, 0.2])

"""
Pretrain data
"""
trans_robot = random_robot(n=4) #rtb.models.DH.UR10() # Puma560()
n_trans_samples = 500
trans_set = RoboKinSet.random_sampling(trans_robot, n_trans_samples)
trans_train_set, trans_val_set = rand_data_split(trans_set, [0.8, 0.2,])
pretraining_params = {'epochs': 300,
                      'batch_size': 512,
                      'lr': 1e-3,
                      'lr_scheduler': True,
                      'optim': partial(torch.optim.Adam, weight_decay=5e-5)}

"""
Definición de modelos
"""
n_prueba = 1 # Cambiar para tener registros separados

base_params = {'input_dim': robot.n, 
                'output_dim': 3}

# TODO: Organizar archivos de conjuntos de hiperparámetros
#       Cada uno con objetivo específico (e.g. tanh vs relu)
#       para no estar atascando aquí
model_params = [{'depth': 3,
                'mid_layer_size': 10,
                'activation': torch.relu},
                {'depth': 3,
                'mid_layer_size': 8,
                'activation': torch.relu},
                {'depth': 3,
                'mid_layer_size': 6,
                'activation': torch.relu},
                {'depth': 6,
                'mid_layer_size': 6,
                'activation': torch.relu},
                {'depth': 6,
                'mid_layer_size': 4,
                'activation': torch.relu},
                {'depth': 8,
                'mid_layer_size': 4,
                'activation': torch.relu},
                {'depth': 10,
                'mid_layer_size': 3,
                'activation': torch.relu}]

training_params = {'epochs': 500,
                   'batch_size': 512,
                   'lr': 1e-3,
                   'lr_scheduler': True,
                   'optim': partial(torch.optim.Adam, weight_decay=5e-5)}

Model_class = MLP
best_of_n = 3 

"""
Entrenamiento
"""
best_global_score = torch.inf

for i, params in enumerate(model_params):

    print(f'Probando conjunto de parámetros {i}')
    best_score = torch.inf

    for _ in range(best_of_n):
        model = Model_class(**base_params, **params)

        # Preentrenamiento
        # train(model, trans_train_set, trans_val_set, **pretraining_params)

        model.fit(train_set, val_set, **training_params,
                  log_dir=f'tb_logs/arq_test{n_prueba}/modelo{i}')

        score = model.test(test_set)

        if score < best_score:
            best_score = score
            best_model = model
    
    if best_score < best_global_score:
        best_global_score = best_score
        best_global_model = best_model
        best_model_idx = i

    print(f'Mejor rendimiento del conjunto: {best_score}')

    torch.save(best_model, f'models/arq_test{n_prueba}_m{i}.pt')

print(f'''Mejor conjunto de params: {best_model_idx}
          {best_global_model}
          Con rendimiento: {best_global_score}''')
import torch
import roboticstoolbox as rtb

from modelos import MLP
from utils import RoboKinSet, rand_data_split, coprime_sines
from experim import repetir_experimento

"""
Conjuntos de datos
"""
robot = rtb.models.DH.Cobra600() #Puma560()
n_samples = 2058

c_sines = coprime_sines(robot.n, n_samples, wiggle=3)
explr_dataset = RoboKinSet(robot, c_sines)

random_dataset = RoboKinSet.random_sampling(robot, n_samples)

test_set = RoboKinSet.random_sampling(robot, 1000)

"""
Definición de modelo y entrenamiento
"""
n_reps = 3

model_params = {'input_dim': robot.n, 
                'output_dim': 3,
                'depth': 2,
                'mid_layer_size': 10,
                'activation': torch.relu}

train_params = {
                'epochs': 300,
                'lr': 1e-3,
                'batch_size': 2058,
                # 'optim': partial(torch.optim.Adam, weight_decay=1e-5),
                # 'lr_scheduler': True
               }

def experim_muestreo_MLP(dataset):
    train_set, val_set = rand_data_split(dataset, [0.8, 0.2])
    model = MLP(**model_params)
    model.fit(train_set, val_set, **train_params)
    score = model.test(test_set)
    return score, model

print('Trayectoria de exploración')
score, _ = repetir_experimento(n_reps, experim_muestreo_MLP, explr_dataset)
print(f'Best score: {score} \n')

print('Muestreo aleatorio')
score, _ = repetir_experimento(n_reps, experim_muestreo_MLP, random_dataset)
print(f'Best score: {score} \n')
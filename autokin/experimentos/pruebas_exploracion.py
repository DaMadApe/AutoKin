import torch

from autokin.modelos import MLP
from autokin.robot import RTBrobot
from autokin.muestreo import FKset
from autokin.trayectorias import coprime_sines
from autokin.experimentos.experim import setup_logging, ejecutar_experimento

"""
Conjuntos de datos
"""
robot = RTBrobot.from_name('Cobra600') #Puma560()
n_samples = 10000

c_sines = coprime_sines(robot.n, n_samples, densidad=10)
explr_dataset = FKset(robot, c_sines)

random_dataset = FKset.random_sampling(robot, n_samples)

test_set = FKset.random_sampling(robot, 1000)

"""
Definición de modelo y entrenamiento
"""
n_reps = 3

model_params = {'input_dim': robot.n, 
                'output_dim': 3,
                'depth': 2,
                'mid_layer_size': 10,
                'activation': 'relu'}

train_params = {
                'epochs': 300,
                'lr': 1e-3,
                'batch_size': 2058,
                # 'optim': partial(torch.optim.Adam, weight_decay=1e-5),
                # 'lr_scheduler': True
               }

def experim_muestreo_MLP(dataset):
    train_set, val_set = dataset.rand_split([0.8, 0.2])
    model = MLP(**model_params)
    model.fit(train_set, val_set, **train_params)
    score = model.test(test_set)
    return score, model


logger = setup_logging()

logger.info('Trayectoria de exploración')
ejecutar_experimento(n_reps, experim_muestreo_MLP, explr_dataset,
                     log_all_products=False, 
                     anotar=False)

logger.info('Muestreo aleatorio')
ejecutar_experimento(n_reps, experim_muestreo_MLP, random_dataset,
                     log_all_products=False)
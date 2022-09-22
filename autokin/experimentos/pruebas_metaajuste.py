import roboticstoolbox as rtb
import torch
from torch.utils.data import ConcatDataset, random_split

from autokin.modelos import FKModel, MLP
from autokin.muestreo import FKset
from autokin.robot import RTBrobot
from autokin.experimentos.experim import setup_logging, ejecutar_experimento


# Arreglo de los robots de rtb según n de GDL
rtb_robots = rtb.models.DH.__all__
max_n = 10
robots_by_n = [[] for _ in range(max_n+1)]
for robot_name in rtb_robots:
    robot = RTBrobot.from_name(robot_name)
    robots_by_n[robot.n].append(robot)


def meta_ajuste(modelo: FKModel,
                n_steps=10,
                n_datasets=8,
                n_samples=100,
                n_post=10,
                lr=1e-4,
                post_lr=1e-4,
                n_epochs=1,
                n_post_epochs=1):
    input_dim = modelo.input_dim
    min_DH = [1, 0, 0, 1]
    max_DH = [10, 2*torch.pi, 2*torch.pi, 10]

    for i in range(n_steps):
        print(f'Paso {i}')

        sample_robots = []
        # sample_robots.extend(robots_by_n[input_dim])

        post_sets = []

        for _ in range(n_datasets): # - len(sample_robots)):
            sample_robots.append(RTBrobot.random(n=input_dim,
                                                 min_DH=min_DH,
                                                 max_DH=max_DH))

        for robot in sample_robots:
            full_set = FKset.random_sampling(robot, n_samples+n_post)
            train_set, post_set = random_split(full_set, [n_samples,
                                                          n_post])
            modelo.fit(train_set, 
                       lr=lr,
                       epochs=n_epochs,
                       silent=True,
                       use_checkpoint=False)
            post_sets.append(post_set)

        post_set = ConcatDataset(post_sets)
        modelo.fit(post_set, 
                   lr=post_lr,
                   epochs=n_post_epochs,
                   silent=True,
                   use_checkpoint=False)


"""
Conjuntos de datos
"""
robot = RTBrobot.from_name('Cobra600') #random(n=4)
n_samples = 1000
full_set = FKset.random_sampling(robot, n_samples)
train_set, val_set, test_set = full_set.rand_split([0.7, 0.2, 0.1])

"""
Definición de modelo y entrenamiento
"""
n_reps = 5

model_params = {'input_dim': robot.n, 
                'output_dim': 3,
                'depth': 3,
                'mid_layer_size': 10,
                'activation': 'tanh'}

train_params = {'epochs': 1000,
                'lr': 1e-3,
                'batch_size': 256,
                'lr_scheduler': True}


def experim_no_ml():
    model = MLP(**model_params)
    model.fit(train_set, val_set, **train_params)
    score = model.test(test_set)

    return score, model

def experim_ml():
    model = MLP(**model_params)
    meta_ajuste(model)
    model.fit(train_set, val_set, **train_params)
    score = model.test(test_set)

    return score, model


logger = setup_logging()

logger.info('Sin MAML')
ejecutar_experimento(n_reps, experim_no_ml,
                     log_all_products=False, anotar=False)

logger.info('Con MAML')
ejecutar_experimento(n_reps, experim_ml,
                     log_all_products=False)
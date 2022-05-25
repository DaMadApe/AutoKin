from functools import partial

import torch

from modelos import MLP
from robot import SofaRobot
from utils import FKset, coprime_sines
from experim import ejecutar_experimento

"""
Conjuntos de datos
"""
robot_name = 'Trunk4C'
exp_name = 'exp1'
n_samples = 3000

robot = SofaRobot()

c_sines = coprime_sines(robot.n, n_samples, wiggle=3)
dataset = FKset(robot, c_sines)
# dataset = FKset.random_sampling(robot, n_samples)
train_set, val_set, test_set = dataset.rand_split([0.7, 0.2, 0.1])

"""
Definición de experimento
"""
def experim_ajuste():
    label = f'{torch.rand(1).item():.4f}'

    model = MLP(input_dim=robot.n,
                output_dim=robot.out_n,
                depth=3,
                mid_layer_size=10,
                activation=torch.tanh)

    model.fit(train_set, val_set=val_set,
              epochs=1000,
              lr=1e-3,
              batch_size=256,
              # optim=partial(torch.optim.Adam, weight_decay=5e-5),
              # lr_scheduler=True,
              log_dir=f'experimentos/tb_logs/p_ajuste_sofa/{robot_name}_{exp_name}/{label}'
             )

    score = model.test(test_set)

    return score, model

n_reps = 3

ejecutar_experimento(n_reps, experim_ajuste,
                     log_all_products=False,
                     model_save_dir=f'models/{robot_name}_{exp_name}.pt'
                    )

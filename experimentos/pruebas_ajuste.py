from functools import partial

import torch

from modelos import MLP
from robot import RTBrobot
from utils import coprime_sines
from muestreo import FKset
from experim import ejecutar_experimento

"""
Conjuntos de datos
"""
robot_name = 'Cobra600' #
exp_name = ''
n_samples = 5000
full_pose = False

robot = RTBrobot.from_name(robot_name, full_pose=full_pose)
c_sines = coprime_sines(robot.n, n_samples, densidad=3)
dataset = FKset(robot, c_sines)
train_set, val_set = dataset.rand_split([0.8, 0.2])
test_set = FKset.random_sampling(robot, n_samples//5)

"""
Definición de experimento
"""
def experim_ajuste():
    label = f'{torch.rand(1).item():.4f}'

    model = MLP(input_dim=robot.n,
                output_dim=6 if full_pose else 3,
                depth=3,
                mid_layer_size=10,
                activation=torch.tanh)

    model.fit(train_set, val_set=val_set,
              epochs=1000,
              lr=1e-3,
              batch_size=256,
              # optim=partial(torch.optim.Adam, weight_decay=5e-5),
              # lr_scheduler=True,
              log_dir=f'experimentos/tb_logs/p_ajuste/{robot_name}_{exp_name}/{label}'
             )

    score = model.test(test_set)

    return score, model

n_reps = 5

ejecutar_experimento(n_reps, experim_ajuste,
                     log_all_products=False,
                     model_save_dir=f'models/{robot_name}_{exp_name}.pt'
                    )

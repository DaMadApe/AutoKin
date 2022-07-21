from functools import partial

import torch
from torch.utils.data import TensorDataset, random_split
import numpy as np

from autokin.modelos import MLP
from autokin.robot import SofaRobot
from autokin.muestreo import FKset
from autokin.utils import coprime_sines, restringir
from autokin.experimentos.experim import ejecutar_experimento

"""
Conjuntos de datos
"""
robot_name = 'Trunk_LL'
exp_name = 'OF'
n_samples = 10000

robot = SofaRobot(config='LL')

def rand_split(self, proportions: list[float]):
    """
    Reparte el conjunto de datos en segmentos aleatoriamente
    seleccionados, acorde a las proporciones ingresadas.

    args:
    dataset (torch Dataset): Conjunto de datos a repartir
    proportions (list[float]): Porcentaje que corresponde a cada partición
    """
    if round(sum(proportions), ndigits=2) != 1:
        raise ValueError('Proporciones ingresadas deben sumar a 1 +-0.01')
    split = [round(prop*len(self)) for prop in proportions]
    return random_split(self, split)



q = coprime_sines(robot.n, n_samples, densidad=6)
q = restringir(q)
# dataset = FKset(robot, q)
dataset = TensorDataset(torch.tensor(np.load('sofa/q_in.npy'), dtype=torch.float),
                        torch.tensor(np.load('sofa/p_out.npy'), dtype=torch.float))

train_set, val_set, test_set = rand_split(dataset, [0.7, 0.2, 0.1])

"""
Definición de experimento
"""
def experim_ajuste():
    label = f'{torch.rand(1).item():.4f}'

    model = MLP(input_dim=robot.n,
                output_dim=robot.out_n,
                depth=3,
                mid_layer_size=10,
                activation='tanh')

    model.fit(train_set, val_set=val_set,
              epochs=1000,
              lr=1e-3,
              batch_size=256,
              # optim=partial(torch.optim.Adam, weight_decay=5e-5),
              # lr_scheduler=True,
              log_dir=f'autokin/experimentos/tb_logs/p_ajuste_sofa/{robot_name}_{exp_name}/{label}'
             )

    score = model.test(test_set)

    return score, model

n_reps = 3

ejecutar_experimento(n_reps, experim_ajuste,
                     log_all_products=False,
                     model_save_dir=f'models/{robot_name}_{exp_name}.pt'
                    )

from functools import partial

import torch
import roboticstoolbox as rtb

from modelos import MLP
from utils import RoboKinSet, rand_data_split

"""
Conjuntos de datos
"""
robot = rtb.models.DH.Cobra600() #Puma560()
n_samples = 500

full_set = RoboKinSet.random_sampling(robot, n_samples)
train_set, val_set, test_set = rand_data_split(full_set, [0.7, 0.2, 0.1])

"""
Definición de modelo y entrenamiento
"""
model = MLP(input_dim=robot.n,
            output_dim=3,
            depth=3,
            mid_layer_size=7,
            activation=torch.tanh)
model.fit(train_set, val_set=val_set,
          epochs=1000,
          lr=1e-3,
          batch_size=256,
          optim=partial(torch.optim.Adam, weight_decay=1e-5),
          lr_scheduler=True,
          log_dir='tb_logs/entrenamiento/cobra600')

print(model.test(test_set))

torch.save(model, 'models/cobra600_500samples.pt')
import os
import torch
import numpy as np
from matplotlib import pyplot as plt

"""
Ver el último dataset recolectado para un robot
"""

robot_nom = 'r3c' # 'LSL'

ds_dir = os.path.join('gui', 'app_data', 'robots', robot_nom, 'datasets')

d_set = torch.load(os.path.join(ds_dir, sorted(os.listdir(ds_dir))[-1]))

p_set = np.concatenate([d_point[1].unsqueeze(0).numpy() for d_point in d_set])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

p_trans = p_set.transpose()
# p_trans = p_trans[:,::10] # Mostrar 1 de cada 10 puntos
ax.scatter(*p_trans, color='royalblue')
ax.plot(*p_trans, color='lightblue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.tight_layout()
plt.show()
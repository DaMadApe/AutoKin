import os
import torch
from torch.nn.functional import pad
from matplotlib import pyplot as plt

from autokin.trayectorias import coprime_sines
from autokin.utils import restringir

trayec = coprime_sines(3, 5000, 7, 0)
q_plot = trayec.t().numpy()

qp = trayec.diff(dim=0).norm(dim=-1)
qp = pad(qp, (0,1), mode='constant')
qpp = trayec.diff(n=2, dim=0).norm(dim=-1)
qpp = pad(qpp, (1,1), mode='constant')

print(qp)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(*q_plot, c=qp, cmap='turbo')
# ax.scatter(*q_plot, c=qpp, cmap='turbo')

plt.tight_layout()
plt.show()
import os
import torch
from torch.nn.functional import pad
from matplotlib import pyplot as plt

from autokin.trayectorias import coprime_sines
from autokin.utils import restringir, suavizar

n = 3
n_puntos = 5000

trayec = coprime_sines(n, n_puntos, 7, 0)
# trayec = suavizar(trayec, trayec[0], 0.1)

qp = trayec.diff(dim=0).norm(dim=-1)
qp = pad(qp, (0,1), mode='constant')
qpp = trayec.diff(n=2, dim=0).norm(dim=-1)
qpp = pad(qpp, (1,1), mode='constant')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

q_plot = trayec.t().numpy()
ax.scatter(*q_plot, c=qpp, cmap='turbo')
# ax.scatter(*q_plot, c=qpp, cmap='turbo')

print(f'Longitud de q: de {n_puntos} a {len(trayec)}')
print(f'Max dq = {qp.max().item()}')
print(f'Max d2q = {qpp.max().item()}')
plt.tight_layout()
plt.show()
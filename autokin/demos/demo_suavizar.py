import os
import torch
from torch.nn.functional import pad
from matplotlib import pyplot as plt

from autokin.trayectorias import coprime_sines
from autokin.utils import restringir, suavizar

n = 3
n_puntos = 60
dq_max = 0.05

trayec = coprime_sines(n, n_puntos, 1, 0)[:n_puntos//2]
trayec_suave = suavizar(trayec, trayec[0], dq_max)

qp = trayec.diff(dim=0).norm(dim=-1)
qp = pad(qp, (0,1), mode='constant')
qpp = trayec.diff(n=2, dim=0).norm(dim=-1)
qpp = pad(qpp, (1,1), mode='constant')

# print('qpp', qpp)
oversteps = (qpp/dq_max).round().int()
# print('oversteps', oversteps)

# print('trayec', trayec)
# print('trayec_suave', trayec_suave)
qppn = trayec_suave.diff(n=2, dim=0).norm(dim=-1)
qppn = pad(qppn, (1,1), mode='constant')
# print('qppn', qppn)
oversteps_n = (qppn/dq_max).round().int()
print('oversteps_n', oversteps_n)

print(f'Longitud de q: de {len(trayec)} a {len(trayec_suave)}')
print(f'Max dq = {qp.max().item()}')
print(f'Max d2q = {qpp.max().item()}')
print(f'Max d2qn = {qppn.max().item()}')

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(*trayec.t().numpy(), c=qpp, cmap='turbo')
# ax.scatter(*trayec.t().numpy(), c=qpp, cmap='turbo')
ax1.plot(*trayec.t().numpy())
ax1.set_xlabel('q1')
ax1.set_ylabel('q2')
ax1.set_zlabel('q3')
plt.tight_layout()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
print(*trayec.t().numpy())
ax2.scatter(*trayec_suave.t().numpy(), c=qppn, cmap='turbo')
ax2.plot(*trayec_suave.t().numpy())
ax2.set_xlabel('q1')
ax2.set_ylabel('q2')
ax2.set_zlabel('q3')
plt.tight_layout()
plt.show()
import torch
import numpy as np
from matplotlib import pyplot as plt

from autokin.robot import SofaRobot
from autokin.trayectorias import coprime_sines
from autokin.utils import restringir


n_samples = 2000

robot = SofaRobot(config='LS')

q = coprime_sines(robot.n, n_samples, densidad=10)
q = torch.cat([q, 0.7*q])
 # Buenos valores: [6:10]
q = restringir(q)
q, p = robot.fkine(q)

q = q
p = p[5:-5]

# q = np.load('sofa/q_in.npy')[5:]
# p = np.load('sofa/p_out.npy')[5:]
# q = q[::5]
# p = p[::5]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(p[:,0], p[:,1], p[:,2], c=p[:,0], cmap='jet')  #c=np.arange(len(q))
# ax.plot(p[:,0], p[:,1], p[:,2], color='k', linewidth=0.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# q = np.array([[q1.item(), q2.item()] for (q1,q2,q3) in q if q3>0.95 and q3<1])
# print(q)

# fig = plt.figure()

# if robot.n ==2:
#     # Para 2 actuadores
#     ax = fig.add_subplot()
#     ax.scatter(q[:,0], q[:,1], c=np.arange(len(q)), cmap='jet')
#     ax.set_xlabel('q1')
#     ax.set_ylabel('q2')

# elif robot.n ==3:
#     # Para 3 actuadores
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(q[:,0], q[:,1], q[:,2], c=np.arange(len(q)), cmap='jet')  
#     ax.plot(q[:,0], q[:,1], q[:,2], linewidth=1.5)
#     ax.set_xlabel('q1')
#     ax.set_ylabel('q2')
#     ax.set_zlabel('q3')

plt.tight_layout()
plt.show()
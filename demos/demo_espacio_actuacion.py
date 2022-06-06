from turtle import width
import numpy as np
from matplotlib import pyplot as plt

from utils import coprime_sines, restringir
from robot import SofaRobot

n_samples = 5000

q = coprime_sines(2, n_samples, densidad=6)#2,8 # Buenos valores: [6:10]
q = restringir(q)

fig = plt.figure()

n_dim = q.size()[-1]
if n_dim ==2:
    # Para 2 actuadores
    ax = fig.add_subplot()
    ax.plot(q[:,0], q[:,1], linewidth=1.5)
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')

elif n_dim ==3:
    # Para 3 actuadores
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(q[:,0], q[:,1], q[:,2])
    ax.plot(q[:,0], q[:,1], q[:,2], linewidth=1.5)
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('q3')

# plt.tight_layout()
plt.show()
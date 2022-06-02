import numpy as np
from matplotlib import pyplot as plt

from utils import coprime_sines
from robot import SofaRobot

n_samples = 6000

robot = SofaRobot()

c_sines = coprime_sines(3, n_samples, wiggle=8)
q, p = robot.fkine(c_sines)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(p[:,0], p[:,1], p[:,2])
ax.plot(p[:,0], p[:,1], p[:,2])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(q[:,0], q[:,1], q[:,2])
ax.plot(q[:,0], q[:,1], q[:,2])

plt.tight_layout()
plt.show()
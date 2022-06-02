import numpy as np
from matplotlib import pyplot as plt

from robot import SofaRobot
from utils import coprime_sines

n_samples = 10000

robot = SofaRobot()

c_sines = coprime_sines(3, n_samples, wiggle=6)
q, p = robot.fkine(c_sines)

q, p


f = np.load('forces_out.npy')

max_f = np.max(f, axis=1)
mean_f = np.mean(f, axis=1)

q_mag = np.linalg.norm(q.numpy(), axis=1)
#q_mag = np.sum(q.numpy()**2, axis=1)

plt.plot(q_mag, max_f)
plt.xlabel('||q||') # = (q1² + q2²)¹/²')
plt.ylabel('Máximo esfuerzo interno')
plt.title('Actuación v. esfuerzo (3 actuadores)')
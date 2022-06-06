import numpy as np
from matplotlib import pyplot as plt

from robot import SofaRobot
from utils import coprime_sines

n_samples = 10000

robot = SofaRobot('LSL')

c_sines = coprime_sines(robot.n, n_samples, densidad=10)
q, p = robot.fkine(c_sines)

f = np.load('sofa/forces_out.npy')[101:]

max_f = np.max(f, axis=1)
mean_f = np.mean(f, axis=1)

q_mag = np.linalg.norm(q.numpy(), axis=1)
#q_mag = np.sum(q.numpy()**2, axis=1)

if robot.n==2:
    plt.figure()
    scatter = plt.scatter(q[:,0], q[:,1], c=max_f, cmap='turbo')
    plt.xlabel('q1')
    plt.ylabel('q2')
    plt.title('Máximo esfuerzo interno (2 actuadores)')
elif robot.n==3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(q[:,0], q[:,1], q[:,2], c=max_f, cmap='turbo')
    #ax.plot(q[:,0], q[:,1], q[:,2], linewidth=1.5)
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('q3')
    ax.set_title('Máximo esfuerzo interno (3 actuadores)')

plt.figure()
plt.plot(q_mag, max_f)
plt.xlabel('||q||') # = (q1² + q2²)¹/²')
plt.ylabel('Máximo esfuerzo interno')
plt.title(f'Actuación v. esfuerzo ({robot.n} actuadores)')

plt.show()
import torch
from matplotlib import pyplot as plt

from autokin.robot import SofaRobot, ModelRobot
from autokin.muestreo import FKset
from autokin.trayectorias import coprime_sines
from autokin.utils import restringir

robot = SofaRobot(config='LLL',
                  p_scale=torch.tensor([0.0126, 0.0098, 0.0163]),
                  p_offset=torch.tensor([ 1.0626,  0.2171, -2.1753]))

model = ModelRobot.load('gui/app_data/robots/LLL/modelos/Mod1.pt',
                        p_scale=torch.tensor([0.0126, 0.0098, 0.0163]),
                        p_offset=torch.tensor([ 1.0626,  0.2171, -2.1753]))

q_in = restringir(coprime_sines(3, 1000, 0, 0))
test_ds = FKset(robot, q_in)

pred_q = []
q_prev = torch.zeros(robot.n)
for _, p_s in test_ds:
    q = model.ikine_pi_jacob(q_start=q_prev,
                             p_target=p_s,
                             #eta=0.1
                            )

    q_prev = q
    pred_q.append(q)

# Valores objetivo
q_s, p_s = test_ds[:][0].t().numpy(), test_ds[:][1].t().numpy()
# q predecida, p resultante
q_m = torch.cat(pred_q)
q_m, p_m = robot.fkine(q_m)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# p_trans = p_trans[:,::10] # Mostrar 1 de cada 10 puntos
ax.scatter(*p_s.t().numpy(), color='royalblue')
ax.plot(*p_s.t().numpy(), color='lightblue')
ax.scatter(*p_m.t().numpy(), color='red')
ax.plot(*p_m.t().numpy(), color='orange')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.tight_layout()
plt.show()
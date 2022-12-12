import os

import torch
from matplotlib import pyplot as plt

from autokin.robot import SofaRobot, ModelRobot
from autokin.muestreo import FKset
from autokin.trayectorias import coprime_sines
from autokin.utils import restringir

# DS_PATH = 'ext_robot/experimentos/dataset_validacion.pt'
DS_PATH = 'dataset_validacion.pt'

nombre_robot = 's3L' # 'r3c'
nombre_modelo = 'mlp1' # 'mod1'

model_path = os.path.join('app_data', 'robots', nombre_robot,
                          'modelos', nombre_modelo+'.pt')

model = ModelRobot.load(model_path,
                        p_scale=torch.tensor([0.0121, 0.0096, 0.0160]),
                        p_offset=torch.tensor([1.0609, 0.1956, -2.1140]))
                        # p_scale=torch.tensor([14.0707, 30.3070, 14.3431]),
                        # p_offset=torch.tensor([-0.5610, 5.6922, 0.3297]))

d_set = torch.load(DS_PATH)
d_set.apply_p_norm = False

# Valores objetivo (tomar 1 de cada n)
n = 100
q_real = d_set[:][0][::n]
p_real = d_set[:][1][::n]

print(p_real)

pred_q = []
q_prev = torch.zeros(model.n)
for i, p_si in enumerate(p_real):
    q = model.ikine_de(
        q_start=q_prev,
        p_target=p_si,
        # strategy= 'best1bin',
        strategy= 'best1exp',
        # strategy= 'rand1exp',
        # strategy= 'randtobest1exp',
        # strategy= 'currenttobest1exp',
        # strategy= 'best2exp',
        # strategy= 'rand2exp',
        # strategy= 'randtobest1bin',
        # strategy= 'currenttobest1bin',
        # strategy= 'best2bin',
        # strategy= 'rand2bin',
        # strategy= 'rand1bin',
    )
    print(f'q_pred{i} = {q}, q_obj = {q_real[i]}')
    q_prev = q
    pred_q.append(q)

# Probar ikine
q_pred = torch.stack(pred_q)

# Probar fidelidad fkine
_, p_pred = model.fkine(q_real)

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.set_title('Modelo_CD(q_real) vs p_real')
ax1.scatter(*p_pred.t().numpy(), color='royalblue')
ax1.plot(*p_pred.t().numpy(), color='lightblue')
ax1.scatter(*p_real.t().numpy(), color='red')
ax1.plot(*p_real.t().numpy(), color='orange')
ax1.legend(['modelo(q_real)', '', 'p_real', ''])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
# ax1.set_xlim((0, 0.15))
# ax1.set_ylim((-0.2, -0.05))
# ax1.set_zlim((-0.05, 0.1))

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.set_title('Parámetros encontrados por ikine vs q_real')
ax2.scatter(*q_pred.t().numpy(), color='royalblue')
ax2.plot(*q_pred.t().numpy(), color='lightblue')
ax2.scatter(*q_real.t().numpy(), color='red')
ax2.plot(*q_real.t().numpy(), color='orange')
ax2.legend(['ikine(p_real)', '', 'q_real', ''])
ax2.set_xlabel('q1')
ax2.set_ylabel('q2')
ax2.set_zlabel('q3')

plt.tight_layout()
plt.show()
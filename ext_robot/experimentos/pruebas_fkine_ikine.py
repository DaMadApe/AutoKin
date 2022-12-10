import os

import torch
from matplotlib import pyplot as plt

from autokin.robot import SofaRobot, ModelRobot
from autokin.muestreo import FKset
from autokin.trayectorias import coprime_sines
from autokin.utils import restringir

DS_PATH = 'ext_robot/experimentos/dataset_validacion.pt'

nombre_robot = 'r3c'
nombre_modelo = 'mod1'

model_path = os.path.join('app_data', 'robots', nombre_robot,
                          'modelos', nombre_modelo+'.pt')

model = ModelRobot.load(model_path,
                        p_scale=torch.tensor([0.5, 0.5, 0.5]),
                        p_offset=torch.tensor([ 1.0, 1.0, 1.0]))

d_set = torch.load(DS_PATH)

# Valores objetivo (tomar 1 de cada n)
n = 10
q_real = d_set[:][0][::n]
p_real = d_set[:][1][::n]

pred_q = []
q_prev = torch.zeros(model.n)
for i, p_si in enumerate(p_real):
    print(q_prev)
    p_target = p_si*model.p_scale + model.p_offset
    q = model.ikine_de(
        q_start=q_prev,
        p_target=p_si,
        #eta=0.1
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
ax1.set_title('Comparación modelo_CD(q_real) vs p_real')
# p_trans = p_trans[:,::10] # Mostrar 1 de cada 10 puntos
ax1.scatter(*p_pred.t().numpy(), color='royalblue')
ax1.plot(*p_pred.t().numpy(), color='lightblue')
ax1.scatter(*p_real.t().numpy(), color='red')
ax1.plot(*p_real.t().numpy(), color='orange')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.set_title('Parámetros encontrados por ikine vs q_real')
# p_trans = p_trans[:,::10] # Mostrar 1 de cada 10 puntos
ax2.scatter(*q_pred.t().numpy(), color='royalblue')
ax2.plot(*q_pred.t().numpy(), color='lightblue')
ax2.scatter(*q_real.t().numpy(), color='red')
ax2.plot(*q_real.t().numpy(), color='orange')
ax2.set_xlabel('q1')
ax2.set_ylabel('q2')
ax2.set_zlabel('q3')

plt.tight_layout()
plt.show()
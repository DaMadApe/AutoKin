import torch
from matplotlib import pyplot as plt

from autokin.robot import SofaRobot, ModelRobot
from autokin.muestreo import FKset
from autokin.trayectorias import coprime_sines
from autokin.utils import restringir

DS_PATH = 'dataset_test_ikine.pt'

robot = SofaRobot(config='LLL',
                  p_scale=torch.tensor([0.0126, 0.0098, 0.0163]),
                  p_offset=torch.tensor([ 1.0626,  0.2171, -2.1753]))

model = ModelRobot.load('gui/app_data/robots/LLL/modelos/Mod1.pt',
                        p_scale=torch.tensor([0.0126, 0.0098, 0.0163]),
                        p_offset=torch.tensor([ 1.0626,  0.2171, -2.1753]))

# Muestreo de prueba
# q_in = restringir(coprime_sines(3, 100, 0, 0))
# d_set = FKset(robot, q_in)

# if isinstance(robot, SofaRobot):
#     robot.stop_instance()
# d_set.robot = None
# torch.save(d_set, DS_PATH)
d_set = torch.load(DS_PATH)

# Valores objetivo (tomar 1 de cada 10)
q_s = d_set[:][0][::10]
p_s = d_set[:][1][::10]

pred_q = []
q_prev = torch.zeros(robot.n)
for i, p_si in enumerate(p_s):
    print(q_prev)
    p_target = p_si*robot.p_scale + robot.p_offset
    q = model.ikine_de(
        q_start=q_prev,
        p_target=p_si,
        #eta=0.1
    )
    print(f'q_pred{i} = {q}, q_obj = {q_s[i]}')
    q_prev = q
    pred_q.append(q)

# q predecida, p resultante
q_m = torch.stack(pred_q)
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
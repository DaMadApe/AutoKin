import torch
import matplotlib.pyplot as plt

from autokin.robot import ModelRobot, RTBrobot
from autokin.trayectorias import coprime_sines

robot = RTBrobot.from_name('Cobra600')
model = ModelRobot.load('gui/app_data/robots/cobra/modelos/sdf.pt')


# q_demo = torch.rand((10, robot.n))
q_demo = coprime_sines(robot.n, 80)[:10]
_, p_demo = robot.fkine(q_demo)

plot_points = p_demo[0].unsqueeze(0)
scatter_points = p_demo[0].unsqueeze(0)

for i in range(len(p_demo)-1):
    q_start = q_demo[i+1]
    q_target = q_demo[i+1]
    _, p_start = robot.fkine(q_start)
    _, p_target = robot.fkine(q_target)
    q_inv, q_list = model.ikine_pi_jacob(q_start, p_target, eta=0.1)

    _, p_reached = robot.fkine(q_inv)
    _, p_passed = robot.fkine(q_list)

    plot_points = torch.cat((plot_points, p_passed))
    scatter_points = torch.cat((scatter_points, p_reached.unsqueeze(0)))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.plot(p_passed[:,0], p_passed[:,1], p_passed[:,2])

# ax.scatter([p_start[0], p_reached[0]],
#            [p_start[1], p_reached[1]],
#            [p_start[2], p_reached[2]])

ax.scatter(scatter_points[:,0], scatter_points[:,1], scatter_points[:,2])
ax.scatter(p_demo[:,0], p_demo[:,1], p_demo[:,2], marker='^')
ax.plot(plot_points[:,0], plot_points[:,1], plot_points[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(['Puntos alcanzados', 'Puntos solicitados', 'Iteraciones de CI'])


dists = [torch.linalg.norm(diff) for diff in (scatter_points-p_demo)]
dist_prom = sum(dists)/len(dists)

print(f'Dist prom: {dist_prom}')

plt.show()
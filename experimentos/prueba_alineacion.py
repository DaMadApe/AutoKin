from math import ceil, floor

import torch
import numpy as np
import matplotlib.pyplot as plt

from robot import SofaRobot
from utils import coprime_sines, linterp

robot = SofaRobot(config='LL')

full_q = coprime_sines(robot.n, 300, densidad=1)
pad = torch.zeros((150, robot.n))
exp_q = torch.concat([pad, full_q, pad, pad])

# _, p_sim = robot.fkine(exp_q)
p_sim = torch.tensor(np.load('sofa/p_out.npy'))

q_resamp = exp_q[::7]
p_resamp = p_sim[::3]


print(q_resamp.size())
print(p_resamp.size())

q_diff = torch.diff(q_resamp, dim=0)
p_diff = torch.diff(p_resamp, dim=0)


def bordes(serie, umbral):
    idx = torch.where(serie>umbral)
    return (idx[0][0], idx[0][-1])

def interp(p1, p2, prop):
    return p1 + prop(p2-p1)

def submuestrear(s1, s2):
    if len(s2) > len(s1):
        s1, s2 = s2, s1

    sub_s1 = torch.zeros((len(s2), s1.size()[-1]))
    samp_prop = len(s1)/len(s2)
    for i in range(len(s2)):
        j0, j1 = floor(samp_prop*i), floor(samp_prop*i)
        if j0 == j1:
            sub_s1[i] = s1[j0]
        else:
            sub_s1[i] = interp(s1[j0], s1[j1], samp_prop*i % 1)

    return sub_s1

qi, qf = bordes(q_diff.norm(dim=-1), 0.2)
pi, pf = bordes(p_diff.norm(dim=-1), 0.8)

q_cut = q_resamp[qi:qf+1]
p_cut = p_resamp[pi:pf+1]

sub_p = submuestrear(p_cut, q_cut)
sub_q_diff = torch.diff(q_cut, dim=0)
sub_p_diff = torch.diff(sub_p, dim=0)

plt.figure()
plt.bar(torch.arange(q_diff.size()[0]), q_diff.norm(dim=-1))
plt.bar([qi, qf], [1, 1], width=0.5)
fig, ax = plt.subplots()
ax.invert_yaxis()
ax.bar(torch.arange(p_diff.size()[0]), p_diff.norm(dim=-1), color='royalblue')
ax.bar([pi, pf], [2, 2], width=0.5)

plt.figure()
plt.bar(torch.arange(len(sub_p_diff)), sub_p_diff.norm(dim=-1))
fig, ax = plt.subplots()
ax.bar(torch.arange(len(sub_q_diff)), sub_q_diff.norm(dim=-1), color='royalblue')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

i=20
q_test = q_cut[i]
p_test = sub_p[i]
m_i = torch.where(exp_q == q_test)[0]

print(torch.where(exp_q == q_test)[0])
print(torch.where((p_sim-p_test).norm()<0.01)[0])
# print(q_test, p_test)
# print(exp_q[m_i-20:m_i], p_sim[m_i-20:m_i])



# # t = torch.linspace(0, torch.pi, 300)

# static_p = static_point(robot, q_sim)
# _, dyn_p = robot.fkine(q_sim.repeat(100,1))

# print(p_sim)
# print(static_p)
# print(dyn_p[-2])
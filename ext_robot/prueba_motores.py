import time

from ext_robot.client import ExtInstance

inst = ExtInstance()
for i in range(5):
    print('Ciclo ', i)
    for j in range(1, 4):
        q = [0]*4
        q[j] = 80
        print('enviando q')
        inst.send_q_esp(q)
        print('enviado')
        print('fin espera')

inst.send_q_esp([0]*4)
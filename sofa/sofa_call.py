import os
import platform
import subprocess

import numpy as np

# Ajustar paths según compu en la que se ejecuta el programa
if platform.system() == 'Windows':
    SOFA_ROOT = os.path.join('C:', 'Users', 'ralej', 'Downloads', 'SofaSoftRobot')
    RUN_PATH = os.path.join(SOFA_ROOT, 'bin', 'runSofa.exe')
else:
    SOFA_ROOT = os.path.join('/home', 'damadape', 'SOFA_robosoft')
    RUN_PATH = os.path.join(SOFA_ROOT, 'bin', 'runSofa')

PYTHONPATH = os.path.join(SOFA_ROOT, 'plugins', 'SofaPython3', 'lib', 'python3', 'site-packages')

SIM_PATH = os.path.join(os.path.dirname(__file__), 'sofa_sim.py')
IN_FILE = os.path.join('sofa', 'q_in.npy')
OUT_FILE = os.path.join('sofa', 'p_out.npy')
CONFIG_FILE = os.path.join('sofa', 'config.txt')


def sofa_fkine(q, config='LSL', headless=True):
    n_wait = 100
    wait = np.zeros((n_wait, q.shape[-1]))
    q = np.concatenate([wait, q])

    with open(CONFIG_FILE, 'w') as output:
        output.write(config)

    np.save(IN_FILE, q)

    op = '-g batch' if headless else '-a'

    # env = os.environ.copy()
    # env.update({'SOFA_ROOT': SOFA_ROOT,
    #             'PYTHONPATH': PYTHONPATH})

    subprocess.call(f'{RUN_PATH} {op} -n {q.shape[0]} \"{SIM_PATH}\"',
                    shell=True) #, env=env)

    p = np.load(OUT_FILE)
    os.remove(IN_FILE)
    os.remove(OUT_FILE)
    return p[n_wait:]


# def ramp(q1, q2, N):
#     assert q1.size() == q2.size(), 'Tensores de diferente tamaño'
#     n_dim = q1.size()[-1]
#     ramp = np.zeros(N, n_dim)

#     for i in range(n_dim):
#         ramp[:,i] = np.linspace


if __name__ == "__main__":
    # export PYTHONPATH="/home/damadape/SOFA_robosoft/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH"
    # export SOFA_ROOT="/home/damadape/SOFA_robosoft"

    import matplotlib.pyplot as plt
    from autokin.trayectorias import coprime_sines

    N = 1000
    # q = [np.zeros(N), np.linspace(0, 1, N), np.ones(N), np.linspace(1, 0, N)]
    # q = np.stack(q, axis=0).T
    q  = np.linspace([0,0,0], [0, 0, 1], 100)

    # q = coprime_sines(3, N, densidad=2).numpy()
    # qs = np.zeros((10, 3))
    
    # q = np.full((20, 3), 0.7)
    # q = np.repeat([[0.9, 0.9, 0.9]], 20, axis=0)
    # q = np.zeros((100, 3))
    
    #q = np.concatenate([qs, q])
    p = sofa_fkine(q, headless=False)
    # p = np.load('p_out.npy')

    print(q.shape)
    print(p.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p[:,0], p[:,1], p[:,2])
    ax.plot(p[:,0], p[:,1], p[:,2])
    
    #ax.set_xlabel('x')
    #ax.set_zlabel('z')
    plt.show()
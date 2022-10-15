import os
import numpy as np


SIM_PATH = os.path.join(os.path.dirname(__file__), 'sofa_sim.py')

def sofa_fkine(q, config='LSL', headless=True):
    n_wait = 100
    wait = np.zeros((n_wait, q.shape[-1]))
    q = np.concatenate([wait, q])

    with open('sofa/config.txt', 'w') as output:
        output.write(config)

    np.save('sofa/q_in.npy', q) # -n {len(q)}

    op = '-g batch' if headless else '-a'

    os.system(f'~/SOFA_robosoft/bin/runSofa {op} -n {q.shape[0]} \'{SIM_PATH}\'')

    p = np.load('sofa/p_out.npy')
    os.remove('sofa/q_in.npy')
    os.remove('sofa/p_out.npy')
    return p[n_wait:]

def setup(): # No funciona: no quedan las variables
    os.system('export SOFA_ROOT=\"/home/damadape/SOFA_robosoft\"')
    os.system('export PYTHONPATH=\"/home/damadape/SOFA_robosoft/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH\"')

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
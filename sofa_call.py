import os
import numpy as np


def sofa_fkine(q):
    np.save('q_data', q) # -n {len(q)}
    os.system(f'~/SOFA_robosoft/bin/runSofa -g batch -n {len(q)} \'/home/damadape/Documents/Autokin/sofa_tst2.py\'')
    p = np.load('p_out.npy')
    return p


def setup():
    os.system('export SOFA_ROOT=\"/home/damadape/SOFA_robosoft\"')
    os.system('export PYTHONPATH=\"/home/damadape/SOFA_robosoft/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH\"')


if __name__ == "__main__":

    N = 100
    q = [np.arange(N), np.zeros(N), np.ones(N), np.arange(N)]
    q = np.stack(q, axis=0).T
    p = sofa_fkine(q)

    print(p)

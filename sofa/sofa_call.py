import os
import time
import socket
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

PORT = 6969
BUFFER_SIZE = 1024 # bytes


class Sofa_instance:
    def __init__(self, config='LSL', headless=True):
        self.proc = None
        self.headless = headless

        # HACK: Esta implementación previene que existan
        #       instancias simultáneas, repetirían config
        with open(CONFIG_FILE, 'w') as output:
            output.write(config)

    def fkine(self, q : np.ndarray):
        q = q.astype(float)

        if self.proc is None or not self.is_alive():
            self.start_proc()
            time.sleep(1)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("localhost", PORT))

            q_encoded = q.tobytes()
            # Enviar tamaño de q
            sock.send(len(q_encoded).to_bytes(4, byteorder='big'))
            # Enviar q
            sock.sendall(q_encoded)

            data = bytearray()
            while True:
                msg = sock.recv(BUFFER_SIZE)
                if not msg:
                    break
                data.extend(msg)

            p_out = np.frombuffer(data).reshape(-1, 3)

        return q, p_out

    def start_proc(self):
        # op = '-g batch' if self.headless else '-a'
        op = ('-g', 'batch') if False else ('-a')
        env = os.environ.copy()
        env.update({'PYTHONPATH': PYTHONPATH})

        # Funciona en windows, proc.kill() no cierra Sofa
        # command = f'{RUN_PATH} {op} \"{SIM_PATH}\"' # -n {q.shape[0]}
        # self.proc = subprocess.Popen(command, shell=True, env=env)

        # No incluye opción headless, no está probado en windows
        # Pero proc.kill() sí cierra Soffa
        self.proc = subprocess.Popen([RUN_PATH, op, SIM_PATH], 
                                     shell=False, env=env)

    def stop(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def is_alive(self):
        return self.proc.poll() is None


# def ramp(q1, q2, N):
#     assert q1.size() == q2.size(), 'Tensores de diferente tamaño'
#     n_dim = q1.size()[-1]
#     ramp = np.zeros(N, n_dim)

#     for i in range(n_dim):
#         ramp[:,i] = np.linspace


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from autokin.trayectorias import coprime_sines

    N = 1000
    # q = [np.zeros(N), np.linspace(0, 1, N), np.ones(N), np.linspace(1, 0, N)]
    # q = np.stack(q, axis=0).T
    q  = np.linspace([0,0,0], [0, 5, 10], 100)

    # q = coprime_sines(3, N, densidad=2).numpy()
    # qs = np.zeros((10, 3))
    
    # q = np.full((20, 3), 0.7)
    # q = np.repeat([[0.9, 0.9, 0.9]], 20, axis=0)
    # q = np.zeros((100, 3))
    
    #q = np.concatenate([qs, q])

    instance = Sofa_instance(headless=False)

    time.sleep(5)
    p, q = instance.fkine(q)

    # Graficar resultado
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p[:,0], p[:,1], p[:,2])
    ax.plot(p[:,0], p[:,1], p[:,2])
    
    #ax.set_xlabel('x')
    #ax.set_zlabel('z')
    plt.show()
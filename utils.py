import numpy as np

import torch
from torch.utils.data import Dataset

import roboticstoolbox as rtb


class RoboKinSet(Dataset):
    """
    Producir un conjunto de puntos (configuración,posición) de un robot
    definido con la interfaz de un robot DH de Peter Corke.
    
    Los puntos se escogen aleatoriamente en el espacio de parámetros.

    robot () : Cadena cinemática para producir ejemplos
    n_samples (int) : Número de ejemplos
    normed_q (bool) : Devolver ejemplos de q normalizados respecto al robot
    output_transform (callable) : Transformación que aplicar a vectores
                                  de posición devueltos
    """
    def __init__(self, robot, n_samples: int, normed_q=True,
                 output_transform=None):
        self.robot = robot
        self.n = self.robot.n # Número de ejes
        self.normed_q_vecs = np.random.rand(n_samples, robot.n)

        self.output_transform = output_transform
        self.normed_q = normed_q
        self.generate_labels()

    @classmethod
    def grid_sampling(cls, robot, n_samples: list, normed_q=True):
        """
        Constructor alternativo
        Toma muestras del espacio de juntas en un patrón de cuadrícula
        """
        dataset = cls(robot, 0, normed_q)

        # Magia negra para producir todas las combinaciones de puntos
        q_vecs = np.meshgrid(*[np.linspace(0,1, int(n)) for n in n_samples])
        q_vecs = np.stack(q_vecs, -1).reshape(-1, robot.n)

        dataset.normed_q_vecs = q_vecs
        dataset.generate_labels()
        return dataset

    def generate_labels(self):
        self.q_vecs = denorm_q(self.robot, self.normed_q_vecs)
        # Hacer cinemática directa
        self.poses = [self.robot.fkine(q_vec).t for q_vec in self.q_vecs]

        # Acomodar en tensores con tipo float
        self.poses = torch.tensor(np.array(self.poses), dtype=torch.float)
        self.normed_q_vecs = torch.tensor(self.normed_q_vecs, dtype=torch.float)
        self.q_vecs = torch.tensor(self.q_vecs, dtype=torch.float)

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        if self.normed_q:
            q_vec = self.normed_q_vecs[idx]
        else:
            q_vec = self.q_vecs[idx]
        pos = self.poses[idx]
        if self.output_transform is not None:
            pos = self.output_transform(pos)
        return q_vec, pos


def norm_q(robot, q_vec):
    """
    Normalizar vector de actuación respecto a los límites en
    las juntas del robot
    """
    q_min, q_max = robot.qlim # Límites de las juntas
    return (q_vec - q_min) / (q_max - q_min)


def denorm_q(robot, q_vec):
    """
    Extender un vector de valores 0 a 1 al rango completo de
    actuación del robot.
    """
    q_min, q_max = robot.qlim
    return q_vec * (q_max - q_min) + q_min


def random_robot(min_DH, max_DH, p_P=0.5, min_n=2, max_n=9, n=None):
    """
    Robot creado a partir de parámetros DH aleatorios.
    
    args:
    min_DH (list) : Mínimos valores posibles de [d, alpha, theta, a]
    max_DH (list) : Máximos valores posibles de [d, alpha, theta, a]
    p_P (float) : Probabilidad de una junta prismática
    min_n (int) : Mínimo número posible de juntas
    max_n (int) : Máximo número posible de juntas
    n (int) : Número de juntas; si se define, se ignora min_n, max_n
    """
    # rtb.DHLink([d, alpha, theta, a, joint_type])  rev=0, prism=1
    min_DH = np.array(min_DH)
    max_DH = np.array(max_DH)

    links = []

    if n is not None:
        n_joints = n
    else:
        n_joints = np.random.randint(min_n, max_n+1)

    for _ in range(n_joints):
        DH_vals = (np.random.rand(4) - min_DH) / (max_DH - min_DH)
        d, alpha, theta, a = DH_vals
        is_prism = np.random.rand() < p_P

        if is_prism:
            links.append(rtb.DHLink(alpha=alpha,theta=theta, a=a, sigma=1,
                                    qlim=[0, 1.5*max_DH[0]]))
        else:
            links.append(rtb.DHLink(d=d, alpha=alpha, a=a, sigma=0))
                         #qlim=np.array([0, 1.5*max_DH[0]])))
    return rtb.DHRobot(links)
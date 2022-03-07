from multiprocessing.sharedctypes import Value
import numpy as np

import torch
from torch.utils.data import Dataset, random_split

import roboticstoolbox as rtb


class RoboKinSet(Dataset):
    """
    Producir un conjunto de puntos (configuración,posición) de un robot
    definido con la interfaz de un robot DH de Peter Corke.
    
    Los puntos se escogen aleatoriamente en el espacio de parámetros.

    robot () : Cadena cinemática para producir ejemplos
    q_vecs (int) : Número de ejemplos
    normed_q (bool) : Devolver ejemplos de q normalizados respecto al robot
    output_transform (callable) : Transformación que aplicar a vectores
                                  de posición devueltos
    uniform_q_noise (float) : Cantidad de ruido uniforme aplicado a ejemplos q
        Se aplica antes de estirar (denorm) q a los límites del robot
    normal_q_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a ejemplos q
        Se aplica antes de estirar (denorm) q a los límites del robot
    """
    def __init__(self, robot, q_vecs: torch.Tensor, normed_q=True,
                 output_transform=None,
                 uniform_q_noise=0, normal_q_noise=0,
                 uniform_p_noise=0, normal_p_noise=0):

        is_q_normed = np.all(q_vecs>=0) and np.all(q_vecs<=1)
        if not(is_q_normed):
            raise ValueError('q_vecs debe ir normalizado a intervalo [0,1]')

        self.robot = robot
        self.q_vecs = q_vecs
        self.normed_q = normed_q
        self.output_transform = output_transform

        self.n = self.robot.n # Número de ejes

        self.q_noise = (uniform_q_noise*torch.rand(len(self), self.n) +
                        normal_q_noise*torch.randn(len(self), self.n))

        self.p_noise = (uniform_p_noise*torch.rand(len(self), self.n) +
                        normal_p_noise*torch.randn(len(self), self.n))

        self._generate_labels()

    @classmethod
    def random_sampling(cls, robot, n_samples: int, **kwargs):
        """
        Constructor alternativo
        Toma muestras uniformemente distribuidas en el espacio de juntas

        args:
        robot () : Cadena cinemática para producir ejemplos
        n_samples (int) : Número de ejemplos
        """
        q_vecs = np.random.rand(n_samples, robot.n)
        return cls(robot, q_vecs, **kwargs)

    @classmethod
    def grid_sampling(cls, robot, n_samples: list, **kwargs):
        """
        Constructor alternativo
        Toma muestras del espacio de juntas en un patrón de cuadrícula

        args:
        robot () : Cadena cinemática para producir ejemplos
        n_samples (int list) : Número de divisiones por junta
        """
        # Magia negra para producir todas las combinaciones de puntos
        q_vecs = np.meshgrid(*[np.linspace(0,1, int(n)) for n in n_samples])
        q_vecs = np.stack(q_vecs, -1).reshape(-1, robot.n)
        return cls(robot, q_vecs, **kwargs)

    def _generate_labels(self):
        self.denormed_q_vecs = denorm_q(self.robot, self.q_vecs)
        # Hacer cinemática directa
        self.pos_vecs = [self.robot.fkine(q_vec).t for q_vec in self.denormed_q_vecs]

        # Acomodar en tensores con tipo float
        self.pos_vecs = torch.tensor(np.array(self.pos_vecs), dtype=torch.float)
        self.q_vecs = torch.tensor(self.q_vecs, dtype=torch.float)
        self.denormed_q_vecs = torch.tensor(self.denormed_q_vecs, dtype=torch.float)
        
        # self.q_vecs = self.q_vecs + self.q_noise
        # self.pos_vecs = self.pos_vecs + self.p_noise

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        if self.normed_q:
            q_vec = self.q_vecs[idx]
        else:
            q_vec = self.denormed_q_vecs[idx]

        pos = self.pos_vecs[idx]
        if self.output_transform is not None:
            pos = self.output_transform(pos)

        return q_vec, pos


def rand_data_split(dataset: Dataset, proportions: list[float]):
    """
    Reparte un conjunto de datos en segmentos aleatoriamente
    seleccionados, acorde a las proporciones ingresadas.

    args:
    dataset (torch Dataset): Conjunto de datos a repartir
    proportions (list[float]): Porcentaje que corresponde a cada partición
    """
    if round(sum(proportions), ndigits=2) != 1:
        raise ValueError('Proporciones ingresadas deben sumar a 1 +-0.01')
    split = [round(prop*len(dataset)) for prop in proportions]
    return random_split(dataset, split)


def norm_q(robot, q_vec):
    """
    Normalizar vector de actuación respecto a los límites en
    las juntas del robot
    """
    q_min, q_max = robot.qlim.astype(np.float32) # Límites de las juntas
    return (q_vec - q_min) / (q_max - q_min)


def denorm_q(robot, q_vec):
    """
    Extender un vector de valores 0 a 1 al rango completo de
    actuación del robot.
    """
    q_min, q_max = robot.qlim.astype(np.float32)
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


def rand_data_split(dataset: Dataset, proportions: list[float]):
    """
    Reparte un conjunto de datos en segmentos aleatoriamente
    seleccionados, acorde a las proporciones ingresadas.

    args:
    dataset (torch Dataset): Conjunto de datos a repartir
    proportions (list[float]): Porcentaje que corresponde a cada partición
    """
    if round(sum(proportions), ndigits=2) != 1:
        raise ValueError('Proporciones ingresadas deben sumar a 1 +-0.01')
    split = [round(prop*len(dataset)) for prop in proportions]
    return random_split(dataset, split)
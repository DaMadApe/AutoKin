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
    q_vecs (torch.Tensor) : Lista de vectores de actuación normalizados para generar ejemplos
    normed_q (bool) : Devolver ejemplos de q normalizados respecto al robot
    output_transform (callable) : Transformación que aplicar a vectores
                                  de posición devueltos
    q_uniform_noise (float) : Cantidad de ruido uniforme aplicado a ejemplos q
        Se aplica antes de estirar (denorm) q a los límites del robot
    q_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a ejemplos q
        Se aplica antes de estirar (denorm) q a los límites del robot
    p_uniform_noise (float) : Cantidad de ruido uniforme aplicado a etiquetas pos
    p_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a etiquetas pos
    """
    def __init__(self, robot, q_vecs: torch.Tensor, 
                 normed_q=True, output_transform=None,
                 q_uniform_noise=0, q_normal_noise=0,
                 p_uniform_noise=0, p_normal_noise=0):

        is_q_normed = torch.all(q_vecs>=0) and torch.all(q_vecs<=1)
        if not(is_q_normed):
            raise ValueError('q_vecs debe ir normalizado a intervalo [0,1]')

        self.robot = robot
        self.q_vecs = q_vecs
        self.normed_q = normed_q
        self.output_transform = output_transform

        self.n = self.robot.n # Número de ejes

        self.q_noise = (q_uniform_noise*torch.rand(len(self), self.n) +
                        q_normal_noise*torch.randn(len(self), self.n))

        self.p_noise = (p_uniform_noise*torch.rand(len(self), 3) +
                        p_normal_noise*torch.randn(len(self), 3))

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
        q_vecs = torch.rand(n_samples, robot.n)
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
        q_vecs = torch.meshgrid(*[torch.linspace(0,1, int(n)) for n in n_samples],
                                indexing='ij')
        q_vecs = torch.stack(q_vecs, -1).reshape(-1, robot.n)
        return cls(robot, q_vecs, **kwargs)

    def _generate_labels(self):
        self.denormed_q_vecs = denorm_q(self.robot, self.q_vecs)
        # Hacer cinemática directa
        self.pos_vecs = [self.robot.fkine(q_vec.numpy()).t for q_vec in self.denormed_q_vecs]

        # Acomodar en tensores con tipo float
        self.pos_vecs = torch.tensor(np.array(self.pos_vecs), dtype=torch.float)

        self.q_vecs = self.q_vecs + self.q_noise
        self.pos_vecs = self.pos_vecs + self.p_noise

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


def norm_q(robot, q_vec: torch.Tensor):
    """
    Normalizar vector de actuación respecto a los límites en
    las juntas del robot
    """
    q_min, q_max = torch.tensor(robot.qlim, dtype=torch.float32)
    # q_min, q_max = robot.qlim.astype(np.float32)
    return (q_vec - q_min) / (q_max - q_min)


def denorm_q(robot, q_vec: torch.Tensor):
    """
    Extender un vector de valores 0 a 1 al rango completo de
    actuación del robot.
    """
    q_min, q_max = torch.tensor(robot.qlim, dtype=torch.float32)
    # q_min, q_max = robot.qlim.astype(np.float32)
    return q_vec * (q_max - q_min) + q_min


def random_robot(min_DH=None, max_DH=None, p_P=0.5, min_n=2, max_n=9, n=None):
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

    if min_DH is None:
        min_DH = [0, 0, 0, 0]
    if max_DH is None:
        max_DH = [1, 2*np.pi, 2*np.pi, 1]

    min_DH = np.array(min_DH)
    max_DH = np.array(max_DH)

    if np.any(min_DH > max_DH):
        raise ValueError('Parámetros mínimos de DH no son menores a los máximos')

    links = []

    if n is not None:
        n_joints = n
    else:
        n_joints = np.random.randint(min_n, max_n+1)

    for _ in range(n_joints):
        DH_vals = np.random.rand(4) * (max_DH - min_DH) + min_DH
        d, alpha, theta, a = DH_vals
        is_prism = np.random.rand() < p_P

        if is_prism:
            links.append(rtb.DHLink(alpha=alpha,theta=theta, a=a, sigma=1,
                                    qlim=[0, 1.5*max_DH[0]]))
        else:
            links.append(rtb.DHLink(d=d, alpha=alpha, a=a, sigma=0))
                         #qlim=np.array([0, 1.5*max_DH[0]])))
    return rtb.DHRobot(links)
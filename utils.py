import numpy as np

import torch
from torch.utils.data import Dataset, random_split

import roboticstoolbox as rtb

# TODO: Mandar a muestreo.py
class FKset(Dataset):
    """
    Producir un conjunto de puntos (configuración,posición) de un robot
    definido con la interfaz de un robot DH de Peter Corke.
    
    Los puntos se escogen aleatoriamente en el espacio de parámetros.

    robot () : Cadena cinemática para producir ejemplos
    q_vecs (torch.Tensor) : Lista de vectores de actuación para generar ejemplos
    q_uniform_noise (float) : Cantidad de ruido uniforme aplicado a ejemplos q
    q_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a ejemplos q
    p_uniform_noise (float) : Cantidad de ruido uniforme aplicado a etiquetas pos
    p_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a etiquetas pos
    """
    def __init__(self, robot, q_vecs: torch.Tensor,
                 q_uniform_noise=0, q_normal_noise=0,
                 p_uniform_noise=0, p_normal_noise=0):

        is_q_normed = torch.all(q_vecs>=0) and torch.all(q_vecs<=1)
        if not(is_q_normed):
            raise ValueError('q_vecs debe ir normalizado a intervalo [0,1]')

        self.robot = robot
        self.q_vecs = q_vecs

        self.n = self.robot.n # Número de ejes

        self.q_uniform_noise = q_uniform_noise
        self.q_normal_noise = q_normal_noise
        self.p_uniform_noise = p_uniform_noise
        self.p_normal_noise = p_normal_noise

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
        # Hacer cinemática directa
        self.q_vecs, self.p_vecs = self.robot.fkine(self.q_vecs)

        q_noise = (self.q_uniform_noise*torch.rand(len(self), self.n) +
                   self.q_normal_noise*torch.randn(len(self), self.n))

        p_noise = (self.p_uniform_noise*torch.rand(len(self), 3) +
                   self.p_normal_noise*torch.randn(len(self), 3))

        self.q_vecs = self.q_vecs + q_noise
        self.p_vecs = self.p_vecs + p_noise

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        return self.q_vecs[idx], self.p_vecs[idx]

    def rand_split(self, proportions: list[float]):
        """
        Reparte el conjunto de datos en segmentos aleatoriamente
        seleccionados, acorde a las proporciones ingresadas.

        args:
        dataset (torch Dataset): Conjunto de datos a repartir
        proportions (list[float]): Porcentaje que corresponde a cada partición
        """
        if round(sum(proportions), ndigits=2) != 1:
            raise ValueError('Proporciones ingresadas deben sumar a 1 +-0.01')
        split = [round(prop*len(self)) for prop in proportions]
        return random_split(self, split)

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
        max_DH = [1, 2*torch.pi, 2*torch.pi, 1]

    min_DH = torch.tensor(min_DH)
    max_DH = torch.tensor(max_DH)

    if torch.any(min_DH > max_DH):
        raise ValueError('Parámetros mínimos de DH no son menores a los máximos')

    links = []

    if n is not None:
        n_joints = n
    else:
        n_joints = torch.randint(min_n, max_n+1, (1,))

    for _ in range(n_joints):
        DH_vals = torch.rand(4) * (max_DH - min_DH) + min_DH
        d, alpha, theta, a = DH_vals
        is_prism = torch.rand(1) < p_P

        if is_prism:
            links.append(rtb.DHLink(alpha=alpha,theta=theta, a=a, sigma=1,
                                    qlim=[0, 1.5*max_DH[0]]))
        else:
            links.append(rtb.DHLink(d=d, alpha=alpha, a=a, sigma=0))
                         #qlim=np.array([0, 1.5*max_DH[0]])))
    return rtb.DHRobot(links)


def coprime_sines(n_dim, n_points, wiggle=0):
    # https://www.desmos.com/calculator/m4pjhqjgz6
    """
    Genera trayectoria paramétrica explorando cada dimensión
    con sinusoides de frecuencias coprimas.

    El muestreo de estas curvas suele concentrarse en los
    límites del espacio, y pasa por múltiples
    coordenadas con valor 0, por lo que podría atinarle a
    las singularidades de un robot si se usan las curvas
    en el espacio de parámetros.
    """
    coefs = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    coefs = torch.tensor(coefs) * 2*torch.pi
    points = torch.zeros((n_points, n_dim))

    t = torch.linspace(0, 1, n_points)
    t += 0.5 * torch.rand((n_points)) / n_points
    #points = 0.3*torch.rand((n_dim, n_points))

    for i in range(n_dim):
        points[:, i] = torch.sin(coefs[i+wiggle]*t) /2 + 0.5
    return points
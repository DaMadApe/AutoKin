import logging

import torch
from torch.utils.data import (Dataset, TensorDataset,
                              ConcatDataset, random_split)

from autokin.robot import Robot


class FKset(Dataset):
    """
    Producir un conjunto de puntos (configuración,posición) de un robot
    definido con la interfaz de un robot DH de Peter Corke.
    
    Los puntos se escogen aleatoriamente en el espacio de parámetros.

    robot () : Cadena cinemática para producir ejemplos
    q_vecs (torch.Tensor) : Lista de vectores de actuación para generar ejemplos
    full_pose (bool) : Usar posición + orientación o sólo posición
    q_uniform_noise (float) : Cantidad de ruido uniforme aplicado a ejemplos q
    q_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a ejemplos q
    p_uniform_noise (float) : Cantidad de ruido uniforme aplicado a etiquetas pos
    p_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a etiquetas pos
    """
    def __init__(self, robot: Robot, q_vecs: torch.Tensor,
                 q_uniform_noise=0, q_normal_noise=0,
                 p_uniform_noise=0, p_normal_noise=0):

        is_q_normed = torch.all(q_vecs>=0) and torch.all(q_vecs<=1)
        if not(is_q_normed):
            raise ValueError('q_vecs debe ir normalizado a intervalo [0,1]')

        self.robot = robot
        self.q_in_vecs = q_vecs

        self.n = self.robot.n # Número de ejes
        self.out_n = self.robot.out_n

        self.q_uniform_noise = q_uniform_noise
        self.q_normal_noise = q_normal_noise
        self.p_uniform_noise = p_uniform_noise
        self.p_normal_noise = p_normal_noise

        self.apply_p_norm = True
        self.include_dq = False

        self.p_scale = self.robot.p_scale.clone()
        self.p_offset = self.robot.p_offset.clone()

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
        self.q_vecs, self.p_vecs = self.robot.fkine(self.q_in_vecs)
        # Agregar ruido según configuración
        q_noise = (self.q_uniform_noise*torch.rand(len(self), self.n) +
                   self.q_normal_noise*torch.randn(len(self), self.n))
        p_noise = (self.p_uniform_noise*torch.rand(len(self), self.out_n) +
                   self.p_normal_noise*torch.randn(len(self), self.out_n))
        self.q_vecs = self.q_vecs + q_noise
        self.p_vecs = self.p_vecs + p_noise
        # Generar vector de dq para entrenamiento de SelPropEnsemble
        self.q_diff = self.q_vecs.diff(dim=0)
        if len(self.q_diff): # Para casos de len(q)==1
            self.q_diff = torch.cat([self.q_diff[0].unsqueeze(0), self.q_diff])
        else:
            self.q_diff = torch.zeros(1, self.q_vecs.shape[-1])

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        q, p = self.q_vecs[idx], self.p_vecs[idx]
        if self.apply_p_norm:
            p = p * self.p_scale + self.p_offset
        # hasattr para compatibilidad con datasets creados antes del cambio
        if hasattr(self, 'include_dq') and self.include_dq:
            q = torch.concat([q, self.q_diff[idx]])
        return q, p

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

        # HACK: Compensa por algunos valores que no suman la longitud original
        split[0] += (len(self) - sum(split))

        return random_split(self, split)
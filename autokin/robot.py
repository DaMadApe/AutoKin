import logging
from typing import Union

import torch
from torch.autograd.functional import jacobian

import roboticstoolbox as rtb

from autokin.robot_mixins import IkineMixin
from autokin.utils import random_robot, suavizar
from sofa.sofa_call import SofaInstance
from ext_robot.client import ExtInstance


logger = logging.getLogger('autokin')


class Robot(IkineMixin):
    """
    Clase base para definir un robot que interactúe con
    el resto del programa.
    El resto del código requiere que las q estén en el
    rango [0, 1], por lo que se debe hacer normalización
    si se enlaza con un robot que opere con otros valores
    """
    def __init__(self, n_act: int, out_n: int):
        super().__init__()
        self.n = n_act # Número de actuadores
        self.out_n = out_n # Número de salidas

        self.q_min: torch.Tensor
        self.q_max: torch.Tensor
        self.p_scale: torch.Tensor
        self.p_offset: torch.Tensor

    def _denorm_q(self, q: torch.Tensor):
        return q * (self.q_max - self.q_min) + self.q_min

    def fkine(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Toma un tensor q [N,M] de N vectores de actuación de M dimensión.
        Devuelve un par de tensores q',p de tamaños [N',M], [N',3], donde
        N'>=N.
        """
        raise NotImplementedError

    def jacob(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RTBrobot(Robot):
    """
    Clase de interfaz para usar los robots de la librería
    de Peter Corke. 
    
    Para usar otra librería robótica con el resto del
    programa, se debe definir una interfaz como esta.
    """
    def __init__(self,
                 robot: rtb.DHRobot,
                 full_pose=False,
                 p_scale: torch.Tensor = None,
                 p_offset: torch.Tensor = None):
        self.robot = robot
        self.full_pose = full_pose
        super().__init__(robot.n, out_n=6 if full_pose else 3)

        self.q_min, self.q_max = torch.tensor(self.robot.qlim)

        if p_scale is None:
            p_scale = torch.ones(self.out_n)
        if p_offset is None:
            p_offset = torch.zeros(self.out_n)

        self.p_scale = p_scale
        self.p_offset = p_offset

    @classmethod
    def from_name(cls, name, **rtb_kwargs):
        robot = getattr(rtb.models.DH, name)()
        return cls(robot, **rtb_kwargs)

    @classmethod
    def random(cls, full_pose=False, *args, **kwargs):
        return cls(random_robot(*args, **kwargs), full_pose)

    def __repr__(self):
        return self.robot.__repr__()

    def fkine(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        denormed_q = self._denorm_q(q)
        se3_pose = self.robot.fkine(denormed_q.detach().numpy())

        if self.full_pose:
            pos = torch.tensor(se3_pose.t)
            ori = torch.tensor(se3_pose.eul())
            p = torch.cat((pos, ori), dim=1) # Juntar pos y rpy en un vector de 6 elems
        else:
            p = torch.tensor(se3_pose.t) # Extraer componente de traslación

        p = p.float()
        return q, p

    def jacob(self, q: torch.Tensor) -> torch.Tensor:
        J = self.robot.jacob0(q.numpy())

        if not self.full_pose:
            J = J[:3]

        return torch.tensor(J).float()


class SofaRobot(Robot):
    """
    Interfaz para iniciar y controlar un robot suave simulado en SOFA
    """
    def __init__(self,
                 config = 'LSL',
                 headless : bool = True,
                 q_min: torch.Tensor = None,
                 q_max: torch.Tensor = None,
                 p_scale: torch.Tensor = None,
                 p_offset: torch.Tensor = None,
                 max_dq: float = 0.1):
        if config not in ['LLLL','LSLS','LSSL', 'LSL', 'SLS', 'LLL', 'LS', 'LL']:
            raise ValueError('Configuración inválida')

        super().__init__(n_act=len(config), out_n=3)
        self.config = config
        self._headless = headless
        
        if q_min is None:
            q_min = torch.zeros(self.n)
        if q_max is None:
            q_max = 15 * torch.ones(self.n)
        if p_scale is None:
            p_scale = torch.ones(self.out_n)
        if p_offset is None:
            p_offset = torch.zeros(self.out_n)

        self.q_min = q_min
        self.q_max = q_max
        self.p_scale = p_scale
        self.p_offset = p_offset
        self.max_dq = max_dq

        self.q_prev = torch.zeros(self.n)

        self.SofaInstance = SofaInstance(config=config,
                                         headless=headless)

    @property
    def headless(self):
        return self._headless

    @headless.setter
    def headless(self, val: bool):
        if self._headless != val:
            self._headless = val
            self.SofaInstance.headless=val
            self.stop_instance()

    def fkine(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compatibilidad para vectores individuales
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        soft_q = suavizar(q=q,
                          q_prev=self.q_prev,
                          dq_max=self.max_dq)

        denormed_q = self._denorm_q(soft_q)

        p_out = self.SofaInstance.fkine(denormed_q.numpy())
        p_out = torch.tensor(p_out, dtype=torch.float)

        self.q_prev = soft_q[-1]

        return soft_q, p_out

    def start_instance(self):
        self.SofaInstance.start_proc()

    def stop_instance(self):
        self.SofaInstance.stop()

    def running(self) -> bool:
        return self.SofaInstance.is_alive()


class ExternRobot(Robot):
    """
    Interfaz para operar a un robot físico (el objetivo
    experimental). Esta interfaz define la interacción con
    los actuadores del robot, 

    Para conectar otro robot o usar conjuntos disintos de
    sensores, se debe definir otra interfaz como esta.
    """
    def __init__(self, 
                 n: int,
                 q_min: torch.Tensor = None,
                 q_max: torch.Tensor = None,
                 p_scale: torch.Tensor = None,
                 p_offset: torch.Tensor = None,
                 max_dq: float = 0.1):
                 
        super().__init__(n, out_n=3)

        if q_min is None:
            q_min = torch.zeros(self.n, dtype=int)
        if q_max is None:
            q_max = 100 * torch.ones(self.n, dtype=int)
        if p_scale is None:
            p_scale = torch.ones(self.out_n)
        if p_offset is None:
            p_offset = torch.zeros(self.out_n)

        self.q_min = q_min
        self.q_max = q_max
        self.p_scale = p_scale
        self.p_offset = p_offset
        self.max_dq = max_dq

        self.q_prev = torch.zeros(self.n)

        self.client = ExtInstance()

    def fkine(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compatibilidad para vectores individuales
        if q.ndim == 1:
            q = q.unsqueeze(0)

        soft_q = suavizar(q=q,
                          q_prev=self.q_prev,
                          dq_max=self.max_dq)
        self.q_prev = soft_q[-1]

        denormed_q = self._denorm_q(soft_q)

        p_out = self.client.fkine(denormed_q)
        # Tomar sólo la porción muestreada
        q_out = soft_q[:len(p_out)]

        return q_out, p_out

    def status(self) -> dict:
        mcu_status, cam_status = self.client.status()
        return {'Microcontrolador': mcu_status,
                'Sistema de cámaras': cam_status}

    def running(self) -> bool:
        mcu_status, cam_status = self.client.status()
        return mcu_status and cam_status


class ModelRobot(Robot):
    """
    Interfaz para operar una red neuronal que aproxima
    la cinemática de otro robot.
    """
    def __init__(self, 
                 model,
                 p_scale: torch.Tensor = None,
                 p_offset: torch.Tensor = None):

        super().__init__(model.input_dim, model.output_dim)
        self.model = model

        if p_scale is None:
            p_scale = torch.ones(self.out_n)
        if p_offset is None:
            p_offset = torch.zeros(self.out_n)

        self.p_scale = p_scale
        self.p_offset = p_offset

    @classmethod
    def load(cls, 
             model_dir,
             p_scale: torch.Tensor = None,
             p_offset: torch.Tensor = None):
        model = torch.load(model_dir)
        return cls(model, p_scale, p_offset)

    def fkine(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.model.eval()
            p = self.model(q)
            p = (p - self.p_offset) / self.p_scale
        return q, p

    def jacob(self, q: torch.Tensor) -> torch.Tensor:
        return jacobian(self.model, q).squeeze()
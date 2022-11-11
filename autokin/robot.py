import torch
from torch.autograd.functional import jacobian

import roboticstoolbox as rtb

from autokin.robot_mixins import IkineMixin
from autokin.utils import random_robot
from sofa.sofa_call import SofaInstance
from ext_robot.client import ExtInstance


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

    def fkine(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Toma un tensor q [N,M] de N vectores de actuación de M dimensión.
        Devuelve un par de tensores q',p de tamaños [N',M], [N',3], donde
        N'>=N.
        """
        raise NotImplementedError

    def jacob(self, q):
        raise NotImplementedError


class RTBrobot(Robot):
    """
    Clase de interfaz para usar los robots de la librería
    de Peter Corke. 
    
    Para usar otra librería robótica con el resto del
    programa, se debe definir una interfaz como esta.
    """
    def __init__(self, robot: rtb.DHRobot, full_pose=False):
        self.robot = robot
        self.full_pose = full_pose
        super().__init__(robot.n, out_n=6 if full_pose else 3)

    @classmethod
    def from_name(cls, name, full_pose=False):
        robot = getattr(rtb.models.DH, name)()
        return cls(robot, full_pose)

    @classmethod
    def random(cls, full_pose=False, *args, **kwargs):
        return cls(random_robot(*args, **kwargs), full_pose)

    def __repr__(self):
        return self.robot.__repr__()

    def fkine(self, q: torch.Tensor):
        if len(q.shape) == 1:
            q.unsqueeze(0)

        denormed_q = self.denorm(q)
        se3_pose = self.robot.fkine(denormed_q.detach().numpy())

        if self.full_pose:
            pos = torch.tensor(se3_pose.t)
            ori = torch.tensor(se3_pose.eul())
            p = torch.cat((pos, ori), dim=1) # Juntar pos y rpy en un vector de 6 elems
        else:
            p = torch.tensor(se3_pose.t) # Extraer componente de traslación

        p = p.float()
        return q, p

    def jacob(self, q):
        J = self.robot.jacob0(q.numpy())

        if not self.full_pose:
            J = J[:3]

        return torch.tensor(J).float()

    def denorm(self, q):
        q_min, q_max = torch.tensor(self.robot.qlim, dtype=torch.float32)
        return q * (q_max - q_min) + q_min


class SofaRobot(Robot):
    """
    Interfaz para iniciar y controlar un robot suave simulado en SOFA
    """
    def __init__(self,
                 config = 'LSL',
                 headless : bool = True,
                 q_min: list[float] = None,
                 q_max: list[float] = None,
                 p_scale : float = 1):
        if config not in ['LLLL','LSLS','LSSL', 'LSL', 'SLS', 'LLL', 'LS', 'LL']:
            raise ValueError('Configuración inválida')
        if q_min is None:
            q_min = [0] * len(config)
        if q_max is None:
            q_max = [10] * len(config)

        self.config = config
        self._headless = headless
        self.q_min = torch.tensor(q_min)
        self.q_max = torch.tensor(q_max)
        self.p_scale = p_scale

        self.SofaInstance = SofaInstance(config=config,
                                           headless=headless)
        super().__init__(n_act=len(config), out_n=3)

    @property
    def headless(self):
        return self._headless

    @headless.setter
    def headless(self, val: bool):
        if self._headless != val:
            self._headless = val
            self.SofaInstance.headless=val
            self.stop_instance()

    def fkine(self, q: torch.Tensor):
        # Compatibilidad para vectores individuales
        if len(q.shape) == 1:
            q.unsqueeze(0)
        
        scaled_q = (q + self.q_min) * (self.q_max - self.q_min)
        q_out, p_out = self.SofaInstance.fkine(scaled_q.numpy())
        p_out = torch.tensor(p_out, dtype=torch.float)
        p_out = self.p_scale * p_out
        
        return q_out, p_out

    def start_instance(self):
        self.SofaInstance.start_proc()

    def stop_instance(self):
        self.SofaInstance.stop()

    def running(self):
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
                 q_min: list[float] = None,
                 q_max: list[float] = None,
                 p_scale : float = 1):
        super().__init__(n, out_n=3)
        if q_min is None:
            q_min = [0] * n
        if q_max is None:
            q_max = [100] * n

        self.q_min = torch.tensor(q_min)
        self.q_max = torch.tensor(q_max)
        self.p_scale = p_scale

        self.client = ExtInstance()

    def fkine(self, q: torch.Tensor):
        # Compatibilidad para vectores individuales
        if len(q.shape) == 1:
            q.unsqueeze(0)

        scaled_q = (q + self.q_min) * (self.q_max - self.q_min)
        q_out, p_out = self.client.fkine(scaled_q.numpy())
        p_out = torch.tensor(p_out, dtype=torch.float)
        p_out = self.p_scale * p_out

    def status(self):
        mcu_status, cam_status = self.client.status()
        return {'Microcontrolador': mcu_status,
                'Sistema de cámaras': cam_status}


class ModelRobot(Robot):
    # Meter métodos a un mixin?
    """
    Interfaz para operar una red neuronal que aproxima
    la cinemática de otro robot.
    """
    def __init__(self, model):
        self.model = model
        super().__init__(model.input_dim, model.output_dim)

    @classmethod
    def load(cls, model_dir):
        model = torch.load(model_dir)
        model.eval() # TODO: revisar ubicación de esto
        return cls(model)

    def fkine(self, q: torch.Tensor):
        with torch.no_grad():
            p = self.model(q)
        return q, p

    def jacob(self, q: torch.Tensor):
        return jacobian(self.model, q).squeeze()
from multiprocessing.sharedctypes import Value
import torch
from torch.autograd.functional import jacobian

import roboticstoolbox as rtb

from robot_mixins import IkineMixin
from utils import random_robot
from sofa.sofa_call import sofa_fkine


class Robot(IkineMixin):
    """
    Clase base para definir un robot que interactúe con
    el resto del programa.
    El resto del código requiere que las q estén en el
    rango [0, 1], por lo que se debe hacer normalización
    si se enlaza con un robot que opere con otros valores
    """
    def __init__(self, n_act, out_n):
        self.n = n_act # Número de actuadores
        self.out_n = out_n # Número de salidas

    def fkine(self, q: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, robot, full_pose=False):
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

    def fkine(self, q):
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

    def __init__(self, config='LSL'):
        if config not in ['LLLL','LSLS','LSSL', 'LSL', 'SLS', 'LLL', 'LS', 'LL']:
            raise ValueError('Configuración inválida')
        self.config = config
        super().__init__(n_act=len(config), out_n=3)

    def fkine(self, q, headless=True):
        p = sofa_fkine(q.numpy(), headless=headless, config=self.config)
        p = torch.tensor(p, dtype=torch.float)
        return q, p


class ExternRobot(Robot):
    """
    Interfaz para operar a un robot físico (el objetivo
    experimental). Esta interfaz define la interacción con
    los actuadores del robot, 

    Para conectar otro robot o usar conjuntos disintos de
    sensores, se debe definir otra interfaz como esta.
    """
    def __init__(self, n, name):
        super().__init__(n)
        self.name = name

    def fkine(self, q):
        # cam.start()
        # mcu.write(q, params)
        # q_full = mcu.read()
        # p_full = cam.stop()
        # q, p = align(q_full, p_full)
        # return q, p # Para poder devolver q más grande que la de entrada
        pass


class ModelRobot(Robot):
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

    def fkine(self, q):
        return self.model(q)

    def jacob(self, q):
        return jacobian(self.model, q).squeeze()
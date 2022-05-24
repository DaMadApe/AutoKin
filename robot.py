import torch
from torch.autograd.functional import jacobian

import roboticstoolbox as rtb

from utils import random_robot
from ikine import ikine_pi_jacob


class Robot():
    """
    Clase base para definir un robot que interactúe con
    el resto del programa.
    El resto del código requiere que las q estén en el
    rango [0, 1], por lo que se debe hacer normalización
    si se enlaza con un robot que opere con otros valores
    """
    def __init__(self, n):
        self.n = n # Número de actuadores

    def fkine(self, q: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """
        Toma un tensor q [N,M] de N vectores de actuación de M dimensión.
        Devuelve un par de tensores q',p de tamaños [N',M], [N',3], donde
        N'>=N.
        """
        raise NotImplementedError

    # def jacobian(self, q) ?

    def ikine(self, p): # Hacer método de Robot?
        raise NotImplementedError


class RTBrobot(Robot):
    """
    Clase de interfaz para usar los robots de la librería
    de Peter Corke. 
    
    Para usar otra librería robótica con el resto del
    programa, se debe definir una interfaz como esta.
    """
    def __init__(self, robot):
        super().__init__(robot.n)
        self.robot = robot

    @classmethod
    def from_name(cls, name):
        robot = getattr(rtb.models.DH, name)()
        return cls(robot)

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(random_robot(*args, **kwargs))

    def __repr__(self):
        return self.robot.__repr__()

    def fkine(self, q):
        denormed_q = self.denorm(q)
        p = self.robot.fkine(denormed_q.detach().numpy()).t
        p = torch.tensor(p, dtype=torch.float)
        return q, p

    def jacobian(self, q):
        return torch.tensor(self.robot.jacob0(q.numpy())[:3]).float()

    def ikine(self, q_start, p_target):
        return ikine_pi_jacob(q_start, p_target, eta=0.1,
                              fkine=self.fkine, 
                              jacob=self.jacobian)

    def denorm(self, q):
        q_min, q_max = torch.tensor(self.robot.qlim, dtype=torch.float32)
        return q * (q_max - q_min) + q_min


class RTBnonLinear(Robot):

    def __init__(self, n_joints):
        self.virtual_bot = RTBrobot.random(min_n=n_joints*2, max_n=n_joints*4)

    def q_trans(self, q):
        trans_q = q
        return trans_q

    def fkine(self, q):
        trans_q = self.q_trans(q)
        return self.virtual_bot.fkine(trans_q)

    def jacobian(self, q):
        trans_q = self.q_trans(q)
        return self.virtual_bot.jacobian(trans_q)


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
        super().__init__(model.input_dim)

    @classmethod
    def load(cls, model_dir):
        model = torch.load(model_dir)
        model.eval() # TODO: revisar ubicación de esto
        return cls(model)

    def fkine(self, q):
        return self.model(q)

    def jacobian(self, q):
        return jacobian(self.model, q).squeeze()
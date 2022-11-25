from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeVar, Optional

from autokin.robot import *
from autokin.modelos import *


@dataclass
class Reg:
    """
    Base de registro de un objeto preinicializado
    """
    inits = {} # {cls_id1 : Cls1, ...}

    nombre: str
    cls_id : str
    kwargs : dict

    def init_obj(self):
        cls = self.inits[self.cls_id]
        return cls(**self.kwargs)


R = TypeVar('R', bound=Reg)


class SelectionList(list[R]):
    """
    Base para la lista de robots y la lista de modelos de cada uno.
    """
    def __init__(self):
        self._select : Optional[int] = None

    def agregar(self, nueva_entrada: R) -> bool:
        for entrada in self:
            if entrada.nombre == nueva_entrada.nombre:
                return False

        self.append(nueva_entrada)
        return True

    def selec(self) -> Optional[R]:
        return None if self._select is None else self[self._select]

    def seleccionar(self, idx: int):
        self._select = idx

    def copiar(self, origen: int, nuevo_nombre: str) -> bool:
        for entry in self:
            if entry.nombre == nuevo_nombre:
                return False

        nueva_entrada = deepcopy(self[origen])
        nueva_entrada.nombre = nuevo_nombre
        self.append(nueva_entrada)
        return True

    def eliminar(self, idx: int):
        self.pop(idx)
        if self._select is not None and idx < self._select:
            self._select -=1
        elif self._select == idx:
            self._select = None


@dataclass
class ModelReg(Reg):
    """
    Registro de modelo 
    """
    inits = {cls.__name__ : cls for cls in [MLP, 
                                            ResNet,
                                            MLPEnsemble,
                                            ResNetEnsemble]}

    trains: list = field(default_factory=list)


@dataclass
class RoboReg(Reg):
    """
    Registro de cada robot guardado junto con sus modelos.

    Se guardan los parámetros de inicialización del robot y
    su lista de modelos
    """
    inits = {"Externo" : ExternRobot,
             "Sim. RTB" : RTBrobot.from_name,
             "Sim. SOFA" : SofaRobot}

    modelos: SelectionList[ModelReg] = field(default_factory=SelectionList)

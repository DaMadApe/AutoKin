from copy import deepcopy
from dataclasses import dataclass, field

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


class SelectionList:
    """
    Base para la lista de robots y la lista de modelos de cada uno.
    """
    def __init__(self):
        self._data : list[Reg] = []
        self._select : int = None

    def __getitem__(self, idx):
        return self._data[idx]

    def __delitem__(self, idx):
        del self._data[idx]

    def __len__(self):
        return len(self._data)

    def agregar(self, nueva_entrada: Reg):
        for entrada in self:
            if entrada.nombre == nueva_entrada.nombre:
                return False

        self._data.append(nueva_entrada)
        return True

    def selec(self):
        if self._select is None:
            return None
        else:
            return self._data[self._select]

    def seleccionar(self, idx: int):
        self._select = idx

    def copiar(self, origen: int, nuevo_nombre: str):
        for entry in self:
            if entry.nombre == nuevo_nombre:
                return False

        nueva_entrada = deepcopy(self._data[origen])
        nueva_entrada.nombre = nuevo_nombre
        self._data.append(nueva_entrada)
        return True

    def eliminar(self, idx: int):
        self._data.pop(idx)
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

    epochs: int = 0


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

    modelos: SelectionList = field(default_factory=SelectionList)

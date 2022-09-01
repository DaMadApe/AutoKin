import torch

import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field

from autokin.robot import Robot
from autokin.modelos import FKModel
"""
TODO: Renombrar a "app_state.py", "control.py" o algo así?
"""
DB_SAVE_DIR = 'gui/app_data/robotRegs'


@dataclass
class Reg:
    """
    Base de registro
    """
    nombre: str

@dataclass
class ModelReg(Reg):
    """
    Registro de modelo 
    """
    modelo: FKModel
    train_kwargs: dict = field(default_factory=dict)
    epocs: int = 0


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
class RoboReg(Reg):
    """
    Registro de cada robot guardado junto con sus modelos.

    Cada entrada se identifica con un nombre y corresponde a un
    objeto Robot, con una lista de modelos asociados, almacenados
    como identificadores textuales únicos
    """
    robot: Robot
    modelos: SelectionList = field(default_factory=SelectionList)

    def _model_filename(self, idx):
        return f'{self.nombre}_m{self.model_regs[idx].nombre}.pt'

    def _log_filename(self, idx):
        return f'{self.nombre}_m{self.model_regs[idx]}.log'

    def load_model(self, idx):
        return torch.load(self._model_filename(idx))


class Singleton(type):
    """
    Metaclase para asegurar que cada incialización de la clase
    devuelva la misma instancia en lugar de crear una nueva
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class UIController(metaclass=Singleton):
    """
    Controlador para acoplar GUI con lógica del programa.
    """
    def __init__(self):
        self.pickle_path = DB_SAVE_DIR + '.pkl'
        self.cargar()

    def cargar(self):
        if os.path.isfile(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                self.robots = pickle.load(f)
        else:
            self.robots = SelectionList()

    def guardar(self):
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.robots, f)

    def robot_selec(self):
        return self.robots.selec()

    def modelos(self):
        return self.robots.selec().modelos

    def modelo_selec(self):
        robot_selec = self.robots.selec()
        if robot_selec is None:
            return None
        else:
            return robot_selec.modelos.selec()

    def agregar_robot(self, nombre: str, robot: Robot):
        nuevo = RoboReg(nombre, robot)
        return self.robots.agregar(nuevo)

    def agregar_modelo(self, nombre: str, modelo: FKModel):
        return self.modelos().agregar(ModelReg(nombre, modelo))


if __name__ == "__main__":
    from autokin.robot import RTBrobot

    robot = RTBrobot.from_name('Cobra600')

    controlador = UIController()
    # controlador.agregar_robot("qwe", robot)
    print(controlador.robot_selec())
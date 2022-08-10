import torch

import os
import pickle
import string
import random
from copy import deepcopy
from dataclasses import dataclass, field

from autokin.robot import Robot

"""
TODO: Renombrar a "app_state.py", "control.py" o algo así?
"""

@dataclass
class ModelReg:
    """
    Registro de modelo 
    
    """
    model_id: str
    train_kwargs: dict = field(default_factory=dict)
    quick_log: dict = field(default_factory=dict)

    def __init__(self) -> None:
        pass


@dataclass
class RoboReg:
    """
    Registro de cada robot guardado junto con sus modelos.

    Cada entrada se identifica con un nombre y corresponde a un
    objeto Robot, con una lista de modelos asociados, almacenados
    como identificadores textuales únicos
    """
    nombre: str
    robot: Robot
    model_regs: list[ModelReg] = field(default_factory=list)

    def _model_filename(self, idx):
        return f'{self.nombre}_{self.model_ids[idx]}.pt'

    def _log_filename(self, idx):
        return f'{self.nombre}_{self.model_ids[idx]}.log'

    def _new_id(self):
        alphanum = string.ascii_letters+string.digits
        new_id = str().join(random.choices(alphanum, k=8))
        for model_reg in self.model_regs:
            if model_reg.model_id == new_id:
                return self._new_id()
        return new_id

    def load_model(self, idx):
        return torch.load(self._model_filename(idx))

    def new_model(self, model):
        self.model_regs.append(ModelReg(model, self._new_id()))


class RobotDatabase:

    def __init__(self):
        self._data : list[RoboReg] = []
        self._select : int = None

    def __getitem__(self, idx):
        return self._data[idx]

    def __delitem__(self, idx):
        del self._data[idx]

    def __len__(self):
        return len(self._data)

    def seleccionar(self, idx: int):
        self._select = idx

    def agregar(self, nombre: str, robot: Robot):
        for entry in self:
            if entry.nombre == nombre:
                return False

        self._data.append(RoboReg(nombre, robot))
        return True

    def copiar(self, origen: int, nuevo_nombre: str):
        for entry in self:
            if entry.nombre == nuevo_nombre:
                return False

        nueva_entrada = deepcopy(self._data[origen])
        nueva_entrada.nombre = nuevo_nombre
        self._data.append(nueva_entrada)
        return True

    def eliminar(self, idx: int):
        if idx < self._select:
            self._select -=1


DB_SAVE_DIR = 'gui/app_data/robotRegs'

class UIController:
    """
    Controlador para acoplar GUI con lógica del programa.
    """
    def __init__(self) -> None:
        self.pickle_path = DB_SAVE_DIR + '.pkl'
        if os.path.isfile(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                self.robots = pickle.load(f)
        else:
            self.robots = RobotDatabase()

    def _update_datafile(self):
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.robots, f)

    def agregar_robot(self, nombre: str, robot: Robot):
        nuevo = RoboReg(nombre, robot)
        agregado = self.robots.agregar(nuevo)
        if agregado:
            self._update_datafile()
        return agregado

    def seleccionar_robot(self, idx: int):
        self.robots.seleccionar(idx)
        self._update_datafile()

    def copiar_robot(self, origen: int, nuevo_nombre: str):
        agregado = self.robots.copiar(origen, nuevo_nombre)
        if agregado:
            self._update_datafile()
        return agregado

    def eliminar_robot(self, idx: int):
        self.robots.eliminar(idx)
        self._update_datafile()

    def agregar_modelo(self, nombre: str, modelo: FKModel):
        seleccionado = self.robots.selec()
        agregado = seleccionado.modelos.agregar(ModelReg(nombre, modelo))
        if agregado:
            self._update_datafile()
        return agregado

    def seleccionar_modelo(self, idx: int):
        seleccionado = self.robots.selec()
        seleccionado.modelos.seleccionar(idx)
        self._update_datafile()

    def copiar_modelo(self, origen: int, nuevo_nombre: str):
        seleccionado = self.robots.selec()
        agregado = seleccionado.modelos.copiar(origen, nuevo_nombre)
        if agregado:
            self._update_datafile()
        return agregado

    def eliminar_modelo(self, idx: int):
        seleccionado = self.robots.selec()
        seleccionado.modelos.eliminar(idx)
        self._update_datafile()


if __name__ == "__main__":
    database = RobotDatabase()
    print(list(database))
    print(type(database._data))
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field

from autokin.robot import Robot


@dataclass
class RoboReg:
    """
    Registro para cada robot guardado.

    Cada entrada se identifica con un nombre y corresponde a un
    objeto Robot, que puede tener múltiples modelos asociados.
    """
    nombre: str
    robot: Robot
    model_ids: list[str] = field(default_factory=list)


class RobotDatabase:

    def __init__(self, path):
        self.path = path + '.pkl' 

        if os.path.isfile(self.path):
            print("Cargando...")
            with open(self.path, 'rb') as f:
                self._data = pickle.load(f)
        else:
            print("Nuevo archivo")
            self._data = []
            self.update_datafile()

    def __getitem__(self, idx):
        return self._data[idx]

    def __delitem__(self, idx):
        del self._data[idx]
        self.update_datafile()

    def __len__(self):
        return len(self._data)

    def update_datafile(self):
        print("Guardando...")
        with open(self.path, 'wb') as f:
            pickle.dump(self._data, f)

    def agregar(self, nombre: str, robot: Robot):
        for entry in self:
            if entry.nombre == nombre:
                return False

        self._data.append(RoboReg(nombre, robot))
        self.update_datafile()
        return True

    def copiar(self, origen: int, nuevo_nombre: str):
        for entry in self:
            if entry.nombre == nuevo_nombre:
                return False

        nueva_entrada = deepcopy(self._data[origen])
        nueva_entrada.nombre = nuevo_nombre
        self._data.append(nueva_entrada)
        self.update_datafile()
        return True

    def table_repr(self):
        for entry in self:
            yield (entry.nombre,
                   entry.robot.n,
                   len(entry.model_ids))


if __name__ == "__main__":
    from autokin.robot import RTBrobot
    robot = RTBrobot.from_name('Cobra600')

    database = RobotDatabase('gui/app_data/robotRegs')
    database.agregar('asd', robot)
    print(list(database))
    print(type(database._data))
import os
import pickle

from autokin.robot import Robot
from autokin.modelos import FKModel
from gui.robot_database import SelectionList, ModelReg, RoboReg


DB_SAVE_DIR = 'gui/app_data/robotRegs'


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
        self.train_kwargs = {}
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

    def set_train_kwargs(self, train_kwargs):
        self.train_kwargs = train_kwargs

    def entrenar(self):
        pass
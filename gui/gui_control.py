import os
import pickle

import numpy as np

from autokin.robot import ExternRobot, RTBrobot, SofaRobot
from autokin import modelos
from autokin.muestreo import FKset
from autokin.loggers import GUIprogress
from gui.robot_database import SelectionList, ModelReg, RoboReg


SAVE_DIR = 'gui/app_data'


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
        self.pickle_dir = os.path.join(SAVE_DIR, 'robotRegs.pkl')
        self.tb_dir = os.path.join(SAVE_DIR, 'tb_logs')
        self.trayec_dir = os.path.join(SAVE_DIR, 'trayec')
        self.train_kwargs = {}
        self.datasets = {}
        self.cargar()

    """
    Bases de datos robots/modelos
    TODO: Separar en 3 controladores según responsabilidad?
    """
    def cargar(self):
        if os.path.isfile(self.pickle_dir):
            with open(self.pickle_dir, 'rb') as f:
                self.robots = pickle.load(f)
        else:
            self.robots = SelectionList()

    def guardar(self):
        with open(self.pickle_dir, 'wb') as f:
            pickle.dump(self.robots, f)

    @property
    def robot_selec(self):
        return self.robots.selec()

    @property
    def modelos(self):
        return self.robots.selec().modelos

    @property
    def modelo_selec(self):
        if self.robot_selec is None:
            return None
        else:
            return self.robot_selec.modelos.selec()

    def agregar_robot(self, nombre: str, robot_args: dict) -> bool:
        robot_inits = {"Externo" : ExternRobot,
                       "Sim. RTB" : RTBrobot.from_name,
                       "Sim. SOFA" : SofaRobot}

        cls_id = robot_args.pop('cls_id')
        robot_cls = robot_inits[cls_id]
        robot = robot_cls(**robot_args)

        agregado = self.robots.agregar(RoboReg(nombre, robot))
        self.guardar()
        return agregado

    def agregar_modelo(self, nombre: str, model_args: dict) -> bool:
        cls_id = model_args.pop('cls_id')
        model_cls = getattr(modelos, cls_id)

        model_args.update(input_dim=self.robot_selec.robot.n,
                          output_dim=3)
        modelo = model_cls(**model_args)

        agregado = self.modelos().agregar(ModelReg(nombre, modelo))
        self.guardar()
        return agregado

    """
    Entrenamiento
    """
    def set_train_kwargs(self, train_kwargs):
        self.train_kwargs = train_kwargs

    def set_sample(self, sample, sample_split):
        self.sample = sample
        self.split = list(sample_split.values())

    def entrenar(self, stage_callback, step_callback, close_callback):
        # if(muestreo_activo):
        #     modelo = ensemble(modelo)

        train_set, val_set, test_set = self._muestreo_inicial()
        stage_callback() 

        self._ajuste_inicial(self.modelo_selec,
                             train_set, val_set,
                             step_callback, close_callback)
        stage_callback()

        # if(muestreo_activo):
        #     modelo = max(ensemble, max_score)

    def _meta_ajuste(self):
        pass

    def _muestreo_inicial(self):
        # Generar dataset y repartirlo
        dataset = FKset(self.robot_selec.robot, self.sample)
        return dataset.rand_split(self.split)

    def _ajuste_inicial(self, model_reg, train_set, val_set,
                        step_callback, close_callback):
        fit_kwargs = self.train_kwargs['Ajuste inicial']
        epocas = fit_kwargs['epochs']
        log_dir = f'{self.tb_dir}/{self.robot_selec.nombre}_{model_reg.nombre}'

        model_reg.modelo.fit(train_set=train_set, val_set=val_set,
                             log_dir=log_dir,
                             loggers=[GUIprogress(epocas, step_callback,
                                                  close_callback)],
                             # silent=True,
                             **fit_kwargs)

    """
    Control
    """
    def listas_puntos(self):
        return [os.path.splitext(n)[0] for n in os.listdir(self.trayec_dir)]

    def guardar_puntos(self, nombre, puntos):
        save_path = os.path.join(self.trayec_dir, nombre)
        np.save(save_path, np.array(puntos))

    def cargar_puntos(self, nombre):
        nombre = nombre + '.npy'
        load_path = os.path.join(self.trayec_dir, nombre)

        if nombre and os.path.exists(load_path):
            return np.load(load_path).tolist()
        else:
            return None
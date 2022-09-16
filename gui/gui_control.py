import os
import pickle
import time

import numpy as np
import torch

from autokin.robot import ModelRobot
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


class CtrlRobotDB:
    """
    Métodos para la selección y carga de robots y modelos
    """
    def __init__(self):
        super().__init__()
        self.pickle_dir = os.path.join(SAVE_DIR, 'robotDB.pkl')
        self.model_dir = os.path.join(SAVE_DIR, 'modelos')
        self._robots = None
        self._modelos = None
        self._robot_s = None
        self._modelo_s = None

    def guardar(self):
        with open(self.pickle_dir, 'wb') as f:
            pickle.dump(self._robots, f)

    """ Robots """
    @property
    def robots(self):
        if self._robots is None:
            if os.path.isfile(self.pickle_dir):
                with open(self.pickle_dir, 'rb') as f:
                    self._robots = pickle.load(f)
            else:
                self._robots = SelectionList()
        return self._robots

    @property
    def robot_selec(self):
        """
        Registro de datos del robot seleccionado
        """
        return self.robots.selec()

    @property
    def robot_s(self):
        """
        Robot seleccionado
        """
        if self._robot_s is None:
            self._robot_s = self.robot_selec.init_obj()
        return self._robot_s

    def seleccionar_robot(self, indice: int):
        self.robots.seleccionar(indice)
        self._robot_s = None
        self._modelo_s = None

    def agregar_robot(self, nombre: str, robot_args: dict) -> bool:
        cls_id = robot_args.pop('cls_id')
        agregado = self.robots.agregar(RoboReg(nombre, cls_id, robot_args))
        self.guardar()
        return agregado

    def copiar_robot(self,
                     origen: int,
                     nombre: str,
                     copiar_modelos: bool) -> bool:
        agregado = self.robots.copiar(origen, nombre)
        if agregado and not copiar_modelos:
            self.robots[-1].modelos = SelectionList()
        return agregado

    def eliminar_robot(self, indice: int):
        self.robots.eliminar(indice)

    """ Modelos """
    def _model_filename(self):
        robot_nom = self.robot_selec.nombre
        model_nom = self.modelo_selec.nombre
        return f'{robot_nom}_{model_nom}.pt'

    @property
    def modelos(self):
        return self.robots.selec().modelos

    @property
    def modelo_selec(self):
        """
        Registo de datos del modelo seleccionado
        """
        if self.robot_selec is None:
            return None
        else:
            return self.robot_selec.modelos.selec()

    @property
    def modelo_s(self):
        """
        Modelo seleccionado
        """
        if self._modelo_s is None:
            model_path = os.path.join(self.model_dir, self._model_filename())
            if os.path.isfile(model_path):
                self._modelo_s = torch.load(model_path)
            else:
                self._modelo_s = self.modelo_selec.init_obj()
        return self._modelo_s
        
    def seleccionar_modelo(self, indice: int):
        self.modelos.seleccionar(indice)
        self._model_s = None

    def agregar_modelo(self, nombre: str, model_args: dict) -> bool:
        model_args.update(input_dim=self.robot_s.n,
                          output_dim=3)

        cls_id = model_args.pop('cls_id')

        agregado = self.modelos.agregar(ModelReg(nombre, cls_id, model_args))
        self.guardar()
        return agregado

    def copiar_modelo(self, indice: int, nombre: str) -> bool:
        return self.modelos.copiar(indice, nombre)

    def eliminar_modelo(self, indice: int):
        self.modelos.eliminar(indice)

    def guardar_modelo(self):
        if self._modelo_s is not None:
            torch.save(self._modelo_s, self._model_filename())
            self.guardar()


class CtrlEntrenamiento:
    """
    Métodos para coordinar el entrenamiento de los modelos con GUI
    """
    def __init__(self):
        super().__init__()
        self.train_kwargs = {}
        self.datasets = {}
        self.tb_dir = os.path.join(SAVE_DIR, 'tb_logs')

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

        self._ajuste_inicial(train_set, val_set,
                             step_callback, close_callback)
        stage_callback()

        # if(muestreo_activo):
        #     modelo = max(ensemble, max_score)

    def _meta_ajuste(self):
        pass

    def _muestreo_inicial(self):
        # Generar dataset y repartirlo
        dataset = FKset(self.robot_s, self.sample)
        return dataset.rand_split(self.split)

    def _ajuste_inicial(self, train_set, val_set,
                        step_callback, close_callback):
        fit_kwargs = self.train_kwargs['Ajuste inicial']
        epocas = fit_kwargs['epochs']
        log_dir = os.path.join(self.tb_dir, self._model_filename())

        self.modelo_s.fit(train_set=train_set, val_set=val_set,
                          log_dir=log_dir,
                          loggers=[GUIprogress(step_callback,
                                               close_callback)],
                          silent=True,
                          **fit_kwargs)
        self.modelo_selec.epochs += epocas
        self.guardar_modelo()


class CtrlEjecucion:
    """
    Métodos para coordinar control punto a punto del robot.
    """
    def __init__(self):
        super().__init__()
        self.trayec_dir = os.path.join(SAVE_DIR, 'trayec')
        self.puntos = None

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

    def set_trayec(self, puntos):
        self.puntos = puntos

    def ejecutar_trayec(self, reg_callback):
        model_robot = ModelRobot(self.modelo_s)
        q_prev = torch.zeros(model_robot.n)
        for x, y, z, t_t, t_s in self.puntos:
            target = torch.Tensor([x,y,z])
            q = model_robot.ikine_pi_jacob(q_start=q_prev,
                                           p_target=target)
            _, p = self.robot_s.fkine(q)
            q_prev = q
            # time.sleep(t_s)
            reg_callback(p.tolist())


class UIController(CtrlRobotDB,
                   CtrlEntrenamiento,
                   CtrlEjecucion,
                   metaclass=Singleton):
    """
    Controlador para acoplar GUI con lógica del programa.
    """
    def __init__(self):
        super().__init__()

    def get_ext_status(self):
        # Revisar estado de conexión BT, cámaras, etc.
        return (False, False)
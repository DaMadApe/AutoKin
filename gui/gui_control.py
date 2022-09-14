import os
import pickle
import time

import numpy as np
import torch

from autokin.robot import ExternRobot, RTBrobot, SofaRobot, ModelRobot
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


class CtrlRobotDB:
    """
    Métodos para la selección y carga de robots y modelos
    """
    def __init__(self):
        super().__init__()
        self.pickle_dir = os.path.join(SAVE_DIR, 'robotDB.pkl')
        self._robots = None
        self._modelos = None

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
        return self.robots.selec()

    def seleccionar_robot(self, indice: int):
        self.robots.seleccionar(indice)

    def agregar_robot(self, nombre: str, robot_args: dict) -> bool:
        robot_inits = {"Externo" : ExternRobot,
                       "Sim. RTB" : RTBrobot.from_name,
                       "Sim. SOFA" : SofaRobot}

        cls_id = robot_args.pop('cls_id')
        robot_cls = robot_inits[cls_id]
        robot = robot_cls(**robot_args)

        agregado = self.robots.agregar(RoboReg(nombre, robot))
        # 
        self.guardar()
        return agregado

    def copiar_robot(self,
                     origen: int,
                     nombre: str,
                     copiar_modelos: bool) -> bool:
        agregado = self.robots.copiar(origen, nombre)
        if agregado and copiar_modelos:
            pass # self.robots[-1].modelos = copy()
        return agregado

    def eliminar_robot(self, indice: int):
        self.robots.eliminar(indice)

    """ Modelos """
    @property
    def modelos(self):
        # if self._modelos is not None:
        #     return self._modelos
        # else:
        #     for robot in self.robots:
        #         for model in robot.models:
        #             init(model.kwargs)
        return self.robots.selec().modelos

    @property
    def modelo_selec(self):
        if self.robot_selec is None:
            return None
        else:
            return self.robot_selec.modelos.selec()

    def seleccionar_modelo(self, indice: int):
        self.modelos.seleccionar(indice)

    def agregar_modelo(self, nombre: str, model_args: dict) -> bool:
        # TODO: Hacer método self._init_obj para hacer esto
        model_args.update(input_dim=self.robot_selec.robot.n,
                          output_dim=3)

        cls_id = model_args.pop('cls_id')
        model_cls = getattr(modelos, cls_id)

        modelo = model_cls(**model_args)

        agregado = self.modelos.agregar(ModelReg(nombre, modelo))
        self.guardar()
        return agregado

    def copiar_modelo(self, indice: int, nombre: str) -> bool:
        return self.modelos.copiar(indice, nombre)

    def eliminar_modelo(self, indice: int):
        self.modelos.eliminar(indice)


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
        model_robot = ModelRobot(self.modelo_selec.modelo)
        q_prev = torch.zeros(model_robot.n)
        for x, y, z, t_t, t_s in self.puntos:
            target = torch.Tensor([x,y,z])
            print(x,y,z)
            q = model_robot.ikine_pi_jacob(q_start=q_prev,
                                           p_target=target)
            _, p = self.robot_selec.robot.fkine(q)
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
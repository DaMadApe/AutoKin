import os
import pickle
import time
from collections import namedtuple
from threading import Thread
from queue import Queue

import numpy as np
import torch

from autokin.robot import ModelRobot
from autokin.modelos import FKModel
from autokin.muestreo import FKset
from autokin.loggers import GUIprogress, LastEpochLog
from gui.robot_database import SelectionList, ModelReg, RoboReg


SAVE_DIR = 'gui/app_data'


class SignalQueue(Queue):
    def __init__(self):
        super().__init__()
        self.done = False
        self.pause = False

    def interrupt(self) -> bool:
        while self.pause:
            pass
        return self.done


Msg = namedtuple('Msg', ('head', 'info'), defaults=(None, None))


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
        
        if self._modelo_s is not None:
            model_path = self._model_path(self.robot_selec,
                                          self.modelo_selec)
            torch.save(self._modelo_s, model_path)

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
        if agregado:
            if copiar_modelos:
                for modelo in self.robots[origen].modelos:
                    obj = torch.load(self._model_path(self.robots[origen],
                                                      modelo))
                    torch.save(obj, self._model_path(self.robots[-1],
                                                     modelo))

            else:
                self.robots[-1].modelos = SelectionList()
        return agregado

    def eliminar_robot(self, indice: int):
        self.robots.eliminar(indice)

    """ Modelos """
    def _model_path(self, robot, modelo):
        robot_nom = robot.nombre
        model_nom = modelo.nombre
        filename = f'{robot_nom}_{model_nom}.pt'
        return os.path.join(self.model_dir, filename)

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
            model_path = self._model_path(self.robot_selec,
                                          self.modelo_selec)
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

    def copiar_modelo(self, origen: int, nombre: str) -> bool:
        agregado = self.modelos.copiar(origen, nombre)
        if agregado:
            obj = torch.load(self._model_path(self.robot_selec,
                                              self.modelos[origen]))
            torch.save(obj, self._model_path(self.robot_selec,
                                             self.modelos[-1]))
        return agregado

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
        self.queue = SignalQueue()

    def set_train_kwargs(self, train_kwargs):
        self.train_kwargs = train_kwargs

    def set_sample(self, sample, sample_split):
        self.sample = sample
        self.split = list(sample_split.values())

    def entrenar(self, stage_callback, step_callback, end_callback, after_fn):
        self.queue = SignalQueue()
        log_name = f'{self.robot_selec.nombre}_{self.modelo_selec.nombre}'
        log_dir = os.path.join(self.tb_dir, log_name)

        self.trainer = TrainThread(queue=self.queue,
                                   modelo=self.modelo_s,
                                   robot=self.robot_s,
                                   sample=self.sample,
                                   split=self.split,
                                   train_kwargs=self.train_kwargs,
                                   log_dir=log_dir)
        self.trainer.start()

        self.check_queue(stage_callback, step_callback, 
        end_callback, after_fn)

    def check_queue(self, stage_callback, step_callback, end_callback, after_fn):
        while not self.queue.empty() and not self.queue.done:
            msg = self.queue.get()

            if msg.head == 'stage':
                stage_callback(msg.info)
            elif msg.head == 'step':
                step_callback(*msg.info)
            elif msg.head == 'close':
                self.modelo_selec.epochs += msg.info['Ajuste inicial']
                self.detener(guardar=True)
                end_callback()
                return

        after_fn(100, self.check_queue, 
                 stage_callback, step_callback, end_callback, after_fn)

    def detener(self, guardar: bool):
        self.queue.done = True
        self.queue.pause = False
        self.trainer.join()
        if guardar:
            self.guardar()

    def pausar(self):
        self.queue.pause = True

    def reanudar(self):
        self.queue.pause = False


class TrainThread(Thread):

    def __init__(self, queue: SignalQueue, modelo:FKModel,
                 robot, sample, split, train_kwargs, log_dir):
        super().__init__(name='training',daemon=True)
        self.queue = queue
        self.modelo = modelo
        self.robot = robot
        self.sample = sample
        self.split = split
        self.train_kwargs = train_kwargs
        self.log_dir = log_dir

    def run(self):
        resultados = {}
        for etapa in self.train_kwargs.keys():
            if not self.queue.done:
                method_name = '_' + etapa.lower().replace(' ', '_')
                resultados[etapa] = getattr(self, method_name)()

        self.queue.put(Msg('close', resultados))

    def _meta_ajuste(self):
        self.queue.put(Msg('stage', 100))
        mfit_kwargs = self.train_kwargs['Meta ajuste']
        # self.modelo.meta_fit(**mfit_kwargs)
        for i in range(100):
            time.sleep(0.01)
            self.queue.put(Msg('step', ({},i)))
        print(mfit_kwargs)

    def _ajuste_inicial(self):
        # Muestreo
        self.queue.put(Msg('stage', 0))
        time.sleep(2)            
        dataset = FKset(self.robot, self.sample)
        train_set, val_set, test_set = dataset.rand_split(self.split)

        fit_kwargs = self.train_kwargs['Ajuste inicial']
        self.queue.put(Msg('stage', fit_kwargs['epochs']))

        if not self.queue.done:
            le_log = LastEpochLog()
            gui_logger = GUIprogress(step_callback=lambda *x:
                                        self.queue.put(Msg('step', x)),
                                    close_callback=lambda:None)

            self.modelo.fit(train_set=train_set, val_set=val_set,
                            log_dir=self.log_dir,
                            loggers=[gui_logger, le_log],
                            silent=True,
                            ext_interrupt=self.queue.interrupt,
                            **fit_kwargs)

            return le_log.last_epoch

    def _ajuste_dirigido(self):
        self.queue.put(Msg('stage', 0))
        afit_kwargs = self.train_kwargs['Ajuste dirigido']
        # self.modelo.active_fit(**afit_kwargs)
        print(afit_kwargs)
        for i in range(100):
            time.sleep(0.01)
            self.queue.put(Msg('step', ({},i)))


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
            # after_fn(t_s*1000)
            reg_callback(p.tolist())


class UIController(CtrlRobotDB,
                   CtrlEntrenamiento,
                   CtrlEjecucion):
                   # TODO(?): Refactorizar para composición
                   # self.controlador.robots.seleccionar
                   # self.controlador.entrenamiento.detener
                   # self.controlador.ejecucion.iniciar
    """
    Controlador para acoplar GUI con lógica del programa.
    """
    def __init__(self):
        super().__init__()

    def get_ext_status(self):
        # Revisar estado de conexión BT, cámaras, etc.
        return (False, False)
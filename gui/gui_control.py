import os
import shutil
import pickle
import time
import webbrowser
from threading import Thread
from queue import Queue
from typing import Union
from collections import namedtuple

import numpy as np
import torch
from tensorboard import program

from autokin.robot import Robot, ModelRobot
from autokin.modelos import FKEnsemble, FKModel
from autokin.muestreo import FKset
from autokin.loggers import GUIprogress, LastEpochLog
from gui.robot_database import SelectionList, ModelReg, RoboReg


SAVE_DIR = os.path.join('gui', 'app_data')


class SignalQueue(Queue):
    def __init__(self):
        super().__init__()
        self.done = False
        self.pause = False

    def interrupt(self) -> bool:
        while self.pause:
            time.sleep(0.1)
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
        self.pending_cleanup = []

    def guardar(self):
        with open(self.pickle_dir, 'wb') as f:
            pickle.dump(self._robots, f)
        
        if self._modelo_s is not None:
            model_path = self._model_path(self.robot_selec,
                                          self.modelo_selec)
            torch.save(self._modelo_s, model_path)

    """ Robots """
    @property
    def robots(self) -> SelectionList:
        if self._robots is None:
            if os.path.isfile(self.pickle_dir):
                with open(self.pickle_dir, 'rb') as f:
                    self._robots = pickle.load(f)
            else:
                self._robots = SelectionList()
        return self._robots

    @property
    def robot_selec(self) -> RoboReg:
        """
        Registro de datos del robot seleccionado
        """
        return self.robots.selec()

    @property
    def robot_s(self) -> Robot:
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
        for idx in range(len(self.robots[indice].modelos)):
            self.eliminar_modelo(idx)

        self.robots.eliminar(indice)

    """ Modelos """
    def _model_path(self, robot, modelo):
        robot_nom = robot.nombre
        model_nom = modelo.nombre
        filename = f'{robot_nom}_{model_nom}.pt'
        return os.path.join(self.model_dir, filename)

    def _model_log_dir(self, robot, modelo):
        log_name = f'{robot.nombre}_{modelo.nombre}'
        return os.path.join(self.tb_dir, log_name)

    @property
    def modelos(self) -> SelectionList:
        return self.robots.selec().modelos

    @property
    def modelo_selec(self) -> ModelReg:
        """
        Registo de datos del modelo seleccionado
        """
        if self.robot_selec is None:
            return None
        else:
            return self.robot_selec.modelos.selec()

    @property
    def modelo_s(self) -> Union[FKModel, FKEnsemble]:
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
        self._modelo_s = None

    def agregar_modelo(self, nombre: str, model_args: dict) -> bool:
        model_args.update(input_dim=self.robot_s.n,
                          output_dim=3)

        cls_id = model_args.pop('cls_id')

        agregado = self.modelos.agregar(ModelReg(nombre, cls_id, model_args))
        
        if agregado:
            # Crear directorio de tensorboard
            os.mkdir(self._model_log_dir(self.robot_selec,
                                         self.modelos[-1]))
            self.guardar()
        return agregado

    def copiar_modelo(self, origen: int, nombre: str) -> bool:
        agregado = self.modelos.copiar(origen, nombre)
        if agregado:
            # Copiar modelo (~.pt) si existe
            orig_path = self._model_path(self.robot_selec,
                                         self.modelos[origen])
            dest_path = self._model_path(self.robot_selec,
                                         self.modelos[-1])
            if os.path.isfile(orig_path):
                model = torch.load(orig_path)
                torch.save(model, dest_path)

            # Crear directorio de tensorboard
            os.mkdir(self._model_log_dir(self.robot_selec,
                                         self.modelos[-1]))
        return agregado 

    def eliminar_modelo(self, indice: int):
        # Eliminar registro de tensorboard
        log_dir = self._model_log_dir(self.robot_selec,
                                      self.modelos[indice])
        # shutil.rmtree(log_dir)
        self.pending_cleanup.append(log_dir)

        # Eliminar modelo almacenado
        model_path = self._model_path(self.robot_selec,
                                      self.modelos[indice])
        if os.path.isfile(model_path):
            os.remove(model_path)
        
        self.modelos.eliminar(indice)

    def abrir_tensorboard(self, ver_todos=False):
        # TODO: Proceso de tensorboard se queda abierto, buscar
        #       forma de detener o reemplazar nuevas instancias
        base_dir = self._model_log_dir(self.robot_selec, self.modelo_selec)
        if os.listdir(base_dir) and not ver_todos:
            local_dir = sorted(os.listdir(base_dir))[-1]
            log_dir = os.path.join(base_dir, local_dir)
        else:
            log_dir = base_dir

        tb = program.TensorBoard()
        tb.configure(logdir=log_dir) # '--port', str(6006)])
        url = tb.launch()
        webbrowser.open(url)

    def cerrar(self):
        for path in self.pending_cleanup:
            shutil.rmtree(path)


class CtrlEntrenamiento:
    """
    Métodos para coordinar el entrenamiento de los modelos con GUI
    """
    def __init__(self):
        super().__init__()
        self.train_kwargs = {}
        self.datasets = {}
        # TODO(?): mover dirs a ubicación cental
        self.tb_dir = os.path.join(SAVE_DIR, 'tb_logs')
        self.queue = SignalQueue()

    def set_train_kwargs(self, train_kwargs):
        self.train_kwargs = train_kwargs

    def set_sample(self, sample, sample_split):
        self.sample = sample
        self.split = list(sample_split.values())

    def entrenar(self, stage_callback, step_callback, end_callback, after_fn):
        self.queue = SignalQueue()

        timestamp = time.strftime('%Y%m%d-%H%M%S')
        base_dir = self._model_log_dir(self.robot_selec, self.modelo_selec) 
        log_dir = os.path.join(base_dir, timestamp)

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
                self.trainer.join()
                end_callback() # msg.info['Ajuste inicial']
                return

        after_fn(100, self.check_queue, 
                 stage_callback, step_callback, end_callback, after_fn)

    def detener(self, guardar: bool):
        self.queue.done = True
        self.queue.pause = False
        self.trainer.join()
        if guardar:
            self.modelo_selec.trains.append(self.train_kwargs)
            self.guardar()
        else:
            # Desechar instancia actual (entrenada) del modelo
            self._modelo_s = None
            # Desechar registros de tensorboard
            base_dir = self._model_log_dir(self.robot_selec,
                                           self.modelo_selec)
            local_dir = sorted(os.listdir(base_dir))[-1]
            log_dir = os.path.join(base_dir, local_dir)
            # shutil.rmtree(log_dir)
            self.pending_cleanup.append(log_dir)

    def pausar(self):
        self.queue.pause = True

    def reanudar(self):
        self.queue.pause = False


class TrainThread(Thread):
    """
    Thread para entrenar el modelo en concurrencia con
    las actualizaciones de la GUI
    """
    def __init__(self,
                 queue: SignalQueue,
                 modelo: Union[FKModel, FKEnsemble],
                 robot: Robot,
                 sample: torch.Tensor,
                 split: list[float],
                 train_kwargs: dict,
                 log_dir):
        super().__init__(name='training', daemon=True)
        self.queue = queue
        self.modelo = modelo
        self.robot = robot
        self.sample = sample
        self.split = split
        self.train_kwargs = train_kwargs
        self.log_dir = log_dir
        
        self.gui_logger = GUIprogress(step_callback=lambda *x:
                                          self.queue.put(Msg('step', x)),
                                      close_callback=lambda:None)

    def run(self):
        resultados = {}
        for etapa in self.train_kwargs.keys():
            if not self.queue.done:
                method_name = '_' + etapa.lower().replace(' ', '_')
                log_dir = os.path.join(self.log_dir, method_name)
                resultados[etapa] = getattr(self, method_name)(log_dir)

        self.queue.put(Msg('close', resultados))

    def _meta_ajuste(self, log_dir):
        mfit_kwargs = self.train_kwargs['Meta ajuste']

        steps = mfit_kwargs['n_epochs'] * mfit_kwargs['n_datasets']
        steps += mfit_kwargs['n_post_epochs']
        steps *= mfit_kwargs['n_steps']
        self.queue.put(Msg('stage', steps))

        self.modelo.meta_fit(log_dir=log_dir,
                             loggers=[self.gui_logger],
                             ext_interrupt=self.queue.interrupt,
                             **mfit_kwargs)

    def _ajuste_inicial(self, log_dir):
        # Muestreo
        self.queue.put(Msg('stage', 0))
        time.sleep(2)            
        dataset = FKset(self.robot, self.sample)
        train_set, val_set, test_set = dataset.rand_split(self.split)

        self.train_set = train_set

        # Ajuste
        fit_kwargs = self.train_kwargs['Ajuste inicial']

        steps = fit_kwargs['epochs']
        self.queue.put(Msg('stage', steps))

        if not self.queue.done:
            le_log = LastEpochLog()

            self.modelo.fit(train_set=train_set, val_set=val_set,
                            log_dir=log_dir,
                            loggers=[self.gui_logger, le_log],
                            silent=True,
                            ext_interrupt=self.queue.interrupt,
                            **fit_kwargs)

            return le_log.last_epoch

    def _ajuste_dirigido(self, log_dir):
        afit_kwargs = self.train_kwargs['Ajuste dirigido']

        steps = len(self.modelo.ensemble) * (afit_kwargs['epochs'])
        steps *= afit_kwargs['query_steps']
        self.queue.put(Msg('stage', steps))

        def label_fun(X):
            _, result = self.robot.fkine(X)
            return result

        self.modelo.active_fit(train_set=self.train_set,
                               label_fun=label_fun,
                               loggers=[self.gui_logger],
                               log_dir=log_dir,
                               silent=True,
                               ext_interrupt=self.queue.interrupt,
                               **afit_kwargs)


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
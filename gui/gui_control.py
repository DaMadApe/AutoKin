import os
import time
import shutil
import pickle
import logging
import webbrowser
import multiprocessing as mp
from queue import Queue
from threading import Thread
from typing import Optional, Union
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tensorboard import program

from autokin.robot import Robot, ExternRobot, ModelRobot
from autokin.modelos import FKEnsemble, FKModel
from autokin.muestreo import FKset
from autokin.utils import RobotExecError, rand_split
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
        self.pickle_path = os.path.join(SAVE_DIR, 'robotDB.pkl')
        self.robot_base_dir = os.path.join(SAVE_DIR, 'robots')
        self.trayec_dir = os.path.join(SAVE_DIR, 'trayec')

        self._robots = None # Lista de registros de robots
        self._modelos = None # Lista de registros de modelos
        self._robot_s = None # Robot seleccionado
        self._modelo_s = None # Modelo seleccionado

        self.tb_proc = None

        # Asegurar estructura de directorios
        for dir in [SAVE_DIR, self.robot_base_dir, self.trayec_dir]:
            if not os.path.exists(dir):
                os.mkdir(dir)

    """ Directorios """
    def _robot_dir(self, robot: RoboReg):
        return os.path.join(self.robot_base_dir, robot.nombre)

    def _model_dir(self, robot: RoboReg):
        return os.path.join(self._robot_dir(robot), 'modelos')

    def _dataset_dir(self, robot: RoboReg):
        return os.path.join(self._robot_dir(robot), 'datasets')

    def _tb_dir(self, robot: RoboReg):
        return os.path.join(self._robot_dir(robot), 'tb_logs')

    def _model_path(self, robot: RoboReg, modelo: ModelReg):
        basedir = self._model_dir(robot)
        filename = modelo.nombre + '.pt'
        return os.path.join(basedir, filename)

    def _model_log_dir(self, robot: RoboReg, modelo: ModelReg):
        return os.path.join(self._tb_dir(robot), modelo.nombre)

    def guardar(self):
        # Guardar lista de registros y modelo de torch
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self._robots, f)
        
        if self._modelo_s is not None:
            model_path = self._model_path(self.robot_reg_s,
                                          self.modelo_reg_s)
            torch.save(self._modelo_s, model_path)

    """ Robots """
    @property
    def robots(self) -> SelectionList[RoboReg]:
        if self._robots is None:
            if os.path.isfile(self.pickle_path):
                with open(self.pickle_path, 'rb') as f:
                    self._robots = pickle.load(f)
            else:
                self._robots = SelectionList()
        return self._robots

    @property
    def robot_reg_s(self) -> RoboReg:
        """
        Registro de datos del robot seleccionado
        """
        if not self.is_robot_selected():
            raise RuntimeError('Robot no seleccionado')
        return self.robots.selec()

    @property
    def robot_s(self) -> Robot:
        """
        Robot seleccionado
        """
        if self._robot_s is None:
            logging.info(f'Creando nueva instancia de robot')
            self._robot_s = self.robot_reg_s.init_obj()
        return self._robot_s

    def is_robot_selected(self) -> bool:
        return self.robots.selec() is not None

    def seleccionar_robot(self, indice: int):
        self.robots.seleccionar(indice)
        self._robot_s = None
        self._modelo_s = None

    def agregar_robot(self, nombre: str, robot_args: dict) -> bool:
        cls_id = robot_args.pop('cls_id')
        agregado = self.robots.agregar(RoboReg(nombre, cls_id, robot_args))

        if agregado:
            # Crear directorios
            os.mkdir(self._robot_dir(self.robots[-1]))
            os.mkdir(self._model_dir(self.robots[-1]))
            os.mkdir(self._dataset_dir(self.robots[-1]))
            os.mkdir(self._tb_dir(self.robots[-1]))
            self.guardar()
        return agregado

    def copiar_robot(self,
                     origen: int,
                     nombre: str,
                     copiar_modelos: bool,
                     copiar_datasets: bool) -> bool:
        agregado = self.robots.copiar(origen, nombre)

        if agregado:
            # Crear directorios
            os.mkdir(self._robot_dir(self.robots[-1]))

            if copiar_modelos:
                shutil.copytree(src=self._model_dir(self.robots[origen]),
                                dst=self._model_dir(self.robots[-1])), #dirs_exist_ok=True)
                # Copiar logs de tensorboard
                shutil.copytree(src=self._tb_dir(self.robots[origen]),
                                dst=self._tb_dir(self.robots[-1]))
            else:
                self.robots[-1].modelos = SelectionList()
                os.mkdir(self._model_dir(self.robots[-1]))
                os.mkdir(self._tb_dir(self.robots[-1]))

            if copiar_datasets:
                shutil.copytree(src=self._dataset_dir(self.robots[origen]),
                                dst=self._dataset_dir(self.robots[-1]))
            else:
                os.mkdir(self._dataset_dir(self.robots[-1]))

        return agregado

    def eliminar_robot(self, indice: int):
        # Eliminar modelos, datasets y logs
        shutil.rmtree(self._robot_dir(self.robots[indice]))

        self.robots.eliminar(indice)

        if self.robots.selec is None:
            self._robot_s = None

    def config_robot(self, config: dict):
        logging.info(f"Config robot: {config}")
        self.robot_reg_s.kwargs.update(config)

        if self._robot_s is not None:
            for key, val in config.items():
                setattr(self._robot_s, key, val)

    def set_pnorm(self):
        pass

    """ Modelos """
    @property
    def modelos(self) -> SelectionList[ModelReg]:
        if self.robots.selec() is None:
            raise RuntimeError('Intento de acceso a modelos sin robot seleccionado')
        return self.robots.selec().modelos

    @property
    def modelo_reg_s(self) -> ModelReg:
        """
        Registo de datos del modelo seleccionado
        """
        if not self.is_model_selected():
            raise RuntimeError('Modelo no seleccionado')
        return self.robot_reg_s.modelos.selec()

    @property
    def modelo_s(self) -> Union[FKModel, FKEnsemble]:
        """
        Modelo seleccionado
        """
        if self._modelo_s is None:
            model_path = self._model_path(self.robot_reg_s,
                                          self.modelo_reg_s)
            if os.path.isfile(model_path):
                logging.info(f'Cargando robot en {model_path}')
                self._modelo_s = torch.load(model_path)
            else:
                logging.info(f'Creando nueva instancia de modelo')
                self._modelo_s = self.modelo_reg_s.init_obj()
        return self._modelo_s

    def is_model_selected(self) -> bool:
        return self.is_robot_selected() and self.robot_reg_s.modelos.selec() is not None

    def seleccionar_modelo(self, indice: int):
        self.modelos.seleccionar(indice)
        self._modelo_s = None

    def agregar_modelo(self, nombre: str, model_args: dict) -> bool:
        model_args.update(input_dim=self.robot_s.n,
                          output_dim=3)

        cls_id = model_args.pop('cls_id')

        agregado = self.modelos.agregar(ModelReg(nombre, cls_id, model_args))
        
        if agregado:
            # Crear directorio para guardar logs
            os.mkdir(self._model_log_dir(self.robot_reg_s,
                                         self.modelos[-1]))
            self.guardar()
        return agregado

    def copiar_modelo(self, origen: int, nombre: str) -> bool:
        agregado = self.modelos.copiar(origen, nombre)
        if agregado:
            # Copiar modelo (~.pt) si existe
            orig_path = self._model_path(self.robot_reg_s,
                                         self.modelos[origen])
            dest_path = self._model_path(self.robot_reg_s,
                                         self.modelos[-1])
            if os.path.isfile(orig_path):
                model = torch.load(orig_path)
                torch.save(model, dest_path)

            # Copiar registros de tensorboard
            shutil.copytree(src=self._model_log_dir(self.robot_reg_s,
                                                    self.modelos[origen]),
                            dst=self._model_log_dir(self.robot_reg_s,
                                                    self.modelos[-1]))
        return agregado 

    def eliminar_modelo(self, indice: int):
        # Eliminar registro de tensorboard
        log_dir = self._model_log_dir(self.robot_reg_s,
                                      self.modelos[indice])
        shutil.rmtree(log_dir)

        # Eliminar modelo almacenado
        model_path = self._model_path(self.robot_reg_s,
                                      self.modelos[indice])
        if os.path.isfile(model_path):
            os.remove(model_path)
        
        self.modelos.eliminar(indice)

        if self.modelos.selec is None:
            self._modelo_s = None

    def abrir_tensorboard(self, ver_todos=False):
        self.cerrar_tensorboard()

        base_dir = self._model_log_dir(self.robot_reg_s, self.modelo_reg_s)
        if os.listdir(base_dir) and not ver_todos:
            local_dir = sorted(os.listdir(base_dir))[-1]
            log_dir = os.path.join(base_dir, local_dir)
        else:
            log_dir = base_dir

        def abrir():
            tb = program.TensorBoard()
            tb.configure(logdir=log_dir, port=6006)
            tb.main()

        self.tb_proc = mp.Process(target=abrir)
        self.tb_proc.start()
        time.sleep(2) # Incrementar si no conecta a la primera
        webbrowser.open('http://localhost:6006/')

    def cerrar_tensorboard(self):
        if self.tb_proc is not None:
            self.tb_proc.terminate()
            self.tb_proc.join()

    def get_datasets(self) -> dict[str, FKset]:
        datasets = {}
        dataset_dir = self._dataset_dir(self.robot_reg_s)
        for filename in sorted(os.listdir(dataset_dir)):
            dataset_path = os.path.join(dataset_dir, filename)
            datasets[filename] = torch.load(dataset_path)
        return datasets

    def get_ext_status(self) -> Optional[dict]:
        """
        Revisar estado de conexión BT, cámaras, etc.
        """
        if self.is_robot_selected() and isinstance(self.robot_s, ExternRobot):
            return self.robot_s.status()
        else:
            return None


class CtrlEntrenamiento:
    """
    Métodos para coordinar el entrenamiento de los modelos con GUI
    """
    def __init__(self):
        super().__init__()
        self.train_kwargs = {}
        self.extra_datasets = {}
        self.queue = SignalQueue()

    def set_train_kwargs(self, train_kwargs: dict):
        self.train_kwargs = train_kwargs

    def set_sample(self, sample: torch.Tensor, sample_split: dict):
        self.sample = sample
        self.split = list(sample_split.values())

    def set_extra_datasets(self, datasets: dict[str, Dataset]):
        self.extra_datasets = datasets

    def entrenar(self, stage_callback, step_callback, end_callback, after_fn):
        self.queue = SignalQueue()

        timestamp = time.strftime('%Y%m%d-%H%M%S')
        base_dir = self._model_log_dir(self.robot_reg_s, self.modelo_reg_s) 
        log_dir = os.path.join(base_dir, timestamp)

        self.trainer = TrainThread(queue=self.queue,
                                   modelo=self.modelo_s,
                                   robot=self.robot_s,
                                   sample=self.sample,
                                   split=self.split,
                                   train_kwargs=self.train_kwargs,
                                   log_dir=log_dir,
                                   prev_datasets=list(self.extra_datasets.values()))
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
            elif msg.head == 'fail':
                print('fail called')
                self.trainer.join()
                end_callback(fail=True)
                return

        after_fn(100, self.check_queue, 
                 stage_callback, step_callback, end_callback, after_fn)

    def detener(self, guardar_entrenamiento: bool, guardar_dataset: bool):
        self.queue.done = True
        self.queue.pause = False
        self.trainer.join()

        if guardar_entrenamiento:
            # Guardar parámetros usados en el registro del modelo
            self.modelo_reg_s.trains.append(self.train_kwargs)
            # Guardar todos los registros
            self.guardar()
        else:
            # Desechar instancia actual (entrenada) del modelo
            self._modelo_s = None
            # TODO: Desechar logs recolectados de tensorboard

        if guardar_dataset:
            dataset = self.trainer.get_dataset()
            if dataset is not None:
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                filename = f'dataset_{timestamp}.pt'
                data_save_dir = os.path.join(self._dataset_dir(self.robot_reg_s), filename)
                torch.save(dataset, data_save_dir)

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
                 log_dir,
                 prev_datasets: list[FKset] = None):
        super().__init__(name='training', daemon=True)
        self.queue = queue
        self.modelo = modelo
        self.robot = robot
        self.sample = sample
        self.split = split
        self.train_kwargs = train_kwargs
        self.log_dir = log_dir

        self.prev_datasets = prev_datasets
        if prev_datasets is None:
            self.prev_datasets = []

        for dataset in self.prev_datasets:
            dataset.p_scale = robot.p_scale
            dataset.p_offset = robot.p_offset

        self.sampled_dataset = None
        
        self.gui_logger = GUIprogress(step_callback=lambda *x:
                                          self.queue.put(Msg('step', x)),
                                      close_callback=lambda:None)

    def run(self):
        resultados = {}
        for etapa in self.train_kwargs.keys():
            if not self.queue.done:
                method_name = '_' + etapa.lower().replace(' ', '_')
                log_dir = os.path.join(self.log_dir, method_name)
                resultado = getattr(self, method_name)(log_dir)
                if resultado == 'fail':
                    return
                else:
                    resultados[etapa] = resultado
                self.gui_logger.steps = 0

        # Agregar dataset generado a resultados
        self.queue.put(Msg('close', resultados))

    def _meta_ajuste(self, log_dir):
        mfit_kwargs = self.train_kwargs['Meta ajuste']

        steps = mfit_kwargs['n_epochs'] * mfit_kwargs['n_datasets']
        steps += mfit_kwargs['n_post_epochs']
        steps *= mfit_kwargs['n_steps']
        if issubclass(type(self.modelo), FKEnsemble):
            steps *= len(self.modelo.ensemble)
        self.queue.put(Msg('stage', steps))

        self.modelo.meta_fit(log_dir=log_dir,
                             loggers=[self.gui_logger],
                             ext_interrupt=self.queue.interrupt,
                             **mfit_kwargs)

    def _ajuste_inicial(self, log_dir):
        # Muestreo
        self.queue.put(Msg('stage', 0))

        if len(self.sample) > 0:
            try:
                sampled_dataset = FKset(self.robot, self.sample)
            except RobotExecError:
                logging.info('RobotExecError durante muestreo')
                self.queue.put(Msg('fail', 0))
                return 'fail'

            self.sampled_dataset = sampled_dataset
        else:
            self.sampled_dataset = []

        full_dataset = ConcatDataset([self.sampled_dataset, *self.prev_datasets])
        self.train_set, self.val_set, self.test_set = rand_split(full_dataset, self.split)

        # Ajuste
        fit_kwargs = self.train_kwargs['Ajuste inicial']
        # Calcular número de pasos
        steps = fit_kwargs['epochs']
        if issubclass(type(self.modelo), FKEnsemble):
            steps *= len(self.modelo.ensemble)
        self.queue.put(Msg('stage', steps))

        if not self.queue.done:
            le_log = LastEpochLog()

            self.modelo.fit(train_set=self.train_set, val_set=self.val_set,
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

        try:
            self.modelo.active_fit(train_set=self.train_set,
                                label_fun=label_fun,
                                loggers=[self.gui_logger],
                                log_dir=log_dir,
                                silent=True,
                                ext_interrupt=self.queue.interrupt,
                                **afit_kwargs)
        except RobotExecError:
            self.queue.put(Msg('fail', 0))
            return 'fail'

    def get_dataset(self):
        # HACK: Para que no hayan problemas al guardar el dataset con pickle
        self.sampled_dataset.robot = None
        return self.sampled_dataset


class CtrlEjecucion:
    """
    Métodos para coordinar control punto a punto del robot.
    """
    def __init__(self):
        super().__init__()
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

    def ejecutar_trayec(self, reg_callback, error_callback):
        model_robot = ModelRobot(self.modelo_s)
        q_prev = torch.zeros(model_robot.n)
        for x, y, z, t_t, t_s in self.puntos:
            target = torch.Tensor([x,y,z])
            q = model_robot.ikine_pi_jacob(q_start=q_prev,
                                           p_target=target,
                                           #eta=0.1
                                           )

            try:
                _, p = self.robot_s.fkine(q)
            except RobotExecError:
                error_callback()
                return

            q_prev = q

            # Tomar última posición si se registró más de una
            if len(p.shape) == 2:
                p = p[-1]

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
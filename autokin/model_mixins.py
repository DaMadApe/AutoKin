import inspect
import logging
from copy import deepcopy
from random import choice
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, Subset,
                              TensorDataset, ConcatDataset,
                              random_split)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autokin.loggers import Logger, TqdmDisplay, TBLogger
from autokin.robot import RTBrobot
from autokin.muestreo import FKset
from autokin.utils import suavizar


logger = logging.getLogger('autokin')


class HparamsMixin:
    """
    Clase auxiliar para almacenar los parámetros con los que se define un
    modelo. Se guarda un diccionario en el atributo hparams del modelo.
    """
    def __init__(self):
        super().__init__()
        # Conseguir los argumentos con los que se definió el modelo
        frame = inspect.currentframe()
        frame = frame.f_back.f_back
        hparams = inspect.getargvalues(frame).locals

        # Quitar valor inecesario
        hparams.pop('self')

        # Si el argumento es una función o módulo, usar su nombre
        primitive_types = (int, float, str, bool)
        for key, val in hparams.items():
            if not isinstance(val, primitive_types):
                hparams[key] = val.__name__

        # Renombrar atributo __class__ por conveniencia
        hparams['tipo'] = hparams.pop('__class__')

        self.hparams = hparams


class DataFitMixin:
    """
    Implementa métodos para entrenar 
    """
    def __init__(self):
        super().__init__()
        self.checkpoint = {}

    def _set_out_bias(self, reference_set):
        """
        Ajustar el bias de salida a los promedios de salida
        de un set de referencia. Acelera convergencia de fit()
        """
        out_mean = torch.zeros(self.output_dim)
        for _, y in reference_set:
            out_mean += y

        out_mean /= len(reference_set)

        if hasattr(self, 'ensemble'):
            for model in self.ensemble:
                model.layers[-1].bias = nn.Parameter(out_mean)
        else:
            self.layers[-1].bias = nn.Parameter(out_mean)

    def _train_step(self, batch):
        X, Y = batch
        pred = self(X)
        train_loss = self.criterion(pred, Y)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        return train_loss

    def meta_fit(self,
                 task_datasets: list[Dataset],
                 n_steps: int = 5,
                 n_epochs_step: int = 1,
                 n_dh_datasets: int = 5,
                 n_samples_task: int = 500,
                 lr: float = 1e-4,
                 eps: float = 0.1,
                 ext_interrupt: Callable = None,
                 **fit_kwargs):

        for _ in range(n_dh_datasets):
            robot = RTBrobot.random(n=self.input_dim,
                                    min_DH=[1, 0, 0, 1],
                                    max_DH=[10, 2*torch.pi, 2*torch.pi, 10])
            robot_samples = FKset.random_sampling(robot, n_samples_task)
            task_datasets.append(robot_samples)

        for _ in range(n_steps):
            # Tomar aleatoriamente una de las tareas (datasets)
            task_ds = choice(task_datasets)
            # Tomar sólo el número solicitado de muestras por dataset
            task_ds = Subset(task_ds, range(n_samples_task))
            # Copiar modelo en estado actual para calcular params ajustados
            adjusted = deepcopy(self)
            # Encontrar params ajustados a la tarea
            adjusted.fit(task_ds, 
                         lr=lr,
                         batch_size=1024,
                         epochs=n_epochs_step,
                         silent=True,
                         use_checkpoint=False,
                         preadjust_bias=False,
                         ext_interrupt=ext_interrupt,
                         **fit_kwargs)

            # Aplicar actualización a los meta-parámetros
            with torch.no_grad():
                for p1, p2 in zip(self.parameters(), adjusted.parameters()):
                    p1 += eps*(p2-p1)

            if ext_interrupt is not None and ext_interrupt():
                return

    def fit(self, train_set: Dataset, 
            val_set: Dataset = None,
            epochs: int = 10,
            batch_size: int = 32,
            lr: float = 1e-3,
            weight_decay: float = 1e-3,
            criterion = nn.MSELoss(),
            optim: Type[torch.optim.Optimizer] = torch.optim.AdamW,
            lr_scheduler: bool = False,
            silent: bool = False,
            log_dir: str = None,
            use_checkpoint: bool = True,
            preadjust_bias: bool = True,
            loggers: list[Logger] = None,
            ext_interrupt: Callable = None):
        """
        Rutina de entrenamiento para ajustar a un conjunto de datos
        
        args:

        train_set (Dataset) : Conjunto de datos para entrenamiento
        val_set (Dataset) : Conjunto de datos para validación (opcional)
        epochs (int) : Número de recorridos al dataset
        lr (float) : Learning rate para el optimizador
        batch_size (int) : Número de muestras propagadas a la vez
        criterion (callable) : Función para evaluar la pérdida
        optim () : Clase de optimizador
        lr_scheduler (bool) : Reducir lr al frenar disminución de val_loss
        silent (bool) : Mostrar barra de progreso del entrenamiento
        log_dir (str) : Dirección para almacenar registros de Tensorboard
        checkpoint () : Cargar el estado resultante de un entrenaminto previo

        returns:
        
        checkpoint : Estado del optimizador y lr_scheduler, para reanudar
        entrenamiento
        """
        loggers = loggers if loggers is not None else []
        # Loggers automáticos según argumentos
        if not silent:
            loggers.append(TqdmDisplay(epochs))
        if log_dir is not None:
            loggers.append(TBLogger(log_dir))

        # Asegurar normalización de las muestras
        for dset in (train_set, val_set):
            if hasattr(dset, 'apply_p_norm'):
                dset.apply_p_norm = True

        # Colocar modelo en modo de entrenamiento
        self.train()

        # TODO: Transferir datos y modelo a GPU si está disponible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion

        if preadjust_bias:
            self._set_out_bias(train_set)

        self.optimizer = optim(self.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)
        if use_checkpoint and self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        if lr_scheduler:
            scheduler = ReduceLROnPlateau(self.optimizer)#, patience=5)
            if use_checkpoint:
                sched_state = self.checkpoint.get('sheduler_state_dict')
                if sched_state:
                    scheduler.load_state_dict(sched_state)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True)
        if val_set is not None and len(val_set)>0:
            val_loader = DataLoader(val_set, batch_size=len(val_set))

        for epoch in range(epochs):
            for batch in train_loader:
                train_loss = self._train_step(batch)

            progress_info = {'Loss/train': train_loss.item()}

            # Val step
            if val_set is not None:
                # self.eval()
                with torch.no_grad():
                    for X, Y in val_loader:
                        pred = self(X)
                        val_loss = criterion(pred, Y)

                        if lr_scheduler:
                            scheduler.step(val_loss)

                progress_info.update({'Loss/val': val_loss.item()})

            for logger in loggers:
                logger.log_step(progress_info, epoch)

            if ext_interrupt is not None and ext_interrupt():
                break

        for logger in loggers:
            logger.close()

        # Guardar estado de los optimizadores en el checkpoint
        self.checkpoint.update(
            {'optimizer_state_dict': self.optimizer.state_dict()})
        if lr_scheduler:
            self.checkpoint.update(
                {'sheduler_state_dict': scheduler.state_dict()})


    def test(self, test_set, criterion=nn.MSELoss()):
        test_loader = DataLoader(test_set, batch_size=len(test_set))
        with torch.no_grad():
            self.eval()
            for X, Y in test_loader:
                pred = self(X)
                test_loss = criterion(pred, Y)

        return test_loss


class ActiveFitMixin:
    def __init__(self):
        super().__init__()

    def query(self,
              candidate_batch: torch.Tensor = None,
              n_queries: int = 1) -> torch.Tensor:
        """
        De un conjunto de posibles puntos, devuelve
        el punto que maximiza la desviación estándar
        de las predicciones del grupo de modelos.
        """
        if candidate_batch is None:
            candidate_batch = torch.rand((100*n_queries, self.input_dim))

        with torch.no_grad():
            preds = torch.stack([model(candidate_batch) for model in self.ensemble])

        # La desviación estándar de cada muestra es la suma de la
        # varianza entre modelos de cada coordenada.
        # https://math.stackexchange.com/questions/850228/finding-how-spreaded-a-point-cloud-in-3d
        deviation = torch.sum(torch.var(preds, axis=0), axis=-1)
        candidate_idx = torch.topk(deviation, n_queries).indices
        query = candidate_batch[candidate_idx]

        # Ordenar puntos del query según norma euclidiana (~ menor a mayor tensión)
        indices = query.norm(dim=1).argsort()
        query = query[indices, :]

        return query
        # return torch.topk(deviation, n_queries)

    def query_trayec(self, query: torch.Tensor, n_rep: int = 5) -> torch.Tensor:
        """
        Función para conectar muestras solicitadas por interpolación lineal.
        """
        trayec = torch.cat([torch.zeros(1,self.input_dim),
                            *[q.repeat(n_rep,1) for q in query],
                            torch.zeros(1,self.input_dim),])
        trayec = suavizar(trayec,
                          q_prev = torch.zeros(self.input_dim),
                          dq_max=0.03)
        return trayec

    def active_fit(self, train_set, label_fun,
                   query_steps : int,
                   n_queries: int =1,
                   candidate_batch = None,
                   ext_interrupt=None,
                   **train_kwargs):
        """
        Ciclo para solicitar muestras y ajustar a ellas

        args:
        train_set (Dataset) : Conjunto base de entrenamiento
        label_fun (Callable: Tensor(N,d)->Tensor(N,d)) : Método para obtener las 
            etiquetas de nuevas muestras
        query_steps (int) : Número de veces que se solicitan nuevas muestras
        n_queries (int) : Número de muestras solicitadas en cada paso
        relative_weight (int) : Ponderación extra de las muestras nuevas (repetir en dataset)
        tb_dir (str) : Directorio base para guardar logs de tensorboard de entrenamientos
        **train_kwargs: Argumentos de entrenamiento usados para cada ajuste
        """
        for _ in range(query_steps):

            query = self.query(candidate_batch=candidate_batch,
                               n_queries=n_queries)
            query_trayec = self.query_trayec(query)
            logger.debug(f"Query: {query}, trayec.shape: {query_trayec.shape}")

            new_queries = label_fun(query_trayec)
            train_set = ConcatDataset([train_set, new_queries])

            self.fit(train_set,
                     ext_interrupt=ext_interrupt,
                     preadjust_bias=False,
                     **train_kwargs)

            if ext_interrupt is not None and ext_interrupt():
                return
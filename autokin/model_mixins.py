import inspect

import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader,
                              TensorDataset, ConcatDataset,
                              random_split)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autokin.loggers import Logger, TqdmDisplay, TBLogger
from autokin.robot import RTBrobot
from autokin.muestreo import FKset


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
    # TODO(?): Transferir funcionalidad a una clase Trainer
    def __init__(self):
        super().__init__()
        self.checkpoint = {}

    def _set_out_bias(self, reference_set=None):
        """
        Ajustar el bias de salida a los promedios de salida
        de un set de referencia. Acelera convergencia de fit()
        """
        out_size = reference_set[0][1].size()
        out_mean = torch.zeros(out_size)

        for _, y in reference_set:
            out_mean += y

        out_mean /= len(reference_set)

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
                 n_steps=10,
                 n_datasets=8,
                 n_samples=100,
                 n_post=10,
                 lr=1e-4,
                 post_lr=1e-4,
                 n_epochs=1,
                 n_post_epochs=1,
                 ext_interrupt=None,
                 **fit_kwargs):
        min_DH = [1, 0, 0, 1]
        max_DH = [10, 2*torch.pi, 2*torch.pi, 10]

        for _ in range(n_steps):
            sample_robots = []
            # sample_robots.extend(robots_by_n[input_dim])

            post_sets = []

            for _ in range(n_datasets): # - len(sample_robots)):
                robot = RTBrobot.random(n=self.input_dim,
                                        min_DH=min_DH,
                                        max_DH=max_DH)
                sample_robots.append(robot)

            for robot in sample_robots:
                full_set = FKset.random_sampling(robot, n_samples+n_post)
                train_set, post_set = random_split(full_set, [n_samples,
                                                              n_post])
                
                self.fit(train_set, 
                         lr=lr,
                         epochs=n_epochs,
                         silent=True,
                         use_checkpoint=False,
                         **fit_kwargs)
                post_sets.append(post_set)

                if ext_interrupt is not None and ext_interrupt():
                    return

            post_set = ConcatDataset(post_sets)
            self.fit(post_set, 
                     lr=post_lr,
                     epochs=n_post_epochs,
                     silent=True,
                     use_checkpoint=False,
                     ext_interrupt=ext_interrupt,
                     **fit_kwargs)

    def fit(self, train_set: Dataset, 
            val_set:Dataset = None,
            epochs=10,
            lr=1e-3,
            batch_size=32,
            criterion=nn.MSELoss(),
            optim=torch.optim.Adam,
            lr_scheduler=False,
            silent=False,
            log_dir=None,
            use_checkpoint=True,
            preadjust_bias=True,
            loggers: list[Logger] = None,
            ext_interrupt=None):
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

        # TODO: Transferir datos y modelo a GPU si está disponible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion

        if preadjust_bias:
            self._set_out_bias(train_set)

        self.optimizer = optim(self.parameters(), lr=lr)
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
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=len(val_set))

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                train_loss = self._train_step(batch)

            progress_info = {'Loss/train': train_loss.item()}

            # Val step
            if val_set is not None:
                self.eval()
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


class EnsembleMixin:
    def __init__(self):
        super().__init__()
        self.best_model_idx = None
        self.model_scores = None
    
    def __getitem__(self, idx):
        """
        Facilitar acceso a los modelos individuales desde afuera
        """
        return self.ensemble[idx]
    
    def forward(self, x):
        """
        Este forward (tal vez) no tiene utilidad para entrenamiento
        """
        return torch.stack([model(x) for model in self.ensemble])

    def joint_predict(self, x):
        """
        Devuelve la predicción promedio de todos los modelos
        """
        with torch.no_grad():
            self.ensemble.eval()
            preds = self(x)
        return torch.mean(preds, dim=0)

    def best_predict(self, x):
        """
        Devuelve la predicción del mejor modelo
        (requiere ejecutar antes self.test con un set de prueba)
        """
        if self.best_model_idx is None:
            raise RuntimeError('No se hizo ensemble.test()')
        else:
            with torch.no_grad():
                self.ensemble[self.best_model_idx].eval()
                pred = self.ensemble[self.best_model_idx](x)
            return pred

    def query(self,
              candidate_batch: torch.Tensor = None,
              n_queries: int = 1) -> torch.Tensor:
        """
        De un conjunto de posibles puntos, devuelve
        el punto que maximiza la desviación estándar
        de las predicciones del grupo de modelos.
        """
        if candidate_batch is None:
            candidate_batch = torch.rand((1000, self.input_dim))

        with torch.no_grad():
            self.ensemble.eval()
            preds = self(candidate_batch)

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

    def meta_fit(self, **m_fit_kwargs):
        """
        Entrenar cada uno de los modelos individualmente
        """
        for model in self.ensemble:
            model.meta_fit(**m_fit_kwargs)

    def fit(self, train_set, **train_kwargs):
        """
        Entrenar cada uno de los modelos individualmente
        """
        for model in self.ensemble:
            model.fit(train_set, **train_kwargs)

    def test(self, test_set, **test_kwargs):
        self.model_scores = []
        for model in self.ensemble:
            score = model.test(test_set, **test_kwargs)
            self.model_scores.append(score)

        self.best_model_idx = self.model_scores.index(min(self.model_scores))

        return self.model_scores.copy()

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

            labels = label_fun(query)
            new_queries = TensorDataset(query, labels)
            train_set = ConcatDataset([train_set, new_queries])

            self.fit(train_set,
                     ext_interrupt=ext_interrupt,
                     **train_kwargs)

            if ext_interrupt is not None and ext_interrupt():
                return
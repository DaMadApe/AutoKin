import inspect

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autokin.loggers import Logger, TqdmDisplay, TBLogger


class HparamsMixin():
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


class DataFitMixin():
    # TODO: Transferir funcionalidad a una clase Trainer
    def __init__(self):
        super().__init__()
        self.checkpoint = {}
        self.trained_epochs = 0

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


    def fit(self, train_set, val_set=None,
            epochs=10, lr=1e-3, batch_size=32,
            criterion=nn.MSELoss(), optim=torch.optim.Adam,
            lr_scheduler=False, silent=False, log_dir=None,
            use_checkpoint=True, preadjust_bias=True,
            loggers: list[Logger] = None, ext_interrupt=None):
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

        # Métricas para almacenar junto a hiperparámetros
        if val_set is not None:
            metrics = {'Last val loss': val_loss.item()}
        else:
            metrics = {'Last train loss': train_loss.item()}

        self.hparams.update({'lr':lr, 'batch_size':batch_size})

        for logger in loggers:
            logger.log_hparams(self.hparams, metrics)
            logger.close()
        
        self.trained_epochs += epochs

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
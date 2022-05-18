import inspect
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


class HparamsMixin():
    """
    Clase auxiliar para almacenar los parámetros con los que se define un
    modelo. Se guarda un diccionario en el atributo hparams del modelo.
    """
    def __init__(self):
        super().__init__()
        frame = inspect.currentframe()
        frame = frame.f_back
        hparams = inspect.getargvalues(frame).locals
        hparams.pop('self')

        # Si el argumento es una función o módulo, usar su nombre
        primitive_types = (int, float, str, bool)
        for key, val in hparams.items():
            if not isinstance(val, primitive_types):
                hparams[key] = val.__name__

        self.hparams = hparams


class DataFitMixin():
    def __init__(self):
        super().__init__()
        self.checkpoint = {}
        self.writer = None
        self.trained_epochs = 0

    def set_out_bias(self, reference_set=None):
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


    def fit(self, train_set, val_set=None,
            epochs=10, lr=1e-3, batch_size=32,
            criterion=nn.MSELoss(), optim=torch.optim.Adam,
            lr_scheduler=False, silent=False, log_dir=None,
            use_checkpoint=True, preadjust_bias=True):
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
        
        checkpoint : Estado del optimizador y lr_scheduler, para reanudar entrenamiento
        """
        # TODO: Transferir datos y modelo a GPU si está disponible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train()

        if preadjust_bias:
            self.set_out_bias(train_set)

        if log_dir is not None:
            if self.writer is None or not os.path.samefile(self.writer.log_dir, log_dir):
                self.writer = SummaryWriter(log_dir=log_dir)
            # else:
            #     self.writer.open()

        optimizer = optim(self.parameters(), lr=lr)
        if use_checkpoint and self.checkpoint:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        if lr_scheduler:
            scheduler = ReduceLROnPlateau(optimizer)#, patience=5)
            if use_checkpoint and self.checkpoint:
                scheduler.load_state_dict(self.checkpoint['sheduler_state_dict'])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=len(val_set))

        if silent:
            epoch_iter = range(epochs)
        else:
            epoch_iter = tqdm(range(epochs), desc='Training')

        for epoch in epoch_iter:
            # Train step
            for X, Y in train_loader:
                pred = self(X)
                train_loss = criterion(pred, Y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                if log_dir is not None:
                    self.writer.add_scalar('Loss/train', train_loss.item(),
                                           epoch + self.trained_epochs)

            progress_info = {'Loss': train_loss.item()}

            # Val step
            if val_set is not None:
                with torch.no_grad():
                    self.eval()
                    for X, Y in val_loader:
                        pred = self(X)
                        val_loss = criterion(pred, Y)

                        if lr_scheduler:
                            scheduler.step(val_loss)

                        if log_dir is not None:
                            self.writer.add_scalar('Loss/val', val_loss.item(),
                                                   epoch + self.trained_epochs)

                progress_info.update({'Val': val_loss.item()})

            if not silent:
                epoch_iter.set_postfix(progress_info)

        if log_dir is not None:
            if val_set is not None:
                metrics = {'Last val loss': val_loss.item()}
            else:
                metrics = {'Last train loss': train_loss.item()}

            self.writer.add_hparams({**self.hparams, 'lr':lr, 'batch_size':batch_size},
                            metric_dict=metrics, run_name='.')
            self.writer.close()

        self.trained_epochs += epochs
        self.checkpoint.update({'optimizer_state_dict': optimizer.state_dict()})

        if lr_scheduler:
            self.checkpoint.update({'sheduler_state_dict': scheduler.state_dict()})


    def test(self, test_set, criterion=nn.MSELoss()):
        test_loader = DataLoader(test_set, batch_size=len(test_set))
        with torch.no_grad():
            self.eval()
            for X, Y in test_loader:
                pred = self(X)
                test_loss = criterion(pred, Y)

        return test_loss
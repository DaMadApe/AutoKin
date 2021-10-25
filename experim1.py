from pytorch_lightning.core import lightning
import torch
from torch import nn
from torch.nn import functional as F
#from torchvision.datasets import MNIST
#from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

"""
Módulo del modelo
Regresor para funciones multivariables con salida real
"""
class Regressor(pl.LightningModule):

    def __init__(self, in_size=1, out_size=1,
                 mid_size=30, depth=1,
                 lr=1e-3, activation=F.relu):
        super().__init__()
        self.save_hyperparameters()

        # Armar modelo
        self.layers = nn.ModuleList()
        # Capa de entrada
        self.layers.append(nn.Linear(self.hparams.in_size,
                                     self.hparams.mid_size))
        # Capas intermedias
        for _ in range(self.hparams.depth):
            self.layers.append(nn.Linear(self.hparams.mid_size,
                                         self.hparams.mid_size))
        # Capa de salida
        self.layers.append(nn.Linear(self.hparams.mid_size,
                                     self.hparams.out_size))

        # Se necesita definir para save_graph
        self.example_input_array = torch.zeros(in_size, 1)
        # Nombre específico para LR finder
        self.lr = lr 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        for layer in self.layers:
            x = self.hparams.activation(layer(x))
        return x

    def training_step(self, batch, batch_idx):
        point, target = batch
        pred = self(point)
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        self.log('train_loss', loss)#, prog_bar=True)
        return loss

    #def training_epoch_end(self, outputs):
        

if __name__ == '__main__':

    # Hyperparams
    cpu_cores = 8 # Núcleos usados por el programa
    nn_mid_size = 30 # Tamaño de capas intermedias
    n_samples = 24
    batch_size = 4
    epochs = 100
    lr = 3e-4

    """
    Conjunto de datos
    """
    def f(x): return 0.4*x**3 - 0.8*x**2 + 2*x + 10
    #x = torch.linspace(0, 1, n_samples).view(-1,1)
    x = torch.rand(n_samples, 1)
    y = f(x)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, num_workers=cpu_cores)

    """
    Entrenamiento
    """
    logger = TensorBoardLogger('lightning_logs', 'Exp1',
                               #default_hp_metric=False,
                               log_graph=True)

    model = Regressor(mid_size=nn_mid_size, lr=lr)
    #trainer = pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(max_epochs=epochs,
                         logger=logger)
    trainer.fit(model, train_loader)

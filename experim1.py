from pytorch_lightning.core import lightning
import torch
from torch import nn
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

    def __init__(self, in_size, mid_size, lr):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, 1))
        # Se necesita definir para save_graph
        self.example_input_array = torch.zeros(in_size,1)
        # Nombre específico para LR finder
        self.lr = lr 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        point, target = batch
        pred = self.model(point)
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
    epochs = 50
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

    model = Regressor(in_size=1, mid_size=nn_mid_size, lr=lr)
    #trainer = pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(max_epochs=epochs,
                         logger=logger)
    trainer.fit(model, train_loader)

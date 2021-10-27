import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

"""
Módulo del modelo
Regresor para funciones multivariables con salida real
"""
class Regressor(pl.LightningModule):

    def __init__(self, in_size=1, out_size=1, mid_size=30,
                 depth=1, lr=1e-3, activation=F.relu,
                 *args, **kwargs):
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
        self.example_input_array = torch.zeros(self.hparams.in_size, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
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
        self.log('train_loss', loss)
        self.log('hp_metric', loss)
        return loss


if __name__ == '__main__':

    pl.seed_everything(36)

    # Hyperparams
    cpu_cores = 8 # Núcleos usados por el programa
    depth = 3
    nn_mid_size = 30 # Tamaño de capas intermedias
    n_samples = 24
    batch_size = 4
    epochs = 100
    lr = 6e-4
    view_plot = True

    """
    Conjunto de datos
    """
    def f(x): return torch.sin(10*x**2)
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
                               log_graph=True)

    model = Regressor(mid_size=nn_mid_size, depth=depth, lr=lr)

    # trainer = pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(max_epochs=epochs,
                         logger=logger)
    trainer.fit(model, train_loader)

    """
    Visualización de datos en una dimensión
    """
    if view_plot:
        x_plot = torch.linspace(0, 1, n_samples).view(-1,1)

        with torch.no_grad():
            pred = model(x_plot)
        fig, ax = plt.subplots()
        ax.plot(x_plot, f(x_plot))
        ax.plot(x_plot, pred)
        ax.scatter(x, y)
        ax.legend(['Target F',
                   'Model',
                   'Trainset'])
        plt.show()
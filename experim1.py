import torch
from torch import nn
#from torchvision.datasets import MNIST
#from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

# Hyperparams
cpu_cores = 4 # Núcleos usados por el programa
nn_mid_size = 30 # Tamaño de capas intermedias
n_samples = 24

"""
Módulo del modelo
"""
class Regressor(pl.LightningModule):

    def __init__(self, mid_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, 1))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        point, target = batch
        pred = self.model(point)
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        self.log('train_loss', loss)
        return loss

"""
Conjunto de datos
"""

def f(x): return 2*x#0.4*x**3 - 0.8*x**2 + 2*x + 10
#x = torch.linspace(0, 1, n_samples).view(-1,1)
x = torch.rand(n_samples, 1)
y = f(x)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset,
                          batch_size=2, shuffle=True)
                          #num_workers=cpu_cores)


"""
Entrenamiento
"""
model = Regressor(mid_size=nn_mid_size)
trainer = pl.Trainer(num_processes=cpu_cores)
trainer.fit(model, train_loader)
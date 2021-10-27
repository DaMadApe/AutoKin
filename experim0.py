from typing import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import logging
import matplotlib.pyplot as plt

"""
TODO
- Hacer variable número de capas
- Meter tensorboard con guild
"""
# args
lr = 1e-3
depth = 3
mid_layer_size = 10
n_samples = 10
batch_size = 10
epochs = 1000

input_dim = 1
output_dim = 1
x_min = 0
x_max = 1
view_plot = True

def f(x): return 3*x**3 - 5*x**2 + 2*x


class Regressor(nn.Module):

    def __init__(self, input_dim=1, output_dim=1,
                 depth=1, mid_layer_size=10, activation=F.relu):
        super().__init__()

        # Armar modelo
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim,
                                     mid_layer_size))
        for _ in range(depth):
            self.layers.append(nn.Linear(mid_layer_size,
                                         mid_layer_size))
        self.layers.append(nn.Linear(mid_layer_size,
                                     output_dim))

        self.activation = activation

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

"""
Conjunto de datos
"""
x = torch.linspace(x_min, x_max, n_samples).view(-1, 1)
y = f(x)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


"""
Entrenamiento
"""
model = Regressor(input_dim, 1, 3, 10)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for t in range(epochs):
    for X, Y in train_loader:
        pred = model(X)
        loss = criterion(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if t%(epochs//10) == 0:
        print(f'Epoch {t}: Loss={loss.item()}')

"""
Visualización de datos en una dimensión
"""
if view_plot:
    x_plot = torch.linspace(x_min, x_max, n_samples).view(-1,1)

    with torch.no_grad():
        pred = model(x_plot)
    fig, ax = plt.subplots()
    ax.plot(x_plot, f(x))
    ax.plot(x_plot, pred)
    ax.scatter(x, y)
    ax.legend(['Target F',
               'Model',
               'Trainset'])
    plt.show()
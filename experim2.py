"""
- Pytorch puro
- Regresión con validación
- Conjunto de datos 2D->2D
- Normalización de entradas y etiquetas
- tqdm
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from experim0 import Regressor


if __name__ == '__main__':

    torch.manual_seed(42)

    # args
    lr = 5e-3
    depth = 3
    mid_layer_size = 10
    activation = torch.tanh
    n_samples = 256
    batch_size = 32
    epochs = 2000

    input_dim = 2
    output_dim = 2
    x_min = -1
    x_max = 1
    view_plot = True

    def f(x): return torch.sin(6*x**2)

    """
    Conjunto de datos
    """
    x = (x_max - x_min)*torch.rand(n_samples, input_dim) + x_min
    y = f(x)

    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    """
    Entrenamiento
    """
    model = Regressor(input_dim, output_dim,
                      depth, mid_layer_size,
                      activation)

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
        x1_plot = torch.linspace(x_min, x_max, 256).view(-1,1)
        x2_plot = torch.linspace(x_min, x_max, 256).view(-1,1)

        #grid_x1, grid_x2 = torch.meshgrid(x1_plot, x2_plot)

        # with torch.no_grad():
        #     pred = model(x_plot)

        fig, axes = plt.subplots(1, output_dim,
                                 subplot_kw={'projection': '3d'})
        for i, ax in enumerate(axes):
            # ax.plot(x_plot, f(x_plot))
            # ax.plot(x_plot, pred)
            ax.scatter(x[:,0], x[:,1], y[:,i])

            # ax.legend(['Target F',
            #            'Model',
            #            'Trainset'])
        plt.show()
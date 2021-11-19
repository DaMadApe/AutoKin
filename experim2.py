"""
- Pytorch puro
- Regresión con validación
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
    activation = torch.relu
    n_samples = 512
    batch_size = 512
    epochs = 500

    input_dim = 2
    output_dim = 1
    x_min = -1
    x_max = 1
    view_plot = True

    #def f(x): return torch.sin(x[0]**2 + x[1]**2)
    def f(x):
        r = torch.linalg.norm(4*x**2, dim=1)
        return torch.sin(r).view(-1,1)

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
        res = 256

        x1_plot = torch.linspace(x_min, x_max, res)
        x2_plot = torch.linspace(x_min, x_max, res)

        x1_grid, x2_grid = torch.meshgrid(x1_plot, x2_plot)

        # Magia negra para que las dimensiones ajusten con la fun
        y_plot = f(torch.stack([x1_grid, x2_grid], dim=1))
        # Más magia negra para que la forma sea graficable
        y_plot = y_plot.view(res,res)

        with torch.no_grad():
            pred = model(torch.stack([x1_grid, x2_grid], dim=-1).view(-1, 2))
            pred = pred.view(res, res)

        fig, axes = plt.subplots(output_dim, 2,
                                 subplot_kw={'projection': '3d'})
        if output_dim==1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax[0].plot_surface(x1_grid.numpy(), x2_grid.numpy(),
                               y_plot.numpy(), label='Target', color='r')
            ax[0].plot_wireframe(x1_grid.numpy(), x2_grid.numpy(),
                                 pred.numpy(), label='Trainset', color='b')
            # ax[0].legend(['Target',
            #               'Modelo'])
            ax[1].plot_wireframe(x1_grid.numpy(), x2_grid.numpy(),
                                 pred.numpy(), color='red')
            ax[1].scatter(x[:,0], x[:,1], y[:,i])
        plt.show()
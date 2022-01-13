"""
Probar una arquitectura residual.
"""
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from experim0 import save

class ResBlock(nn.Module):

    def __init__(self, depth=3, block_width=10):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(nn.Linear(block_width,
                                         block_width))

    def forward(self, x):
        identity = x #.copy?
        for layer in self.layers:
            x = layer(x)
        return x + identity


class ResNet(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, depth=3, block_depth=3,
                 block_width=10, activation=torch.tanh):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(input_dim,
                                     block_width))
        for _ in range(depth):
            self.blocks.append(ResBlock(block_depth, block_width))
        
        self.blocks.append(nn.Linear(block_width,
                                     output_dim))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":

    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    torch.manual_seed(42)

    block = ResBlock()
    model = ResNet()


    # args
    lr = 5e-3
    depth = 3
    block_depth = 3
    block_width = 10
    activation = torch.tanh
    n_samples = 32
    batch_size = 32
    epochs = 500

    input_dim = 1
    output_dim = 1
    x_min = -1
    x_max = 1
    view_plot = True

    #def f(x): return 3*x**3 - 3*x**2 + 2*x
    def f(x): return torch.sin(10*x**2 - 10*x)

    """
    Conjunto de datos
    """
    x = torch.linspace(x_min, x_max, n_samples).view(-1, input_dim)
    y = f(x)

    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    """
    Entrenamiento
    """
    model = ResNet(input_dim=input_dim, output_dim=output_dim,
                   depth=depth, block_depth=block_depth, block_width=block_width,
                   activation=activation)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    progress = tqdm(range(epochs), desc='Training')
    for _ in progress:
        # Train step
        # Invertir muestras para fines de exp
        for X, Y in train_loader:
            model.train()
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress.set_postfix(Loss=loss.item())

    """
    Visualización de datos en una dimensión
    """
    if view_plot:
        x_plot = torch.linspace(x_min, x_max, 1000).view(-1,1)

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

    """
    Guardar modelo
    """
    path = 'models/experim13'
    save(model, path, 'v1')
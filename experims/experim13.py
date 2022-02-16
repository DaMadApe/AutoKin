"""
Probar una arquitectura residual.
"""
import torch
from torch import nn

from experim0 import HparamsMixin


class ResBlock(nn.Module):

    def __init__(self, depth, block_width, activation):
        super().__init__()
        
        self.activation = activation
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(nn.Linear(block_width,
                                         block_width))

    def forward(self, x):
        identity = x.clone() # TODO: Confirmar si clone es necesario
        for layer in self.layers:
            x = self.activation(layer(x))
        return x + identity


class ResNet(HparamsMixin, nn.Module):

    def __init__(self, input_dim=1, output_dim=1,
                 depth=3, block_depth=3, block_width=10,
                 activation=torch.tanh):
        super().__init__()
        
        self.input_dim = input_dim
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(input_dim,
                                     block_width))
        for _ in range(depth):
            self.blocks.append(ResBlock(block_depth, block_width, activation))
        
        self.blocks.append(nn.Linear(block_width,
                                     output_dim))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":

    from torch.utils.data import TensorDataset
    import matplotlib.pyplot as plt

    from experim14 import train

    torch.manual_seed(42)

    # args
    n_samples = 32
    x_min = -1
    x_max = 1
    view_plot = True

    #def f(x): return 3*x**3 - 3*x**2 + 2*x
    def f(x): return torch.sin(10*x**2 - 10*x)

    """
    Conjunto de datos
    """
    x = torch.linspace(x_min, x_max, n_samples).view(-1, 1)
    y = f(x)

    train_set = TensorDataset(x, y)

    """
    Entrenamiento
    """
    model = ResNet(input_dim=1, output_dim=1,
                   depth = 3,
                   block_depth = 3,
                   block_width = 8,
                   activation = torch.tanh)

    train(model, train_set,
          epochs=2000,
          lr=5e-3,
          batch_size=32,
          log_dir='tb_logs/exp13')

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

    # Guardar modelo
    torch.save(model, 'models/experim13_v1.pt')
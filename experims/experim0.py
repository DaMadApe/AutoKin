"""
Pruebas de regresión  con pytorch puros
"""
import os
import inspect

import torch
from torch import nn


class HparamsMixin():
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

"""
MLP configurable de ancho constante
"""
class MLP(HparamsMixin, nn.Module):

    def __init__(self, input_dim=1, output_dim=1,
                 depth=1, mid_layer_size=10, activation=torch.tanh):
        super().__init__()

        self.input_dim = input_dim
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
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


if __name__ == '__main__':

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
    model = MLP(input_dim=1, output_dim=1,
                depth=6,
                mid_layer_size=10,
                activation=torch.tanh)

    print(model.hparams)

    train(model, train_set,
          epochs=2000,
          lr=5e-3,
          batch_size=32,
          log_dir='tb_logs/exp0')

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
    """
    Save:
    torch.save(model, PATH)

    Load:
    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()
    """
    torch.save(model, 'models/experim0_v1.pt')
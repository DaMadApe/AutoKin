"""
Pruebas de regresión  con pytorch puros
"""
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

"""
MLP configurable de ancho constante
"""
class MLP(nn.Module):

    def __init__(self, input_dim=1, output_dim=1,
                 depth=1, mid_layer_size=10, activation=torch.tanh):
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
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

def save(model, path, name):
    name += ".pt"
    if not os.path.exists(path):
        print("Path doesn't exist, creating folder...")
        os.makedirs(path)
    save_path = os.path.join(path, name)
    torch.save(model.state_dict(), save_path)

def load(model, path, name):
    name += ".pt"
    load_path = os.path.join(path, name)
    model.load_state_dict(torch.load(load_path))


if __name__ == '__main__':

    torch.manual_seed(42)

    # args
    lr = 5e-3
    depth = 3
    mid_layer_size = 10
    activation = torch.tanh
    n_samples = 32
    batch_size = 32
    epochs = 2000

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
    model = MLP(input_dim, output_dim,
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
    path = 'models/experim0'
    save(model, path, 'v1')

    """
    Cargar modelo
    """
    # model = MLP(*args, **kwargs)
    # cargar(model, path, 'v1')
"""
Pruebas de regresión unidimensional
"""
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from modelos import MLP
from entrenamiento import train

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
      lr=1e-3,
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
# torch.save(model, 'models/experim0_v1.pt')
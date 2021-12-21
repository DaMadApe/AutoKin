"""
Obtener el Jacobiano de una red neuronal que aproxime
una función arbitraria: la que se prueba en experim0.py
y comparar con su derivada analítica
"""
import torch
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt

from experim0 import Regressor, load

"""
Cargar modelo
"""
path = 'models/experim0'
name = 'v1'

# args
depth = 3
mid_layer_size = 10
activation = torch.tanh
input_dim = 1
output_dim = 1

model = Regressor(input_dim, output_dim,
                    depth, mid_layer_size,
                    activation)
load(model, path, name)
model.eval()

# Jacobiano de la red neuronal
def j_model(x):
    return jacobian(model, x, vectorize=True)

"""
Comparar modelo y jacobiano con funciones originales
"""
x_min = -1
x_max = 1
def f(x): return torch.sin(10*x**2 - 10*x)
def df_dx(x): return torch.cos(10*x**2 - 10*x) * (20*x - 10)

x_plot = torch.linspace(x_min, x_max, 1000).view(-1,1)

with torch.no_grad():
    pred = model(x_plot)
    pred_j = j_model(x_plot)
    # La forma del vector de entrada introduce dimensiones
    # adicionales en el resultado [N, 1, N, 1]
    pred_j = pred_j.squeeze()
    # Pasar el batch entero al jacobiano saca las derivadas
    # de cada salida respecto a todas las entradas del batch
    # Sólo es relevante la diagonal, lo demás son ceros.
    pred_j = torch.diagonal(pred_j)

fig, ax = plt.subplots()
ax.plot(x_plot, f(x_plot))
ax.plot(x_plot, pred, '--')
ax.plot(x_plot, df_dx(x_plot))
ax.plot(x_plot, pred_j, '--')
ax.legend(['Target F',
           'Predicted F',
           'Target J(F)',
           'Predicted J(F)'])
plt.show()
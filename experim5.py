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

# Función para calcular el jacobiano para cada muestra de un batch
def batch_jacobian(func, x, create_graph=False, vectorize=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)
    return jacobian(_func_sum, x,
                    create_graph=create_graph,
                    vectorize=vectorize).permute(1,0,2)

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
    pred_j = batch_jacobian(model, x_plot).squeeze()

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
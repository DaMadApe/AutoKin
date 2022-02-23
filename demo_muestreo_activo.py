import torch
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt

from modelos import MLP, ResNet
from muestreo_activo import EnsembleRegressor

"""
Conjunto de datos
"""
x_min = -1
x_max = 1
n_samples = 20
n_models = 3

# Función de prueba
def f(x): return torch.sin(10*x**2 - 10*x)
# def f(x): return torch.cos(10*x**2)

X = torch.linspace(x_min, x_max, n_samples).reshape(-1, 1)
Y = f(X)
full_set = TensorDataset(X, Y)

split_proportions = [0.5, 0.5]
split = [round(prop*len(X)) for prop in split_proportions]

train_set, test_set = random_split(full_set, split)

# Conversión nada elegante para poder graficar ejemplos originales
# Sólo es relevante para graficar ejemplos en 1D
X_train = torch.tensor([])
Y_train = torch.tensor([])
for x, y in train_set:
    X_train = torch.cat((X_train, x.unsqueeze(dim=0)))
    Y_train = torch.cat((Y_train, y.unsqueeze(dim=0)))

"""
Primer entrenamiento
"""
# Declarar conjunto de modelos
models = [MLP(input_dim=1,
                output_dim=1,
                depth=3,
                mid_layer_size=10,
                activation=torch.tanh) for _ in range(n_models)]

ensemble = EnsembleRegressor(models)

# Entrenar el modelo con datos iniciales
ensemble.fit(train_set, lr=3e-3, epochs=1000)
ensemble.rank_models(test_set)

# Para graficar después
X_plot = torch.linspace(x_min, x_max, 1000).view(-1,1)
first_pred = ensemble.best_predict(X_plot).detach()

"""
Afinación con muestras nuevas recomendadas
"""
# Datos disponibles para 'pedir'
X_query = torch.rand(64).view(-1, 1)*(x_max-x_min) + x_min

queries, _ = ensemble.online_fit(train_set,
                                    candidate_batch=X_query,
                                    label_fun=f,
                                    query_steps=4,
                                    n_queries=2,
                                    relative_weight=2,
                                    final_adjust_weight=4,
                                    lr=1e-3, epochs=200)
ensemble.rank_models(test_set)

last_pred = ensemble.best_predict(X_plot).detach()

"""
Graficar resultados
"""
fig, ax = plt.subplots()
ax.plot(X_plot, f(X_plot))
ax.scatter(X_train, Y_train)
ax.scatter(queries, f(queries))
ax.plot(X_plot, first_pred)
ax.plot(X_plot, last_pred)
labels = ['Target F', 'Trainset', 'Queries', 'Primer entrenamiento',
            'Luego de muestreo']
# for i in range(n_models):
#     ax.plot(X_plot, ensemble[i](X_plot).detach())
#     labels.append(f'Model {i}')

# ax.plot(X_plot, ensemble.joint_predict(X_plot))
# labels.append('Joint predict')

ax.legend(labels)
plt.show()
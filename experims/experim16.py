"""
Réplica de experim 12 implementado desde cero (sin modAL)
Evitar numpy, usar torch donde sea posible.
"""
import torch
from torch.utils.data import TensorDataset

from experim14 import train


class EnsembleRegressor(torch.nn.Module):
    """
    Produce n copias de un mismo tipo de modelo.
    """
    def __init__(self, models : list[torch.nn.Module]):
        super().__init__()
        self.ensemble = torch.nn.ModuleList(models)
    
    def forward(self, x):
        return torch.stack([model(x) for model in self.ensemble])

    def predict(self, x):
        with torch.no_grad():
            self.ensemble.eval()
            preds = self(x)
        return torch.mean(preds, dim=0)

    def query(self, candidate_batch, n_queries=1):
        """
        De un conjunto de posibles puntos, devuelve
        el punto que maximiza la desviación estándar
        de las predicciones del grupo de modelos.
        """
        # preds = torch.zeros((len(self.ensemble),
        #                      len(candidate_batch,
        #                      self.output_dim)))
        # with torch.no_grad():
        #     for i, model in enumerate(self.ensemble):
        #         model.eval()
        #         preds[i] = model(candidate_batch).numpy()

        with torch.no_grad():
            self.ensemble.eval()
            preds = self(candidate_batch)

        # La desviación estándar de cada muestra es la suma de la
        # varianza entre modelos de cada coordenada.
        # https://math.stackexchange.com/questions/850228/finding-how-spreaded-a-point-cloud-in-3d
        deviation = torch.sum(torch.var(preds, axis=0), axis=-1)
        candidate_idx = torch.topk(deviation, n_queries).indices
        query = candidate_batch[candidate_idx]
        return candidate_idx, query
        # return torch.topk(deviation, n_queries)

    def fit(self, train_set, **train_kwargs):
        for model in self.ensemble:
            model.train()
            train(model, train_set, **train_kwargs)



class MockModule(torch.nn.Module):
    """
    Módulo auxiliar para probar ensamble
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rand(x.shape[0], 3)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from experim0 import MLP

    x_min = -1
    x_max = 1
    n_samples = 30
    n_models = 5
    n_queries = 10

    #def f(x): return 3*x**3 - 3*x**2 + 2*x
    def f(x): return torch.sin(10*x**2 - 10*x)

    X_train = torch.linspace(x_min, x_max, n_samples).view(-1, 1)
    y_train = f(X_train)

    X_query = torch.linspace(x_min/2, x_max, 100).reshape(-1, 1)/1.2

    models = [MLP(input_dim=1,
                  output_dim=1,
                  depth=3,
                  mid_layer_size=10,
                  activation=torch.tanh) for _ in range(n_models)]

    ensemble = EnsembleRegressor(models)

    # Entrenar el model con datos iniciales
    train_set = TensorDataset(X_train, y_train)
    ensemble.fit(train_set, epochs=100)
    # Solicitar recomenación de nuevas muestras
    _, queries = ensemble.query(X_query, n_queries)

    """
    Graficar conjunto de datos
    """
    X_plot = torch.linspace(x_min, x_max, 1000).view(-1,1)

    fig, ax = plt.subplots()
    ax.plot(X_plot, f(X_plot))
    ax.scatter(X_train, y_train)
    ax.scatter(queries, f(queries))
    ax.plot(X_plot, ensemble.predict(X_plot))
    ax.legend(['Target F',
               'Trainset',
               'Queries',
               'Model'])
    plt.show()
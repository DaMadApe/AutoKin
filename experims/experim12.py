"""
Aprendizaje activo: Selección dinámica de puntos para
mejorar la convergencia de la regresión de experim0
"""
import torch
from torch.utils.data import TensorDataset

from experim14 import train


class FKEstimator():

    def __init__(self, model):
        self.model = model #MLP(args)

    def fit(self, X, y, batch_size=32, **train_kwargs):
        """
        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)
        """
        # Acondicionar datos
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)
        train_set = TensorDataset(X_tensor, y_tensor)

        train(self.model, train_set, silent=True, **train_kwargs)
        return self

    def predict(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float)
        with torch.no_grad():
            pred = self.model(x_tensor).numpy()
        return pred


if __name__ == "__main__":
    import numpy as np
    from modAL.models import ActiveLearner, CommitteeRegressor
    from modAL.disagreement import max_std_sampling
    import matplotlib.pyplot as plt

    from experim0 import MLP

    np.random.seed(64)
    torch.manual_seed(64)

    x_min = -1
    x_max = 1
    n_samples = 20
    n_models = 5
    n_queries = 10

    #def f(x): return 3*x**3 - 3*x**2 + 2*x
    def f(x): return np.sin(10*x**2 - 10*x)

    X_train = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)
    y_train = f(X_train)

    #X_query = np.linspace(x_min/2, x_max, 100).reshape(-1, 1)/1.2
    X_query = np.random.rand(100).reshape(-1, 1)

    learner_list = [ActiveLearner(
        estimator=FKEstimator(MLP(input_dim=1,
                                  output_dim=1,
                                  depth=6,
                                  mid_layer_size=10,
                                  activation=torch.tanh))
            ) for _ in range(n_models)]

    ensemble = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )

    queries = []

    for _ in range(n_queries):
        query_idx, query = ensemble.query(X_query)
        ensemble.teach(query, f(query), only_new=True)
        queries.append(query)

    print(queries)

    """
    Graficar conjunto de datos
    """
    X_plot = torch.linspace(x_min, x_max, 1000).view(-1,1)
    queries = np.array(queries)

    # with torch.no_grad():
    #     pred = model(x_plot)
    fig, ax = plt.subplots()
    ax.plot(X_plot, f(X_plot))
    # ax.plot(x_plot, pred)
    ax.scatter(X_train, y_train)
    ax.scatter(queries, f(queries))
    ax.legend(['Target F',
               'Trainset',
               'Queries'])
    plt.show()
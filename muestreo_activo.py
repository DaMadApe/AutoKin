import torch
from torch.utils.data import ConcatDataset

from entrenamiento import train

# Mover a modelos?
class EnsembleRegressor(torch.nn.Module):
    """
    Agrupa un conjunto de modelos, permite entrenarlos en conjunto,
    y luego predecir qué nuevas muestras serían más efectivas.
    """
    def __init__(self, models : list[torch.nn.Module]):
        super().__init__()
        self.ensemble = torch.nn.ModuleList(models)
        self.best_model_idx = None
    
    def __getitem__(self, idx):
        """
        Facilitar acceso a los modelos individuales desde afuera
        """
        return self.ensemble[idx]
    
    def forward(self, x):
        """
        Este forward (tal vez) no tiene utilidad para entrenamiento
        """
        return torch.stack([model(x) for model in self.ensemble])

    def joint_predict(self, x):
        """
        Devuelve la predicción promedio de todos los modelos
        """
        with torch.no_grad():
            self.ensemble.eval()
            preds = self(x)
        return torch.mean(preds, dim=0)

    def best_predict(self, x):
        """
        Devuelve la predicción del mejor modelo
        (requiere ejecutar antes rank_models con un set de prueba)
        """
        if self.best_model_idx is None:
            pass
        else:
            with torch.no_grad():
                self.ensemble[self.best_model_idx].eval()
                pred = self.ensemble[self.best_model_idx](x)
            return pred

    def query(self, candidate_batch, n_queries=1):
        """
        De un conjunto de posibles puntos, devuelve
        el punto que maximiza la desviación estándar
        de las predicciones del grupo de modelos.
        """
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
        """
        Entrenar cada uno de los modelos individualmente
        """
        print("Ajustando modelos del conjunto")
        for i, model in enumerate(self.ensemble):
            train(model, train_set, **train_kwargs)
        print("Fin del entrenamiento")

    def fit_to_query(self, train_set, query_set, relative_weight:int=1, **train_kwargs):
        """
        Entrenar con el conjunto original aumentado con las nuevas muestras.
        Permite asignar peso adicional a las nuevas muestras.
        """
        augmented_train_set = ConcatDataset([train_set, *relative_weight*[query_set]])
        self.fit(augmented_train_set, **train_kwargs)

    def online_fit(self, train_set, function, n_queries, relative_weight:int=1, **train_kwargs):
        """
        Ciclo para solicitar muestra y reajustar
        """
        queries = torch.zeros(n_queries, 1)

        for i in range(n_queries):

            _, query = ensemble.query(X_query, n_queries=1)
            queries[i] = query
            print(f'Queried: {query.item()}')

            # Aquí se mandaría la q al robot y luego leer posición
            result = function(queries)
            # Agarrar sólo las entradas que han sido asignadas
            query_set = TensorDataset(queries[:i+1], result[:i+1])

            ensemble.fit_to_query(train_set, query_set, relative_weight,
                                  **train_kwargs)

        return queries

    def rank_models(self, test_set):
        """
        Registrar mejor modelo según precisión en un conjunto de prueba
        TODO: Guardar rendimiento de todos los modelos, ponderarlo en la
              selección de nuevas muestras
        """
        best_score = torch.inf
        for i, model in enumerate(self.ensemble):
            score = test(model, test_set)
            if score < best_score:
                self.best_model_idx = i


if __name__ == "__main__":
    
    from torch.utils.data import TensorDataset, random_split
    import matplotlib.pyplot as plt

    from modelos import MLP, ResNet
    from entrenamiento import test

    """
    Conjunto de datos
    """
    x_min = -1
    x_max = 1
    n_samples = 16
    n_models = 3
    n_queries = 10

    # Función de prueba
    def f(x): return torch.sin(10*x**2 - 10*x)
    # def f(x): return torch.cos(10*x**2)

    X = torch.linspace(x_min, x_max, n_samples).reshape(-1, 1)
    Y = f(X)
    full_set = TensorDataset(X, Y)

    split_proportions = [0.5, 0.5]
    split = [round(prop*len(X)) for prop in split_proportions]

    train_set, test_set = random_split(full_set, split)

    # Datos disponibles para 'pedir'
    X_query = torch.rand(30).view(-1, 1)*(x_max-x_min) + x_min

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
    ensemble.fit(train_set, lr=1e-3, epochs=2000)
    ensemble.rank_models(test_set)

    # Para graficar después
    X_plot = torch.linspace(x_min, x_max, 1000).view(-1,1)
    first_pred = ensemble.best_predict(X_plot).detach()

    """
    Afinación con muestras nuevas recomendadas
    """
    queries = ensemble.online_fit(train_set, f, n_queries, relative_weight=1,
                                  lr=1e-3, epochs=300)
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
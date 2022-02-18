import torch

from entrenamiento import train


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

    def rank_models(self, test_set):
        """
        Registrar mejor modelo según precisión en un conjunto de prueba
        """
        best_score = torch.inf
        for i, model in enumerate(self.ensemble):
            score = test(model, test_set)
            if score < best_score:
                self.best_model_idx = i


if __name__ == "__main__":
    
    from torch.utils.data import TensorDataset
    import matplotlib.pyplot as plt

    from modelos import MLP, ResNet
    from entrenamiento import test

    x_min = -1
    x_max = 1
    n_samples = 50
    n_models = 3
    n_queries = 10

    # Función de prueba
    def f(x): return torch.sin(10*x**2 - 10*x)
    # def f(x): return torch.cos(10*x**2)
    
    # Datos de entrenamiento
    X_train = torch.linspace(x_min, x_max, n_samples).reshape(-1, 1)
    y_train = f(X_train)

    # Datos disponibles para 'pedir'
    X_query = torch.rand(100).view(-1, 1)*(x_max-x_min) + x_min

    # Declarar conjunto de modelos
    models = [MLP(input_dim=1,
                  output_dim=1,
                  depth=3,
                  mid_layer_size=10,
                  activation=torch.tanh) for _ in range(n_models)]

    ensemble = EnsembleRegressor(models)

    # Entrenar el model con datos iniciales
    train_set = TensorDataset(X_train, y_train)
    ensemble.fit(train_set, lr=2e-3, epochs=2000)

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
    labels = ['Target F', 'Trainset','Queries']
    for i in range(n_models):
        ax.plot(X_plot, ensemble[i](X_plot).detach())
        labels.append(f'Model {i}')

    # ax.plot(X_plot, ensemble.joint_predict(X_plot))
    # labels.append('Joint predict')

    ax.legend(labels)
    plt.show()
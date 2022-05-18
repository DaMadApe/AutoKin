import torch
from torch.utils.data import TensorDataset, ConcatDataset


class EnsembleRegressor(torch.nn.Module):
    """
    Agrupa un conjunto de modelos, permite entrenarlos en conjunto,
    y luego predecir qué nuevas muestras serían más efectivas.
    """
    def __init__(self, models : list[torch.nn.Module]):
        super().__init__()
        self.input_dim = models[0].input_dim
        for model in models:
            if model.input_dim != self.input_dim:
                raise ValueError('Modelos de dimensiones diferentes')
        self.ensemble = torch.nn.ModuleList(models)
        self.best_model_idx = None
        self.model_scores = None
    
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
        (requiere ejecutar antes self.test con un set de prueba)
        """
        if self.best_model_idx is None:
            raise RuntimeError('No se hizo ensemble.test()')
        else:
            with torch.no_grad():
                self.ensemble[self.best_model_idx].eval()
                pred = self.ensemble[self.best_model_idx](x)
            return pred

    def query(self, candidate_batch=None, n_queries=1):
        """
        De un conjunto de posibles puntos, devuelve
        el punto que maximiza la desviación estándar
        de las predicciones del grupo de modelos.
        """
        if candidate_batch is None:
            candidate_batch = torch.rand((1000, self.input_dim))

        with torch.no_grad():
            self.ensemble.eval()
            preds = self(candidate_batch)

        # La desviación estándar de cada muestra es la suma de la
        # varianza entre modelos de cada coordenada.
        # https://math.stackexchange.com/questions/850228/finding-how-spreaded-a-point-cloud-in-3d
        deviation = torch.sum(torch.var(preds, axis=0), axis=-1)
        candidate_idx = torch.topk(deviation, n_queries).indices
        query = candidate_batch[candidate_idx]
        return query
        # return torch.topk(deviation, n_queries)

    def fit(self, train_set, **train_kwargs):
        """
        Entrenar cada uno de los modelos individualmente
        """
        print("Ajustando modelos del conjunto")

        for model in self.ensemble:
            model.fit(train_set, **train_kwargs)

        print("Fin del entrenamiento")

    def test(self, test_set, **test_kwargs):
        self.model_scores = []
        for model in self.ensemble:
            score = model.test(test_set, **test_kwargs)
            self.model_scores.append(score)

        self.best_model_idx = self.model_scores.index(min(self.model_scores))

        return self.model_scores.copy()

    def online_fit(self, train_set, label_fun, query_steps,
                   candidate_batch=None, n_queries=1,
                   relative_weight:int=1, final_adjust_weight=None,
                   **train_kwargs):
        """
        Ciclo para solicitar muestras y ajustar a ellas

        args:
        train_set (Dataset) : Conjunto base de entrenamiento
        label_fun (Callable: Tensor(N,d)->Tensor(N,d)) : Método para obtener las 
            etiquetas de nuevas muestras
        query_steps (int) : Número de veces que se solicitan nuevas muestras
        n_queries (int) : Número de muestras solicitadas en cada paso
        relative_weight (int) : Ponderación extra de las muestras nuevas (repetir en dataset)
        final_adjust_weight (int) : Asignar para finalizar con un entrenamiento ponderado
        tb_dir (str) : Directorio base para guardar logs de tensorboard de entrenamientos
        **train_kwargs: Argumentos de entrenamiento usados para cada ajuste
        """
        for i in range(query_steps):

            query = self.query(candidate_batch=candidate_batch,
                               n_queries=n_queries)

            labels = label_fun(query)
            new_queries = TensorDataset(query, labels)
            train_set = ConcatDataset([train_set, new_queries])

            self.fit(train_set, **train_kwargs)
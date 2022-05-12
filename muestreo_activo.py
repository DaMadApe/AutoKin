import torch
from torch.utils.data import TensorDataset, ConcatDataset


class EnsembleRegressor(torch.nn.Module):
    """
    Agrupa un conjunto de modelos, permite entrenarlos en conjunto,
    y luego predecir qué nuevas muestras serían más efectivas.
    """
    def __init__(self, models : list[torch.nn.Module]):
        super().__init__()
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

    def fit_to_query(self, train_set, query_set, relative_weight:int=1,
                     **train_kwargs):
        """
        Entrenar con el conjunto original aumentado con las nuevas muestras.
        Permite asignar peso adicional a las nuevas muestras.

        args:
        train_set (Dataset) : Conjunto base de entrenamiento
        query_set (Dataset) : Conjunto de muestras nuevas para ajustar
        relative_weight: Ponderación extra de las muestras nuevas (repetir en dataset)
        **train_kwargs: Argumentos nombrados de la función de entrenamiento
        """
        augmented_train_set = ConcatDataset([train_set, *(relative_weight*[query_set])])
        self.fit(augmented_train_set, **train_kwargs)

    def online_fit(self, train_set, candidate_batch, label_fun, query_steps, n_queries=1,
                   relative_weight:int=1, final_adjust_weight=None, tb_dir=None, use_checkpoint=True, **train_kwargs):
        """
        Ciclo para solicitar muestra y ajustar, una por una.

        args:
        train_set (Dataset) : Conjunto base de entrenamiento
        label_fun (Callable: Tensor(N,d)->Tensor(N,d)) : Método para obtener las 
            etiquetas de nuevas muestras
        query_steps (int) : Número de veces que se solicitan nuevas muestras
        n_queries (int) : Número de muestras solicitadas en cada paso
        relative_weight (int) : Ponderación extra de las muestras nuevas (repetir en dataset)
        final_adjust_weight (int) : Asignar para finalizar con un entrenamiento ponderado
        tb_dir (str) : Directorio base para guardar logs de tensorboard de entrenamientos
        use_checkpoint (bool) : Si se activa, se guarda el estado del optimizador entre ajustes
        **train_kwargs: Argumentos de entrenamiento usados para cada ajuste

        TODO: Sacar candidate_batch nuevo para cada query, no requerir como argumento
        """
        input_dim = train_set[0][0].size()
        queries = torch.zeros(query_steps, n_queries, *input_dim)

        for i in range(query_steps):

            log_dir = tb_dir+f'_s{i}' if tb_dir is not None else None

            _, query = self.query(candidate_batch, n_queries=n_queries)

            queries[i,] = query
            # print(f'Queried: {query}')

            # Agarrar sólo las entradas que han sido asignadas
            flat_current_queries = queries[:i+1].flatten(end_dim=-2)
            # Aquí se mandaría la q al robot y luego leer posición
            result = label_fun(flat_current_queries)
            query_set = TensorDataset(flat_current_queries,
                                      result)

            self.fit_to_query(train_set, query_set, relative_weight,
                              **train_kwargs, log_dir=log_dir,
                              use_checkpoint=use_checkpoint)

        if final_adjust_weight is not None:
            log_dir = tb_dir+'_final' if tb_dir is not None else None
            self.fit_to_query(train_set, query_set,
                              relative_weight=final_adjust_weight,
                              **train_kwargs, log_dir=log_dir,
                              use_checkpoint=use_checkpoint)

        return queries, result
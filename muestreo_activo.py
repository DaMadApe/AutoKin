import torch
from torch.utils.data import TensorDataset, ConcatDataset

from entrenamiento import train, test

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
        self.last_checkpoints = [{}]*len(self.ensemble)
    
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

    def fit(self, train_set, use_checkpoint=False, **train_kwargs):
        """
        Entrenar cada uno de los modelos individualmente
        """
        print("Ajustando modelos del conjunto")

        for i, model in enumerate(self.ensemble):
            if use_checkpoint and all(self.last_checkpoints):
                train_kwargs.update({'checkpoint': self.last_checkpoints[i]})

            checkpoint = train(model, train_set, **train_kwargs)

            if use_checkpoint:
                self.last_checkpoints[i].update(checkpoint)
        print("Fin del entrenamiento")

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
        augmented_train_set = ConcatDataset([train_set, *relative_weight*[query_set]])
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
        **train_kwargs: Argumentos de entrenamiento usados para cada ajuste

        TODO: Sacar candidate_batch nuevo para cada query, no requerir como argumento
        """
        input_dim = train_set[0][0].size()
        queries = torch.zeros(query_steps, n_queries, *input_dim)

        for i in range(query_steps):

            log_dir = tb_dir+f'_s{i}' if tb_dir is not None else None

            _, query = self.query(candidate_batch, n_queries=n_queries)

            queries[i,] = query
            print(f'Queried: {query}')

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
                              **train_kwargs, log_dir=log_dir)

        return queries, result

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
    
    import torch
    from torch.utils.data import random_split
    import roboticstoolbox as rtb

    from modelos import MLP
    from muestreo_activo import EnsembleRegressor
    from utils import RoboKinSet

    """
    Conjuntos de datos
    """
    robot = rtb.models.DH.Cobra600() #Puma560()
    n_samples = 10000

    full_set = RoboKinSet(robot, n_samples)

    # Repartir muestras entre conjuntos
    split_proportions = [0.6, 0.2, 0.2]
    # Convertir proporciones al número de muestras correspondiente
    split = [round(prop*len(full_set)) for prop in split_proportions]

    train_set, val_set, test_set = random_split(full_set, split)

    """
    Definición de modelos
    """
    n_models = 3

    models = [MLP(input_dim=robot.n,
                  output_dim=3,
                  depth=3,
                  mid_layer_size=12,
                  activation=torch.tanh) for _ in range(n_models)]

    ensemble = EnsembleRegressor(models)

    """
    Entrenamiento
    """
    # Primer entrenamiento
    ensemble.fit(train_set, val_set=val_set,
                 lr=1e-3, epochs=36)

    # Ajuste a nuevas muestras
    def label_fun(X):
        result = robot.fkine(X.numpy()).t
        return torch.tensor(result, dtype=torch.float)

    candidate_batch = torch.rand((500, robot.n))

    queries, _ = ensemble.online_fit(train_set,
                                     val_set=val_set,
                                     candidate_batch=candidate_batch,
                                     label_fun=label_fun,
                                     query_steps=6,
                                     n_queries=10,
                                     relative_weight=5,
                                     final_adjust_weight=5,
                                     lr=1e-4, epochs=12,
                                     lr_scheduler=True,
                                     tb_dir='tb_logs/muestreo_activo/cobra600'
                                     )
    ensemble.rank_models(test_set)

    torch.save(ensemble[ensemble.best_model_idx], 'models/cobra600_MA_v1.pt')
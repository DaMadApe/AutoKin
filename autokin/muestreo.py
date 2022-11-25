import logging

import torch
from torch.utils.data import (Dataset, TensorDataset,
                              ConcatDataset, random_split)

from autokin.robot import Robot


class FKset(Dataset):
    """
    Producir un conjunto de puntos (configuración,posición) de un robot
    definido con la interfaz de un robot DH de Peter Corke.
    
    Los puntos se escogen aleatoriamente en el espacio de parámetros.

    robot () : Cadena cinemática para producir ejemplos
    q_vecs (torch.Tensor) : Lista de vectores de actuación para generar ejemplos
    full_pose (bool) : Usar posición + orientación o sólo posición
    q_uniform_noise (float) : Cantidad de ruido uniforme aplicado a ejemplos q
    q_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a ejemplos q
    p_uniform_noise (float) : Cantidad de ruido uniforme aplicado a etiquetas pos
    p_normal_noise (float) : Cantidad de ruido normal(m=0,s=1) aplicado a etiquetas pos
    """
    def __init__(self, robot: Robot, q_vecs: torch.Tensor,
                 q_uniform_noise=0, q_normal_noise=0,
                 p_uniform_noise=0, p_normal_noise=0):

        is_q_normed = torch.all(q_vecs>=0) and torch.all(q_vecs<=1)
        if not(is_q_normed):
            raise ValueError('q_vecs debe ir normalizado a intervalo [0,1]')

        self.robot = robot
        self.q_in_vecs = q_vecs

        self.n = self.robot.n # Número de ejes
        self.out_n = self.robot.out_n

        self.q_uniform_noise = q_uniform_noise
        self.q_normal_noise = q_normal_noise
        self.p_uniform_noise = p_uniform_noise
        self.p_normal_noise = p_normal_noise

        self._generate_labels()

    @classmethod
    def random_sampling(cls, robot, n_samples: int, **kwargs):
        """
        Constructor alternativo
        Toma muestras uniformemente distribuidas en el espacio de juntas

        args:
        robot () : Cadena cinemática para producir ejemplos
        n_samples (int) : Número de ejemplos
        """
        q_vecs = torch.rand(n_samples, robot.n)
        return cls(robot, q_vecs, **kwargs)

    @classmethod
    def grid_sampling(cls, robot, n_samples: list, **kwargs):
        """
        Constructor alternativo
        Toma muestras del espacio de juntas en un patrón de cuadrícula

        args:
        robot () : Cadena cinemática para producir ejemplos
        n_samples (int list) : Número de divisiones por junta
        """
        # Magia negra para producir todas las combinaciones de puntos
        q_vecs = torch.meshgrid(*[torch.linspace(0,1, int(n)) for n in n_samples],
                                indexing='ij')
        q_vecs = torch.stack(q_vecs, -1).reshape(-1, robot.n)
        return cls(robot, q_vecs, **kwargs)

    def _generate_labels(self):
        # Hacer cinemática directa
        self.q_vecs, self.p_vecs = self.robot.fkine(self.q_in_vecs)

        q_noise = (self.q_uniform_noise*torch.rand(len(self), self.n) +
                   self.q_normal_noise*torch.randn(len(self), self.n))

        p_noise = (self.p_uniform_noise*torch.rand(len(self), self.out_n) +
                   self.p_normal_noise*torch.randn(len(self), self.out_n))

        self.q_vecs = self.q_vecs + q_noise
        self.p_vecs = self.p_vecs + p_noise

    def __len__(self):
        return self.q_vecs.shape[0]

    def __getitem__(self, idx):
        return self.q_vecs[idx], self.p_vecs[idx]

    def rand_split(self, proportions: list[float]):
        """
        Reparte el conjunto de datos en segmentos aleatoriamente
        seleccionados, acorde a las proporciones ingresadas.

        args:
        dataset (torch Dataset): Conjunto de datos a repartir
        proportions (list[float]): Porcentaje que corresponde a cada partición
        """
        if round(sum(proportions), ndigits=2) != 1:
            raise ValueError('Proporciones ingresadas deben sumar a 1 +-0.01')
        split = [round(prop*len(self)) for prop in proportions]

        # HACK: Compensa por algunos valores que no suman la longitud original
        split[0] += (len(self) - sum(split))

        return random_split(self, split)


class EnsembleRegressor(torch.nn.Module):
    # TODO: Borrar, reemplazado en autokin.modelos por FKEnsemble
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
        logging.info("Ajustando modelos del conjunto")

        for model in self.ensemble:
            model.fit(train_set, **train_kwargs)

        logging.info("Fin del entrenamiento")

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
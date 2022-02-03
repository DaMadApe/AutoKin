"""
Réplica de experim 12 implementado desde cero (sin modAL)
"""

import torch

class EnsembleRegressor(torch.nn.Module):
    """
    Produce n copias de un mismo tipo de modelo.
    """
    def __init__(self, models : list[torch.nn.Module]):
        super().__init__()
        self.ensemble = torch.nn.ModuleList(models)
    
    def forward (self, x):
        return torch.stack([model(x) for model in self.ensemble])

    def query(self, candidate_batch):
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
        candidate_idx = torch.argmax(deviation)
        query = candidate_batch[candidate_idx]

        return candidate_idx, query


class MockModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rand(x.shape[0], 3)


if __name__ == "__main__":
    
    from experim0 import MLP

    ensemble = EnsembleRegressor([MockModule() for _ in range(5)])

    qs = torch.rand(4, 6)

    print(ensemble(qs))
    print(ensemble.query(qs))
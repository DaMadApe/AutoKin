import torch
import roboticstoolbox as rtb

from modelos import MLP, EnsembleRegressor
from utils import RoboKinSet, rand_data_split

"""
Conjuntos de datos
"""
robot = rtb.models.DH.Cobra600() #Puma560()
n_samples = 10000

full_set = RoboKinSet.random_sampling(robot, n_samples)
train_set, val_set, test_set = rand_data_split(full_set, [0.6, 0.2, 0.2])

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
import torch

from modelos import MLP
from robot import RTBrobot
from muestreo_activo import EnsembleRegressor
from utils import FKset, coprime_sines
from experimentos.experim import ejecutar_experimento

"""
Conjuntos de datos
"""
robot = RTBrobot.from_name('Cobra600') #Puma560()
n_samples = 2000

c_sines = coprime_sines(robot.n, n_samples, wiggle=3)
dataset = FKset(robot, c_sines)

n_models = 3
n_reps = 3

# Ajuste a nuevas muestras
def label_fun(X):
    _, result = robot.fkine(X)
    return result

def experim_muestreo_activo():
    train_set, val_set, test_set = dataset.rand_split([0.6, 0.2, 0.2])

    models = [MLP(input_dim=robot.n,
                output_dim=3,
                depth=3,
                mid_layer_size=12,
                activation=torch.tanh) for _ in range(n_models)]

    ensemble = EnsembleRegressor(models)

    # Primer entrenamiento
    ensemble.fit(train_set, val_set=val_set,
                    lr=1e-3, epochs=36)

    candidate_batch = torch.rand((500, robot.n))

    queries, _ = ensemble.online_fit(train_set,
                                     val_set=val_set,
                                     candidate_batch=candidate_batch,
                                     label_fun=label_fun,
                                     query_steps=6,
                                     n_queries=10,
                                     relative_weight=5,
                                     final_adjust_weight=5,
                                     lr=1e-3, epochs=12,
                                     batch_size=256,
                                     lr_scheduler=True,
                                     tb_dir='tb_logs/muestreo_activo/cobra600'
                                    )
    score = max(ensemble.test(test_set))
    model = ensemble[ensemble.best_model_idx]

    return score, model

ejecutar_experimento(n_reps, experim_muestreo_activo,
                     log_all_products=False,
                     model_save_dir='models/cobra600_MA_v2.pt')
import torch
import roboticstoolbox as rtb
from experimentos.experim import repetir_experimento

from modelos import MLP
from muestreo_activo import EnsembleRegressor
from utils import RoboKinSet, rand_data_split, coprime_sines

"""
Conjuntos de datos
"""
robot = rtb.models.DH.Cobra600() #Puma560()
n_samples = 2058

c_sines = coprime_sines(robot.n, n_samples, wiggle=3)
dataset = RoboKinSet(robot, c_sines)

n_models = 3
n_reps = 3

# Ajuste a nuevas muestras
def label_fun(X):
    result = robot.fkine(X.numpy()).t
    return torch.tensor(result, dtype=torch.float)

def experim_muestreo_activo():
    train_set, val_set, test_set = rand_data_split(dataset, [0.6, 0.2, 0.2])

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

score, best_model = repetir_experimento(n_reps, experim_muestreo_activo)

print(f'Mejor puntaje: {score}')
torch.save(best_model, 'models/cobra600_MA_v2.pt')
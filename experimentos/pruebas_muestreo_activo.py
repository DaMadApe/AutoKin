import torch

from modelos import MLP
from robot import RTBrobot
from muestreo_activo import EnsembleRegressor
from utils import FKset, coprime_sines
from experimentos.experim import ejecutar_experimento

"""
Conjuntos de datos
"""
robot_name = 'Cobra600' #'Puma560'
exp_name = 'MA1'
robot = RTBrobot.from_name(robot_name)
n_samples = 900

# c_sines = coprime_sines(robot.n, n_samples, wiggle=3)
# dataset = FKset(robot, c_sines)

dataset = FKset.random_sampling(robot, n_samples)
train_set, val_set, test_set = dataset.rand_split([0.7, 0.2, 0.1])

n_models = 5
n_reps = 5

# Ajuste a nuevas muestras
def label_fun(X):
    _, result = robot.fkine(X)
    return result

def experim_muestreo_activo():
    # train_set, val_set, test_set = dataset.rand_split([0.6, 0.2, 0.2])

    label = f'{torch.rand(1).item():.4f}'

    models = [MLP(input_dim=robot.n,
                  output_dim=3,
                  depth=3,
                  mid_layer_size=10,
                  activation=torch.tanh) for _ in range(n_models)]

    ensemble = EnsembleRegressor(models)

    # Primer entrenamiento
    print(f'Prefit: {len(train_set)}')
    ensemble.fit(train_set, val_set=val_set,
                 lr=1e-3, epochs=300, batch_size=256,
                 log_dir=f'experimentos/tb_logs/MA/{robot_name}_{exp_name}/{label}')
    print(f'Postfit: {len(train_set)}')
    ensemble.online_fit(train_set,
                        val_set=val_set,
                        label_fun=label_fun,
                        query_steps=6,
                        n_queries=25,
                        #relative_weight=5,
                        #final_adjust_weight=5,
                        lr=1e-3, epochs=80,
                        batch_size=256,
                        # lr_scheduler=True,
                        log_dir=f'experimentos/tb_logs/MA/{robot_name}_{exp_name}/{label}'
                       )
    print(f'Post online: {len(train_set)}')
    score = max(ensemble.test(test_set))
    model = ensemble[ensemble.best_model_idx]

    return score, model

ejecutar_experimento(n_reps, experim_muestreo_activo,
                     log_all_products=False,
                     #model_save_dir='models/{robot_name}_MA.pt'
                    )
import numpy as np

from autokin.modelos import MLP
from autokin.muestreo import EnsembleRegressor, FKset
from autokin.robot import RTBrobot
from autokin.utils import random_robot

def test_ajuste_modelos():
    robot = RTBrobot(random_robot())
    train_set = FKset.random_sampling(robot, n_samples=100)

    model = MLP(input_dim=robot.n, output_dim=3)

    before_score = model.test(train_set)
    model.fit(train_set, epochs=10)
    after_score = model.test(train_set)

    assert after_score < before_score

def test_ajuste_activo():
    robot = RTBrobot(random_robot())
    train_set = FKset.random_sampling(robot, n_samples=100)
    n_models = 5

    models = [MLP(input_dim=robot.n,
                  output_dim=3) for _ in range(n_models)]

    ensemble = EnsembleRegressor(models)

    before_scores = np.array(ensemble.test(train_set))
    ensemble.fit(train_set, epochs=10, use_checkpoint=True)
    mid_scores = np.array(ensemble.test(train_set))
    ensemble.fit(train_set, epochs=10, use_checkpoint=True)
    after_scores = np.array(ensemble.test(train_set))

    assert np.all(mid_scores < before_scores)
    assert np.all(after_scores < mid_scores)
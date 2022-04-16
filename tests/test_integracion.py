import numpy as np

from modelos import MLP
from muestreo_activo import EnsembleRegressor
from utils import random_robot, RoboKinSet, rand_data_split

def test_ajuste_modelos():
    robot = random_robot()
    train_set = RoboKinSet.random_sampling(robot, n_samples=100)

    model = MLP(input_dim=robot.n, output_dim=3)

    before_score = model.test(train_set)
    model.fit(train_set, epochs=10)
    after_score = model.test(train_set)

    assert after_score < before_score

def test_ajuste_activo():
    robot = random_robot()
    train_set = RoboKinSet.random_sampling(robot, n_samples=100)
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
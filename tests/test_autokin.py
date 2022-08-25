from functools import partial

import torch
import torch.testing as tt
import numpy as np

from hypothesis import given
import hypothesis.strategies as some

from autokin.modelos import MLP
from autokin.muestreo import EnsembleRegressor, FKset
from autokin.robot import RTBrobot
from autokin.utils import random_robot, restringir


assert_equal = partial(tt.assert_close, atol=1e-7, rtol=1e-7)
assert_close = partial(tt.assert_close, atol=1e-3, rtol=1e-3)


def test_random_robot():
    robot = random_robot()


def test_restringir():
    for i in [2, 3, 4]:
        q = torch.rand(100, i)
        q_trans = restringir(q)
        max_norm = max(q_trans.norm(dim=-1))
        assert max_norm <= 1


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


""" @given(some.builds(random_robot))
def test_denorm_q_within_range(robot):
    qlim_tensor = torch.tensor(robot.qlim, dtype=torch.float32)
    unit_q_vec = torch.rand((robot.n))

    denormed_q = denorm_q(robot, unit_q_vec)

    assert all(denormed_q >= qlim_tensor[0])
    assert all(denormed_q <= qlim_tensor[1])


@given(some.builds(random_robot))
def test_denorm_q_full_range(robot):
    qlim_tensor = torch.tensor(robot.qlim, dtype=torch.float32)
    zero_vec = torch.zeros((robot.n))
    ones_vec = torch.ones((robot.n))

    denormed_min = denorm_q(robot, zero_vec)
    denormed_max = denorm_q(robot, ones_vec)

    assert_equal(denormed_min, qlim_tensor[0])
    assert_equal(denormed_max, qlim_tensor[1])


@given(some.builds(random_robot))
def test_norm_q_within_range(robot):
    unit_q_vec = torch.rand((robot.n))
    denormed_q = denorm_q(robot, unit_q_vec)

    normed_q = norm_q(robot, denormed_q)

    assert all(normed_q >= 0)
    assert all(normed_q <= 1)


@given(some.builds(random_robot))
def test_norm_q_full_range(robot):
    qlim_tensor = torch.tensor(robot.qlim, dtype=torch.float32)
    zero_vec = torch.zeros((robot.n))
    ones_vec = torch.ones((robot.n))

    normed_min = norm_q(robot, qlim_tensor[0])
    normed_max = norm_q(robot, qlim_tensor[1])

    assert_equal(normed_min, zero_vec)
    assert_equal(normed_max, ones_vec) """
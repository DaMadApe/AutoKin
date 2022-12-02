from functools import partial

import torch
import torch.testing as tt

from hypothesis import given
import hypothesis.strategies as some

from autokin.modelos import MLP, MLPEnsemble
from autokin.muestreo import FKset
from autokin.robot import RTBrobot
from autokin.utils import *


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


def test_suavizar():
    q = torch.rand(250, 4)
    q_prev = torch.zeros(4)
    dq_max = 0.1

    q_s = suavizar(q=q, q_prev=q_prev, dq_max=dq_max)
    q_s_diff = q_s.diff(dim=0)

    # El resultado nunca excede la diferencia solicitada
    assert torch.all(q_s_diff.abs() <= dq_max)
    # El resutado contiene todos los puntos de la entrada
    assert all([(q_i in q_s) for q_i in q])


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

    ensemble = MLPEnsemble(n_modelos=5,
                           input_dim=robot.n,
                           output_dim=3)

    before_scores = ensemble.test(train_set)
    ensemble.fit(train_set, epochs=10, use_checkpoint=True)
    mid_scores = ensemble.test(train_set)
    ensemble.fit(train_set, epochs=10, use_checkpoint=True)
    after_scores = ensemble.test(train_set)

    assert torch.all(mid_scores < before_scores)
    assert torch.all(after_scores < mid_scores)


def test_dset_denorm_p():
    scale = 0.3

    robot = RTBrobot(random_robot())
    robot.p_scale = scale * torch.ones(robot.out_n)
    dataset = FKset.random_sampling(robot, n_samples=10)

    _, pn = dataset[0]

    dataset.apply_p_norm = False
    _, pdn = dataset[0]

    assert_equal(pn, scale*pdn)


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
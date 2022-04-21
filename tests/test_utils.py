from functools import partial

import torch
import torch.testing as tt

import pytest
from hypothesis import given
import hypothesis.strategies as some

from utils import denorm_q, norm_q, random_robot

assert_equal = partial(tt.assert_close, atol=1e-7, rtol=1e-7)
assert_close = partial(tt.assert_close, atol=1e-3, rtol=1e-3)

def test_random_robot():
    robot = random_robot()

@given(some.builds(random_robot))
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
    assert_equal(normed_max, ones_vec)
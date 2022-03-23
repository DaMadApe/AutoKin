import torch
import torch.testing as tt

import pytest
from hypothesis import given
import hypothesis.strategies as some
import hypothesis.extra.array_api as some_arr

from utils import denorm_q, norm_q, random_robot


@given(some.builds(random_robot))
def test_norm_denorm_q_ranges(robot):
    unit_q_vec = torch.rand((robot.n))

    q_lim_tensor = torch.tensor(robot.qlim)

    denormed_q = denorm_q(robot, unit_q_vec)
    assert all(denormed_q >= q_lim_tensor[0])
    assert all(denormed_q <= q_lim_tensor[1])

    normed_q = norm_q(robot, denormed_q)
    assert all(normed_q >= 0)
    assert all(normed_q <= 1)
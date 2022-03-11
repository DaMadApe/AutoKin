from modelos import MLP
from utils import random_robot, RoboKinSet, rand_data_split

def test_ajuste():
    robot = random_robot()
    train_set = RoboKinSet.random_sampling(robot, n_samples=100)

    model = MLP(input_dim=robot.n, output_dim=3)

    before_score = model.test(train_set)
    model.fit(train_set, epochs=10)
    after_score = model.test(train_set)

    assert after_score < before_score
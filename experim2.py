from torchvision.datasets import MNIST

trainset = MNIST('datasets', train=True, download=True)

"""
Aprendizaje activo: Selección dinámica de puntos para
mejorar la convergencia de la regresión de experim0
"""

import modAL

from experim0 import MLP
from experim14 import train


if __name__ == "__main__":
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from modAL.models import ActiveLearner

    x_min = -1
    x_max = 1

    #def f(x): return 3*x**3 - 3*x**2 + 2*x
    def f(x): return torch.sin(10*x**2 - 10*x)
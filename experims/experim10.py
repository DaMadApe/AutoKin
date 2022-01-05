"""
Probar L-BFGS para entrenamiento, comparar con
Adam, SGD, etc.
"""
import torch
import pytorch_lightning as pl

from experimR import RoboKinSet
from experim3 import RegressorPL

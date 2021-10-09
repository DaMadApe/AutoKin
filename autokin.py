import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# Por definir: Algo similar a la librería de Peter Corke
# para producir funciones objetivo.

# Cosas necesarias
# Datos:
#   Función de posición
#   Función de exploración de espacio de configuraciones
#   Integrar conjunto de datos
#   Preprocesamiento de datos
# Modelo:
#   Modelo genérico manual
#   Rutinas para probar hiperparámetros
#   Visualización de resultados, rendimiento
# Cinemática inversa:
#   Método iterativo
#   Trazo de trayectoria

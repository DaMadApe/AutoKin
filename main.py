# Entrenar modelo o Usar modelo

# Usar:
# configurar si recopilar nuevas muestras para entrenar
# 

# Entrenar:
# definir conjunto de hipótesis
# escoger buenas hipótesis de partida
# 
# establecer muestra inicial
# entrenamiento inicial, generar nuevas muestras
# reentrenar
# evaluar
import torch

from modelos import MLP
from utils import coprime_sines

def muestreo_inicial():
    pass

def entrenamiento():
    definir_modelo()
    iniciar_modelo()
    ajuste_inicial()
    ajuste_dirigido()

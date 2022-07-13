import torch
from torch import nn

from autokin.model_mixins import HparamsMixin, DataFitMixin


class MLP(HparamsMixin, # Para almacenar hiperparámetros del modelo
          DataFitMixin, # Para ajustar y probar modelo con conjuntos de datos
          nn.Module # Módulo base de modelo de pytorch
          ):
    """
    Red neuronal simple (Perceptrón Multicapa)

    args:
    input_dim (int) : Tamaño del tensor de entrada
    ouput_dim (int) : Tamaño de la salida
    depth (int) : Número de capas
    mid_layer_size (int) : Número de neuronas en cada capa
    activation (string) : Función de activación aplicada después de cada capa
    """
    def __init__(self, input_dim=1, output_dim=1,
                 depth=1, mid_layer_size=10, activation='tanh'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # Armar modelo
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim,
                                     mid_layer_size))
        for _ in range(depth):
            self.layers.append(nn.Linear(mid_layer_size,
                                         mid_layer_size))
        self.layers.append(nn.Linear(mid_layer_size,
                                     output_dim))

        self.activation = getattr(torch, activation)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class ResBlock(nn.Module):
    """
    Bloque de una ResNet. Equivalente a una MLP con una conexión
    adicional de la entrada a la salida (F'(x) = F(x) + x)
    """
    def __init__(self, depth, block_width, activation):
        super().__init__()
        
        self.activation = activation
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(nn.Linear(block_width,
                                         block_width))

    def forward(self, x):
        identity = x.clone() # TODO: Confirmar si clone es necesario
        for layer in self.layers:
            x = self.activation(layer(x))
        return x + identity


class ResNet(HparamsMixin, DataFitMixin, nn.Module):
    """
    Red residual: Red con conexiones salteadas cada cierto número de capas

    args:
    input_dim (int) : Tamaño del tensor de entrada
    ouput_dim (int) : Tamaño de la salida
    depth (int) : Número de bloques residuales
    block_depth (int) : Número de capas en cada bloque
    block_width (int) : Número de neuronas por capa
    activation (string) : Función de activación aplicada después de cada capa
    """
    def __init__(self, input_dim=1, output_dim=1,
                 depth=3, block_depth=3, block_width=10,
                 activation='tanh'):
        super().__init__()

        self.activation = getattr(torch, activation)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(input_dim,
                                     block_width))
        for _ in range(depth):
            self.blocks.append(ResBlock(block_depth, block_width,
                                        self.activation))
        
        self.blocks.append(nn.Linear(block_width,
                                     output_dim))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class RBFnet(HparamsMixin, DataFitMixin, nn.Module):
    pass


class ELM(HparamsMixin, DataFitMixin, nn.Module):
    pass

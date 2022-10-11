import torch
from torch import nn
from torch.autograd.functional import jacobian

from autokin.model_mixins import HparamsMixin, DataFitMixin, EnsembleMixin

class FKModel(HparamsMixin, # Para almacenar hiperparámetros del modelo
              DataFitMixin, # Para ajustar y probar modelo con datos
              nn.Module # Módulo base de modelo de pytorch
              ):
    """
    Base para definir modelos que aproximan cinemática directa
    """
    def __init__(self):
        super().__init__()
        self.input_dim : int
        self.output_dim : int


class MLP(FKModel):
    """
    Red neuronal simple (Perceptrón Multicapa)

    args:
    input_dim: Tamaño del tensor de entrada
    ouput_dim: Tamaño de la salida
    depth: Número de capas
    mid_layer_size: Número de neuronas en cada capa
    activation: Función de activación aplicada después de cada capa
    """
    def __init__(self,
                 input_dim : int = 1,
                 output_dim : int = 1,
                 depth : int = 1,
                 mid_layer_size : int = 10,
                 activation : str = 'tanh'):
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


class ResNet(FKModel):
    """
    Red residual: Red con conexiones salteadas cada cierto número de capas

    args:
    input_dim: Tamaño del tensor de entrada
    ouput_dim: Tamaño de la salida
    depth: Número de bloques residuales
    block_depth: Número de capas en cada bloque
    block_width: Número de neuronas por capa
    activation: Función de activación aplicada después de cada capa
    """
    def __init__(self,
                 input_dim : int = 1,
                 output_dim : int = 1,
                 depth : int = 3,
                 block_depth : int = 3,
                 block_width : int = 10,
                 activation : str = 'tanh'):
        super().__init__()

        self.activation = getattr(torch, activation)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,
                                     block_width))
        for _ in range(depth):
            self.layers.append(ResBlock(block_depth,
                                        block_width,
                                        self.activation))
        
        self.layers.append(nn.Linear(block_width,
                                     output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RBFnet(FKModel):
    pass


class ELM(FKModel):
    pass


class FKEnsemble(
                 #HparamsMixin,
                 EnsembleMixin,
                 torch.nn.Module):
    """
    Agrupa un conjunto de modelos, permite entrenarlos en conjunto,
    y hacer predicción colectiva de nuevas muestras más efectivas.
    """
    def __init__(self, models : list[FKModel]):
        super().__init__()
        self.input_dim = models[0].input_dim
        for model in models:
            if model.input_dim != self.input_dim:
                raise ValueError('Modelos de dimensiones diferentes')
        self.ensemble = torch.nn.ModuleList(models)


class MLPEnsemble(FKEnsemble):
    def __init__(self,
                 n_modelos : int,
                 input_dim : int = 1,
                 output_dim : int = 1,
                 depth : int = 1,
                 mid_layer_size : int = 10,
                 activation : str = 'tanh'):
        modelos = [MLP(input_dim=input_dim,
                       output_dim=output_dim,
                       depth=depth,
                       mid_layer_size=mid_layer_size,
                       activation=activation) for _ in range(n_modelos)]
        super().__init__(modelos)


class ResNetEnsemble(FKEnsemble):
    def __init__(self,
                 n_modelos : int,
                 input_dim : int = 1,
                 output_dim : int = 1,
                 depth : int = 3,
                 block_depth : int = 3,
                 block_width : int = 10,
                 activation : str = 'tanh'):
        modelos = [ResNet(input_dim=input_dim,
                          output_dim=output_dim,
                          depth=depth,
                          block_depth=block_depth,
                          block_width=block_width,
                          activation=activation) for _ in range(n_modelos)]
        super().__init__(modelos)
import logging

import torch
from torch import nn

from autokin.model_mixins import *


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


class FKEnsemble(DataFitMixin,
                 ActiveFitMixin,
                 torch.nn.Module):
    """
    Agrupa un conjunto de modelos, permite entrenarlos en conjunto,
    y hacer predicción colectiva de nuevas muestras más efectivas.
    """
    def __init__(self, models : list[FKModel]):
        super().__init__()
        self.input_dim = models[0].input_dim
        self.output_dim = models[0].output_dim
        for model in models:
            if model.input_dim != self.input_dim:
                raise ValueError('Modelos de dimensiones diferentes')
        self.ensemble = torch.nn.ModuleList(models)

    @classmethod
    def from_cls(self_cls, model_cls, n_modelos:int, **model_args):
        """
        Para inicialización desde la GUI
        """
        return self_cls([model_cls(**model_args) for _ in range(n_modelos)])

    def __getitem__(self, idx):
        """
        Facilitar acceso a los modelos individuales desde afuera
        """
        return self.ensemble[idx]
    
    def forward(self, x):
        """
        La propagación del conjunto resulta en 
        """
        preds = torch.stack([model(x) for model in self.ensemble])
        return torch.mean(preds, dim=0)


class SelPropEnsemble(FKEnsemble):
    """
    Conjunto de modelos que propagan las muestras selectivamente según
    signo de la derivada de la entrada. El conjunto se inicializa con
    2^n_entradas modelos.
    """
    def __init__(self, models : list[FKModel]):
        super().__init__(models)
        if len(models) != 2**self.input_dim:
            raise ValueError('Número de modelos debe ser 2^n_entradas')

        self.prev_x = torch.zeros(self.input_dim)

    def forward(self, x: torch.Tensor):
        # Redefine el forward de los otros ensembles
        if x.ndim == 1:
            x.unsqueeze_(0)
        logging.debug(f"Forward: x={x}")
        if self.training:
            x = x[:, :self.input_dim]
            dx = x[:, self.input_dim:]
            logging.debug(f"SP Train: x={x}, dx={dx}")
        else:
            ext_x = torch.cat([self.prev_x.unsqueeze(0), x])
            dx = torch.diff()
            self.prev_x = x[-1]
            logging.debug(f"SP Eval: x={x}, dx={dx}")
        # Obtener vector de signos de la derivada
        sign_dx = (dx.sign()/2 + 0.5).int()
        # Convertir vector de signos a entero
        bin_mask = 2 ** torch.arange(self.input_dim)
        model_idx = sum(sign_dx * bin_mask.unsqueeze(0)).item()
        # Propagar al modelo correspondiente
        logging.debug('Propagando a modelo', model_idx)
        return self.ensemble[model_idx].forward(x)
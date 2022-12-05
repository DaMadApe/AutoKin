"""
Diccionarios de parámetros
"""

# Parámetros para los distintos tipos de modelo
model_args = {
    'MLP': {
        'depth': {
            'label': '# de capas',
            'var_type': 'int',
            'default_val': 3,
            'restr_positiv': True,
            'non_zero': True
        },
        'mid_layer_size': {
            'label': '# de neuronas/capa',
            'var_type': 'int',
            'default_val': 10,
            'restr_positiv': True,
            'non_zero': True
        },
    },
    'ResNet': {
        'depth': {
            'label': '# de bloques',
            'var_type': 'int',
            'default_val': 3,
            'restr_positiv': True,
            'non_zero': True
        },
        'block_depth': {
            'label': '# de capas/bloque',
            'var_type': 'int',
            'default_val': 3,
            'restr_positiv': True,
            'non_zero': True
        },
        'block_width': {
            'label': '# de neuronas/capa',
            'var_type': 'int',
            'default_val': 3,
            'restr_positiv': True,
            'non_zero': True
        },
    }
}

model_args.update({
    'MLPEnsemble': {
        'n_modelos': {
            'label': '# de modelos',
            'var_type': 'int',
            'default_val': 3,
            'restr_positiv': True,
            'non_zero': True
        },
        **model_args['MLP']
    },
    'ResNetEnsemble': {
        'n_modelos': {
            'label': '# de modelos',
            'var_type': 'int',
            'default_val': 3,
            'restr_positiv': True,
            'non_zero': True
        },
        **model_args['ResNet']
    }
})

# Parámetros para el proceso de entrenamiento
args_etapas = {
    'Meta ajuste': {
        'n_steps': {
            'label': 'Iteraciones de meta ajuste',
            'var_type': 'int',
            'default_val': 4,
            'restr_positiv': True,
            'non_zero': True
        },
        'n_datasets': {
            'label': 'Datasets por iteración',
            'var_type': 'int',
            'default_val': 8,
            'restr_positiv': True,
            'non_zero': True
        },
        'n_samples': {
            'label': 'Muestras por dataset',
            'var_type': 'int',
            'default_val': 100,
            'restr_positiv': True,
            'non_zero': True
        },
        'n_post': {
            'label': 'Muestras para ajuste final',
            'var_type': 'int',
            'default_val': 10,
            'restr_positiv': True,
            'non_zero': True
        },
        'n_epochs': {
            'label': 'Épocas por dataset',
            'var_type': 'int',
            'default_val': 50,
            'restr_positiv': True,
            'non_zero': True
        },
        'n_post_epochs': {
            'label': 'Épocas de ajuste final',
            'var_type': 'int',
            'default_val': 5,
            'restr_positiv': True,
            'non_zero': True
        },
        'lr': {
            'label': 'Learning rate',
            'var_type': 'float',
            'default_val': 1e-5,
            'restr_positiv': True,
            'non_zero': True
        },
        'post_lr': {
            'label': 'Learning rate para ajuste final',
            'var_type': 'float',
            'default_val': 1e-5,
            'restr_positiv': True,
            'non_zero': True
        },
    },

    'Ajuste inicial': {
        'epochs': {
            'label': '# de épocas',
            'var_type': 'int',
            'default_val': 1000,
            'restr_positiv': True,
            'non_zero': True
        },
        'batch_size': {
            'label': 'Batch size',
            'var_type': 'int',
            'default_val': 256,
            'restr_positiv': True,
            'non_zero': True
        },
        'lr': {
            'label': 'Learning rate',
            'var_type': 'float',
            'default_val': 1e-3,
            'restr_positiv': True,
            'non_zero': True
        },
        'weight_decay': {
            'label': 'Weight decay',
            'var_type': 'float',
            'default_val': 0,
            'restr_positiv': True,
            'non_zero': False
        },
    },

    'Ajuste dirigido': {
        'query_steps': {
            'label': 'Pasos de muestreo',
            'var_type': 'int',
            'default_val': 5,
            'restr_positiv': True,
            'non_zero': True
        },
        'n_queries': {
            'label': 'Muestras por paso',
            'var_type': 'int',
            'default_val': 5,
            'restr_positiv': True,
            'non_zero': True
        }
    }
}

args_etapas['Ajuste dirigido'].update(args_etapas['Ajuste inicial'])

# Parámetros para el muestreo inicial
samp_args = {
    'coprime_sines': {
        'n_points': {
            'label': '# de muestras',
            'var_type': 'int',
            'default_val': 100,
            'restr_positiv': True,
            'non_zero': False
        },
        'densidad': {
            'label': 'densidad de ondas',
            'var_type': 'int',
            'default_val': 1,
            'restr_positiv': True,
            'non_zero': False
        },
        'base_frec': {
            'label': 'frecuencia base',
            'var_type': 'int',
            'default_val': 1,
            'restr_positiv': True,
            'non_zero': False
        }
    }
}
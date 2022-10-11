import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Pantalla, Label_Entry


class PantallaConfigAjuste(Pantalla):

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
            'lr': {
                'label': 'Learning rate',
                'var_type': 'float',
                'default_val': 1e-3,
                'restr_positiv': True,
                'non_zero': True
            },
            'batch_size': {
                'label': 'Batch size',
                'var_type': 'int',
                'default_val': 256,
                'restr_positiv': True,
                'non_zero': True
            }
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

    def __init__(self, parent):
        self.arg_getters = {etapa: {} for etapa in self.args_etapas.keys()}

        super().__init__(parent, titulo="Configurar aproximación")

    def definir_elementos(self):
        # Checkboxes para etapas de entrenamiento
        frame_checks = ttk.Frame(self)
        frame_checks.grid(column=0, row=0, sticky='nsew')

        titulo_checks = ttk.Label(frame_checks,
                                  text="Seleccionar etapas de ajuste",
                                  font=(12))
        titulo_checks.grid(column=0, row=0, padx=5, pady=5)

        self.check_vars = []
        for i, etapa in enumerate(self.args_etapas.keys()):
            check_var = tk.IntVar()
            check_but = ttk.Checkbutton(frame_checks, text=etapa,
                                        variable=check_var,
                                        command=self.actualizar_tabs)
            check_but.grid(column=0, row=i+1,
                           padx=5, pady=5, sticky='w')
            self.check_vars.append(check_var)

        # Configuración de cada etapa
        self.tabs_config = ttk.Notebook(self)
        self.tabs_config.grid(column=0, row=1, sticky='nsew')

        for etapa, args in self.args_etapas.items():
            frame = ttk.Frame(self.tabs_config)

            for i, (arg_name, entry_kwargs) in enumerate(args.items()):
                entry = Label_Entry(frame, **entry_kwargs)
                entry.grid(column=0, row=i, padx=5, pady=5)
                self.arg_getters[etapa][arg_name] = entry.get

            self.tabs_config.add(frame, text=etapa)

        # Ajuste inicial checado por default
        self.check_vars[1].set(1)
        self.actualizar_tabs()

        # Botones inferiores
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=0, row=2, sticky='we')
        frame_botones.columnconfigure(0, weight=1)

        boton_cancelar = ttk.Button(frame_botones,
                                    text="Cancelar",
                                    command=self.parent.regresar)
        boton_cancelar.grid(column=0, row=0, sticky='w')

        boton_ejecutar = ttk.Button(frame_botones,
                                    text="Aceptar",
                                    command=self.aceptar)
        boton_ejecutar.grid(column=1, row=0, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    def actualizar_tabs(self):
        for i, var in enumerate(self.check_vars):
            state = 'normal' if var.get()==1 else 'hidden'
            self.tabs_config.tab(i, state=state)

    def aceptar(self):
            train_kwargs = self.get_train_kwargs()
            if train_kwargs is not None:
                self.controlador.set_train_kwargs(train_kwargs)
                self.parent.avanzar()

    def get_train_kwargs(self):
        train_kwargs = {} # etapa: {} for etapa in self.args_etapas.keys()}
        for i, (etapa, args) in enumerate(self.arg_getters.items()):
            # Sólo añadir params de etapas seleccionadas
            if self.check_vars[i].get() == 1:
                train_kwargs[etapa] = {}
                for arg_name, get_fn in args.items():
                    arg_value = get_fn()
                    train_kwargs[etapa][arg_name] = get_fn()

                    if not bool(arg_value):
                        return None

        return train_kwargs


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaConfigAjuste(root)
    root.mainloop()
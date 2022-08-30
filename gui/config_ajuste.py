import tkinter as tk
from tkinter import ttk

from gui.robot_database import UIController
from gui.gui_utils import Label_Entry


class PantallaConfigAjuste(ttk.Frame):

    args_etapas = {
        'Meta ajuste': {
            'epochs': {
                'label': '# de épocas',
                'var_type': 'int',
                'default_val': 1000,
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
                'label': 'learning rate',
                'var_type': 'float',
                'default_val': 1e-3,
                'restr_positiv': True,
                'non_zero': True
            },
            'batch_size': {
                'label': 'batch size',
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

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Configurar aproximación")

        self.arg_getters = {etapa: {} for etapa in self.args_etapas.keys()}

        self.controlador = UIController()

        self.definir_elementos()

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

        # Comportamiento default: Check de a. inicial bloqueado en True
        #                         Tab de a. inicial por default
        #                         Desactivar tabs no usadas
        # tabs_config.select(1)
        # tabs_config.tab(0, state='hidden')
        # tabs_config.tab(2, state='hidden')

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
                                    text="Ejecutar",
                                    command=self.ejecutar)
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

    def ejecutar(self):
            train_kwargs = self.get_train_kwargs()
            if train_kwargs is not None:
                # self.controlador.set_train_kwargs(train_kwargs)
                self.parent.avanzar()

    def get_train_kwargs(self):
        train_kwargs = {etapa: {} for etapa in self.args_etapas.keys()}
        for etapa, args in self.arg_getters.items():
            for arg_name, get_fn in args.items():
                arg_value = get_fn()
                train_kwargs[etapa][arg_name] = get_fn()

                if not bool(arg_value):
                    return None

        return train_kwargs


if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('800x450+100+100')
    root.minsize(550,330)
    root.maxsize(1200,800)
    pant1 = PantallaConfigAjuste(root)
    root.mainloop()
import tkinter as tk
from tkinter import Label, ttk

from gui.gui_utils import Label_Entry


class PantallaConfigAjuste(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Parámetros de aproximación")

        self.model_args = {
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
                    'default_val': 3,
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

        self.train_args = {
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
            },
        }

        self.arg_getters = None
        self.t_arg_getters = {}

        self.definir_elementos()


    def definir_elementos(self):
        # Selección tipo de modelo
        frame_tipo = ttk.Frame(self)
        frame_tipo.grid(column=0, row=0, sticky='nsew')

        label_tipo = ttk.Label(frame_tipo, text="Tipo de modelo")
        label_tipo.grid(column=0, row=0)

        self.combo_model_cls = ttk.Combobox(frame_tipo, state='readonly')
        self.combo_model_cls.grid(column=0, row=1)
        self.combo_model_cls['values'] = list(self.model_args.keys())
        self.combo_model_cls.bind('<<ComboboxSelected>>', self.definir_panel_hparams)

        # Parámetros de modelo
        self.frame_mod_params = ttk.LabelFrame(self, text='Parámetros de modelo')
        self.frame_mod_params.grid(column=1, row=0, sticky='nsew')
        place_label = ttk.Label(self.frame_mod_params, text="\n\n")
        place_label.grid(column=0, row=0)

        # Parámetros de ajuste
        frame_t_params = ttk.LabelFrame(self, text='Parámetros de ajuste')
        frame_t_params.grid(column=1, row=1, sticky='nsew')

        for i, (arg_name, entry_kwargs) in enumerate(self.train_args.items()):
            entry = Label_Entry(frame_t_params, **entry_kwargs)
            entry.grid(column=0, row=i)
            self.t_arg_getters[arg_name] = entry.get

        # Botones inferiores
        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.destroy)
        boton_cancelar.grid(column=0, row=2)

        boton_cancelar = ttk.Button(self, text="Ejecutar",
                                    command=self.ejecutar)
        boton_cancelar.grid(column=1, row=2)

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

        for child in frame_t_params.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

    def definir_panel_hparams(self, event):
        for widget in self.frame_mod_params.winfo_children():
            widget.destroy()

        tipo_modelo = self.combo_model_cls.get()
        args = self.model_args[tipo_modelo]
        self.arg_getters = {}

        for i, (arg_name, entry_kwargs) in enumerate(args.items()):
            entry = Label_Entry(self.frame_mod_params,**entry_kwargs)
            entry.grid(column=0, row=i)
            self.arg_getters[arg_name] = entry.get

        f_act_label = ttk.Label(self.frame_mod_params, text="Función de activación")
        f_act_label.grid(column=0, row=len(args))
        f_act_combo = ttk.Combobox(self.frame_mod_params,state='readonly')
        f_act_combo.grid(column=1, row=len(args))
        f_act_combo['values'] = ('relu', 'tanh')
        f_act_combo.set('relu')

        self.arg_getters['activation'] = f_act_combo.get

        for child in self.frame_mod_params.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def ejecutar(self):
        #if self.combo_model_cls.get() != '':
        if self.arg_getters is not None:
            model_cls = self.combo_model_cls.get()
            model_kwargs = self.get_model_kwargs()
            train_kwargs = self.get_train_kwargs()
            if not (None in model_kwargs.values()):
                self.parent.set_model(model_cls, model_kwargs, train_kwargs)
                self.parent.avanzar(self.__class__)

    def get_model_kwargs(self):
        model_kwargs = {}
        for arg_name, get_fn in self.arg_getters.items():
            model_kwargs[arg_name] = get_fn()
        return model_kwargs

    def get_train_kwargs(self):
        train_kwargs = {}
        for arg_name, get_fn in self.t_arg_getters.items():
            train_kwargs[arg_name] = get_fn()
        return train_kwargs


if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('800x450+100+100')
    root.minsize(550,330)
    root.maxsize(1200,800)
    pant1 = PantallaConfigAjuste(root)
    root.mainloop()
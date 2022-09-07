import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Label_Entry
from autokin import modelos


class Popup_agregar_modelo(tk.Toplevel):

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

    def __init__(self, parent, callback):
        super().__init__()
        self.parent = parent
        self.callback = callback

        self.wm_title("Nuevo modelo")

        self.arg_getters = None

        self.definir_elementos()
        # Centrar pantalla
        x_pos = parent.winfo_rootx() + parent.winfo_width()//2 - 120
        y_pos = parent.winfo_rooty() + parent.winfo_height()//2 - 50
        self.geometry(f'+{x_pos}+{y_pos}')
        self.resizable(False, False)

    def definir_elementos(self):
        # Entrada para el nombre del modelo
        self.nom_entry = Label_Entry(self, label="Nombre", 
                                var_type='str', width=20)
        self.nom_entry.grid(column=0, row=0)

        # Selección tipo de modelo
        label_tipo = ttk.Label(self, text="Tipo de modelo")
        label_tipo.grid(column=0, row=1)

        self.combo_model_cls = ttk.Combobox(self, state='readonly')
        self.combo_model_cls.grid(column=1, row=1)
        self.combo_model_cls['values'] = list(self.model_args.keys())
        self.combo_model_cls.bind('<<ComboboxSelected>>', self.definir_panel_hparams)

        # Parámetros de modelo
        self.frame_mod_params = ttk.LabelFrame(self, text='Parámetros')
        self.frame_mod_params.grid(column=0, row=2,
                                   sticky='nsew', columnspan=2)
        # Para que el labelframe no esté vacío y sí aparezca
        place_label = ttk.Label(self.frame_mod_params,
                                text="\n      \n      ")
        place_label.grid(column=0, row=0)

        # Botones de abajo
        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.destroy)
        boton_cancelar.grid(column=0, row=3)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.ejecutar)
        boton_aceptar.grid(column=1, row=3)

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def definir_panel_hparams(self, event):
        # Producir automáticamente los widgets según el dict
        # de model_params para cada tipo de modelo
        for widget in self.frame_mod_params.winfo_children():
            widget.destroy()

        tipo_modelo = self.combo_model_cls.get()
        args = self.model_args[tipo_modelo]
        self.arg_getters = {}

        for i, (arg_name, entry_kwargs) in enumerate(args.items()):
            entry = Label_Entry(self.frame_mod_params,
                                width=10, **entry_kwargs)
            entry.grid(column=0, row=i)
            self.arg_getters[arg_name] = entry.get

        f_act_label = ttk.Label(self.frame_mod_params, text="Función de activación")
        f_act_label.grid(column=0, row=len(args))
        f_act_combo = ttk.Combobox(self.frame_mod_params,state='readonly',
                                   width=10)
        f_act_combo.grid(column=1, row=len(args))
        f_act_combo['values'] = ('relu', 'tanh')
        f_act_combo.set('relu')

        self.arg_getters['activation'] = f_act_combo.get

        for child in self.frame_mod_params.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def get_model_kwargs(self):
        model_kwargs = {}
        for arg_name, get_fn in self.arg_getters.items():
            model_kwargs[arg_name] = get_fn()
        return model_kwargs

    def ejecutar(self):
        nombre = self.nom_entry.get()
        if self.arg_getters is not None and nombre != '':
            model_kwargs = self.get_model_kwargs()
            if not (None in model_kwargs.values()):
                model_kwargs.update(cls_id=self.combo_model_cls.get())
                agregado = self.callback(nombre, model_kwargs)
                if agregado:
                    self.destroy()


if __name__ == '__main__':

    root = tk.Tk()
    root.minsize(550,330)
    root.maxsize(1200,800)

    win_width = 800
    win_height = 450
    x_pos = int(root.winfo_screenwidth()/2 - win_width/2)
    y_pos = int(root.winfo_screenheight()/2 - win_height/2)

    geom = f'{win_width}x{win_height}+{x_pos}+{y_pos}'
    root.geometry(geom)

    Popup_agregar_modelo(root, lambda x,y: True)
    root.mainloop()
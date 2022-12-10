import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Popup, Label_Entry
from gui.const import model_args


class Popup_agregar_modelo(Popup):

    def __init__(self, parent, callback):
        self.callback = callback
        self.arg_getters = None
        self.model_args = model_args
        super().__init__(title="Nuevo modelo", parent=parent)

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

        # Checkbutton para crear conjunto de modelos
        self.ensemble_var = tk.IntVar()
        check_but = ttk.Checkbutton(self,
                                    text="Conjunto de modelos",
                                    variable=self.ensemble_var,
                                    command=self.ensemble_check_fun)
        check_but.grid(column=0, row=2, columnspan=2, sticky='w')

        # Parámetros de modelo
        self.frame_mod_params = ttk.LabelFrame(self, text='Parámetros')
        self.frame_mod_params.grid(column=0, row=3,
                                   sticky='nsew', columnspan=2)
        # Para que el labelframe no esté vacío y sí aparezca
        place_label = ttk.Label(self.frame_mod_params,
                                text="\n      \n      ")
        place_label.grid(column=0, row=0)

        # Botones de abajo
        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.destroy)
        boton_cancelar.grid(column=0, row=4)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.ejecutar)
        boton_aceptar.grid(column=1, row=4)

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.bind('<Return>', self.ejecutar)

    def definir_panel_hparams(self, *args):
        # Producir automáticamente los widgets según el dict
        # de model_params para cada tipo de modelo
        for widget in self.frame_mod_params.winfo_children():
            widget.destroy()

        tipo_modelo = self.combo_model_cls.get()
        args = self.model_args[tipo_modelo].copy()
        self.arg_getters = {}
        self.entries = {}

        # Añadir parámetro de n_modelos para ensembles
        if self.ensemble_var.get():
            args.update({
                'n_modelos': {
                    'label': '# de modelos',
                    'var_type': 'int',
                    'default_val': 3,
                    'restr_positiv': True,
                    'non_zero': True
                }
            })

        # Selección de función de activación
        f_act_label = ttk.Label(self.frame_mod_params, text="Función de activación")
        f_act_label.grid(column=0, row=0, sticky='w')
        f_act_combo = ttk.Combobox(self.frame_mod_params,state='readonly',
                                   width=10)
        f_act_combo.grid(column=1, row=0)
        f_act_combo['values'] = ('relu', 'tanh')
        f_act_combo.set('relu')

        # Entradas para parámetros numéricos
        for i, (arg_name, entry_kwargs) in enumerate(args.items()):
            entry = Label_Entry(self.frame_mod_params,
                                width=10, **entry_kwargs)
            entry.grid(column=0, row=i+1)
            self.entries[arg_name] = entry
            self.arg_getters[arg_name] = entry.get

        self.arg_getters['activation'] = f_act_combo.get

        # Checkbox de propagación selectiva
        if self.ensemble_var.get():
            self.sel_prop_var = tk.IntVar()
            check_but = ttk.Checkbutton(self.frame_mod_params,
                                        text="Propagar según signo de dq",
                                        variable=self.sel_prop_var,
                                        command=self.sel_prop_check_fun)
            check_but.grid(column=0, row=len(args)+1, columnspan=2, sticky='w')

        for child in self.frame_mod_params.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def ensemble_check_fun(self):
        if self.combo_model_cls.get():
            self.definir_panel_hparams()

    def sel_prop_check_fun(self):
        state = 'disabled' if self.sel_prop_var.get() else 'normal'
        entry = self.entries['n_modelos']
        entry.entry['state'] = state

    def get_model_kwargs(self):
        model_kwargs = {}
        for arg_name, get_fn in self.arg_getters.items():
            model_kwargs[arg_name] = get_fn()
        return model_kwargs

    def ejecutar(self, *args):
        nombre = self.nom_entry.get()
        if self.arg_getters is not None and nombre != '':
            model_kwargs = self.get_model_kwargs()
            if not (None in model_kwargs.values()):

                cls_id = self.combo_model_cls.get()

                if self.ensemble_var.get():
                    if self.sel_prop_var.get():
                        cls_id += '_SPEnsemble'
                        n_modelos = 2**self.parent.controlador.robot_s.n
                        model_kwargs['n_modelos'] = n_modelos
                    else:
                        cls_id += '_Ensemble'

                model_kwargs.update(cls_id=cls_id)
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
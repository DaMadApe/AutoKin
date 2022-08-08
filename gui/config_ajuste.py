import tkinter as tk
from tkinter import Label, ttk

from gui.gui_utils import Label_Entry


class PantallaConfigAjuste(ttk.Frame):

    train_args = {
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

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Configurar aproximación")

        self.arg_getters = {}

        self.definir_elementos()

    def definir_elementos(self):
        # Parámetros de ajuste
        frame_t_params = ttk.LabelFrame(self, text='Parámetros de ajuste')
        frame_t_params.grid(column=1, row=1, sticky='nsew')

        for i, (arg_name, entry_kwargs) in enumerate(self.train_args.items()):
            entry = Label_Entry(frame_t_params, **entry_kwargs)
            entry.grid(column=0, row=i)
            self.arg_getters[arg_name] = entry.get

        # Botones inferiores
        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.parent.regresar)
        boton_cancelar.grid(column=0, row=2, sticky='w')

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

    def ejecutar(self):
            train_kwargs = self.get_train_kwargs()
            if not (None in train_kwargs.values()):
                self.controlador.set_train_kwargs(train_kwargs)
                self.parent.avanzar()

    def get_train_kwargs(self):
        train_kwargs = {}
        for arg_name, get_fn in self.arg_getters.items():
            train_kwargs[arg_name] = get_fn()
        return train_kwargs


if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('800x450+100+100')
    root.minsize(550,330)
    root.maxsize(1200,800)
    pant1 = PantallaConfigAjuste(root)
    root.mainloop()